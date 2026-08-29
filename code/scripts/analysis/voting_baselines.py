"""D8 —— 投票基线后处理（协议 §7 基线 4–6）

对 `results/g0_environment_grid/` 已落盘的 G0 网格产物做**纯后处理**，计算协议 §7 中
从未被计算过的三条 source-only 基线，并按协议 §10 出指标。**不重新训练任何模型**、
不读取任何原始特征、不使用任何随机数。

    §7.4  hard voting              —— 三基模型 predictions.csv 多数票
    §7.5  等权 soft voting         —— 三基模型 pred_proba.csv 等权平均后 argmax
    §7.6  校准后 soft voting       —— 每个基模型先在**源域** OOF 上拟合 temperature，
                                      施加于测试 pred_proba 后等权平均 argmax

执行口径见 `docs/EXECUTION_PLAN_20260829.md` 决策 D8。本脚本不做任何设计决策：
所有口径均取协议原文或 D8 原文，凡协议未定死之处（平局规则、温度参数化、优化器边界）
一律在下方写明为**执行细节**并落盘审计。

--------------------------------------------------------------------------------
1. 输入（全部为 G0 已落盘产物，只读）
--------------------------------------------------------------------------------

    results/g0_environment_grid/summary_metrics.csv
        网格元数据唯一来源：task / task_type / n_sources / target_env /
        grid_kind / split_mode。本脚本**不重新推导**这些字段（§11 唯一实现纪律），
        只与各任务 metrics.json 的 train_rounds/test_rounds/task_type 交叉校验。

    results/g0_environment_grid/raw_all/<task>/all_features/<model>/
        predictions.csv   —— 含 true_label / predicted_label（测试集，逐窗口）
        pred_proba.csv    —— 含 proba_<class>（5 列，行和为 1）与 true_label
        metrics.json      —— 含 macro_f1 / accuracy / per_class_f1（自检基准）
        stacking/oof_meta.csv —— 含 oof_<model>_<class>（源域 OOF 概率）与
                                 true_label（**训练标签**，无需从特征表重建）

    行对齐：同一任务四个模型的 predictions.csv / pred_proba.csv 行数、行序
    （source_file, round, window_id, window_start）与 true_label 全部逐行一致；
    本脚本每任务重新断言一次，不一致即报错退出。

--------------------------------------------------------------------------------
2. 类别轴与概率列序
--------------------------------------------------------------------------------

    LABELS = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']

    与 `code/configs/research_experiments.json` 的 `labels`（`metric_summary` 的
    labels 参数）逐字一致。pred_proba.csv / oof_meta.csv 的概率列**列名自带类别名**
    （`proba_<cls>` / `oof_<model>_<cls>`），本脚本按列名取列，不依赖列位置；
    读入时校验列名齐全，缺列即报错退出。

--------------------------------------------------------------------------------
3. 平局规则（确定性，执行细节）
--------------------------------------------------------------------------------

hard voting（3 票 5 类）只有两种局面：

  (a) 某类得 2 或 3 票 —— 胜者唯一，概率不参与，就是该类；
  (b) 三个模型各投一类（1-1-1）—— 三个候选并列。

  **平局规则**：在并列（各得 1 票）的类中取**三模型平均概率**（未校准的原始
  pred_proba 等权平均，即 §7.5 的同一张概率表）最大者。若该平均概率仍完全相等
  （浮点意义下的精确相等），取类别轴上**字典序最靠前**者。

  实现为 `argmax(votes + 0.5 * mean_proba)`：votes ∈ {0,1,2,3} 相差至少 1，
  而 0.5 * mean_proba ∈ [0, 0.5]，故票数差永远压过概率项 —— 该式与上述规则
  逐字等价，且 numpy argmax 在完全相等时返回最小下标（= 字典序最靠前的类）。

soft voting（§7.5 / §7.6）的 argmax 平局同样由 numpy argmax 取最小下标解决，
即类别轴字典序最靠前者。三处平局规则统一，全流程无随机数。

--------------------------------------------------------------------------------
4. 校准（§7.6）：temperature scaling —— 参数化、优化与数据边界
--------------------------------------------------------------------------------

协议 §7.6 允许 temperature 或 isotonic，**D8 选 temperature**（执行细节，记于本
docstring 与 VOTING_BASELINES_NOTE.md）。

**伪 logit（近似）**：G0 落盘的是概率而非 logit，故取

        z = log(p + 1e-12)

作为伪 logit。这是一个**近似**：真实 logit 只在相差一个逐行常数的意义下与 log p
相等，而 softmax 对逐行常数不变，因此当 p 是行和为 1 的 softmax 输出时该近似是
精确的；但 RF 的概率是叶节点投票频率、并非任何 softmax 的输出，对它而言
"log p 是 logit" 只是一个工作假设。1e-12 的偏置使 p = 0 的项映到 -27.63 而非
-inf，这本身也改变了该项相对其它项的间距。两点都记为已知近似。

**参数化**：等价地写成幂缩放。令 w = 1/T，则

        softmax(z / T) = softmax(w · log p) = p^w / Σ_c p_c^w

    w = 1（T = 1）**精确还原**未校准概率，故 §7.6 在 T = 1 处退化为 §7.5。
    w < 1（T > 1）平滑（治过自信），w > 1（T < 1）锐化。

**拟合**：以训练标签的多类交叉熵（NLL）为目标，对 u = log w 在
[log 1e-2, log 1e2]（即 T ∈ [0.01, 100]）上做有界 Brent 一维搜索
（`scipy.optimize.minimize_scalar(method="bounded", xatol=1e-10)`）。
NLL 对 w 是凸的（固定伪 logit 上的单权重 softmax 回归），经 w = e^u 单调重参数化
后保持单峰，故有界 Brent 稳定且确定性。边界命中会被标记并落盘。
每个任务 × 每个基模型独立拟合一个标量 T（共 162 × 3 个）。

**数据边界（硬约束，§7「校准只在源域内做」/ §8.2 / §9.2）**：

    拟合温度**只**读 stacking/oof_meta.csv 的两样东西 ——
        · 源域 OOF 概率 oof_<model>_<cls>（按 §9.1 分组 CV 生成，无相邻窗口泄漏）
        · 同文件的 true_label（= **训练**轮次标签）
    测试集的任何标签**只**在最后一步 metric_row() 打分时出现，绝不进入温度拟合。
    代码上由 `fit_temperature()` 的签名结构性保证：它只接受 OOF 概率矩阵与训练
    标签，函数体内无任何测试集对象可达。
    逐任务逐模型的 T、NLL(前/后)、样本数、是否触界落盘为
    `voting_calibration_temperatures.csv`，供审计。

--------------------------------------------------------------------------------
5. 指标（协议 §10）
--------------------------------------------------------------------------------

    macro_f1_5class   主指标 1 —— 5 类 macro-F1，labels 固定为上述 5 类、
                      average='macro'、zero_division=0，与
                      `robust_iot_research.metric_summary` 逐字同口径
    macro_f1_4class   主指标 2 —— 去 Socket 后 4 类的 macro-F1。口径 = 在**完整
                      测试集**上算 5 个逐类 F1，再对 4 个非 Socket 类取无权平均
                      （等价于 sklearn 传 labels=非 Socket 四类、average='macro'；
                      sklearn 的 labels 只选取要平均的类，不删样本，故真值为
                      Socket 的样本仍作为其它类的 FP 计入）。**不**过滤测试样本。
    worst_class_f1    主指标 3 —— 5 个逐类 F1 的最小值。同时给出去 Socket 后
                      4 类的最小值 worst_class_f1_4class（Socket 恒为 1.0 时两者
                      相等；分列以免"最差类"在 5 类/4 类两种读法上产生歧义）。
    accuracy          辅助指标
    gain_vs_best_base 主指标 4 的一般化 —— 本行 macro_f1_5class 减去同任务三个
                      基模型 macro_f1_5class 的最大值。method='stacking' 行即为
                      §10 定义的 stacking gain。

--------------------------------------------------------------------------------
6. 输出的 method 取值
--------------------------------------------------------------------------------

    method_kind='base_model' : rf / xgboost / lightgbm         （§7.1 及其余两个基模型）
    method_kind='stacking'   : stacking                        （§7.3）
    method_kind='reference'  : best_base_posthoc               （§7.2，事后上界；
                               按 macro_f1_5class 选，并列取模型名字典序最靠前，
                               所选模型记于 selected_base_model 列）
    method_kind='voting'     : hard_voting / soft_voting_equal /
                               soft_voting_calibrated          （§7.4 / §7.5 / §7.6）

前三类由已落盘产物按**同一套**指标函数重算，只为让投票行与对照行可直接相减；
它们不是新结果，其 macro_f1_5class 已由第 7 节的自检逐一对齐 metrics.json。

--------------------------------------------------------------------------------
7. 管道自检（硬门，D8 验收）
--------------------------------------------------------------------------------

出任何产物之前，对 162 任务 × 4 模型从 predictions.csv 重推
    · 5-class macro-F1  与 metrics.json 的 macro_f1
    · accuracy          与 metrics.json 的 accuracy
    · 5 个逐类 F1       与 metrics.json 的 per_class_f1
逐一比对，容差 1e-6（绝对）。任一不过 → 立即退出、**不写任何产物**。
（后两项是 4-class / 最差类 F1 的直接依据，故一并纳入硬门。）

--------------------------------------------------------------------------------
8. 输出
--------------------------------------------------------------------------------

    results/g0_environment_grid/voting_baselines.csv
    results/g0_environment_grid/voting_calibration_temperatures.csv
    results/g0_environment_grid/voting_baselines_run_metadata.json   （§19.2）

--------------------------------------------------------------------------------
9. 用法
--------------------------------------------------------------------------------

    python code/scripts/analysis/voting_baselines.py                 # 全量 162 任务
    python code/scripts/analysis/voting_baselines.py --selfcheck-only
    python code/scripts/analysis/voting_baselines.py --limit 4 --dry-run   # smoke

本脚本是**单进程**的（读 CSV + numpy + 一维标量优化），并在导入 numpy 前把各
BLAS 线程数钉为 1 —— 服务器上有 E1 网格与深度模型训练在跑，不得抢 CPU。
"""
from __future__ import annotations

import os

# 必须在 import numpy 之前：G0 服务器上有训练任务在跑，本后处理限制为单线程。
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
GRID_ROOT = REPO_ROOT / "results" / "g0_environment_grid"
RAW_ROOT = GRID_ROOT / "raw_all"
SUMMARY_CSV = GRID_ROOT / "summary_metrics.csv"

OUT_CSV = GRID_ROOT / "voting_baselines.csv"
OUT_CALIB_CSV = GRID_ROOT / "voting_calibration_temperatures.csv"
OUT_META_JSON = GRID_ROOT / "voting_baselines_run_metadata.json"

# 类别轴：与 code/configs/research_experiments.json 的 "labels" 逐字一致。
LABELS = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]
NON_SOCKET = [c for c in LABELS if c != "Socket"]
BASE_MODELS = ["rf", "xgboost", "lightgbm"]
ALL_MODELS = BASE_MODELS + ["stacking"]
FEATURE_SET = "all_features"
ENCODING = "utf-8-sig"

# 自检容差（D8 验收：1e-6）
SELFCHECK_TOL = 1e-6

# 校准执行细节（见 docstring §4）
PROBA_EPS = 1e-12          # 伪 logit z = log(p + PROBA_EPS)
TEMP_BOUNDS = (1e-2, 1e2)  # T 的搜索区间；等价 w = 1/T ∈ [1e-2, 1e2]
BRENT_XATOL = 1e-10        # 对 u = log w 的绝对收敛容差

KEY_COLS = ["source_file", "round", "window_id", "window_start"]
PROBA_COLS = [f"proba_{c}" for c in LABELS]


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------
def load_grid_metadata() -> pd.DataFrame:
    """从 summary_metrics.csv 取每任务一行的网格元数据（§11：不重新推导）。"""
    df = pd.read_csv(SUMMARY_CSV, encoding=ENCODING)
    need = [
        "task", "task_type", "train_rounds", "test_rounds", "train_samples",
        "test_samples", "model", "feature_set", "n_sources", "target_env",
        "grid_kind", "split_mode",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[STOP] summary_metrics.csv 缺列: {missing}")
    df = df[df["feature_set"] == FEATURE_SET]
    meta_cols = ["task_type", "train_rounds", "test_rounds", "train_samples",
                 "test_samples", "n_sources", "target_env", "grid_kind", "split_mode"]
    # 同一任务的 4 个模型行必须给出同一份元数据
    nun = df.groupby("task")[meta_cols].nunique()
    bad = nun[(nun > 1).any(axis=1)]
    if len(bad):
        raise SystemExit(f"[STOP] summary_metrics.csv 同任务模型行元数据不一致:\n{bad}")
    out = df.drop_duplicates("task").set_index("task")[meta_cols]
    return out


def load_task(task: str) -> dict:
    """读单任务的全部输入并断言结构假设。只读，不写。"""
    tdir = RAW_ROOT / task / FEATURE_SET
    preds, probas, metrics = {}, {}, {}
    for m in ALL_MODELS:
        mdir = tdir / m
        p = pd.read_csv(
            mdir / "predictions.csv",
            encoding=ENCODING,
            usecols=KEY_COLS + ["true_label", "predicted_label"],
        )
        q = pd.read_csv(mdir / "pred_proba.csv", encoding=ENCODING)
        missing = [c for c in PROBA_COLS if c not in q.columns]
        if missing:
            raise SystemExit(f"[STOP] {task}/{m}/pred_proba.csv 缺概率列: {missing}")
        with (mdir / "metrics.json").open(encoding="utf-8") as f:
            metrics[m] = json.load(f)
        preds[m], probas[m] = p, q

    ref = preds[BASE_MODELS[0]]
    n = len(ref)
    for m in ALL_MODELS:
        if len(preds[m]) != n or len(probas[m]) != n:
            raise SystemExit(
                f"[STOP] {task}/{m} 行数不一致: predictions={len(preds[m])} "
                f"pred_proba={len(probas[m])} ref={n}"
            )
        if not preds[m][KEY_COLS].reset_index(drop=True).equals(
            ref[KEY_COLS].reset_index(drop=True)
        ):
            raise SystemExit(f"[STOP] {task}/{m} predictions.csv 行序与 rf 不一致")
        if not probas[m][KEY_COLS].reset_index(drop=True).equals(
            preds[m][KEY_COLS].reset_index(drop=True)
        ):
            raise SystemExit(f"[STOP] {task}/{m} pred_proba.csv 与 predictions.csv 行序不一致")
        if not (probas[m]["true_label"].to_numpy() == preds[m]["true_label"].to_numpy()).all():
            raise SystemExit(f"[STOP] {task}/{m} pred_proba 与 predictions 的 true_label 不一致")
        if not (preds[m]["true_label"].to_numpy() == ref["true_label"].to_numpy()).all():
            raise SystemExit(f"[STOP] {task}/{m} true_label 与 rf 不一致")

    P = {m: probas[m][PROBA_COLS].to_numpy(dtype=float) for m in ALL_MODELS}
    for m in ALL_MODELS:
        s = P[m].sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-6):
            raise SystemExit(
                f"[STOP] {task}/{m} pred_proba 行和不为 1（min={s.min()} max={s.max()}）"
            )

    oof = pd.read_csv(tdir / "stacking" / "oof_meta.csv", encoding=ENCODING)
    if "true_label" not in oof.columns:
        raise SystemExit(
            f"[STOP] {task}/stacking/oof_meta.csv 无 true_label 列 —— "
            "D8 规格要求此时改由任务定义重建训练标签，属设计决策，停并报告。"
        )
    n_train = int(metrics["stacking"]["train_samples"])
    if len(oof) != n_train:
        raise SystemExit(
            f"[STOP] {task} oof_meta 行数 {len(oof)} != train_samples {n_train}"
        )
    oof_p = {}
    for m in BASE_MODELS:
        cols = [f"oof_{m}_{c}" for c in LABELS]
        missing = [c for c in cols if c not in oof.columns]
        if missing:
            raise SystemExit(f"[STOP] {task}/stacking/oof_meta.csv 缺列: {missing}")
        Q = oof[cols].to_numpy(dtype=float)
        s = Q.sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-6):
            raise SystemExit(
                f"[STOP] {task} oof_{m} 行和不为 1（min={s.min()} max={s.max()}）"
            )
        oof_p[m] = Q

    y_oof = oof["true_label"].to_numpy()
    unknown = sorted(set(y_oof) - set(LABELS))
    if unknown:
        raise SystemExit(f"[STOP] {task} oof true_label 出现未知类别: {unknown}")
    y_test = ref["true_label"].to_numpy()
    unknown = sorted(set(y_test) - set(LABELS))
    if unknown:
        raise SystemExit(f"[STOP] {task} 测试 true_label 出现未知类别: {unknown}")

    return {
        "task": task,
        "y_test": y_test,
        "pred_label": {m: preds[m]["predicted_label"].to_numpy() for m in ALL_MODELS},
        "proba": P,
        "oof_proba": oof_p,
        "y_oof": y_oof,
        "metrics": metrics,
        "n_test": n,
        "n_oof": len(oof),
    }


# --------------------------------------------------------------------------
# 指标（协议 §10）
# --------------------------------------------------------------------------
def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """§10 指标。逐类 F1 的口径与 robust_iot_research.metric_summary 逐字一致。"""
    _, _, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    per = {c: float(v) for c, v in zip(LABELS, per_f1)}
    ns = [per[c] for c in NON_SOCKET]
    row = {
        "macro_f1_5class": float(np.mean(per_f1)),
        "macro_f1_4class": float(np.mean(ns)),
        "worst_class_f1": float(np.min(per_f1)),
        "worst_class_f1_4class": float(np.min(ns)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for c in LABELS:
        row[f"f1_{c}"] = per[c]
    return row


# --------------------------------------------------------------------------
# 自检（硬门）
# --------------------------------------------------------------------------
def selfcheck_task(td: dict) -> list[dict]:
    """从 predictions.csv 重推指标并与 metrics.json 比对（容差 1e-6）。"""
    rows = []
    for m in ALL_MODELS:
        got = metric_row(td["y_test"], td["pred_label"][m])
        ref = td["metrics"][m]
        d_f1 = abs(got["macro_f1_5class"] - float(ref["macro_f1"]))
        d_acc = abs(got["accuracy"] - float(ref["accuracy"]))
        d_per = max(
            abs(got[f"f1_{c}"] - float(ref["per_class_f1"][c])) for c in LABELS
        )
        rows.append({
            "task": td["task"], "model": m,
            "recomputed_macro_f1": got["macro_f1_5class"],
            "metrics_json_macro_f1": float(ref["macro_f1"]),
            "abs_diff_macro_f1": d_f1,
            "abs_diff_accuracy": d_acc,
            "abs_diff_max_per_class_f1": d_per,
            "passed": bool(max(d_f1, d_acc, d_per) <= SELFCHECK_TOL),
        })
    return rows


# --------------------------------------------------------------------------
# 投票基线（协议 §7.4 / §7.5 / §7.6）
# --------------------------------------------------------------------------
def _idx(labels: np.ndarray) -> np.ndarray:
    lut = {c: i for i, c in enumerate(LABELS)}
    return np.array([lut[v] for v in labels], dtype=int)


def hard_voting(td: dict) -> np.ndarray:
    """§7.4。多数票；1-1-1 平局取并列类中三模型平均概率最大者（见 docstring §3）。"""
    n = td["n_test"]
    votes = np.zeros((n, len(LABELS)), dtype=float)
    for m in BASE_MODELS:
        votes[np.arange(n), _idx(td["pred_label"][m])] += 1.0
    mean_proba = np.mean([td["proba"][m] for m in BASE_MODELS], axis=0)
    # votes 相差 >= 1，0.5 * mean_proba ∈ [0, 0.5] → 票数永远压过概率项。
    return np.array(LABELS)[np.argmax(votes + 0.5 * mean_proba, axis=1)]


def soft_voting_equal(td: dict) -> np.ndarray:
    """§7.5。三基模型概率等权平均后 argmax。"""
    mean_proba = np.mean([td["proba"][m] for m in BASE_MODELS], axis=0)
    return np.array(LABELS)[np.argmax(mean_proba, axis=1)]


def _nll(Z: np.ndarray, y_idx: np.ndarray, w: float) -> float:
    S = Z * w
    S = S - S.max(axis=1, keepdims=True)
    lse = np.log(np.exp(S).sum(axis=1))
    return float(np.mean(lse - S[np.arange(len(y_idx)), y_idx]))


def fit_temperature(oof_proba: np.ndarray, y_oof_idx: np.ndarray) -> dict:
    """§7.6 温度拟合。**只**接受源域 OOF 概率与训练标签；函数体内测试集不可达。

    伪 logit z = log(p + 1e-12)，对 w = 1/T 做幂缩放 softmax(w·z) = p^w / Σ p^w，
    在 u = log w 上做有界 Brent 搜索最小化训练 NLL。w = 1 精确还原未校准概率。
    """
    Z = np.log(oof_proba + PROBA_EPS)
    lo, hi = np.log(1.0 / TEMP_BOUNDS[1]), np.log(1.0 / TEMP_BOUNDS[0])

    def obj(u: float) -> float:
        return _nll(Z, y_oof_idx, float(np.exp(u)))

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                          options={"xatol": BRENT_XATOL})
    u = float(res.x)
    w = float(np.exp(u))
    return {
        "temperature": 1.0 / w,
        "w_inv_temperature": w,
        "nll_uncalibrated": _nll(Z, y_oof_idx, 1.0),
        "nll_calibrated": float(res.fun),
        "n_oof": int(len(y_oof_idx)),
        "optimizer_converged": bool(res.success),
        "bound_hit": bool(min(abs(u - lo), abs(u - hi)) < 1e-6),
    }


def apply_temperature(proba: np.ndarray, w: float) -> np.ndarray:
    Z = np.log(proba + PROBA_EPS) * w
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def soft_voting_calibrated(td: dict) -> tuple[np.ndarray, list[dict]]:
    """§7.6。逐基模型在源域 OOF 上拟合 T，施加于测试概率后等权平均 argmax。"""
    y_oof_idx = _idx(td["y_oof"])
    calib_rows, cal_probas = [], []
    for m in BASE_MODELS:
        fit = fit_temperature(td["oof_proba"][m], y_oof_idx)   # 只见 OOF + 训练标签
        calib_rows.append({"task": td["task"], "model": m, **fit})
        cal_probas.append(apply_temperature(td["proba"][m], fit["w_inv_temperature"]))
    mean_proba = np.mean(cal_probas, axis=0)
    return np.array(LABELS)[np.argmax(mean_proba, axis=1)], calib_rows


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_rows(td: dict, meta: pd.Series) -> tuple[list[dict], list[dict]]:
    base = {
        "task": td["task"],
        "task_type": meta["task_type"],
        "grid_kind": meta["grid_kind"],
        "split_mode": meta["split_mode"],
        "n_sources": int(meta["n_sources"]),
        "target_env": meta["target_env"],
        "train_rounds": meta["train_rounds"],
        "test_rounds": meta["test_rounds"],
        "n_train": int(meta["train_samples"]),
        "n_test": int(td["n_test"]),
        "n_oof": int(td["n_oof"]),
    }
    y = td["y_test"]
    preds = {m: td["pred_label"][m] for m in ALL_MODELS}
    preds["hard_voting"] = hard_voting(td)
    preds["soft_voting_equal"] = soft_voting_equal(td)
    preds["soft_voting_calibrated"], calib_rows = soft_voting_calibrated(td)

    scored = {k: metric_row(y, v) for k, v in preds.items()}
    best_base_f1 = max(scored[m]["macro_f1_5class"] for m in BASE_MODELS)
    # §7.2 事后最佳单基模型：按 macro_f1_5class 选，并列取模型名字典序最靠前。
    best_base = min(
        m for m in BASE_MODELS if scored[m]["macro_f1_5class"] == best_base_f1
    )

    kind = {
        "rf": "base_model", "xgboost": "base_model", "lightgbm": "base_model",
        "stacking": "stacking",
        "best_base_posthoc": "reference",
        "hard_voting": "voting", "soft_voting_equal": "voting",
        "soft_voting_calibrated": "voting",
    }
    order = ALL_MODELS + ["best_base_posthoc", "hard_voting",
                          "soft_voting_equal", "soft_voting_calibrated"]
    scored["best_base_posthoc"] = dict(scored[best_base])

    rows = []
    for name in order:
        r = dict(base)
        r["method"] = name
        r["method_kind"] = kind[name]
        r["selected_base_model"] = best_base if name == "best_base_posthoc" else ""
        r.update(scored[name])
        r["gain_vs_best_base"] = r["macro_f1_5class"] - best_base_f1
        rows.append(r)
    return rows, calib_rows


def git_hash() -> dict:
    try:
        h = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": h, "dirty": bool(dirty)}
    except Exception as exc:  # pragma: no cover
        return {"commit": None, "dirty": None, "error": str(exc)}


def main() -> None:
    global GRID_ROOT, RAW_ROOT, SUMMARY_CSV, OUT_CSV, OUT_CALIB_CSV, OUT_META_JSON

    ap = argparse.ArgumentParser(description="D8 投票基线后处理（协议 §7 基线 4–6）")
    ap.add_argument("--grid-root", default=str(GRID_ROOT))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务（smoke）")
    ap.add_argument("--selfcheck-only", action="store_true", help="只跑硬门自检")
    ap.add_argument("--dry-run", action="store_true", help="算但不写盘")
    args = ap.parse_args()

    GRID_ROOT = Path(args.grid_root)
    RAW_ROOT = GRID_ROOT / "raw_all"
    SUMMARY_CSV = GRID_ROOT / "summary_metrics.csv"
    OUT_CSV = GRID_ROOT / "voting_baselines.csv"
    OUT_CALIB_CSV = GRID_ROOT / "voting_calibration_temperatures.csv"
    OUT_META_JSON = GRID_ROOT / "voting_baselines_run_metadata.json"

    t0 = time.time()
    meta_df = load_grid_metadata()
    tasks = sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir())
    missing = sorted(set(tasks) - set(meta_df.index))
    if missing:
        raise SystemExit(f"[STOP] summary_metrics.csv 缺任务: {missing}")
    extra = sorted(set(meta_df.index) - set(tasks))
    if extra:
        raise SystemExit(f"[STOP] summary_metrics.csv 有目录不存在的任务: {extra}")
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[D8] tasks={len(tasks)}  grid_root={GRID_ROOT}")

    # ---- 单遍：逐任务自检 + 投票，全部只在内存里；硬门通过前不写任何产物 ----
    check_rows, rows, calib = [], [], []
    for i, t in enumerate(tasks, 1):
        td = load_task(t)
        m = meta_df.loc[t]
        if ast.literal_eval(str(m["train_rounds"])) != td["metrics"]["rf"]["train_rounds"]:
            raise SystemExit(f"[STOP] {t} train_rounds: summary 与 metrics.json 不一致")
        if ast.literal_eval(str(m["test_rounds"])) != td["metrics"]["rf"]["test_rounds"]:
            raise SystemExit(f"[STOP] {t} test_rounds: summary 与 metrics.json 不一致")
        if str(m["task_type"]) != td["metrics"]["rf"]["task_type"]:
            raise SystemExit(f"[STOP] {t} task_type: summary 与 metrics.json 不一致")
        check_rows.extend(selfcheck_task(td))
        if not args.selfcheck_only:
            r, c = build_rows(td, m)
            rows.extend(r)
            calib.extend(c)
        del td
        if i % 20 == 0 or i == len(tasks):
            print(f"  [{i}/{len(tasks)}] {t}", flush=True)

    chk = pd.DataFrame(check_rows)
    n_fail = int((~chk["passed"]).sum())
    print(f"[D8] selfcheck: {len(chk)} 项（{len(tasks)} 任务 × {len(ALL_MODELS)} 模型），"
          f"失败 {n_fail}，最大 |Δmacro_f1| = {chk['abs_diff_macro_f1'].max():.3e}，"
          f"最大 |Δaccuracy| = {chk['abs_diff_accuracy'].max():.3e}，"
          f"最大 |Δper_class_f1| = {chk['abs_diff_max_per_class_f1'].max():.3e}")
    if n_fail:
        print(chk[~chk["passed"]].to_string())
        raise SystemExit(
            f"[STOP] 管道自检未通过（{n_fail} 项超出容差 {SELFCHECK_TOL}）——不写任何产物。"
        )
    if args.selfcheck_only:
        print("[D8] --selfcheck-only：自检通过，退出。")
        return

    out = pd.DataFrame(rows)
    cal = pd.DataFrame(calib)
    elapsed = time.time() - t0

    if args.dry_run:
        print(out.groupby("method")["macro_f1_5class"].mean().to_string())
        print("[D8] --dry-run：不写盘。")
        return

    out.to_csv(OUT_CSV, index=False, encoding=ENCODING)
    cal.to_csv(OUT_CALIB_CSV, index=False, encoding=ENCODING)
    meta = {
        "script": "code/scripts/analysis/voting_baselines.py",
        "protocol_refs": ["§7.4", "§7.5", "§7.6", "§10", "§19.2"],
        "execution_plan": "docs/EXECUTION_PLAN_20260829.md D8",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "command_line": " ".join([sys.executable] + sys.argv),
        "cwd": str(Path.cwd()),
        "random_seed": None,
        "determinism": "no RNG anywhere; all ties broken deterministically",
        "inputs": {
            "grid_root": str(GRID_ROOT),
            "summary_metrics": str(SUMMARY_CSV),
            "n_tasks": len(tasks),
            "models_read": ALL_MODELS,
        },
        "outputs": {
            "voting_baselines_csv": str(OUT_CSV),
            "calibration_csv": str(OUT_CALIB_CSV),
            "rows": int(len(out)),
        },
        "labels": LABELS,
        "calibration": {
            "family": "temperature scaling (power scaling on pseudo-logits)",
            "pseudo_logit": "z = log(p + 1e-12)",
            "parametrization": "softmax(w*z) = p^w / sum(p^w), w = 1/T; w=1 == uncalibrated",
            "objective": "multiclass NLL on source-domain OOF probs + training labels",
            "optimizer": "scipy minimize_scalar(bounded Brent) over u = log(w)",
            "temperature_bounds": list(TEMP_BOUNDS),
            "xatol": BRENT_XATOL,
            "data_boundary": "fit sees ONLY stacking/oof_meta.csv (source-domain OOF "
                             "probabilities + training labels); test labels enter only "
                             "at final scoring",
        },
        "selfcheck": {
            "tolerance": SELFCHECK_TOL,
            "n_checks": int(len(chk)),
            "n_failed": n_fail,
            "max_abs_diff_macro_f1": float(chk["abs_diff_macro_f1"].max()),
            "max_abs_diff_accuracy": float(chk["abs_diff_accuracy"].max()),
            "max_abs_diff_per_class_f1": float(chk["abs_diff_max_per_class_f1"].max()),
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_env": {v: os.environ.get(v) for v in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    with OUT_META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[D8] wrote {OUT_CSV}  ({len(out)} rows)")
    print(f"[D8] wrote {OUT_CALIB_CSV}  ({len(cal)} rows)")
    print(f"[D8] wrote {OUT_META_JSON}")
    print(f"[D8] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
