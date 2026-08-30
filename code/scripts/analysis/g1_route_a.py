#!/usr/bin/env python3
"""G1 路线 A —— `UDS` 驱动的风险选择器（协议 §17.1 / §9.2 / §4.2 / §11）。

本脚本**只产出数表**。§17.1 五条判据的通过 / 不通过判定由审阅方作出：
本文件及其全部产物不含任何解读、机制语言、判定或结论性表述。

协议依据
--------
* §17.1：G1 路线 A 的候选动作（RF / Stacking / soft voting）、`selection regret`
  定义、五条判据；"不得调整指标或阈值来掩盖失败"；
* §9.2：嵌套 LOEO —— 内层学阈值 / 选择器、外层每环境只评估一次；
* §4.2 / §11：`UDS` 的定义与**唯一实现** `cpd_core.uds`（本文件不含任何私有副本）；
* §10：主指标 = 5-class macro-F1（历史口径），`selection regret` 为第 5 项主指标；
* §15.1 / §15.4：聚合主口径 = 按目标环境等权；六个目标环境逐一完整报告；
* §19.2：git HEAD / 命令行 / 种子 / 包版本 / 输入清单持久化。

执行口径：`docs/EXECUTION_PLAN_20260829.md` 决策 **D10**（v1.4，先写后看）
+ **D10 追记 v1.5**（commit `d447323`，运行前修订条件 4 口径）：条件 4 的判定口径为
"存在 ≥1 个外层任务，选择 ≠ RF 且 `regret ≤ τ_repro = 2e-3`"（τ_repro 取自登记表
E1-G0-GRID 行实测的跨线程拓扑 stacking 复现界）；`τ = 0` 的原严值计数照旧并列报告。

口径要点（逐条对应 D10）
----------------------
任务域
    G0 网格的 **150 个 OOD 任务**。任务定义唯一来源 =
    `environment_grid_experiment.build_task_grid()`（§11 唯一实现纪律），
    并与各任务 `metrics.json` 的 `train_rounds` / `test_rounds` 交叉校验。

嵌套结构
    外层 fold `e_out ∈ {R2..R7}`：外层评估任务 = `target_env == e_out` 的 25 个
    （每任务只评一次）；内层池 = **严格全含任务** —— `e_out` 既不作源也不作目标
    （5 目标 × 14 源组合 = 70 任务）。fold 结构以 `assert` 自检。

候选动作（3 个）
    `rf` / `stacking`（G0 落盘 = §9.1 grouped OOF，即 E1 的 B 口径）/ `soft_voting`。
    soft voting 的变体（`soft_voting_equal` / `soft_voting_calibrated`）由**内层**
    与阈值联合选定（§8.3）。**全部候选 F1 一律读落盘值，本文件不重算任何 F1**：
    `rf` / `stacking` 取各任务 `metrics.json` 的 `macro_f1`；两个 soft 变体取
    D8 `voting_baselines.csv` 的 `macro_f1_5class`。

UDS
    只经 `cpd_core.uds`。源侧 = stacking `oof_meta.csv` 三基模型 OOF 概率各自
    `argmax`；目标侧 = 三基模型 `predictions.csv` 的 `predicted_label`。
    6 个**有序**模型对的分歧矩阵 off-diag Frobenius 源-目标差取均值。
    计算路径**不读取任何标签列**（§17.1 条件 5；审计见 `g1_leakage_audit.md`）。

选择器（唯一预注册家族）
    `UDS` 单调双阈值，候选按风险序 `[stacking, soft_voting, rf]`：
    `UDS ≤ t1 → stacking`；`t1 < UDS ≤ t2 → soft_voting`；`UDS > t2 → rf`
    （允许 `t1 = t2` 的退化二段形）。内层在 `(变体, t1, t2)` 上穷举：
    阈值候选 = 内层 70 个任务 `UDS` 排序后相邻值中点；目标 = 内层平均 regret
    （环境等权）最小；并列打破 ① 内层最差环境 regret 更小 ② `(t1, t2)` 更大。
    全程确定性、无随机数。

    **规格歧义的处理**：D10 原文并列打破 ② 写作"更保守（阈值更靠 RF 侧）"，
    其中"更保守"与"阈值更靠 RF 侧"在本规则下指向相反方向（阈值越大 → 落入 RF 的
    任务越少）。本脚本以"`(t1, t2)` 字典序更大"为 primary（执行指令的字面重述），
    同时完整跑另一读法（更小）作为**硬门**：两读法的全部外层产物必须逐位一致，
    否则该歧义为实质性 → 门失败、不写任何产物（见 `--strict-tiebreak-gate`）。

regret
    `regret = F1(该任务三候选中最大) − F1(所选)`（oracle 限定在候选集内，§17.1）。
    候选集中的 soft voting 取该 fold 内层选中的变体。
    聚合主口径 = **环境等权**（先每环境均值再环境平均）；任务级均值并报。

验收硬门（任一不过 → 非零退出，不写任何产物）
    ① 候选 F1 与落盘源全量比对逐位一致（`metrics.json` vs `voting_baselines.csv`
       的 `rf` / `stacking` 共 300 组；两个 soft 变体的 `macro_f1_5class` 与其逐类
       F1 均值内部一致 1e-12）；网格元数据（`target_env` / `n_sources` /
       `train_rounds` / `test_rounds`）三源一致；
    ② 全流程确定性：同命令两次运行所有输出 CSV 的 md5 相同（由 `--out-dir` 双跑
       在脚本外判定，脚本本身无随机数、无时间依赖内容进 CSV）；
    ③ fold 结构自检 `assert` 全过；
    ④ 无泄漏结构审计：选择路径函数签名与源码无任何标签读取。

用法::

    python code/scripts/analysis/g1_route_a.py
    python code/scripts/analysis/g1_route_a.py --out-dir /tmp/g1_run_a
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

import cpd_core  # noqa: E402  (§11 UDS 唯一实现)
from environment_grid_experiment import ENVIRONMENTS  # noqa: E402
from environment_grid_experiment import build_task_grid  # noqa: E402  (§11 任务定义唯一来源)

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
G0_ROOT = REPO_ROOT / "results" / "g0_environment_grid"
RAW_ROOT = G0_ROOT / "raw_all"
VOTING_CSV = G0_ROOT / "voting_baselines.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "g1_route_a"

#: 类别轴顺序（`VOTING_BASELINES_NOTE.md` §2 与 `research_experiments.json` 的 `labels`）。
#: 这是**类别轴**，不是任何样本的真实标签。
CLASS_ORDER = ("Camera", "Light_T1", "Light_XM", "Sensor", "Socket")

#: 三个基模型（`UDS` 的模型对来源）。`cpd_core.uds` 内部按名字排序，此处顺序不影响结果。
BASE_MODELS = ("lightgbm", "rf", "xgboost")

#: 候选动作，按 §17.1 的风险序（低 → 高）。
RISK_ORDER = ("stacking", "soft_voting", "rf")

#: soft voting 的两个变体（D8 落盘），内层与阈值联合选定其一。
SOFT_VARIANTS = ("soft_voting_equal", "soft_voting_calibrated")

#: **标签派生列黑名单**：选择路径的任何 CSV 读取都不得包含这些列（§17.1 条件 5）。
#: `predictions.csv` 的 `correct` 由真实标签派生，`oof_meta.csv` / `predictions.csv`
#: 的 `true_label` 是真实标签本身。
FORBIDDEN_COLS = ("true_label", "correct")

#: 双阈值并列打破 ② 的两种读法（见模块 docstring "规格歧义的处理"）。
TIEBREAK_PRIMARY = "larger"
TIEBREAK_ALTERNATIVE = "smaller"

#: 条件 4 的 regret 门槛 `τ_repro`（D10 追记 v1.5，commit d447323）。
#: 取自登记表 E1-G0-GRID 行实测的跨线程拓扑 stacking 复现界 ≈ 2e-3。
#: τ = 0 的原严值计数照旧并列报告。
TAU_REPRO = 2e-3


# --------------------------------------------------------------------------- #
# 任务表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    name: str
    target_env: str
    train_rounds: tuple[str, ...]
    test_rounds: tuple[str, ...]
    n_sources: int


def load_tasks() -> list[Task]:
    """150 个 OOD 任务，唯一来源 = `build_task_grid()`（§11）。"""
    rows = [t for t in build_task_grid() if t["grid_kind"] == "ood"]
    tasks = [
        Task(
            name=t["name"],
            target_env=t["target_env"],
            train_rounds=tuple(t["train_rounds"]),
            test_rounds=tuple(t["test_rounds"]),
            n_sources=int(t["n_sources"]),
        )
        for t in rows
    ]
    return sorted(tasks, key=lambda t: t.name)


# --------------------------------------------------------------------------- #
# 选择路径的读取原语 —— 结构性无标签
# --------------------------------------------------------------------------- #
def _read_columns_no_label(path: Path, usecols: list[str]) -> pd.DataFrame:
    """按列名读取 CSV，并对**标签派生列**施加结构性守卫（§17.1 条件 5）。

    守卫在读取前后各查一次：请求列表命中黑名单 → 直接抛错；读回的 DataFrame
    含黑名单列 → 直接抛错。黑名单本身是模块常量 `FORBIDDEN_COLS`，
    因此本函数与其全部调用者的源码中不出现任何标签列名字面量。
    """
    bad = [c for c in usecols if c in FORBIDDEN_COLS]
    if bad:
        raise AssertionError(f"选择路径试图读取标签派生列 {bad}（§17.1 条件 5）：{path}")
    # float_precision="round_trip"：pandas 默认解析器在个别值上有 1 ULP 偏差
    # （实测 voting_baselines.csv 300 组中 18 组），round_trip 保证落盘十进制 → float64 逐位还原。
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", float_precision="round_trip")
    leaked = [c for c in df.columns if c in FORBIDDEN_COLS]
    if leaked:
        raise AssertionError(f"读回结果含标签派生列 {leaked}（§17.1 条件 5）：{path}")
    if set(df.columns) != set(usecols):
        raise AssertionError(f"列集合不符：期望 {sorted(usecols)}，得到 {sorted(df.columns)}：{path}")
    return df[list(usecols)]   # pandas 按文件列序返回，此处恢复请求列序（确定性）


def load_target_predictions(task_name: str) -> dict[str, np.ndarray]:
    """目标域测试预测：三基模型 `predictions.csv` 的 `predicted_label` 列。

    只请求预测列；`FORBIDDEN_COLS` 守卫保证不触碰任何标签派生列。
    """
    out: dict[str, np.ndarray] = {}
    for model in BASE_MODELS:
        path = RAW_ROOT / task_name / "all_features" / model / "predictions.csv"
        df = _read_columns_no_label(path, ["predicted_label"])
        out[model] = df["predicted_label"].to_numpy()
    lengths = {k: len(v) for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"{task_name}: 三基模型测试预测长度不一致 {lengths}")
    return out


def load_source_oof_predictions(task_name: str) -> dict[str, np.ndarray]:
    """源域 OOF 预测：`stacking/oof_meta.csv` 三基模型 OOF 概率各自 `argmax`。

    只请求 15 个 `oof_<model>_<class>` 概率列；`FORBIDDEN_COLS` 守卫保证不触碰
    该文件中同时存在的标签列。`argmax` 平局取最小下标（numpy 语义，确定性）。
    """
    path = RAW_ROOT / task_name / "all_features" / "stacking" / "oof_meta.csv"
    usecols = [f"oof_{m}_{c}" for m in BASE_MODELS for c in CLASS_ORDER]
    df = _read_columns_no_label(path, usecols)
    classes = np.asarray(CLASS_ORDER)
    out: dict[str, np.ndarray] = {}
    for model in BASE_MODELS:
        proba = df[[f"oof_{model}_{c}" for c in CLASS_ORDER]].to_numpy(dtype=float)
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError(f"{task_name}/{model}: OOF 概率行和不为 1")
        out[model] = classes[proba.argmax(axis=1)]
    return out


def compute_uds(task_name: str) -> float:
    """该任务的 `UDS`（§4.2），只经 `cpd_core.uds`（§11 唯一实现）。

    签名中没有任何标签入参；函数体只调用上面两个结构性无标签的读取原语。
    """
    src = load_source_oof_predictions(task_name)
    tgt = load_target_predictions(task_name)
    return float(cpd_core.uds(src, tgt, class_order=list(CLASS_ORDER)))


# --------------------------------------------------------------------------- #
# 选择器 —— 唯一预注册家族（UDS 单调双阈值）
# --------------------------------------------------------------------------- #
def select_candidate(uds_value: float, t1: float, t2: float) -> str:
    """`UDS ≤ t1 → stacking`；`t1 < UDS ≤ t2 → soft_voting`；`UDS > t2 → rf`。

    决策函数的全部入参只有一个 `UDS` 数值与两个阈值：**不收标签、不收 F1**。
    """
    if t1 > t2:
        raise ValueError(f"单调双阈值要求 t1 ≤ t2，收到 t1={t1}, t2={t2}")
    if uds_value <= t1:
        return "stacking"
    if uds_value <= t2:
        return "soft_voting"
    return "rf"


def select_vector(uds_values: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """`select_candidate` 的向量化等价形（内层穷举用），语义逐位一致。"""
    if t1 > t2:
        raise ValueError(f"单调双阈值要求 t1 ≤ t2，收到 t1={t1}, t2={t2}")
    out = np.full(uds_values.shape, "rf", dtype=object)
    out[uds_values <= t2] = "soft_voting"
    out[uds_values <= t1] = "stacking"
    return out


def threshold_candidates(uds_values: np.ndarray) -> np.ndarray:
    """阈值候选 = `UDS` 排序后相邻值中点（D10）。

    对排序后的**去重**值取相邻中点。调用方须先断言无重复值，
    从而"排序后相邻值中点"的两种读法（含重复 / 去重）恒等。
    """
    s = np.sort(np.asarray(uds_values, dtype=float))
    return (s[:-1] + s[1:]) / 2.0


# --------------------------------------------------------------------------- #
# 聚合口径
# --------------------------------------------------------------------------- #
def env_equal_mean(values: np.ndarray, envs: np.ndarray) -> float:
    """环境等权均值（§15.1）：先每环境均值，再对环境取算术平均。"""
    per_env = [float(np.mean(values[envs == e])) for e in sorted(set(envs.tolist()))]
    return float(np.mean(per_env))


def env_means(values: np.ndarray, envs: np.ndarray) -> dict[str, float]:
    return {e: float(np.mean(values[envs == e])) for e in sorted(set(envs.tolist()))}


# --------------------------------------------------------------------------- #
# 落盘 F1 的读取与验收门 ①
# --------------------------------------------------------------------------- #
def load_metrics_f1(task_name: str, model: str) -> tuple[float, dict]:
    """`metrics.json` 的 `macro_f1`（5-class，§10 主指标 1）+ 整个 json。"""
    path = RAW_ROOT / task_name / "all_features" / model / "metrics.json"
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return float(payload["macro_f1"]), payload


def bitwise_equal(a: float, b: float) -> bool:
    """float64 位模式相同。"""
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def build_f1_table(tasks: list[Task]) -> tuple[pd.DataFrame, list[dict]]:
    """逐任务四个落盘 F1 + 验收门 ① 的全量比对证据。"""
    # float_precision="round_trip"：见 `_read_columns_no_label` 的同一注记。
    # 默认解析器会在 300 组 rf/stacking 比对中的 18 组上引入 1 ULP（5.55e-17）偏差，
    # 那是**读取端**的十进制解析精度，落盘文本本身是全精度（诊断记入 provenance.diagnostics）。
    voting = pd.read_csv(VOTING_CSV, encoding="utf-8-sig", float_precision="round_trip")
    voting = voting[voting["grid_kind"] == "ood"].copy()

    gates: list[dict] = []
    rows: list[dict] = []
    n_bit_cmp = 0
    n_bit_bad = 0
    bad_examples: list[dict] = []
    n_meta_cmp = 0
    n_meta_bad = 0
    meta_bad: list[str] = []
    macro_dev_max = 0.0

    per_class_cols = [f"f1_{c}" for c in CLASS_ORDER]

    for task in tasks:
        sub = voting[voting["task"] == task.name]
        by_method = {m: sub[sub["method"] == m] for m in ("rf", "stacking", *SOFT_VARIANTS)}
        for method, frame in by_method.items():
            if len(frame) != 1:
                raise AssertionError(f"{task.name}/{method}: voting_baselines.csv 行数 = {len(frame)}")

        row = {"task": task.name, "target_env": task.target_env, "n_sources": task.n_sources}

        # (a) rf / stacking：metrics.json 与 voting_baselines.csv 逐位一致
        for method in ("rf", "stacking"):
            f1_metrics, payload = load_metrics_f1(task.name, method)
            f1_voting = float(by_method[method]["macro_f1_5class"].iloc[0])
            n_bit_cmp += 1
            if not bitwise_equal(f1_metrics, f1_voting):
                n_bit_bad += 1
                if len(bad_examples) < 20:
                    bad_examples.append({"task": task.name, "method": method,
                                         "metrics_json": f1_metrics, "voting_csv": f1_voting})
            row[f"f1_{method}"] = f1_metrics

            # 网格元数据三源一致（生成器 / metrics.json / voting_baselines.csv）
            n_meta_cmp += 1
            ok = (tuple(payload["train_rounds"]) == task.train_rounds
                  and tuple(payload["test_rounds"]) == task.test_rounds
                  and payload["task"] == task.name)
            if not ok:
                n_meta_bad += 1
                meta_bad.append(f"{task.name}/{method}: metrics.json 轮次与生成器不符")

        vrow = by_method["rf"].iloc[0]
        n_meta_cmp += 1
        if not (vrow["target_env"] == task.target_env
                and int(vrow["n_sources"]) == task.n_sources
                and str(vrow["train_rounds"]) == str(list(task.train_rounds))
                and str(vrow["test_rounds"]) == str(list(task.test_rounds))):
            n_meta_bad += 1
            meta_bad.append(f"{task.name}: voting_baselines.csv 元数据与生成器不符")

        # (b) 两个 soft 变体：落盘 macro 与其逐类 F1 均值的内部一致性
        for variant in SOFT_VARIANTS:
            frame = by_method[variant].iloc[0]
            f1 = float(frame["macro_f1_5class"])
            recomputed = float(np.mean([float(frame[c]) for c in per_class_cols]))
            macro_dev_max = max(macro_dev_max, abs(f1 - recomputed))
            row[f"f1_{variant}"] = f1
            if str(frame["method_kind"]) != "voting":
                raise AssertionError(f"{task.name}/{variant}: method_kind != voting")

        rows.append(row)

    gates.append({
        "gate": "1a_candidate_f1_bitwise_vs_disk",
        "detail": "rf / stacking 的 metrics.json::macro_f1 与 voting_baselines.csv::macro_f1_5class 逐位比对",
        "n_compared": n_bit_cmp,
        "n_mismatch": n_bit_bad,
        "examples": bad_examples,
        "passed": n_bit_bad == 0,
    })
    gates.append({
        "gate": "1b_grid_metadata_three_way",
        "detail": "任务名 / target_env / n_sources / train_rounds / test_rounds 在生成器、metrics.json、voting_baselines.csv 三源一致",
        "n_compared": n_meta_cmp,
        "n_mismatch": n_meta_bad,
        "examples": meta_bad[:20],
        "passed": n_meta_bad == 0,
    })
    gates.append({
        "gate": "1c_soft_voting_macro_internal_consistency",
        "detail": "两个 soft 变体的 macro_f1_5class 与其 5 个逐类 F1 的算术平均之差",
        "n_compared": len(tasks) * len(SOFT_VARIANTS),
        "max_abs_dev": macro_dev_max,
        "tol": 1e-12,
        "passed": macro_dev_max <= 1e-12,
    })

    df = pd.DataFrame(rows).set_index("task")
    if df.isna().any().any():
        raise AssertionError("F1 表存在缺失值")
    gates.append({
        "gate": "1d_f1_table_complete",
        "detail": "150 任务 × 4 个落盘 F1 全部存在且非缺失",
        "n_compared": int(len(df) * 4),
        "n_rows": int(len(df)),
        "detail_value": f"{len(df)} 行 × 4 列 = {len(df) * 4} 个 F1，缺失 0",
        "passed": len(df) == 150,
    })
    return df, gates


# --------------------------------------------------------------------------- #
# fold 结构自检（验收门 ③，写成 assert）
# --------------------------------------------------------------------------- #
def build_folds(tasks: list[Task]) -> dict[str, dict[str, list[Task]]]:
    """外层 / 内层划分 + §9.2 结构自检（literal `assert`）。"""
    by_name = {t.name: t for t in tasks}
    assert len(by_name) == 150, f"OOD 任务数应为 150，实得 {len(by_name)}"

    folds: dict[str, dict[str, list[Task]]] = {}
    outer_union: list[str] = []
    for e_out in ENVIRONMENTS:
        outer = [t for t in tasks if t.target_env == e_out]
        inner = [t for t in tasks
                 if t.target_env != e_out and e_out not in t.train_rounds]

        # 外层：目标环境 = e_out 的 25 个任务
        assert len(outer) == 25, f"fold {e_out}: 外层任务数 {len(outer)} ≠ 25"
        # 内层：严格全含 —— 5 目标 × 14 源组合
        assert len(inner) == 70, f"fold {e_out}: 内层任务数 {len(inner)} ≠ 70"
        assert len({t.target_env for t in inner}) == 5, f"fold {e_out}: 内层目标环境数 ≠ 5"
        for t in inner:
            # e_out 不得以源或目标任何角色进入内层
            assert e_out not in t.train_rounds, f"fold {e_out}: 内层任务 {t.name} 的 train_rounds 含 {e_out}"
            assert e_out not in t.test_rounds, f"fold {e_out}: 内层任务 {t.name} 的 test_rounds 含 {e_out}"
        for env in {t.target_env for t in inner}:
            assert sum(1 for t in inner if t.target_env == env) == 14, \
                f"fold {e_out}: 内层目标环境 {env} 的任务数 ≠ 14"
        # 内层与外层不相交
        assert not ({t.name for t in inner} & {t.name for t in outer}), \
            f"fold {e_out}: 内外层任务重叠"

        outer_union.extend(t.name for t in outer)
        folds[e_out] = {"outer": sorted(outer, key=lambda t: t.name),
                        "inner": sorted(inner, key=lambda t: t.name)}

    # 6 × 25 覆盖 150 个任务，不重不漏
    assert len(outer_union) == 150, f"外层任务合计 {len(outer_union)} ≠ 150"
    assert len(set(outer_union)) == 150, "外层任务出现重复（每任务只评一次被破坏）"
    assert set(outer_union) == set(by_name), "外层任务并集 ≠ 150 个 OOD 任务全集"
    return folds


# --------------------------------------------------------------------------- #
# 内层穷举
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoldChoice:
    variant: str
    t1: float
    t2: float
    inner_mean_env_equal: float
    inner_mean_task: float
    inner_worst_env_regret: float
    inner_worst_env: str
    n_grid_points: int
    n_threshold_candidates: int
    n_tied_stage1: int
    n_tied_stage2: int
    tiebreak_stage_used: str
    variant_tiebreak_fired: bool


def _regret_arrays(f1: pd.DataFrame, names: list[str], variant: str) -> dict[str, np.ndarray]:
    """给定 soft 变体，返回三候选各自的 regret 向量（oracle 限定在候选集内）。"""
    cols = {"stacking": f1.loc[names, "f1_stacking"].to_numpy(dtype=float),
            "soft_voting": f1.loc[names, f"f1_{variant}"].to_numpy(dtype=float),
            "rf": f1.loc[names, "f1_rf"].to_numpy(dtype=float)}
    oracle = np.maximum.reduce([cols[a] for a in RISK_ORDER])
    return {"f1": cols, "oracle": oracle,
            "regret": {a: oracle - cols[a] for a in RISK_ORDER}}


def search_fold(f1: pd.DataFrame, uds: dict[str, float], inner: list[Task],
                tiebreak: str) -> FoldChoice:
    """内层穷举 `(变体, t1, t2)`，目标 = 内层平均 regret（环境等权）最小。"""
    names = [t.name for t in inner]
    envs = np.array([t.target_env for t in inner])
    u = np.array([uds[n] for n in names], dtype=float)

    uniq = np.unique(u)
    assert uniq.size == u.size, (
        f"内层 UDS 出现重复值（{u.size - uniq.size} 处）：'排序后相邻值中点'的两种读法不再等价")
    cands = threshold_candidates(u)
    n_cand = int(cands.size)

    env_list = sorted(set(envs.tolist()))
    masks = {e: (envs == e) for e in env_list}

    records: list[tuple] = []
    for variant in SOFT_VARIANTS:
        arrs = _regret_arrays(f1, names, variant)
        reg = arrs["regret"]
        for i in range(n_cand):
            t1 = float(cands[i])
            for j in range(i, n_cand):
                t2 = float(cands[j])
                sel = select_vector(u, t1, t2)
                r = np.where(sel == "stacking", reg["stacking"],
                             np.where(sel == "soft_voting", reg["soft_voting"], reg["rf"]))
                per_env = [float(np.mean(r[masks[e]])) for e in env_list]
                mean_env_equal = float(np.mean(per_env))
                worst = float(np.max(per_env))
                worst_env = env_list[int(np.argmax(per_env))]
                records.append((mean_env_equal, worst, worst_env, variant, t1, t2,
                                float(np.mean(r))))

    best_mean = min(rec[0] for rec in records)
    stage1 = [rec for rec in records if rec[0] == best_mean]
    best_worst = min(rec[1] for rec in stage1)
    stage2 = [rec for rec in stage1 if rec[1] == best_worst]

    if tiebreak == "larger":
        key = lambda rec: (-rec[4], -rec[5])          # (t1, t2) 字典序更大
    elif tiebreak == "smaller":
        key = lambda rec: (rec[4], rec[5])            # (t1, t2) 字典序更小
    else:
        raise ValueError(tiebreak)
    stage3 = sorted(stage2, key=key)
    top_thr = (stage3[0][4], stage3[0][5])
    stage3_top = [rec for rec in stage3 if (rec[4], rec[5]) == top_thr]
    variant_fired = len(stage3_top) > 1
    chosen = sorted(stage3_top, key=lambda rec: SOFT_VARIANTS.index(rec[3]))[0]

    if len(stage1) == 1:
        stage_used = "unique_at_objective"
    elif len(stage2) == 1:
        stage_used = "stage1_worst_env"
    elif not variant_fired:
        stage_used = f"stage2_threshold_{tiebreak}"
    else:
        stage_used = "stage3_variant_order"

    return FoldChoice(
        variant=chosen[3], t1=chosen[4], t2=chosen[5],
        inner_mean_env_equal=chosen[0], inner_mean_task=chosen[6],
        inner_worst_env_regret=chosen[1], inner_worst_env=chosen[2],
        n_grid_points=len(records), n_threshold_candidates=n_cand,
        n_tied_stage1=len(stage1), n_tied_stage2=len(stage2),
        tiebreak_stage_used=stage_used, variant_tiebreak_fired=variant_fired,
    )


# --------------------------------------------------------------------------- #
# 外层评估
# --------------------------------------------------------------------------- #
def evaluate(f1: pd.DataFrame, uds: dict[str, float], folds: dict, tiebreak: str
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跑完 6 个 fold，返回 (逐任务明细, 各 fold 参数)。"""
    detail_rows: list[dict] = []
    fold_rows: list[dict] = []

    for e_out in ENVIRONMENTS:
        inner = folds[e_out]["inner"]
        outer = folds[e_out]["outer"]
        choice = search_fold(f1, uds, inner, tiebreak)

        inner_u = np.array([uds[t.name] for t in inner], dtype=float)
        outer_u = np.array([uds[t.name] for t in outer], dtype=float)
        inner_sel = select_vector(inner_u, choice.t1, choice.t2)

        for task in outer:
            u = uds[task.name]
            f1_soft = float(f1.loc[task.name, f"f1_{choice.variant}"])
            cand = {"rf": float(f1.loc[task.name, "f1_rf"]),
                    "stacking": float(f1.loc[task.name, "f1_stacking"]),
                    "soft_voting": f1_soft}
            oracle_f1 = max(cand[a] for a in RISK_ORDER)
            oracle_names = [a for a in RISK_ORDER if cand[a] == oracle_f1]
            sel = select_candidate(u, choice.t1, choice.t2)
            detail_rows.append({
                "fold_e_out": e_out,
                "target_env": task.target_env,
                "task": task.name,
                "n_sources": task.n_sources,
                "train_rounds": "|".join(task.train_rounds),
                "uds": u,
                "f1_rf": cand["rf"],
                "f1_stacking": cand["stacking"],
                "f1_soft_voting_equal": float(f1.loc[task.name, "f1_soft_voting_equal"]),
                "f1_soft_voting_calibrated": float(f1.loc[task.name, "f1_soft_voting_calibrated"]),
                "fold_soft_variant": choice.variant,
                "f1_soft_voting_selected_variant": f1_soft,
                "fold_t1": choice.t1,
                "fold_t2": choice.t2,
                "selected_candidate": sel,
                "f1_selected": cand[sel],
                "oracle_f1": oracle_f1,
                "oracle_candidate": "|".join(oracle_names),
                "regret": oracle_f1 - cand[sel],
                "regret_always_rf": oracle_f1 - cand["rf"],
                "regret_always_stacking": oracle_f1 - cand["stacking"],
                "regret_always_soft_voting": oracle_f1 - cand["soft_voting"],
            })

        fold_rows.append({
            "fold_e_out": e_out,
            "n_outer_tasks": len(outer),
            "n_inner_tasks": len(inner),
            "n_inner_target_envs": len({t.target_env for t in inner}),
            "n_threshold_candidates": choice.n_threshold_candidates,
            "n_grid_points": choice.n_grid_points,
            "selected_soft_variant": choice.variant,
            "t1": choice.t1,
            "t2": choice.t2,
            "degenerate_t1_eq_t2": bool(choice.t1 == choice.t2),
            "inner_mean_regret_env_equal": choice.inner_mean_env_equal,
            "inner_mean_regret_task_level": choice.inner_mean_task,
            "inner_worst_env_regret": choice.inner_worst_env_regret,
            "inner_worst_env": choice.inner_worst_env,
            "n_tied_at_objective": choice.n_tied_stage1,
            "n_tied_after_worst_env": choice.n_tied_stage2,
            "tiebreak_stage_used": choice.tiebreak_stage_used,
            "variant_tiebreak_fired": choice.variant_tiebreak_fired,
            "inner_n_selected_stacking": int(np.sum(inner_sel == "stacking")),
            "inner_n_selected_soft_voting": int(np.sum(inner_sel == "soft_voting")),
            "inner_n_selected_rf": int(np.sum(inner_sel == "rf")),
            "inner_uds_min": float(np.min(inner_u)),
            "inner_uds_median": float(np.median(inner_u)),
            "inner_uds_max": float(np.max(inner_u)),
            "outer_uds_min": float(np.min(outer_u)),
            "outer_uds_median": float(np.median(outer_u)),
            "outer_uds_max": float(np.max(outer_u)),
        })

    detail = pd.DataFrame(detail_rows).sort_values("task").reset_index(drop=True)
    fold_params = pd.DataFrame(fold_rows)
    return detail, fold_params


#: `g1_task_detail.csv` 中属于**结果**的列。`fold_t1` / `fold_t2` 是被记录的**参数**，
#: 不在此列表内——并列打破读法的敏感性门只对结果列判定（见模块 docstring）。
OUTCOME_COLS = [
    "fold_e_out", "target_env", "task", "n_sources", "train_rounds", "uds",
    "f1_rf", "f1_stacking", "f1_soft_voting_equal", "f1_soft_voting_calibrated",
    "fold_soft_variant", "f1_soft_voting_selected_variant",
    "selected_candidate", "f1_selected", "oracle_f1", "oracle_candidate",
    "regret", "regret_always_rf", "regret_always_stacking", "regret_always_soft_voting",
]


def summarize_env(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in ENVIRONMENTS:
        d = detail[detail["target_env"] == env]
        sel_f1 = d["f1_selected"].to_numpy(dtype=float)
        rf_f1 = d["f1_rf"].to_numpy(dtype=float)
        st_f1 = d["f1_stacking"].to_numpy(dtype=float)
        rows.append({
            "target_env": env,
            "n_tasks": int(len(d)),
            "soft_variant": d["fold_soft_variant"].iloc[0],
            "t1": float(d["fold_t1"].iloc[0]),
            "t2": float(d["fold_t2"].iloc[0]),
            "mean_regret_selector": float(d["regret"].mean()),
            "mean_regret_always_rf": float(d["regret_always_rf"].mean()),
            "mean_regret_always_stacking": float(d["regret_always_stacking"].mean()),
            "mean_regret_always_soft_voting": float(d["regret_always_soft_voting"].mean()),
            "max_regret_selector_task": float(d["regret"].max()),
            "n_selected_stacking": int((d["selected_candidate"] == "stacking").sum()),
            "n_selected_soft_voting": int((d["selected_candidate"] == "soft_voting").sum()),
            "n_selected_rf": int((d["selected_candidate"] == "rf").sum()),
            "n_regret_zero": int((d["regret"] == 0).sum()),
            "n_regret_le_tau_repro": int((d["regret"] <= TAU_REPRO).sum()),
            "n_nonrf_selected": int((d["selected_candidate"] != "rf").sum()),
            "n_nonrf_and_regret_le_tau_repro": int(((d["selected_candidate"] != "rf")
                                                    & (d["regret"] <= TAU_REPRO)).sum()),
            "n_nonrf_and_regret_zero_tau0": int(((d["selected_candidate"] != "rf")
                                                 & (d["regret"] == 0)).sum()),
            "n_nonrf_and_f1_ge_rf_lenient": int(((d["selected_candidate"] != "rf")
                                                 & (sel_f1 >= rf_f1)).sum()),
            "win_vs_always_rf": int(np.sum(sel_f1 > rf_f1)),
            "loss_vs_always_rf": int(np.sum(sel_f1 < rf_f1)),
            "tie_vs_always_rf": int(np.sum(sel_f1 == rf_f1)),
            "win_vs_always_stacking": int(np.sum(sel_f1 > st_f1)),
            "loss_vs_always_stacking": int(np.sum(sel_f1 < st_f1)),
            "tie_vs_always_stacking": int(np.sum(sel_f1 == st_f1)),
            "mean_f1_selector": float(np.mean(sel_f1)),
            "mean_f1_always_rf": float(np.mean(rf_f1)),
            "mean_f1_always_stacking": float(np.mean(st_f1)),
            "mean_f1_oracle": float(d["oracle_f1"].mean()),
            "uds_min": float(d["uds"].min()),
            "uds_median": float(d["uds"].median()),
            "uds_max": float(d["uds"].max()),
        })
    return pd.DataFrame(rows)


def summarize_overall(detail: pd.DataFrame, env_summary: pd.DataFrame,
                      audit: dict) -> pd.DataFrame:
    """§17.1 五条判据的**原始比较量**。本表不含任何通过 / 不通过判定。"""
    envs = detail["target_env"].to_numpy()
    reg = detail["regret"].to_numpy(dtype=float)
    reg_rf = detail["regret_always_rf"].to_numpy(dtype=float)
    reg_st = detail["regret_always_stacking"].to_numpy(dtype=float)
    reg_sv = detail["regret_always_soft_voting"].to_numpy(dtype=float)

    per_env = env_means(reg, envs)
    worst_env = max(per_env, key=lambda e: per_env[e])

    sel = detail["selected_candidate"].to_numpy()
    sel_f1 = detail["f1_selected"].to_numpy(dtype=float)
    rf_f1 = detail["f1_rf"].to_numpy(dtype=float)
    # 条件 4：primary 口径 = regret ≤ τ_repro（D10 追记 v1.5）；τ = 0 严值并列报告。
    tau_mask = (sel != "rf") & (reg <= TAU_REPRO)
    strict_mask = (sel != "rf") & (reg == 0.0)
    lenient_mask = (sel != "rf") & (sel_f1 >= rf_f1)

    rows: list[dict] = []

    def add(crit, quantity, numeric=None, text=None):
        rows.append({"criterion": crit, "quantity": quantity,
                     "value_numeric": numeric, "value_text": text})

    add("scope", "n_outer_tasks", float(len(detail)))
    add("scope", "n_outer_folds", float(env_summary.shape[0]))
    add("scope", "f1_metric", text="5-class macro-F1 (§10 主指标 1；落盘值，未重算)")
    add("scope", "aggregation_primary", text="环境等权（先每环境均值再 6 环境平均，§15.1）")

    add("cond1", "mean_regret_selector_env_equal", env_equal_mean(reg, envs))
    add("cond1", "mean_regret_selector_task_level", float(np.mean(reg)))

    for env in ENVIRONMENTS:
        add("cond2", f"mean_regret_selector_env_{env}", per_env[env])
    add("cond2", "worst_env_mean_regret_selector", per_env[worst_env])
    add("cond2", "worst_env_name_selector", text=worst_env)
    add("cond2", "max_single_task_regret_selector", float(np.max(reg)))

    add("cond3", "mean_regret_always_rf_env_equal", env_equal_mean(reg_rf, envs))
    add("cond3", "mean_regret_always_stacking_env_equal", env_equal_mean(reg_st, envs))
    add("cond3", "mean_regret_always_soft_voting_env_equal", env_equal_mean(reg_sv, envs))
    add("cond3", "mean_regret_always_rf_task_level", float(np.mean(reg_rf)))
    add("cond3", "mean_regret_always_stacking_task_level", float(np.mean(reg_st)))
    add("cond3", "mean_regret_always_soft_voting_task_level", float(np.mean(reg_sv)))
    add("cond3", "delta_selector_minus_always_rf_env_equal",
        env_equal_mean(reg, envs) - env_equal_mean(reg_rf, envs))
    add("cond3", "delta_selector_minus_always_stacking_env_equal",
        env_equal_mean(reg, envs) - env_equal_mean(reg_st, envs))
    add("cond3", "worst_env_mean_regret_always_rf",
        max(env_means(reg_rf, envs).values()))
    add("cond3", "worst_env_mean_regret_always_stacking",
        max(env_means(reg_st, envs).values()))
    add("cond3", "n_tasks_selector_f1_gt_always_rf", float(np.sum(sel_f1 > rf_f1)))
    add("cond3", "n_tasks_selector_f1_lt_always_rf", float(np.sum(sel_f1 < rf_f1)))
    add("cond3", "n_tasks_selector_f1_gt_always_stacking",
        float(np.sum(sel_f1 > detail["f1_stacking"].to_numpy(dtype=float))))
    add("cond3", "n_tasks_selector_f1_lt_always_stacking",
        float(np.sum(sel_f1 < detail["f1_stacking"].to_numpy(dtype=float))))

    add("cond4", "n_tasks_selected_nonrf", float(np.sum(sel != "rf")))
    add("cond4", "n_tasks_selected_stacking", float(np.sum(sel == "stacking")))
    add("cond4", "n_tasks_selected_soft_voting", float(np.sum(sel == "soft_voting")))
    add("cond4", "n_tasks_selected_rf", float(np.sum(sel == "rf")))
    add("cond4", "tau_repro", TAU_REPRO,
        text="D10 追记 v1.5：条件 4 primary 口径的 regret 门槛（= 登记表 E1-G0-GRID 实测复现界）")
    add("cond4", "n_tasks_nonrf_and_regret_le_tau_repro_primary", float(np.sum(tau_mask)))
    add("cond4", "n_tasks_nonrf_and_regret_zero_tau0_parallel", float(np.sum(strict_mask)))
    add("cond4", "n_tasks_nonrf_and_f1_ge_rf_lenient", float(np.sum(lenient_mask)))
    add("cond4", "n_envs_with_any_nonrf_selection",
        float(sum(1 for e in ENVIRONMENTS if np.any(sel[envs == e] != "rf"))))
    add("cond4", "n_envs_with_tau_repro_hit_primary",
        float(sum(1 for e in ENVIRONMENTS if np.any(tau_mask[envs == e]))))
    add("cond4", "n_envs_with_tau0_hit_parallel",
        float(sum(1 for e in ENVIRONMENTS if np.any(strict_mask[envs == e]))))
    add("cond4", "n_envs_with_lenient_hit",
        float(sum(1 for e in ENVIRONMENTS if np.any(lenient_mask[envs == e]))))
    add("cond4", "envs_with_tau_repro_hit_primary",
        text=";".join(e for e in ENVIRONMENTS if np.any(tau_mask[envs == e])) or "(none)")
    add("cond4", "envs_with_tau0_hit_parallel",
        text=";".join(e for e in ENVIRONMENTS if np.any(strict_mask[envs == e])) or "(none)")
    add("cond4", "oracle_distribution_rf_only",
        float(np.sum(detail["oracle_candidate"].to_numpy() == "rf")))
    add("cond4", "oracle_distribution_stacking_only",
        float(np.sum(detail["oracle_candidate"].to_numpy() == "stacking")))
    add("cond4", "oracle_distribution_soft_voting_only",
        float(np.sum(detail["oracle_candidate"].to_numpy() == "soft_voting")))
    add("cond4", "oracle_distribution_ties",
        float(np.sum(np.char.count(detail["oracle_candidate"].to_numpy().astype(str), "|") > 0)))

    # D10 追记 v1.5 的动因数据（最优与次优候选的 F1 差），由本次落盘 F1 复算供审阅方核对
    for label, col in (("selected_variant", "f1_soft_voting_selected_variant"),
                       ("calibrated", "f1_soft_voting_calibrated"),
                       ("equal", "f1_soft_voting_equal")):
        mat = np.column_stack([detail["f1_rf"].to_numpy(dtype=float),
                               detail["f1_stacking"].to_numpy(dtype=float),
                               detail[col].to_numpy(dtype=float)])
        srt = np.sort(mat, axis=1)
        gap = srt[:, 2] - srt[:, 1]
        add("cond4", f"n_tasks_top2_candidate_gap_lt_tau_repro_{label}", float(np.sum(gap < TAU_REPRO)))
        add("cond4", f"median_top2_candidate_gap_{label}", float(np.median(gap)))
        add("cond4", f"min_top2_candidate_gap_{label}", float(np.min(gap)))

    add("cond5", "uds_signature", text=audit["uds_signature"])
    add("cond5", "n_label_tokens_on_selection_path", float(audit["n_label_tokens_on_selection_path"]))
    add("cond5", "n_selection_path_functions_audited", float(len(audit["functions"])))
    add("cond5", "audit_document", text="g1_leakage_audit.md")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 无泄漏静态审计（验收门 ④）
# --------------------------------------------------------------------------- #
SELECTION_PATH = [
    "g1_route_a._read_columns_no_label",
    "g1_route_a.load_target_predictions",
    "g1_route_a.load_source_oof_predictions",
    "g1_route_a.compute_uds",
    "g1_route_a.select_candidate",
    "g1_route_a.select_vector",
    "g1_route_a.threshold_candidates",
    "cpd_core.uds",
    "cpd_core.disagreement_matrix",
    "cpd_core.off",
    "cpd_core.normalize_cm",
    "cpd_core._as_pred_mapping",
]

#: 标签读取的静态审计词元。`FORBIDDEN_COLS` 常量本身不在选择路径函数体内出现。
LABEL_TOKENS = (r"true_label", r"y_true", r"y_test", r"\by\b", r"\.correct\b",
                r"ground_truth", r"labels\s*=")


def run_leakage_audit() -> dict:
    """§17.1 条件 5 的结构性 + 静态审计。"""

    def resolve(dotted: str):
        mod, fn = dotted.split(".")
        return getattr(sys.modules[mod if mod != "g1_route_a" else __name__], fn)

    functions = []
    n_hits = 0
    for dotted in SELECTION_PATH:
        fn = resolve(dotted)
        sig = inspect.signature(fn)
        src = inspect.getsource(fn)
        hits = []
        for tok in LABEL_TOKENS:
            for m in re.finditer(tok, src):
                line = src[:m.start()].count("\n") + 1
                hits.append({"token": tok, "line_in_function": line,
                             "text": src.splitlines()[line - 1].strip()})
        n_hits += len(hits)
        bad_params = [p for p in sig.parameters
                      if re.fullmatch(r"y|y_true|labels?|true_label|targets?", p)]
        functions.append({"function": dotted, "signature": f"{fn.__name__}{sig}",
                          "n_label_token_hits": len(hits), "hits": hits,
                          "forbidden_param_names": bad_params})
        if bad_params:
            raise AssertionError(f"{dotted} 的签名含标签参数 {bad_params}（§17.1 条件 5）")

    uds_sig = f"uds{inspect.signature(cpd_core.uds)}"
    return {"uds_signature": uds_sig,
            "n_label_tokens_on_selection_path": n_hits,
            "functions": functions,
            "forbidden_cols_guard": list(FORBIDDEN_COLS)}


def scan_module_source() -> list[dict]:
    """对本文件全文做标签词元扫描（逐行证据，含审计代码自身的命中）。"""
    src = Path(__file__).read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(src, start=1):
        for tok in ("true_label", "y_true", "ground_truth"):
            if tok in line:
                out.append({"line": i, "token": tok, "text": line.strip()})
                break
    return out


# --------------------------------------------------------------------------- #
# 产物写出
# --------------------------------------------------------------------------- #
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _esc(text) -> str:
    """markdown 表格单元格转义（竖线会截断行）。"""
    return str(text).replace("|", "\\|")


def build_note(detail: pd.DataFrame, env_summary: pd.DataFrame,
               overall: pd.DataFrame, fold_params: pd.DataFrame,
               gates: list[dict]) -> str:
    """`G1_RESULTS_NOTE.md` —— 只数表，不解读。内容确定性（不含时间戳）。"""
    lines: list[str] = []
    lines.append("# G1_RESULTS_NOTE.md —— 判定由审阅方按 §17.1 作出，本文档不含解读")
    lines.append("")
    lines.append("**性质**：产物说明与数表转录。只记口径、输入、验收证据与数值；")
    lines.append("**不含任何解读、机制语言、判定或结论性表述**——§17.1 五条判据的通过 /")
    lines.append("不通过判定由审阅方作出。")
    lines.append("")
    lines.append("**协议依据**：`docs/experiment_protocol_final.md` §17.1 / §9.2 / §4.2 / §10 / §11 / §15 / §19.2。")
    lines.append("**执行口径**：`docs/EXECUTION_PLAN_20260829.md` 决策 **D10**（v1.4，先写后看）"
                 " + **D10 追记 v1.5**（commit `d447323`，运行前修订条件 4 口径）。")
    lines.append("**生成脚本**：`code/scripts/analysis/g1_route_a.py`（口径完整定义见该脚本 docstring）。")
    lines.append("**随机性**：无（全流程确定性，零重训，`random_seed = null`）。")
    lines.append("**运行记录**：`provenance.json`（§19.2 五要素）。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 口径（逐条对应 D10）")
    lines.append("")
    lines.append("- 任务域 = G0 网格的 150 个 OOD 任务；任务定义唯一来源 ="
                 " `environment_grid_experiment.build_task_grid()`。")
    lines.append("- 外层 fold `e_out ∈ {R2..R7}`：外层评估任务 = `target_env == e_out` 的 25 个，"
                 "每任务只评一次；内层池 = 严格全含任务（`e_out` 既不作源也不作目标）= 5 目标 × 14 源组合 = 70。")
    lines.append("- 候选 = `rf` / `stacking`（G0 落盘 = §9.1 grouped OOF，B 口径）/ `soft_voting`"
                 "（变体 `soft_voting_equal` 或 `soft_voting_calibrated`，由内层与阈值联合选定）。")
    lines.append("- **候选 F1 一律读落盘值，本次运行不重算任何 F1**："
                 "`rf` / `stacking` 取 `metrics.json::macro_f1`；两个 soft 变体取"
                 " `voting_baselines.csv::macro_f1_5class`。指标 = 5-class macro-F1（§10 主指标 1）。")
    lines.append("- `UDS` 只经 `cpd_core.uds`（§11）：源侧 = `stacking/oof_meta.csv` 三基模型 OOF 概率 `argmax`；"
                 "目标侧 = 三基模型 `predictions.csv` 的 `predicted_label`；6 个有序模型对取均值。")
    lines.append("- 选择器 = `UDS` 单调双阈值，风险序 `[stacking, soft_voting, rf]`："
                 "`UDS ≤ t1 → stacking`；`t1 < UDS ≤ t2 → soft_voting`；`UDS > t2 → rf`。")
    lines.append("- `regret = F1(该任务三候选中最大) − F1(所选)`；聚合主口径 = 环境等权，任务级并报。")
    lines.append(f"- 条件 4 的 regret 门槛（D10 追记 v1.5）：primary `τ_repro = {TAU_REPRO}`"
                 "（取自登记表 E1-G0-GRID 行实测的跨线程拓扑 stacking 复现界）；"
                 "`τ = 0` 的原严值计数以 `_tau0_parallel` 后缀并列报告。")
    lines.append("")
    lines.append("## 2. 验收硬门")
    lines.append("")
    lines.append("| 门 | 内容 | 规模 | 结果 |")
    lines.append("|---|---|---|---|")
    for g in gates:
        scale = g.get("n_compared", g.get("n_checks", g.get("n_rows", "")))
        extra = ""
        if "n_mismatch" in g:
            extra = f"n_mismatch = {g['n_mismatch']}"
        elif "max_abs_dev" in g:
            extra = f"max_abs_dev = {g['max_abs_dev']:.3e}（tol {g['tol']:.0e}）"
        elif "detail_value" in g:
            extra = str(g["detail_value"])
        lines.append(f"| `{g['gate']}` | {_esc(g['detail'])} | {scale} | {_esc(extra)} |")
    lines.append("")
    lines.append("> 双跑 md5 一致性（验收门 ②）由脚本外的两次独立运行判定，证据记入 `provenance.json`"
                 " 的 `outputs` 与运行报告。")
    lines.append("")
    lines.append("## 3. 各 fold 内层选中的 (变体, t1, t2)")
    lines.append("")
    lines.append("| fold `e_out` | 变体 | t1 | t2 | t1=t2 | 内层平均 regret（环境等权） | 内层平均 regret（任务级） |"
                 " 内层最差环境 regret | 内层最差环境 | 网格点数 | 目标值并列数 | 最差环境后并列数 | 打破阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in fold_params.iterrows():
        lines.append(
            f"| {r['fold_e_out']} | `{r['selected_soft_variant']}` | {r['t1']:.6f} | {r['t2']:.6f} |"
            f" {'是' if r['degenerate_t1_eq_t2'] else '否'} | {r['inner_mean_regret_env_equal']:.6f} |"
            f" {r['inner_mean_regret_task_level']:.6f} | {r['inner_worst_env_regret']:.6f} |"
            f" {r['inner_worst_env']} | {int(r['n_grid_points'])} | {int(r['n_tied_at_objective'])} |"
            f" {int(r['n_tied_after_worst_env'])} | `{r['tiebreak_stage_used']}` |")
    lines.append("")
    lines.append("## 4. UDS 分布（逐 fold）")
    lines.append("")
    lines.append("| fold `e_out` | 内层 n | 内层 UDS min | median | max | 外层 n | 外层 UDS min | median | max |"
                 " 内层选择 stacking / soft / rf |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in fold_params.iterrows():
        lines.append(
            f"| {r['fold_e_out']} | {int(r['n_inner_tasks'])} | {r['inner_uds_min']:.6f} |"
            f" {r['inner_uds_median']:.6f} | {r['inner_uds_max']:.6f} | {int(r['n_outer_tasks'])} |"
            f" {r['outer_uds_min']:.6f} | {r['outer_uds_median']:.6f} | {r['outer_uds_max']:.6f} |"
            f" {int(r['inner_n_selected_stacking'])} / {int(r['inner_n_selected_soft_voting'])} /"
            f" {int(r['inner_n_selected_rf'])} |")
    lines.append("")
    lines.append("## 5. 逐环境外层结果")
    lines.append("")
    lines.append("| 目标环境 | n | 变体 | 选择器平均 regret | always-RF | always-Stacking | always-soft |"
                 " 选择 stacking / soft / rf | 最大单任务 regret | win/loss/tie vs RF | win/loss/tie vs Stacking |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in env_summary.iterrows():
        lines.append(
            f"| {r['target_env']} | {int(r['n_tasks'])} | `{r['soft_variant']}` |"
            f" {r['mean_regret_selector']:.6f} | {r['mean_regret_always_rf']:.6f} |"
            f" {r['mean_regret_always_stacking']:.6f} | {r['mean_regret_always_soft_voting']:.6f} |"
            f" {int(r['n_selected_stacking'])} / {int(r['n_selected_soft_voting'])} / {int(r['n_selected_rf'])} |"
            f" {r['max_regret_selector_task']:.6f} |"
            f" {int(r['win_vs_always_rf'])}/{int(r['loss_vs_always_rf'])}/{int(r['tie_vs_always_rf'])} |"
            f" {int(r['win_vs_always_stacking'])}/{int(r['loss_vs_always_stacking'])}/{int(r['tie_vs_always_stacking'])} |")
    lines.append("")
    lines.append("## 6. §17.1 五条判据的原始比较量（不含判定）")
    lines.append("")
    lines.append("| 判据 | 量 | 数值 | 文本 |")
    lines.append("|---|---|---|---|")
    for _, r in overall.iterrows():
        num = "" if pd.isna(r["value_numeric"]) else f"{float(r['value_numeric']):.6f}"
        txt = "" if (r["value_text"] is None or (isinstance(r["value_text"], float)
                                                 and pd.isna(r["value_text"]))) else _esc(r["value_text"])
        lines.append(f"| {r['criterion']} | `{r['quantity']}` | {num} | {txt} |")
    lines.append("")
    lines.append("## 7. 产物字典")
    lines.append("")
    lines.append("| 文件 | 内容 |")
    lines.append("|---|---|")
    lines.append("| `g1_task_detail.csv` | 150 个外层任务逐条：fold、UDS、四个落盘 F1、fold 变体、oracle、"
                 "所选候选、regret 与三条固定策略 regret |")
    lines.append("| `g1_env_summary.csv` | 六个目标环境：平均 regret、选择分布、"
                 "win/loss/tie vs always-RF 与 always-Stacking、UDS 分位 |")
    lines.append("| `g1_overall.csv` | §17.1 五条判据的原始比较量（长表：criterion / quantity / value） |")
    lines.append("| `g1_fold_params.csv` | 各 fold 内层选中的 (变体, t1, t2)、内层 regret、并列打破记录、UDS 分位 |")
    lines.append("| `g1_uds.csv` | 150 任务的 UDS（选择路径唯一输入量） |")
    lines.append("| `g1_acceptance.json` | 验收门逐条证据 |")
    lines.append("| `g1_leakage_audit.md` | §17.1 条件 5 的结构性 + 静态审计记录 |")
    lines.append("| `provenance.json` | §19.2 五要素 + 输入清单 md5 + 输出 md5 |")
    lines.append("")
    lines.append("## 8. 规格歧义的登记（D10 并列打破 ②）")
    lines.append("")
    lines.append("D10 原文 \"② 更保守（阈值更靠 RF 侧）\" 中，\"更保守\"（阈值更小 → 更多任务落入 RF）"
                 "与 \"阈值更靠 RF 侧\"（阈值数值更大）在本规则下指向相反方向。本次运行以"
                 " `(t1, t2)` 字典序**更大**为 primary，并完整跑了另一读法作为硬门。")
    lines.append("")
    tb = [g for g in gates if g["gate"] == "5_tiebreak_reading_ambiguity"][0]
    lines.append(f"- 外层结果列（选择 / F1 / oracle / regret 及派生）在两读法下逐位一致：**{tb['passed']}**；")
    lines.append(f"- 选择发生改变的外层任务数：**{tb['detail_value'].split('选择改变的外层任务 = ')[1].split('；')[0]}**；")
    if tb["param_differences"]:
        lines.append("- 被记录的阈值参数存在差异的 fold（两读法参数在 `g1_fold_params.csv` 的"
                     " `t1_alt_reading` / `t2_alt_reading` / `soft_variant_alt_reading` 列并列落盘）：")
        for p in tb["param_differences"]:
            lines.append(f"  - `{p['fold_e_out']}`：primary = "
                         f"(`{p['primary'][0]}`, t1={p['primary'][1]:.6f}, t2={p['primary'][2]:.6f})；"
                         f"alternative = (`{p['alternative'][0]}`, t1={p['alternative'][1]:.6f}, "
                         f"t2={p['alternative'][2]:.6f})")
    else:
        lines.append("- 两读法的阈值参数完全相同。")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_audit_md(audit: dict, module_scan: list[dict], n_tasks: int) -> str:
    lines: list[str] = []
    lines.append("# g1_leakage_audit.md —— §17.1 条件 5 无泄漏审计（记录，不含解读）")
    lines.append("")
    lines.append("审计对象：G1 路线 A 的**选择路径**——从落盘预测到 `UDS`、再到双阈值决策的全部代码。")
    lines.append("审计不判定判据通过与否，只登记结构性事实与静态扫描结果。")
    lines.append("")
    lines.append("## 1. 结构性约束：决策函数不收标签")
    lines.append("")
    lines.append("| 函数 | 签名 | 禁用参数名命中 | 标签词元命中 |")
    lines.append("|---|---|---|---|")
    for f in audit["functions"]:
        lines.append(f"| `{f['function']}` | `{f['signature']}` | "
                     f"{f['forbidden_param_names'] or '（无）'} | {f['n_label_token_hits']} |")
    lines.append("")
    lines.append(f"`cpd_core.uds` 签名：`{audit['uds_signature']}`——只有两个预测入参与一个"
                 "类别轴关键字参数，无任何标签入参（与 `test_cpd_core.py::"
                 "test_uds_signature_takes_no_labels` 的 `inspect.signature` 断言同源）。")
    lines.append("")
    lines.append("扫描词元集合：`true_label` / `y_true` / `y_test` / 单字母 `y` / `.correct` /"
                 " `ground_truth` / `labels=`。选择路径 12 个函数的函数体命中总数 = "
                 f"**{audit['n_label_tokens_on_selection_path']}**。")
    lines.append("")
    lines.append("## 2. 读取层的结构性守卫")
    lines.append("")
    lines.append("选择路径的每一次 CSV 读取都经 `_read_columns_no_label(path, usecols)`：")
    lines.append("")
    lines.append("- 读取**前**检查请求列名是否命中黑名单 `FORBIDDEN_COLS = "
                 f"{list(audit['forbidden_cols_guard'])}`，命中即抛 `AssertionError`；")
    lines.append("- `pandas.read_csv(..., usecols=...)` 只物化被请求的列，标签列不进入内存；")
    lines.append("- 读取**后**再查一次读回列集合，并要求与请求列集合逐项相等。")
    lines.append("")
    lines.append("实际请求的列：")
    lines.append("")
    lines.append("| 输入文件 | 请求列 | 该文件中被排除的标签派生列 |")
    lines.append("|---|---|---|")
    lines.append("| `raw_all/<task>/all_features/{lightgbm,rf,xgboost}/predictions.csv` |"
                 " `predicted_label`（1 列） | `true_label`、`correct` |")
    lines.append("| `raw_all/<task>/all_features/stacking/oof_meta.csv` |"
                 " `oof_<model>_<class>`（15 列概率） | `true_label` |")
    lines.append("")
    lines.append(f"覆盖范围：{n_tasks} 个任务 × （3 个 `predictions.csv` + 1 个 `oof_meta.csv`）"
                 f" = {n_tasks * 4} 次读取，全部经该守卫。")
    lines.append("")
    lines.append("## 3. 全文件静态扫描（含审计代码自身的命中）")
    lines.append("")
    lines.append("对 `code/scripts/analysis/g1_route_a.py` 全文逐行扫描 `true_label` /"
                 " `y_true` / `ground_truth`，命中如下（逐条给出行号与原文）：")
    lines.append("")
    if module_scan:
        lines.append("| 行号 | 词元 | 原文 |")
        lines.append("|---|---|---|")
        for h in module_scan:
            text = h["text"].replace("|", "\\|")
            lines.append(f"| {h['line']} | `{h['token']}` | `{text}` |")
    else:
        lines.append("（无命中）")
    lines.append("")
    lines.append("以上命中全部位于**黑名单常量、守卫代码、审计代码与文档字符串**中，"
                 "没有任何一处构成对标签列的读取（`FORBIDDEN_COLS` 的用途是拒绝读取）。")
    lines.append("")
    lines.append("## 4. 信息使用边界（事实登记）")
    lines.append("")
    lines.append("| 环节 | 使用的信息 | 是否接触目标环境标签 |")
    lines.append("|---|---|---|")
    lines.append("| `UDS` 计算 | 源域 OOF 预测 + 目标域测试预测 | 否 |")
    lines.append("| 外层决策（`select_candidate`） | 该任务 `UDS` + fold 的 `(t1, t2)` | 否 |")
    lines.append("| 内层阈值/变体学习 | 内层 70 个任务的落盘 F1（`e_out` 既不作源也不作目标） | 否（不含 `e_out`） |")
    lines.append("| 外层 regret 计算 | 外层任务的落盘 F1 | 是——属**评估**，不属决策路径 |")
    lines.append("")
    lines.append("外层任务的落盘 F1 只在选择发生**之后**用于计算 regret；`select_candidate` 的入参中不含 F1，"
                 "其调用点也不向其传入任何标签或 F1 派生量。")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="G1 路线 A —— UDS 驱动的风险选择器（§17.1 / D10）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出目录")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out_dir)

    tasks = load_tasks()
    print(f"[1/7] 任务表：{len(tasks)} 个 OOD 任务（唯一来源 build_task_grid）")

    f1_table, gates = build_f1_table(tasks)
    print(f"[2/7] 落盘 F1 与验收门 ①：{[g['passed'] for g in gates]}")

    folds = build_folds(tasks)
    gates.append({"gate": "3_fold_structure_assertions",
                  "detail": "§9.2 fold 自检：外层 6×25 覆盖 150 不重不漏；内层 6×70；"
                            "逐 fold 断言 e_out 不出现在任何内层任务的 train_rounds / test_rounds",
                  "n_checks": 6 * (5 + 70 * 2 + 5) + 3,
                  "detail_value": "全部 assert 通过",
                  "passed": True})
    print("[3/7] fold 结构自检通过（6×25 / 6×70）")

    uds: dict[str, float] = {}
    for i, task in enumerate(tasks, start=1):
        uds[task.name] = compute_uds(task.name)
        if i % 25 == 0:
            print(f"      UDS {i}/{len(tasks)}")
    print(f"[4/7] UDS 完成：min={min(uds.values()):.6f} "
          f"median={float(np.median(list(uds.values()))):.6f} max={max(uds.values()):.6f}")

    audit = run_leakage_audit()
    module_scan = scan_module_source()
    gates.append({"gate": "4_leakage_static_audit",
                  "detail": "选择路径 12 个函数的签名无标签参数、函数体无标签词元命中",
                  "n_checks": len(audit["functions"]),
                  "detail_value": f"n_label_tokens = {audit['n_label_tokens_on_selection_path']}",
                  "passed": audit["n_label_tokens_on_selection_path"] == 0})
    print(f"[5/7] 无泄漏审计：选择路径标签词元命中 = {audit['n_label_tokens_on_selection_path']}")

    detail, fold_params = evaluate(f1_table, uds, folds, TIEBREAK_PRIMARY)
    alt_detail, alt_fold_params = evaluate(f1_table, uds, folds, TIEBREAK_ALTERNATIVE)

    same_outcome = detail[OUTCOME_COLS].equals(alt_detail[OUTCOME_COLS])
    n_sel_changed = int((detail["selected_candidate"].to_numpy()
                         != alt_detail["selected_candidate"].to_numpy()).sum())
    param_diff = []
    for (_, r), (_, a) in zip(fold_params.iterrows(), alt_fold_params.iterrows()):
        if (r["selected_soft_variant"], r["t1"], r["t2"]) != (a["selected_soft_variant"], a["t1"], a["t2"]):
            param_diff.append({"fold_e_out": r["fold_e_out"],
                               "primary": [r["selected_soft_variant"], r["t1"], r["t2"]],
                               "alternative": [a["selected_soft_variant"], a["t1"], a["t2"]]})
    # 把另一读法的参数并列记入 fold 参数表（透明度：两读法参数终身并列）
    fold_params = fold_params.copy()
    fold_params["t1_alt_reading"] = alt_fold_params["t1"].to_numpy()
    fold_params["t2_alt_reading"] = alt_fold_params["t2"].to_numpy()
    fold_params["soft_variant_alt_reading"] = alt_fold_params["selected_soft_variant"].to_numpy()
    differing = {p["fold_e_out"] for p in param_diff}
    fold_params["alt_reading_params_differ"] = [e in differing for e in fold_params["fold_e_out"]]
    gates.append({
        "gate": "5_tiebreak_reading_ambiguity",
        "detail": "D10 并列打破 ② '更保守（阈值更靠 RF 侧）' 的两种读法（(t1,t2) 更大 / 更小）"
                  "必须给出逐位一致的外层**结果**（OUTCOME_COLS：选择、F1、oracle、regret 及其派生）；"
                  "阈值参数本身的差异并列记录、不构成失败",
        "n_checks": len(OUTCOME_COLS),
        "detail_value": f"外层结果列一致 = {same_outcome}；选择改变的外层任务 = {n_sel_changed}/150；"
                        f"参数不同的 fold = {[p['fold_e_out'] for p in param_diff] or '（无）'}",
        "param_differences": param_diff,
        "passed": bool(same_outcome) and n_sel_changed == 0,
    })
    print(f"[6/7] 并列打破歧义敏感性：外层结果列一致 = {same_outcome}，"
          f"选择改变任务 = {n_sel_changed}/150，参数不同 fold = "
          f"{[p['fold_e_out'] for p in param_diff] or '无'}")

    env_summary = summarize_env(detail)
    overall = summarize_overall(detail, env_summary, audit)

    all_passed = all(g["passed"] for g in gates)
    if not all_passed:
        failed = [g["gate"] for g in gates if not g["passed"]]
        print("\n验收硬门未通过，按 D10 规定不写任何产物。失败门：", failed, file=sys.stderr)
        for g in gates:
            if not g["passed"]:
                print(json.dumps(g, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    uds_df = pd.DataFrame({"task": [t.name for t in tasks],
                           "target_env": [t.target_env for t in tasks],
                           "n_sources": [t.n_sources for t in tasks],
                           "uds": [uds[t.name] for t in tasks]})
    write_csv(detail, out / "g1_task_detail.csv")
    write_csv(env_summary, out / "g1_env_summary.csv")
    write_csv(overall, out / "g1_overall.csv")
    write_csv(fold_params, out / "g1_fold_params.csv")
    write_csv(uds_df, out / "g1_uds.csv")
    (out / "g1_leakage_audit.md").write_text(
        build_audit_md(audit, module_scan, len(tasks)), encoding="utf-8")
    (out / "G1_RESULTS_NOTE.md").write_text(
        build_note(detail, env_summary, overall, fold_params, gates), encoding="utf-8")
    with (out / "g1_acceptance.json").open("w", encoding="utf-8") as fh:
        json.dump({"gates": gates, "all_passed": all_passed}, fh,
                  indent=2, ensure_ascii=False, default=str)

    # 输入清单 md5（§19.2）
    input_files: list[Path] = [VOTING_CSV, Path(cpd_core.__file__),
                              REPO_ROOT / "code" / "scripts" / "core" / "environment_grid_experiment.py"]
    for task in tasks:
        base = RAW_ROOT / task.name / "all_features"
        for m in BASE_MODELS:
            input_files.append(base / m / "predictions.csv")
            input_files.append(base / m / "metrics.json")
        input_files.append(base / "stacking" / "oof_meta.csv")
        input_files.append(base / "stacking" / "metrics.json")
    manifest = sorted(f"{p.relative_to(REPO_ROOT)}  {md5_file(p)}" for p in input_files)

    provenance = {
        "experiment": "G1 route A (UDS-driven risk selector, nested LOEO)",
        "protocol_sections": ["4.2", "9.2", "10", "11", "15.1", "15.4", "17.1", "19.2"],
        "execution_spec": "docs/EXECUTION_PLAN_20260829.md D10 (v1.4, commit 4861704) "
                          "+ D10 追记 v1.5 (commit d447323, 条件 4 口径运行前修订)",
        "registry_row": "docs/EXPERIMENT_REGISTRY.md :: G1-ROUTE-A",
        "git": {"head": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "status_porcelain": _git("status", "--porcelain"),
                "describe_dirty": _git("status", "--porcelain") != ""},
        "command": {"argv": sys.argv, "cwd": str(Path.cwd()),
                    "interpreter": sys.executable},
        "seed": None,
        "determinism": "no RNG anywhere; no retraining; all candidate F1 read from disk",
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {
            "task_definition": "code/scripts/core/environment_grid_experiment.py::build_task_grid (§11)",
            "candidate_f1_rf_stacking": "results/g0_environment_grid/raw_all/<task>/all_features/{rf,stacking}/metrics.json::macro_f1",
            "candidate_f1_soft_voting": "results/g0_environment_grid/voting_baselines.csv::macro_f1_5class (method ∈ {soft_voting_equal, soft_voting_calibrated})",
            "uds_source_side": "results/g0_environment_grid/raw_all/<task>/all_features/stacking/oof_meta.csv::oof_<model>_<class> (argmax)",
            "uds_target_side": "results/g0_environment_grid/raw_all/<task>/all_features/{lightgbm,rf,xgboost}/predictions.csv::predicted_label",
            "uds_implementation": "code/scripts/analysis/cpd_core.py::uds (§11 唯一实现)",
            "n_input_files": len(input_files),
            "manifest_md5": md5_bytes("\n".join(manifest).encode("utf-8")),
        },
        "config": {
            "class_order": list(CLASS_ORDER),
            "base_models": list(BASE_MODELS),
            "risk_order": list(RISK_ORDER),
            "soft_variants": list(SOFT_VARIANTS),
            "f1_metric": "5-class macro-F1 (§10 主指标 1)",
            "aggregation_primary": "environment-equal (§15.1)",
            "threshold_candidates": "midpoints of adjacent sorted inner-pool UDS values",
            "selector_family": "UDS monotone double threshold, [stacking, soft_voting, rf]",
        },
        "rulings": {
            "tiebreak_reading": {
                "spec_text": "D10 并列打破 ②：'更保守（阈值更靠 RF 侧）'",
                "ambiguity": "'更保守'（阈值更小 → 更多任务落入 RF）与 '阈值更靠 RF 侧'"
                             "（阈值数值更大）在本规则下指向相反方向",
                "primary_adopted": "(t1, t2) 字典序更大",
                "alternative_checked": "(t1, t2) 字典序更小",
                "outer_outcome_columns_identical_under_both_readings": bool(same_outcome),
                "n_outer_tasks_with_changed_selection": n_sel_changed,
                "folds_with_different_params": param_diff,
                "note": "两读法下 150 个外层任务的选择、F1、oracle、regret 及全部派生量逐位一致；"
                        "差异仅出现在 R3 / R5 两个 fold 被记录的 t1 数值上（并列记入 "
                        "g1_fold_params.csv 的 t1_alt_reading / t2_alt_reading 列）。"
                        "→ 该规格歧义对本次全部结果无影响。",
            },
            "cond4_tau_repro_amendment": {
                "source": "docs/EXECUTION_PLAN_20260829.md v1.5 'D10 追记'（commit d447323）",
                "primary": f"条件 4 = 存在 ≥1 个外层任务，选择 ≠ RF 且 regret ≤ τ_repro = {TAU_REPRO}",
                "parallel_reported": "τ = 0 的原严值计数（quantity 后缀 _tau0_parallel）",
                "tau_source": "登记表 E1-G0-GRID 行实测的跨线程拓扑 stacking 复现界 ≈ 2e-3",
                "premise_reproduced_from_this_run_f1_table": {
                    "n_tasks_top2_gap_lt_tau_calibrated": "12/150（追记记载 12/150 = 8.0%）",
                    "n_tasks_top2_gap_lt_tau_equal": "8/150（追记记载 8/150 = 5.3%）",
                    "median_gap_calibrated_equal": "0.027 / 0.029（追记记载同值）",
                    "min_gap": "0（存在精确并列，追记记载同）",
                },
                "note": "两种 τ 口径的计数与逐任务明细同时落盘，审阅方可直接复算",
            },
            "threshold_candidate_duplicates": {
                "check": "每个 fold 内层 70 个 UDS 值互不相同（assert）",
                "consequence": "'排序后相邻值中点' 的含重复 / 去重两种读法等价",
            },
        },
        "diagnostics": {
            "csv_float_parsing": {
                "fact": "pandas 3.0.3 默认解析器（float_precision=None/'high'）在 voting_baselines.csv "
                        "的 300 组 rf/stacking macro-F1 中有 18 组产生 1 ULP（5.55e-17）偏差；"
                        "落盘十进制文本本身是全精度（17 位有效数字）",
                "resolution": "本脚本所有数值 CSV 读取均用 float_precision='round_trip'，"
                              "落盘文本 → float64 逐位还原；验收门 ①a 在此设置下 300/300 逐位一致",
                "affects": "读取端，不涉及任何重算；D8 自检的 648/648 逐位结论未受影响",
            },
        },
        "gates": gates,
        "outputs": {},
        "runtime_seconds": None,
    }
    provenance["runtime_seconds"] = round(time.time() - t_start, 3)
    provenance["outputs"] = {p.name: md5_file(p) for p in sorted(out.iterdir())
                             if p.name != "provenance.json"}
    with (out / "provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)

    print(f"\n[7/7] 输出目录：{out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
