#!/usr/bin/env python3
"""E2 —— CPD 条件解释力的分层回归（协议 §13）。

本脚本**只产出数表**。任何解读、机制语言、结论性表述都不在本文件（及其产物）范围内：
协议 §13 的预注册解读分支由审阅方作出判定。

协议依据
--------
* §13：分层回归 M0 / M1 / M2 的定义、观测量（增量 R²、按环境聚类 bootstrap、
  留一环境 CV 预测误差改善）、统计纪律（不追求"凑显著"、不做未校正多重检验）；
* §15：统计单位 = 测试环境；正式区间用按测试环境聚类的 bootstrap（§15.1）；
  禁止点级 bootstrap / 把 n=150 当独立样本（§15.2）；禁止未校正多重检验（§15.3）；
  主表须同时给出六个目标环境各自的结果（§15.4）；leave-one-environment-out 敏感性（§15.5）；
* §4.2 / §4.3：`CPD_y` / `CPD_dir` 定义与逐行 `n_err ≥ 20` 门槛；
* §11：`CPD_y` / `CPD_dir` 的唯一实现是 `cpd_core`，本文件不含任何私有 CPD 公式副本；
* §19.2：git HEAD / 命令行 / 种子 / 包版本 / 输入清单持久化；
* §7：RF 口径（基线 1，n_estimators=500, class_weight=balanced，由 G0 落盘产物给定）。

执行口径：`docs/EXECUTION_PLAN_20260829.md` 决策 **D9**（先写后看，commit cac9b3f）。

口径要点（逐条对应 D9）
----------------------
因变量 DV
    `results/e1_oof_arms_g0/e1_arms_raw_all.csv` 的 **B 臂** `gain_absolute`，seed 42，
    150 个 G0 OOD 网格任务。A 臂 DV 作敏感性重跑（历史口径对照）。

任务定义
    唯一来源 = `environment_grid_experiment.build_task_grid()`（§11 唯一实现纪律，
    与 D2 同一约束）。本文件不重复实现任务生成，也不从任务名反解源集合。

M0（7 维，逐字按 §13）
    对每类 c：`recall_src_IID(c) − recall_tgt(c)`（5 维）+ 该 5 维向量的 L2 范数 + 最大值。
    * `recall_src_IID(c)` = 源集合 S 中各环境 G0 IID（**time_block**，§8.4 诚实域内参照）
      混淆矩阵逐类 recall 的**算术均值**；敏感性变体用 `random`。
    * `recall_tgt(c)` = 该任务测试混淆矩阵逐类 recall。
    * 模型口径 = **RF**；敏感性 = rf / xgboost / lightgbm 三基模型逐类 recall 的均值。
    * "最大值" 取该 5 维差向量的**最大元素**（字面读法，非最大绝对值）；
      两种读法不一致的任务数记入 provenance 的 `diagnostics`。

M1 = M0 + `CPD_y`；M2 = M0 + `CPD_dir`
    `ref` = 源集合 S 各环境的 IID CM，`tgt` = 该任务 CM。**多源参照构造沿用 0.8397
    历史口径的同一构造**：对每个参照 CM 分别算指标，再对这些值取算术平均
    （不是先平均 CM）。出处：`code/scripts/analysis/test_cpd_core.py` 第 100-101 行
    `per_iid = [cpd_y(load_cm(ROOT_V2, t), tgt) for t in IID_TASKS]; got = float(np.mean(per_iid))`，
    文档转录见 `docs/CPD_DEFINITIONS.md` §4.1 表。

    `CPD_dir` 的 NaN 传播（D9 原文未规定；**裁定 B**：primary = `lenient`，
    `strict` 并列完整报告并带秩亏标记，两变体终身并列、不合并、不取优）：
      * `strict`  —— 逐字用 `np.mean`：任一参照未定义 → 该任务未定义（NaN）；
      * `lenient` —— 用 `np.nanmean`：在已定义的参照上取均值，全部未定义才 NaN。
    M2 及其 M0 对照在**同一子集**上重新拟合（否则增量不可比）；覆盖率随产物报告。

估计与观测量（§13 双通道）
    ① 标准化 OLS（X、y 各自 z-score 后最小二乘；标准化系数即 β）：
       R²、增量 R²、Cohen's f² = ΔR² / (1 − R²_full)、按目标环境聚类的 bootstrap 95% CI；
    ② 留一目标环境 CV：6 折（每折 = 1 个目标环境），标准化参数只在训练折上估计，
       报 pooled 样本外 MSE 与逐折 MSE。
    另报每 |S| 分层的描述性均值（**不加回归项**）与逐目标环境完整结果表（§15.4）。
    **不报 p 值**（§15.3）。预注册通道之外不加任何协变量。

验收硬门（不过即非零退出，不写任何输出）
    a) M0 特征抽 2 个任务（1 个 |S|=1、1 个 |S|=3）由**独立代码路径手算**比对，容差 1e-12；
    b) E2 的 `CPD_y` 在全部 30 个 |S|=1 任务上与
       `results/g0_environment_grid/env_topology_cpd_y_ref_time_block_rf.csv`
       对应单元格**逐位一致**（float64 位模式相同）；按**裁定 A**，该门只在 canonical
       环境（iotcls，`code/requirements-lock.txt`）内判定；
    c) bootstrap 同种子双跑的复制统计量 md5 相同（裁定 B 并列要求：两个 `CPD_dir` 变体都跑）；
    d) 行数核对：主表 n = 150；M2 子表 n = 对应 `CPD_dir` 变体的覆盖数。

裁定（`docs/EXECUTION_PLAN_20260829.md` v1.3 "D9 追记"，逐条落在 provenance 的 `rulings`）
    A canonical 分析环境 = iotcls；跨环境 ULP 差异按可复现性事实登记。
    B `CPD_dir` NaN 传播 primary = lenient，strict 并列且带秩亏标记。
    C `d_recall_max` 维持字面读法；与最大绝对值读法不一致的任务逐条记入 provenance。

用法::

    python3 code/scripts/analysis/e2_conditional_explanation.py
    python3 code/scripts/analysis/e2_conditional_explanation.py --dry-run
    python3 code/scripts/analysis/e2_conditional_explanation.py --bootstrap-b 1000
"""
from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

import cpd_core  # noqa: E402
from cpd_core import cpd_dir, cpd_y  # noqa: E402
from environment_grid_experiment import build_task_grid  # noqa: E402  (§11 任务定义唯一来源)

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
#: 类别轴顺序（`docs/CPD_DEFINITIONS.md` §1.2 的一致约定）。
CLASS_ORDER = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]

#: 六个物理环境（§3.1）。
ENVIRONMENTS = ["R2", "R3", "R4", "R5", "R6", "R7"]

#: IID 参照的两个划分变体（§8.4）。primary = time_block（诚实域内参照，D4/D9 同一口径）。
IID_VARIANTS = ("time_block", "random")

#: 三个基模型（§7 基线 1-3）。primary 只用 rf；三模型均值为 M0 的敏感性变体。
BASE_MODELS = ("rf", "xgboost", "lightgbm")

FEATURE_SET = "all_features"

E1_CSV = REPO_ROOT / "results" / "e1_oof_arms_g0" / "e1_arms_raw_all.csv"
G0_RAW = REPO_ROOT / "results" / "g0_environment_grid" / "raw_all"
TOPOLOGY_CPD_Y_TIME_BLOCK = (
    REPO_ROOT / "results" / "g0_environment_grid" / "env_topology_cpd_y_ref_time_block_rf.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "e2_conditional"

#: M0 的 7 个预测变量名（顺序固定，回归设计矩阵按此列序）。
M0_TERMS = [f"d_recall_{c}" for c in CLASS_ORDER] + ["d_recall_l2", "d_recall_max"]

#: `CPD_dir` 的两种 NaN 传播变体（D9 原文未规定，并列输出）。
DIR_VARIANTS = ("strict", "lenient")

#: 裁定 B（`docs/EXECUTION_PLAN_20260829.md` v1.3 "D9 追记"）：primary = `lenient`（nanmean），
#: `strict` 并列完整报告且必须带秩亏标记。两变体**终身并列**——本脚本所有表格同时输出两者，
#: 本映射只标注角色，不改变任何计算、不删除任何一侧。
DIR_VARIANT_ROLE = {"lenient": "primary", "strict": "parallel_reported"}

HANDCHECK_TOL = 1e-12


# --------------------------------------------------------------------------- #
# 输入读取
# --------------------------------------------------------------------------- #
def load_cm(task: str, model: str = "rf") -> np.ndarray:
    """读取 G0 落盘的**原始计数**混淆矩阵，并校验类别轴顺序。

    CSV 带 UTF-8 BOM，用 utf-8-sig 读。轴序不一致会让逐元素比较静默出错，故硬校验。
    """
    path = G0_RAW / task / FEATURE_SET / model / "confusion_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"混淆矩阵缺失：{path}")
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    idx = [str(i) for i in df.index]
    cols = [str(c) for c in df.columns]
    if idx != CLASS_ORDER or cols != CLASS_ORDER:
        raise ValueError(f"{path} 类别轴不符：index={idx} columns={cols} 期望 {CLASS_ORDER}")
    return df.values.astype(float)


def iid_task_name(env: str, variant: str) -> str:
    if variant not in IID_VARIANTS:
        raise ValueError(f"未知 IID 划分变体 {variant!r}，只允许 {IID_VARIANTS}")
    return f"g0_iid_{env}_{variant}"


def recall_vector(cm: np.ndarray) -> np.ndarray:
    """逐类 recall：`C_ii / Σ_j C_ij`。

    行和为 0 时该行除以 1（沿用 `cpd_core.normalize_cm` 的历史口径，§1.2），
    此时 recall = 0 而不是 NaN。
    """
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    denom = np.where(row_sums == 0, 1.0, row_sums)
    return np.diag(cm) / denom


def recall_vector_multi_model(task: str, models=BASE_MODELS) -> np.ndarray:
    """多个基模型逐类 recall 的算术均值（M0 的 `recall_model=base_mean` 敏感性变体）。"""
    return np.mean([recall_vector(load_cm(task, m)) for m in models], axis=0)


def load_grid_tasks() -> list[dict]:
    """从 G0 生成器取 150 个 OOD 任务定义（§11：任务定义唯一来源，不在此重复实现）。"""
    tasks = [t for t in build_task_grid() if t["grid_kind"] == "ood"]
    if len(tasks) != 150:
        raise ValueError(f"G0 生成器返回 {len(tasks)} 个 OOD 任务，期望 150")
    return tasks


def load_dv_table() -> pd.DataFrame:
    """E1-G0-GRID 三臂 gain 表 → 逐任务 DV（A / B 臂，seed 42）。"""
    df = pd.read_csv(E1_CSV, encoding="utf-8-sig")
    df = df[df["seed"] == 42]
    piv = df.pivot_table(index="task", columns="arm", values="gain_absolute")
    missing = [a for a in ("A", "B") if a not in piv.columns]
    if missing:
        raise ValueError(f"{E1_CSV} 缺臂：{missing}")
    if piv[["A", "B"]].isna().any().any():
        raise ValueError("E1 gain 表存在缺失值")
    return piv


# --------------------------------------------------------------------------- #
# 特征构建
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """一个分析变体（primary 或某个敏感性臂）。"""

    name: str
    dv_arm: str            # "A" | "B"
    iid_variant: str       # "time_block" | "random"
    recall_model: str      # "rf" | "base_mean"
    note: str


VARIANTS = (
    Variant("primary", "B", "time_block", "rf",
            "D9 主口径：B 臂 DV，time_block IID 参照，RF recall"),
    Variant("sens_dv_a_arm", "A", "time_block", "rf",
            "敏感性：DV 换 A 臂（历史随机 OOF 口径）"),
    Variant("sens_ref_random", "B", "random", "rf",
            "敏感性：IID 参照换 random 划分（recall_src_IID 与 CPD 参照同时切换）"),
    Variant("sens_recall_3model", "B", "time_block", "base_mean",
            "敏感性：M0 的 recall 换三基模型均值；CPD 仍为 RF 口径（D9 只对 M0 指定该敏感性）"),
)


def build_features(variant: Variant, tasks: list[dict], dv: pd.DataFrame) -> pd.DataFrame:
    """构造一个变体的完整逐任务特征表（150 行）。"""
    rows: list[dict] = []
    for t in tasks:
        task = t["name"]
        srcs = list(t["train_rounds"])
        tgt_env = t["target_env"]
        cm_tgt_rf = load_cm(task, "rf")

        # ---- M0：逐类 recall 变化（模型口径由 variant.recall_model 决定）----------
        if variant.recall_model == "rf":
            r_tgt = recall_vector(cm_tgt_rf)
            r_src = np.mean(
                [recall_vector(load_cm(iid_task_name(s, variant.iid_variant), "rf")) for s in srcs],
                axis=0,
            )
        elif variant.recall_model == "base_mean":
            r_tgt = recall_vector_multi_model(task)
            r_src = np.mean(
                [recall_vector_multi_model(iid_task_name(s, variant.iid_variant)) for s in srcs],
                axis=0,
            )
        else:
            raise ValueError(f"未知 recall_model={variant.recall_model!r}")
        delta = r_src - r_tgt

        # ---- M1 / M2：CPD 参照恒为 RF CM（D9 的 CPD 口径），多源 = 逐参照算后取均值 ----
        ref_cms = [load_cm(iid_task_name(s, variant.iid_variant), "rf") for s in srcs]
        per_ref_y = [cpd_y(ref, cm_tgt_rf) for ref in ref_cms]
        dir_res = [cpd_dir(ref, cm_tgt_rf, min_err=cpd_core.DEFAULT_MIN_ERR) for ref in ref_cms]
        per_ref_dir = [r.value for r in dir_res]
        n_ref_defined = sum(int(r.is_defined) for r in dir_res)

        rec = {
            "task": task,
            "n_sources": t["n_sources"],
            "source_envs": ";".join(srcs),
            "target_env": tgt_env,
            "dv_gain_absolute": float(dv.loc[task, variant.dv_arm]),
        }
        for i, c in enumerate(CLASS_ORDER):
            rec[f"recall_src_iid_{c}"] = float(r_src[i])
            rec[f"recall_tgt_{c}"] = float(r_tgt[i])
            rec[f"d_recall_{c}"] = float(delta[i])
        rec["d_recall_l2"] = float(np.linalg.norm(delta))
        rec["d_recall_max"] = float(np.max(delta))
        rec["cpd_y"] = float(np.mean(per_ref_y))
        rec["cpd_dir_strict"] = float(np.mean(per_ref_dir))
        rec["cpd_dir_lenient"] = (
            float(np.nanmean(per_ref_dir)) if n_ref_defined > 0 else float("nan")
        )
        rec["n_ref"] = len(srcs)
        rec["n_ref_dir_defined"] = n_ref_defined
        rec["per_ref_cpd_y"] = ";".join(f"{v!r}" for v in per_ref_y)
        rec["per_ref_cpd_dir"] = ";".join(f"{v!r}" for v in per_ref_dir)
        rec["dir_included_rows"] = "|".join(
            ";".join(CLASS_ORDER[i] for i in r.included_rows) for r in dir_res
        )
        # 诊断用，不作为回归项：字面 max 与 max-abs 两种读法是否一致
        rec["_diag_d_recall_max_abs"] = float(np.max(np.abs(delta)))
        rows.append(rec)

    df = pd.DataFrame(rows)
    if len(df) != 150:
        raise ValueError(f"特征表行数 {len(df)} ≠ 150")
    return df


# --------------------------------------------------------------------------- #
# 标准化 OLS
# --------------------------------------------------------------------------- #
def _zscore(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe = np.where(std > 0, std, 1.0)
    return (a - mean) / safe


def fit_std_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """标准化 OLS：X、y 各自 z-score 后带截距最小二乘。

    返回标准化系数 `beta`（即效应量口径的 β）、`r2`、`adj_r2`、`n`、`k`。
    零方差列的 std 记 1（该列 z 分全 0，不贡献自由度）；设计阵秩亏时 `lstsq`
    给最小范数解（bootstrap 重抽可能出现），并置 `degenerate=True`。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    xm, xs = x.mean(axis=0), x.std(axis=0, ddof=0)
    ym, ys = y.mean(), y.std(ddof=0)
    xz = _zscore(x, xm, xs)
    yz = (y - ym) / (ys if ys > 0 else 1.0)
    design = np.column_stack([np.ones(n), xz])
    rank = np.linalg.matrix_rank(design)
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)
    pred = design @ coef
    sse = float(((yz - pred) ** 2).sum())
    sst = float(((yz - yz.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 and np.isfinite(r2) else float("nan")
    return {
        "beta": coef[1:],
        "intercept": float(coef[0]),
        "r2": float(r2),
        "adj_r2": float(adj),
        "n": int(n),
        "k": int(k),
        "design_rank": int(rank),
        "design_n_params": int(design.shape[1]),
        "n_zero_variance_cols": int((xs == 0).sum()),
        "degenerate": bool(rank < design.shape[1]),
    }


def predict_std_ols(fit_x: np.ndarray, fit_y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    """在训练折上估计标准化参数与系数，回到 DV 原尺度对新样本预测（LOEO-CV 用）。"""
    fit_x = np.asarray(fit_x, dtype=float)
    fit_y = np.asarray(fit_y, dtype=float)
    new_x = np.asarray(new_x, dtype=float)
    xm, xs = fit_x.mean(axis=0), fit_x.std(axis=0, ddof=0)
    ym, ys = fit_y.mean(), fit_y.std(ddof=0)
    ys_safe = ys if ys > 0 else 1.0
    xz = _zscore(fit_x, xm, xs)
    yz = (fit_y - ym) / ys_safe
    design = np.column_stack([np.ones(len(xz)), xz])
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)
    nz = _zscore(new_x, xm, xs)
    new_design = np.column_stack([np.ones(len(nz)), nz])
    return (new_design @ coef) * ys_safe + ym


# --------------------------------------------------------------------------- #
# 模型规格
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """一个回归规格：名字 + 预测变量列 + 拟合子集限定列（None = 全 150 行）。"""

    name: str
    terms: tuple[str, ...]
    subset_col: str | None


def model_specs(dir_variant: str) -> dict[str, ModelSpec]:
    """M0 / M1 全样本；M2 与其 M0 对照在同一 `CPD_dir` 可用子集上（D9）。"""
    dir_col = f"cpd_dir_{dir_variant}"
    return {
        "M0": ModelSpec("M0", tuple(M0_TERMS), None),
        "M1": ModelSpec("M1", tuple(M0_TERMS) + ("cpd_y",), None),
        "M0_sub": ModelSpec("M0_sub", tuple(M0_TERMS), dir_col),
        "M2": ModelSpec("M2", tuple(M0_TERMS) + (dir_col,), dir_col),
    }


def subset_frame(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    return df if spec.subset_col is None else df[df[spec.subset_col].notna()]


def design(df: pd.DataFrame, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    sub = subset_frame(df, spec)
    return sub[list(spec.terms)].to_numpy(dtype=float), sub["dv_gain_absolute"].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# 按目标环境聚类的 bootstrap（§15.1）
# --------------------------------------------------------------------------- #
def cluster_bootstrap(df: pd.DataFrame, dir_variant: str, b: int, seed: int) -> dict:
    """重抽 **6 个目标环境**（有放回），每次重拟合全部规格，记录复制统计量。

    §15.2 禁止点级 bootstrap：这里重抽的单位是环境（cluster），被抽中的环境整块进入，
    同一环境被抽中两次即其全部任务重复计入。

    Returns:
        {"stats": (B, n_stat) float64 数组, "names": [...], "n_degenerate": ..., "md5": ...}
    """
    specs = model_specs(dir_variant)
    envs = sorted(df["target_env"].unique().tolist())
    by_env = {e: df[df["target_env"] == e] for e in envs}
    rng = np.random.default_rng(seed)

    stat_names: list[str] = []
    for m in ("M0", "M1", "M0_sub", "M2"):
        stat_names.append(f"r2_{m}")
    stat_names += ["dr2_M1_vs_M0", "dr2_M2_vs_M0sub"]
    for m in ("M1", "M2"):
        for term in specs[m].terms:
            stat_names.append(f"beta_{m}_{term}")

    out = np.full((b, len(stat_names)), np.nan, dtype=float)
    n_degenerate = 0
    deg_by_model = {m: 0 for m in ("M0", "M1", "M0_sub", "M2")}
    for it in range(b):
        draw = rng.integers(0, len(envs), size=len(envs))
        boot = pd.concat([by_env[envs[i]] for i in draw], ignore_index=True)
        vals: dict[str, float] = {}
        deg = False
        fits: dict[str, dict] = {}
        for key in ("M0", "M1", "M0_sub", "M2"):
            spec = specs[key]
            x, y = design(boot, spec)
            if len(y) <= len(spec.terms) + 1:
                fits[key] = None
                continue
            fit = fit_std_ols(x, y)
            fits[key] = fit
            deg_by_model[key] += int(fit["degenerate"])
            deg = deg or fit["degenerate"]
            vals[f"r2_{key}"] = fit["r2"]
            if key in ("M1", "M2"):
                for j, term in enumerate(spec.terms):
                    vals[f"beta_{key}_{term}"] = float(fit["beta"][j])
        if fits["M0"] and fits["M1"]:
            vals["dr2_M1_vs_M0"] = fits["M1"]["r2"] - fits["M0"]["r2"]
        if fits["M0_sub"] and fits["M2"]:
            vals["dr2_M2_vs_M0sub"] = fits["M2"]["r2"] - fits["M0_sub"]["r2"]
        n_degenerate += int(deg)
        for j, name in enumerate(stat_names):
            if name in vals:
                out[it, j] = vals[name]

    md5 = hashlib.md5(np.ascontiguousarray(out).tobytes()).hexdigest()
    return {"stats": out, "names": stat_names, "n_degenerate": n_degenerate,
            "n_degenerate_by_model": deg_by_model, "md5": md5,
            "b": b, "seed": seed, "n_envs": len(envs)}


def percentile_ci(stats: np.ndarray, names: list[str], name: str) -> tuple[float, float, int]:
    col = stats[:, names.index(name)]
    col = col[np.isfinite(col)]
    if col.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(col, [2.5, 97.5])
    return float(lo), float(hi), int(col.size)


# --------------------------------------------------------------------------- #
# 留一目标环境 CV（§15.5）
# --------------------------------------------------------------------------- #
def loeo_cv(df: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, float]:
    """6 折（每折 = 1 个目标环境）样本外预测；标准化参数只在训练折上估计。

    Returns:
        (逐折表, pooled MSE)。
    """
    sub = subset_frame(df, spec)
    envs = sorted(sub["target_env"].unique().tolist())
    recs, all_err = [], []
    for e in envs:
        te = sub[sub["target_env"] == e]
        tr = sub[sub["target_env"] != e]
        if len(tr) <= len(spec.terms) + 1 or len(te) == 0:
            recs.append({"target_env": e, "n_test": len(te), "n_train": len(tr),
                         "mse": float("nan")})
            continue
        pred = predict_std_ols(tr[list(spec.terms)].to_numpy(float),
                               tr["dv_gain_absolute"].to_numpy(float),
                               te[list(spec.terms)].to_numpy(float))
        err = te["dv_gain_absolute"].to_numpy(float) - pred
        all_err.append(err)
        recs.append({"target_env": e, "n_test": int(len(te)), "n_train": int(len(tr)),
                     "mse": float((err ** 2).mean())})
    pooled = float((np.concatenate(all_err) ** 2).mean()) if all_err else float("nan")
    return pd.DataFrame(recs), pooled


# --------------------------------------------------------------------------- #
# 一个变体的完整分析
# --------------------------------------------------------------------------- #
def analyse_variant(variant: Variant, df: pd.DataFrame, b: int, seed: int) -> dict:
    res = {"variant": variant.name, "dir": {}}
    for dv_name in DIR_VARIANTS:
        specs = model_specs(dv_name)
        boot = cluster_bootstrap(df, dv_name, b=b, seed=seed)
        fits, reg_rows, coef_rows, cv_rows = {}, [], [], []
        for key in ("M0", "M1", "M0_sub", "M2"):
            spec = specs[key]
            x, y = design(df, spec)
            fits[key] = fit_std_ols(x, y)
            fold, pooled = loeo_cv(df, spec)
            fold.insert(0, "model", key)
            fold.insert(0, "cpd_dir_variant", dv_name)
            fold.insert(0, "variant", variant.name)
            cv_rows.append(fold)
            fits[key]["cv_mse_pooled"] = pooled
            for j, term in enumerate(spec.terms):
                if key in ("M1", "M2"):
                    lo, hi, nb = percentile_ci(boot["stats"], boot["names"], f"beta_{key}_{term}")
                else:
                    lo, hi, nb = float("nan"), float("nan"), 0
                coef_rows.append({
                    "variant": variant.name, "cpd_dir_variant": dv_name,
                    "cpd_dir_role": DIR_VARIANT_ROLE[dv_name], "model": key,
                    "term": term, "std_beta": float(fits[key]["beta"][j]),
                    "boot_ci_lo": lo, "boot_ci_hi": hi, "n_boot_finite": nb,
                    # 裁定 B：strict 侧的秩亏必须随系数一同可见
                    "design_rank_deficient": bool(fits[key]["degenerate"]),
                    "n_boot_degenerate_replicates_this_model":
                        int(boot["n_degenerate_by_model"][key]),
                    "B": int(boot["b"]),
                })
        for key, base in (("M0", None), ("M1", "M0"), ("M0_sub", None), ("M2", "M0_sub")):
            f = fits[key]
            sub = subset_frame(df, specs[key])
            zero_var = [t for t in specs[key].terms
                        if float(sub[t].std(ddof=0)) == 0.0]
            row = {
                "variant": variant.name, "cpd_dir_variant": dv_name,
                "cpd_dir_role": DIR_VARIANT_ROLE[dv_name], "model": key,
                "n": f["n"], "k_predictors": f["k"], "r2": f["r2"], "adj_r2": f["adj_r2"],
                "cv_mse_pooled_loeo": f["cv_mse_pooled"],
                "design_rank": f["design_rank"], "design_n_params": f["design_n_params"],
                "design_rank_deficient": f["degenerate"],
                "zero_variance_terms": ";".join(zero_var),
                "n_boot_degenerate_replicates_this_model":
                    int(boot["n_degenerate_by_model"][key]),
                "n_boot_replicates": int(boot["b"]),
            }
            r2lo, r2hi, _ = percentile_ci(boot["stats"], boot["names"], f"r2_{key}")
            row["r2_boot_ci_lo"], row["r2_boot_ci_hi"] = r2lo, r2hi
            if base is None:
                row.update({"baseline_model": "", "delta_r2": float("nan"),
                            "delta_r2_boot_ci_lo": float("nan"), "delta_r2_boot_ci_hi": float("nan"),
                            "cohens_f2": float("nan"), "cv_mse_improvement_vs_baseline": float("nan"),
                            "cv_mse_rel_improvement": float("nan")})
            else:
                d = f["r2"] - fits[base]["r2"]
                stat = "dr2_M1_vs_M0" if key == "M1" else "dr2_M2_vs_M0sub"
                lo, hi, _ = percentile_ci(boot["stats"], boot["names"], stat)
                imp = fits[base]["cv_mse_pooled"] - f["cv_mse_pooled"]
                row.update({
                    "baseline_model": base, "delta_r2": d,
                    "delta_r2_boot_ci_lo": lo, "delta_r2_boot_ci_hi": hi,
                    "cohens_f2": d / (1.0 - f["r2"]) if f["r2"] < 1 else float("nan"),
                    "cv_mse_improvement_vs_baseline": imp,
                    "cv_mse_rel_improvement": imp / fits[base]["cv_mse_pooled"]
                    if fits[base]["cv_mse_pooled"] > 0 else float("nan"),
                })
            reg_rows.append(row)
        res["dir"][dv_name] = {
            "regression": pd.DataFrame(reg_rows),
            "coefficients": pd.DataFrame(coef_rows),
            "cv": pd.concat(cv_rows, ignore_index=True),
            "bootstrap": boot,
            "fits": fits,
        }
    return res


# --------------------------------------------------------------------------- #
# 描述性表
# --------------------------------------------------------------------------- #
def table_by_source_count(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """每 |S| 分层的描述性均值（D9：**不加回归项**）。"""
    recs = []
    for k, g in df.groupby("n_sources", sort=True):
        rec = {"variant": variant, "n_sources": int(k), "n_tasks": int(len(g)),
               "dv_mean": float(g["dv_gain_absolute"].mean()),
               "dv_std": float(g["dv_gain_absolute"].std(ddof=1)),
               "dv_min": float(g["dv_gain_absolute"].min()),
               "dv_max": float(g["dv_gain_absolute"].max())}
        for col in M0_TERMS + ["cpd_y"]:
            rec[f"{col}_mean"] = float(g[col].mean())
        for dvv in DIR_VARIANTS:
            c = f"cpd_dir_{dvv}"
            rec[f"{c}_mean"] = float(g[c].mean()) if g[c].notna().any() else float("nan")
            rec[f"{c}_n_defined"] = int(g[c].notna().sum())
        recs.append(rec)
    return pd.DataFrame(recs)


def table_by_target_env(df: pd.DataFrame, variant: str, analysis: dict) -> pd.DataFrame:
    """逐目标环境完整结果表（§15.4）：描述统计 + 各模型该环境作留出折的 MSE。"""
    cv_all = pd.concat([analysis["dir"][d]["cv"] for d in DIR_VARIANTS], ignore_index=True)
    recs = []
    for e, g in df.groupby("target_env", sort=True):
        rec = {"variant": variant, "target_env": e, "n_tasks": int(len(g)),
               "dv_mean": float(g["dv_gain_absolute"].mean()),
               "dv_std": float(g["dv_gain_absolute"].std(ddof=1)),
               "dv_min": float(g["dv_gain_absolute"].min()),
               "dv_max": float(g["dv_gain_absolute"].max()),
               "cpd_y_mean": float(g["cpd_y"].mean()),
               "d_recall_l2_mean": float(g["d_recall_l2"].mean()),
               "d_recall_max_mean": float(g["d_recall_max"].mean())}
        for dvv in DIR_VARIANTS:
            c = f"cpd_dir_{dvv}"
            rec[f"{c}_n_defined"] = int(g[c].notna().sum())
            rec[f"{c}_mean"] = float(g[c].mean()) if g[c].notna().any() else float("nan")
        # M0 / M1 与 cpd_dir 变体无关（全 150 行拟合），只取第一个变体的记录；
        # M0_sub / M2 依赖 cpd_dir 变体的子集，两个变体分列。
        for dvv in DIR_VARIANTS:
            for m in ("M0", "M1", "M0_sub", "M2"):
                if m in ("M0", "M1") and dvv != DIR_VARIANTS[0]:
                    continue
                col = f"loeo_mse_{m}_{dvv}" if m in ("M0_sub", "M2") else f"loeo_mse_{m}"
                sel = cv_all[(cv_all["cpd_dir_variant"] == dvv) & (cv_all["model"] == m)
                             & (cv_all["target_env"] == e)]
                rec[col] = float(sel["mse"].iloc[0]) if len(sel) else float("nan")
                rec[col.replace("loeo_mse", "loeo_n_test")] = (
                    int(sel["n_test"].iloc[0]) if len(sel) else 0)
        recs.append(rec)
    return pd.DataFrame(recs)


def table_dir_coverage(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    cols = ["task", "n_sources", "source_envs", "target_env", "n_ref", "n_ref_dir_defined",
            "cpd_dir_strict", "cpd_dir_lenient", "per_ref_cpd_dir", "dir_included_rows"]
    out = df[cols].copy()
    out.insert(0, "variant", variant)
    out["defined_strict"] = out["cpd_dir_strict"].notna()
    out["defined_lenient"] = out["cpd_dir_lenient"].notna()
    return out


# --------------------------------------------------------------------------- #
# 验收硬门
# --------------------------------------------------------------------------- #
def handcheck_m0(task: str, srcs: list[str], iid_variant: str) -> dict:
    """**独立代码路径**手算 M0 的 7 维特征（验收 a）。

    刻意不复用本文件的 `load_cm` / `recall_vector` / numpy 聚合：用标准库 `csv`
    逐行读、纯 Python 算术求和，避免"同一 bug 两边一致"的假通过。
    """
    def read_counts(t: str) -> list[list[float]]:
        p = G0_RAW / t / FEATURE_SET / "rf" / "confusion_matrix.csv"
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(_csv.reader(fh))
        header = [h.strip() for h in rows[0][1:]]
        if header != CLASS_ORDER:
            raise ValueError(f"{p} 列轴 {header} ≠ {CLASS_ORDER}")
        body = []
        for r in rows[1:]:
            if not r or not r[0].strip():
                continue
            body.append([float(v) for v in r[1:]])
        if len(body) != len(CLASS_ORDER):
            raise ValueError(f"{p} 行数 {len(body)} ≠ {len(CLASS_ORDER)}")
        return body

    def recalls(t: str) -> list[float]:
        m = read_counts(t)
        out = []
        for i in range(len(CLASS_ORDER)):
            s = 0.0
            for v in m[i]:
                s += v
            out.append(m[i][i] / (s if s != 0 else 1.0))
        return out

    r_tgt = recalls(task)
    per_src = [recalls(iid_task_name(s, iid_variant)) for s in srcs]
    r_src = [sum(p[i] for p in per_src) / len(per_src) for i in range(len(CLASS_ORDER))]
    delta = [r_src[i] - r_tgt[i] for i in range(len(CLASS_ORDER))]
    l2 = math.sqrt(sum(d * d for d in delta))
    out = {f"d_recall_{c}": delta[i] for i, c in enumerate(CLASS_ORDER)}
    out["d_recall_l2"] = l2
    out["d_recall_max"] = max(delta)
    return out


def gate_a_handcheck(df: pd.DataFrame, tasks: list[dict], iid_variant: str) -> dict:
    """抽 1 个 |S|=1 与 1 个 |S|=3 任务（按任务名排序取第一个，确定性）手算比对。"""
    by_name = {t["name"]: t for t in tasks}
    picks = []
    for k in (1, 3):
        cand = sorted(t["name"] for t in tasks if t["n_sources"] == k)
        picks.append(cand[0])
    details, ok = [], True
    for task in picks:
        hand = handcheck_m0(task, list(by_name[task]["train_rounds"]), iid_variant)
        row = df[df["task"] == task].iloc[0]
        devs = {t: abs(float(row[t]) - hand[t]) for t in M0_TERMS}
        mx = max(devs.values())
        ok = ok and mx <= HANDCHECK_TOL
        details.append({"task": task, "n_sources": int(by_name[task]["n_sources"]),
                        "source_envs": ";".join(by_name[task]["train_rounds"]),
                        "max_abs_dev": mx, "tol": HANDCHECK_TOL, "passed": mx <= HANDCHECK_TOL,
                        "pipeline": {t: float(row[t]) for t in M0_TERMS},
                        "handcomputed": {t: hand[t] for t in M0_TERMS},
                        "per_term_abs_dev": devs})
    return {"name": "a_m0_handcheck", "passed": ok, "tasks": details}


def _recompute_topology_matrix_via_d4() -> pd.DataFrame:
    """在**当前解释器**下用 D4 的生成器重算 time_block 拓扑矩阵。

    直接 import `six_env_confusion_similarity`（D4 唯一实现），不复制其逻辑：
    这样 b2 检验的是"E2 与 D4 口径是否同一"，而与落盘文件的产出环境无关。
    """
    import six_env_confusion_similarity as d4  # noqa: PLC0415
    pair = d4.load_pair_cms(G0_RAW, "rf", FEATURE_SET)
    iid = d4.load_iid_cms(G0_RAW, "time_block", "rf", FEATURE_SET)
    return d4.build_cpd_y_matrix(pair, iid)


def gate_b_cpd_y_vs_topology(df: pd.DataFrame, tasks: list[dict]) -> dict:
    """|S|=1 的 30 个任务：`CPD_y` 与 D4 拓扑矩阵（time_block）逐位一致。

    两个子检验，**都是硬门**：
      * `b1` —— vs **落盘** `env_topology_cpd_y_ref_time_block_rf.csv`（D9 验收 ② 的字面要求）；
      * `b2` —— vs 由 D4 生成器在**当前解释器**下重算的同一矩阵（口径同一性的等价检验，
        不受落盘文件产出环境影响）。
    b1 与 b2 若判定不同，差异归因于落盘文件的产出环境而非 E2 实现；两者的逐格证据都落盘。
    """
    stored = pd.read_csv(TOPOLOGY_CPD_Y_TIME_BLOCK, index_col=0, encoding="utf-8-sig",
                         float_precision="round_trip")
    stored.index = [str(i) for i in stored.index]
    stored.columns = [str(c) for c in stored.columns]
    recomputed = _recompute_topology_matrix_via_d4()

    by_name = {t["name"]: t for t in tasks}
    cells = []
    ok1 = ok2 = True
    max_dev1 = max_dev2 = 0.0
    for task in sorted(t["name"] for t in tasks if t["n_sources"] == 1):
        src = by_name[task]["train_rounds"][0]
        tgt = by_name[task]["target_env"]
        got = float(df.loc[df["task"] == task, "cpd_y"].iloc[0])
        want_stored = float(stored.loc[src, tgt])
        want_recomp = float(recomputed.loc[src, tgt])
        eq1 = struct.pack("<d", got) == struct.pack("<d", want_stored)
        eq2 = struct.pack("<d", got) == struct.pack("<d", want_recomp)
        max_dev1 = max(max_dev1, abs(got - want_stored))
        max_dev2 = max(max_dev2, abs(got - want_recomp))
        ok1, ok2 = ok1 and eq1, ok2 and eq2
        cells.append({"task": task, "cell": f"[{src},{tgt}]", "e2_cpd_y": repr(got),
                      "stored_cell": repr(want_stored), "d4_recomputed_cell": repr(want_recomp),
                      "bitwise_equal_vs_stored": eq1, "abs_dev_vs_stored": abs(got - want_stored),
                      "bitwise_equal_vs_d4_recompute": eq2,
                      "abs_dev_vs_d4_recompute": abs(got - want_recomp)})
    return {
        "name": "b_cpd_y_bitwise_vs_topology",
        "passed": bool(ok1 and ok2),
        "b1_vs_stored_csv": {"passed": bool(ok1), "n_cells": len(cells),
                             "n_bitwise_equal": sum(c["bitwise_equal_vs_stored"] for c in cells),
                             "max_abs_dev": max_dev1},
        "b2_vs_d4_generator_recomputed_here": {
            "passed": bool(ok2), "n_cells": len(cells),
            "n_bitwise_equal": sum(c["bitwise_equal_vs_d4_recompute"] for c in cells),
            "max_abs_dev": max_dev2},
        "interpreter": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "executable": sys.executable},
        "stored_matrix": str(TOPOLOGY_CPD_Y_TIME_BLOCK.relative_to(REPO_ROOT)),
        "cells": cells,
    }


def gate_c_bootstrap_reproducible(df: pd.DataFrame, b: int, seed: int) -> dict:
    """同种子双跑，复制统计量 md5 相同（验收 c）。

    裁定 B 的并列要求：`strict` 与 `lenient` **两个变体都跑**，两者都必须复现。
    """
    per_variant = {}
    ok = True
    for dvv in DIR_VARIANTS:
        r1 = cluster_bootstrap(df, dvv, b=b, seed=seed)
        r2 = cluster_bootstrap(df, dvv, b=b, seed=seed)
        same = r1["md5"] == r2["md5"]
        ok = ok and same
        per_variant[dvv] = {"passed": bool(same), "md5_run1": r1["md5"], "md5_run2": r2["md5"],
                            "n_degenerate_run1": r1["n_degenerate"],
                            "n_degenerate_by_model_run1": r1["n_degenerate_by_model"]}
    return {"name": "c_bootstrap_md5_reproducible", "passed": bool(ok),
            "b": b, "seed": seed, "by_cpd_dir_variant": per_variant}


def gate_d_row_counts(df: pd.DataFrame, analysis: dict) -> dict:
    checks, ok = [], True
    n_main = int(len(df))
    checks.append({"item": "main_table_n", "expected": 150, "got": n_main, "passed": n_main == 150})
    for dvv in DIR_VARIANTS:
        cov = int(df[f"cpd_dir_{dvv}"].notna().sum())
        for m in ("M0", "M1"):
            got = int(analysis["dir"][dvv]["fits"][m]["n"])
            checks.append({"item": f"{m}_n[{dvv}]", "expected": 150, "got": got,
                           "passed": got == 150})
        for m in ("M0_sub", "M2"):
            got = int(analysis["dir"][dvv]["fits"][m]["n"])
            checks.append({"item": f"{m}_n[{dvv}]", "expected": cov, "got": got,
                           "passed": got == cov})
    ok = all(c["passed"] for c in checks)
    return {"name": "d_row_counts", "passed": ok, "checks": checks}


# --------------------------------------------------------------------------- #
# §19.2 provenance
# --------------------------------------------------------------------------- #
def _git_head(path: Path) -> dict:
    out = {"path": str(path.relative_to(REPO_ROOT)) if path != REPO_ROOT else ".",
           "head": None, "dirty": None}
    try:
        out["head"] = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
        out["dirty"] = bool(st.strip())
    except Exception as exc:                      # noqa: BLE001
        out["error"] = str(exc)
    return out


def _versions() -> dict:
    import sklearn
    v = {"python": sys.version.split()[0], "platform": platform.platform(),
         "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for name in ("xgboost", "lightgbm", "scipy", "joblib"):
        try:
            v[name] = __import__(name).__version__
        except Exception:                         # noqa: BLE001
            v[name] = None
    return v


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cm_input_digest(tasks: list[dict]) -> dict:
    """全部输入混淆矩阵 CSV 的合并摘要（逐文件 sha256 按路径排序后再 sha256）。

    逐文件哈希有 150×3 + 12×3 条，展开进 provenance 过长；这里给出**可复算**的
    汇总摘要 + 文件计数，复算方式写在 `rule` 里。
    """
    paths = []
    for t in tasks:
        for m in BASE_MODELS:
            paths.append(G0_RAW / t["name"] / FEATURE_SET / m / "confusion_matrix.csv")
    for v in IID_VARIANTS:
        for e in ENVIRONMENTS:
            for m in BASE_MODELS:
                paths.append(G0_RAW / iid_task_name(e, v) / FEATURE_SET / m
                             / "confusion_matrix.csv")
    per_file = {}
    for p in sorted(paths):
        if p.exists():
            per_file[str(p.relative_to(REPO_ROOT))] = _sha256(p)
    agg = hashlib.sha256()
    for rel_path in sorted(per_file):
        agg.update(f"{rel_path}:{per_file[rel_path]}\n".encode())
    return {
        "n_files_hashed": len(per_file),
        "n_files_expected": len(paths),
        "aggregate_sha256": agg.hexdigest(),
        "rule": "sha256 over concatenated '<repo-relative path>:<file sha256>\\n' lines, "
                "paths sorted lexicographically",
    }


def build_provenance(args, tasks, features, gates, boot_md5) -> dict:
    inputs = {
        "dv_csv": {"path": str(E1_CSV.relative_to(REPO_ROOT)), "sha256": _sha256(E1_CSV),
                   "dv_column": "gain_absolute", "arm_primary": "B", "arm_sensitivity": "A",
                   "seed": 42},
        "topology_crosscheck_csv_sha256": _sha256(TOPOLOGY_CPD_Y_TIME_BLOCK),
        "cm_csv_sha256": _cm_input_digest(tasks),
        "confusion_matrix_root": str(G0_RAW.relative_to(REPO_ROOT)),
        "confusion_matrix_pattern":
            "<root>/<task>/all_features/<model>/confusion_matrix.csv",
        "n_task_cms": len(tasks),
        "n_iid_ref_cms": len(ENVIRONMENTS) * len(IID_VARIANTS),
        "iid_ref_tasks": [iid_task_name(e, v) for v in IID_VARIANTS for e in ENVIRONMENTS],
        "topology_crosscheck_csv": str(TOPOLOGY_CPD_Y_TIME_BLOCK.relative_to(REPO_ROOT)),
        "task_definition_source":
            "code/scripts/core/environment_grid_experiment.py::build_task_grid (§11)",
        "cpd_implementation": "code/scripts/analysis/cpd_core.py (§11 唯一实现)",
        "multisource_reference_construction": {
            "rule": "per-reference metric then arithmetic mean (NOT mean of CMs)",
            "source": "code/scripts/analysis/test_cpd_core.py:100-101 "
                      "(test_hist_0_8397_vs_iid_mean)",
            "doc": "docs/CPD_DEFINITIONS.md §4.1（0.8397 = 三个 IID CM 逐个算 CPD_y 后取算术平均）",
            "cpd_dir_nan_policy": {v: ("np.mean（任一参照未定义→任务未定义）" if v == "strict"
                                       else "np.nanmean（在已定义参照上取均值）")
                                   for v in DIR_VARIANTS},
        },
    }
    prim = features["primary"]
    max_differs = prim.loc[
        prim["d_recall_max"] != prim["_diag_d_recall_max_abs"],
        ["task", "d_recall_max", "_diag_d_recall_max_abs"]]
    diagnostics = {
        # 裁定 C：`d_recall_max` 维持字面读法（变化向量的最大元素）；与"最大绝对值"读法
        # 不一致的任务在此逐条登记为事实（不改变任何计算）。
        "d_recall_max_reading": {
            "adopted": "literal max element of the delta vector (裁定 C)",
            "alternative_not_adopted": "max absolute value of the delta vector",
            "n_tasks_where_readings_differ": int(len(max_differs)),
            "n_tasks_total": int(len(prim)),
            "tasks_where_readings_differ": [
                {"task": r["task"], "literal_max": float(r["d_recall_max"]),
                 "max_abs": float(r["_diag_d_recall_max_abs"])}
                for _, r in max_differs.sort_values("task").iterrows()],
        },
        "n_tasks_where_max_differs_from_max_abs": int(len(max_differs)),
        "cpd_dir_coverage": {v: int(prim[f"cpd_dir_{v}"].notna().sum()) for v in DIR_VARIANTS},
        "cpd_dir_coverage_by_n_sources": {
            v: {int(k): int(g[f"cpd_dir_{v}"].notna().sum())
                for k, g in prim.groupby("n_sources", sort=True)} for v in DIR_VARIANTS},
    }
    rulings = {
        "source": "docs/EXECUTION_PLAN_20260829.md v1.3, section 'D9 追记'",
        "A_canonical_environment": {
            "ruling": "canonical analysis environment = iotcls (code/requirements-lock.txt); "
                      "D4 topology CSVs regenerated under it; all 'bitwise identical' gates "
                      "are executed inside the canonical environment only",
            "interpreter_used_for_this_run": {
                "executable": sys.executable, "python": sys.version.split()[0],
                "numpy": np.__version__, "pandas": pd.__version__},
            "registered_fact": "initial D4 CSVs were produced under anaconda base "
                               "(numpy 1.23); 9/30 cells differed at max 2.22e-16 "
                               "(fro-norm reduction path change numpy 1.23→2.4); "
                               "D4's own 1e-6 gate passed in both",
        },
        "B_cpd_dir_nan_propagation": {
            "primary": "lenient (np.nanmean over defined references)",
            "parallel_reported": "strict (np.mean; any undefined reference ⇒ task undefined)",
            "requirement": "both variants reported side by side in every table and in the NOTE; "
                           "strict must carry rank-deficiency markers",
            "role_map": DIR_VARIANT_ROLE,
            "rank_deficiency_markers": [
                "e2_regression_main.csv: design_rank / design_n_params / "
                "design_rank_deficient / zero_variance_terms / "
                "n_boot_degenerate_replicates_this_model",
                "e2_coefficients.csv: design_rank_deficient / "
                "n_boot_degenerate_replicates_this_model",
                "e2_bootstrap_summary.csv: n_degenerate_replicates_any_model / "
                "n_degenerate_replicates_by_model",
            ],
        },
        "C_d_recall_max_reading": diagnostics["d_recall_max_reading"],
    }
    return {
        "experiment": "E2 (CPD conditional explanation, hierarchical regression)",
        "protocol_sections": ["4.2", "4.3", "7", "11", "13", "15", "19.2"],
        "execution_spec": "docs/EXECUTION_PLAN_20260829.md D9 (commit cac9b3f) "
                          "+ v1.3 'D9 追记' rulings A/B/C",
        "rulings": rulings,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_line": {"argv": list(sys.argv), "executable": sys.executable,
                         "cwd": str(Path.cwd())},
        "git": {"repo_root": _git_head(REPO_ROOT), "code": _git_head(REPO_ROOT / "code")},
        "versions": _versions(),
        "bootstrap": {"kind": "cluster bootstrap over target environments (§15.1)",
                      "resampled_unit": "target_env", "n_clusters": len(ENVIRONMENTS),
                      "B": args.bootstrap_b, "numpy_seed": args.seed,
                      "rng": "numpy.random.default_rng(seed)",
                      "ci": "percentile 2.5 / 97.5",
                      "stats_md5_primary_variant_by_cpd_dir": boot_md5},
        "inputs": inputs,
        "variants": [{"name": v.name, "dv_arm": v.dv_arm, "iid_variant": v.iid_variant,
                      "recall_model": v.recall_model, "note": v.note} for v in VARIANTS],
        "m0_terms": M0_TERMS,
        "diagnostics": diagnostics,
        "acceptance_gates": {g["name"]: bool(g["passed"]) for g in gates},
        "output_dir": _rel(Path(args.output_dir).resolve()),
    }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="E2 分层回归（协议 §13 / EXECUTION_PLAN D9）。只出数表，不含解读。")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--bootstrap-b", type=int, default=10000, help="聚类 bootstrap 次数（D9: 10000）")
    ap.add_argument("--seed", type=int, default=42, help="numpy bootstrap 种子（D9: 42）")
    ap.add_argument("--dry-run", action="store_true", help="只计算与验收，不写文件")
    args = ap.parse_args()

    print("=" * 78)
    print("E2 —— CPD 条件解释力分层回归（协议 §13；执行口径 D9）")
    print("=" * 78)
    print(f"  DV        : {E1_CSV.relative_to(REPO_ROOT)}  gain_absolute  seed=42")
    print(f"  CM 根     : {G0_RAW.relative_to(REPO_ROOT)}  ({FEATURE_SET})")
    print(f"  CPD 实现  : cpd_core（§11），min_err={cpd_core.DEFAULT_MIN_ERR}")
    print(f"  bootstrap : 聚类单位=target_env，B={args.bootstrap_b}，seed={args.seed}")

    tasks = load_grid_tasks()
    dv = load_dv_table()
    print(f"\n任务：{len(tasks)}（G0 生成器）；DV 表任务数 {len(dv)}")

    features, analyses = {}, {}
    for v in VARIANTS:
        print(f"\n[{v.name}] {v.note}")
        df = build_features(v, tasks, dv)
        features[v.name] = df
        analyses[v.name] = analyse_variant(v, df, b=args.bootstrap_b, seed=args.seed)
        for dvv in DIR_VARIANTS:
            reg = analyses[v.name]["dir"][dvv]["regression"]
            r = {row["model"]: row for _, row in reg.iterrows()}
            print(f"  cpd_dir={dvv:7s} "
                  f"R2(M0)={r['M0']['r2']:.4f} R2(M1)={r['M1']['r2']:.4f} "
                  f"ΔR2={r['M1']['delta_r2']:+.4f} | "
                  f"n(M2)={int(r['M2']['n'])} R2(M0sub)={r['M0_sub']['r2']:.4f} "
                  f"R2(M2)={r['M2']['r2']:.4f} ΔR2={r['M2']['delta_r2']:+.4f}")

    # ---- 验收硬门 --------------------------------------------------------- #
    prim_v = VARIANTS[0]
    prim_df = features[prim_v.name]
    gates = [
        gate_a_handcheck(prim_df, tasks, prim_v.iid_variant),
        gate_b_cpd_y_vs_topology(prim_df, tasks),
        gate_c_bootstrap_reproducible(prim_df, b=min(args.bootstrap_b, 2000), seed=args.seed),
        gate_d_row_counts(prim_df, analyses[prim_v.name]),
    ]
    print("\n" + "-" * 78)
    print("验收硬门")
    print("-" * 78)
    for g in gates:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']}")
        if g["name"] == "b_cpd_y_bitwise_vs_topology":
            for sub in ("b1_vs_stored_csv", "b2_vs_d4_generator_recomputed_here"):
                s = g[sub]
                print(f"        [{'PASS' if s['passed'] else 'FAIL'}] {sub}: "
                      f"{s['n_bitwise_equal']}/{s['n_cells']} 逐位一致，"
                      f"max|Δ|={s['max_abs_dev']:.3e}")
            print(f"        解释器：python {g['interpreter']['python']} "
                  f"numpy {g['interpreter']['numpy']} pandas {g['interpreter']['pandas']}")
            print(f"        （裁定 A：canonical 环境 {g['interpreter']['executable']}）")
        if g["name"] == "c_bootstrap_md5_reproducible":
            for dvv, s in g["by_cpd_dir_variant"].items():
                print(f"        [{'PASS' if s['passed'] else 'FAIL'}] cpd_dir={dvv:7s} "
                      f"B={g['b']} md5={s['md5_run1']} "
                      f"（双跑一致={s['md5_run1'] == s['md5_run2']}，"
                      f"秩亏复制={s['n_degenerate_run1']}）")
    if not all(g["passed"] for g in gates):
        print("\n验收未通过 —— 不写任何输出，退出码 2（不自行变通）")
        for g in gates:
            if g["passed"]:
                continue
            slim = {k: v for k, v in g.items() if k not in ("cells", "tasks")}
            print(json.dumps(slim, ensure_ascii=False, indent=2, default=str))
            if g["name"] == "b_cpd_y_bitwise_vs_topology":
                print("  逐位不一致的格：")
                for c in g["cells"]:
                    if not c["bitwise_equal_vs_stored"] or not c["bitwise_equal_vs_d4_recompute"]:
                        print(f"    {c['cell']:10s} e2={c['e2_cpd_y']} "
                              f"stored={c['stored_cell']} d4_recomp={c['d4_recomputed_cell']}")
        return 2

    if args.dry_run:
        print("\n--dry-run：不写文件")
        return 0

    # ---- 输出 -------------------------------------------------------------- #
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def w(df_: pd.DataFrame, name: str) -> None:
        df_.to_csv(out / name, index=False, encoding="utf-8-sig")

    w(pd.concat([features[v.name].assign(variant=v.name) for v in VARIANTS], ignore_index=True),
      "e2_features.csv")
    w(pd.concat([analyses[v.name]["dir"][d]["regression"] for v in VARIANTS for d in DIR_VARIANTS],
                ignore_index=True), "e2_regression_main.csv")
    w(pd.concat([analyses[v.name]["dir"][d]["coefficients"] for v in VARIANTS for d in DIR_VARIANTS],
                ignore_index=True), "e2_coefficients.csv")
    w(pd.concat([analyses[v.name]["dir"][d]["cv"] for v in VARIANTS for d in DIR_VARIANTS],
                ignore_index=True), "e2_loeo_cv.csv")
    w(pd.concat([table_by_source_count(features[v.name], v.name) for v in VARIANTS],
                ignore_index=True), "e2_by_source_count.csv")
    w(pd.concat([table_by_target_env(features[v.name], v.name, analyses[v.name]) for v in VARIANTS],
                ignore_index=True), "e2_by_target_env.csv")
    w(table_dir_coverage(prim_df, prim_v.name), "e2_m2_coverage.csv")

    # 敏感性各出一张小表（只含该臂的回归主行）
    for v in VARIANTS[1:]:
        w(pd.concat([analyses[v.name]["dir"][d]["regression"] for d in DIR_VARIANTS],
                    ignore_index=True), f"e2_{v.name}.csv")

    # bootstrap 明细：逐复制统计量（float64 原样）+ 汇总
    # 裁定 B：两个 CPD_dir 变体的复制明细都落盘，不只 primary 一侧。
    boot_md5 = {}
    for d in DIR_VARIANTS:
        bb = analyses[prim_v.name]["dir"][d]["bootstrap"]
        boot_md5[d] = bb["md5"]
        bdf = pd.DataFrame(bb["stats"], columns=bb["names"])
        bdf.insert(0, "replicate", np.arange(len(bdf)))
        w(bdf, f"e2_bootstrap_replicates_primary_{d}.csv")
    summ = []
    for v in VARIANTS:
        for d in DIR_VARIANTS:
            bb = analyses[v.name]["dir"][d]["bootstrap"]
            for nm in bb["names"]:
                lo, hi, nfin = percentile_ci(bb["stats"], bb["names"], nm)
                col = bb["stats"][:, bb["names"].index(nm)]
                col = col[np.isfinite(col)]
                summ.append({"variant": v.name, "cpd_dir_variant": d,
                             "cpd_dir_role": DIR_VARIANT_ROLE[d], "stat": nm,
                             "boot_mean": float(col.mean()) if col.size else float("nan"),
                             "boot_sd": float(col.std(ddof=1)) if col.size > 1 else float("nan"),
                             "ci_lo": lo, "ci_hi": hi, "n_finite": nfin,
                             "B": bb["b"], "seed": bb["seed"],
                             "n_degenerate_replicates_any_model": bb["n_degenerate"],
                             "n_degenerate_replicates_by_model":
                                 ";".join(f"{k}={v}" for k, v in
                                          bb["n_degenerate_by_model"].items()),
                             "md5": bb["md5"]})
    w(pd.DataFrame(summ), "e2_bootstrap_summary.csv")

    with (out / "e2_acceptance.json").open("w", encoding="utf-8") as fh:
        json.dump({"gates": gates,
                   "all_passed": all(g["passed"] for g in gates)},
                  fh, indent=2, ensure_ascii=False, default=str)
    with (out / "provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(build_provenance(args, tasks, features, gates, boot_md5),
                  fh, indent=2, ensure_ascii=False)

    print(f"\n输出目录：{_rel(out.resolve())}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
