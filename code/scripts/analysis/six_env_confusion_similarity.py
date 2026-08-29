#!/usr/bin/env python3
"""同质环境×环境拓扑矩阵 —— 由 G0 网格 `|S|=1` 的 30 个有序对构成。

协议依据
--------
* §8.5 第 5 条：`|S|=1` 的 30 个有序对构成**同质**的环境×环境拓扑矩阵，
  替代已废弃的六环境 pairwise 矩阵；
* §20.2：本文件的 `env_mapping`（旧第 33-39 行）改读 G0 的 `|S|=1` 结果；
* §11：`CPD_y` / `CPD_dir` 的唯一实现是 `cpd_core`，本文件不得保留私有副本；
* §4.2 / §4.3：`CPD_y`、`CPD_dir` 的定义与逐行 `n_err ≥ 20` 门槛；
* §8.4：`single_round` 的随机分层划分只能称「session 内上界」，必须并列报告
  按 `window_start` 分块的时间划分数值 —— 故 IID 参照出 `time_block` / `random` 两个变体。

执行口径：`docs/EXECUTION_PLAN_20260829.md` 决策 D4。

废弃说明（§4.4 / `docs/CPD_DEFINITIONS.md` §5.2）
------------------------------------------------
旧实现的 `env_mapping` 把 R2/R3/R4 指向 `single_round_*`（IID 模型）、
R5/R6/R7 指向 `position_*` / `jitter_*`（OOD 模型），**矩阵不同质**，已明确废弃。
旧产物 `results/robust_v2/report/six_env_off_diag_frobenius_rf.csv` 保持原样不动
（`test_cpd_core.py::test_hist_0_1521_six_env_pairwise` 仍以它复现历史值 0.1521）。
本文件不再读写该路径。

矩阵语义（D4 已定，不得更改）
----------------------------
6×6；行 = 源环境 i，列 = 目标环境 j（i ≠ j，共 30 个有序对）::

    cell[i, j] = cpd_core.cpd_y(ref=CM(g0_iid_R{i}_{variant}),
                                tgt=CM(g0_R{i}_to_R{j}))

对角线 `i == j` 在 G0 中无对应的 `|S|=1` 任务 → 置 NaN。
参照系两个变体（均为支撑材料，§8.6）：
`time_block` 为 primary（诚实域内参照），`random` 为 secondary（与历史 IID 参照口径可比）。

输入（一律读 G0 落盘 CSV，不重训、不重算预测）
--------------------------------------------
    results/g0_environment_grid/raw_all/g0_R{i}_to_R{j}/{feature_set}/{model}/confusion_matrix.csv
    results/g0_environment_grid/raw_all/g0_iid_R{i}_{variant}/{feature_set}/{model}/confusion_matrix.csv

输出（写入 `--output-dir`，默认 `results/g0_environment_grid/`）
-------------------------------------------------------------
    env_topology_cpd_y_ref_time_block_{model}.csv   6×6 CPD_y（primary 参照）
    env_topology_cpd_y_ref_random_{model}.csv       6×6 CPD_y（secondary 参照）
    env_topology_macro_f1_from_cm_{model}.csv       由 30 个 CM 重算的 macro-F1 核对表
    env_topology_cpd_dir_ref_time_block_{model}.csv 6×6 CPD_dir（仅覆盖率评估，未达准入置 NaN）
    env_topology_cpd_dir_coverage_{model}.csv       CPD_dir 逐对逐行准入明细

核对（硬门）
-----------
重算的 macro-F1 矩阵必须与 G0 落盘的 `env_topology_matrix_{model}.csv` 逐格一致
（容差 1e-6，协议 §22.1 P1 回归容差）。超差即退出码 2，不写任何输出。

用法::

    python3 code/scripts/analysis/six_env_confusion_similarity.py
    python3 code/scripts/analysis/six_env_confusion_similarity.py --model rf --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cpd_core  # noqa: E402
from cpd_core import cpd_dir, cpd_y, off  # noqa: E402

#: 六个物理环境（§3.1）。行 = 源，列 = 目标。
ENVIRONMENTS = ["R2", "R3", "R4", "R5", "R6", "R7"]

#: 类别轴顺序（`CPD_DEFINITIONS.md` §1.2，历史脚本一致约定）。
CLASS_ORDER = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]

#: IID 参照的两个划分变体（§8.4）。primary 在前。
IID_VARIANTS = ("time_block", "random")
PRIMARY_VARIANT = "time_block"

#: 与落盘 macro-F1 拓扑矩阵比对的容差（§22.1 P1）。
CHECK_TOL = 1e-6

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "g0_environment_grid" / "raw_all"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "g0_environment_grid"


# --------------------------------------------------------------------------
# G0 任务名与混淆矩阵读取
# --------------------------------------------------------------------------

def pair_task_name(src: str, tgt: str) -> str:
    """`|S|=1` 有序对任务名（`environment_grid_experiment.build_task_grid` 的命名）。"""
    return f"g0_{src}_to_{tgt}"


def iid_task_name(env: str, variant: str) -> str:
    """同环境 IID 任务名。variant ∈ {`time_block`, `random`}（§8.4 两种划分）。"""
    if variant not in IID_VARIANTS:
        raise ValueError(f"未知 IID 划分变体 {variant!r}，只允许 {IID_VARIANTS}")
    return f"g0_iid_{env}_{variant}"


def load_cm(results_root: Path, task: str, model: str, feature_set: str) -> np.ndarray:
    """读取**原始计数**混淆矩阵。CSV 带 UTF-8 BOM，用 utf-8-sig 读。

    同时校验类别轴顺序与 `CLASS_ORDER` 完全一致——`CPD_y` / `CPD_dir` 逐元素比较，
    轴序不一致会静默给出错误结果。
    """
    path = results_root / task / feature_set / model / "confusion_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"混淆矩阵缺失：{path}")
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    idx = [str(i) for i in df.index]
    cols = [str(c) for c in df.columns]
    if idx != CLASS_ORDER or cols != CLASS_ORDER:
        raise ValueError(
            f"{path} 的类别轴与约定不符：index={idx} columns={cols} "
            f"期望 {CLASS_ORDER}"
        )
    return df.values.astype(float)


def load_pair_cms(results_root: Path, model: str, feature_set: str) -> dict[tuple[str, str], np.ndarray]:
    """加载 30 个 `|S|=1` 有序对的混淆矩阵，键为 `(源, 目标)`。"""
    cms: dict[tuple[str, str], np.ndarray] = {}
    for src in ENVIRONMENTS:
        for tgt in ENVIRONMENTS:
            if src == tgt:
                continue
            cms[(src, tgt)] = load_cm(results_root, pair_task_name(src, tgt), model, feature_set)
    return cms


def load_iid_cms(results_root: Path, variant: str, model: str,
                 feature_set: str) -> dict[str, np.ndarray]:
    """加载 6 个同环境 IID 参照的混淆矩阵（指定划分变体）。"""
    return {
        env: load_cm(results_root, iid_task_name(env, variant), model, feature_set)
        for env in ENVIRONMENTS
    }


def empty_matrix() -> pd.DataFrame:
    """6×6 空矩阵；行 = 源环境，列 = 目标环境，对角线保持 NaN。"""
    mat = pd.DataFrame(np.nan, index=list(ENVIRONMENTS), columns=list(ENVIRONMENTS), dtype=float)
    mat.index.name = "source_env"
    return mat


# --------------------------------------------------------------------------
# CPD_y 拓扑矩阵（§4.2，计算只经 cpd_core）
# --------------------------------------------------------------------------

def build_cpd_y_matrix(pair_cms: dict[tuple[str, str], np.ndarray],
                       iid_cms: dict[str, np.ndarray]) -> pd.DataFrame:
    """`cell[i, j] = cpd_y(ref=CM_iid(i), tgt=CM_{i→j})`（D4 语义）。"""
    mat = empty_matrix()
    for (src, tgt), cm in pair_cms.items():
        mat.loc[src, tgt] = cpd_y(iid_cms[src], cm)
    return mat


# --------------------------------------------------------------------------
# CPD_dir 覆盖率评估（§4.3：n_err ≥ 20 行准入，不达标置 NaN，不强算）
# --------------------------------------------------------------------------

def build_cpd_dir_matrix(pair_cms: dict[tuple[str, str], np.ndarray],
                         iid_cms: dict[str, np.ndarray],
                         ref_variant: str,
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`CPD_dir` 拓扑矩阵 + 逐对逐行准入明细。

    门槛固定为 `cpd_core.DEFAULT_MIN_ERR`（协议 §4.3 的 20），**不提供放宽入口**。
    ref 或 tgt 任一侧 `n_err < min_err` 的行整行剔除；无任何行达标 → 该格 NaN。

    Returns:
        `(matrix, coverage)`；`coverage` 每行对应一个有序对。
    """
    min_err = cpd_core.DEFAULT_MIN_ERR
    mat = empty_matrix()
    records: list[dict] = []

    for src in ENVIRONMENTS:
        for tgt in ENVIRONMENTS:
            if src == tgt:
                continue
            ref_cm = iid_cms[src]
            tgt_cm = pair_cms[(src, tgt)]
            res = cpd_dir(ref_cm, tgt_cm, min_err=min_err)
            mat.loc[src, tgt] = res.value  # 未定义时 cpd_core 已返回 NaN

            reasons = []
            for i in res.excluded_rows:
                ref_low = res.n_err_ref[i] < min_err
                tgt_low = res.n_err_tgt[i] < min_err
                reason = "both_below" if (ref_low and tgt_low) else ("ref_below" if ref_low else "tgt_below")
                reasons.append(f"{CLASS_ORDER[i]}:{reason}")

            rec = {
                "source_env": src,
                "target_env": tgt,
                "ref_task": iid_task_name(src, ref_variant),
                "tgt_task": pair_task_name(src, tgt),
                "min_err": min_err,
                "cpd_dir": res.value,
                "is_defined": bool(res.is_defined),
                "n_included_rows": len(res.included_rows),
                "included_classes": ";".join(CLASS_ORDER[i] for i in res.included_rows),
                "excluded_classes": ";".join(CLASS_ORDER[i] for i in res.excluded_rows),
                "exclusion_reasons": ";".join(reasons),
            }
            for i, cls in enumerate(CLASS_ORDER):
                rec[f"n_err_ref_{cls}"] = res.n_err_ref[i]
            for i, cls in enumerate(CLASS_ORDER):
                rec[f"n_err_tgt_{cls}"] = res.n_err_tgt[i]
            records.append(rec)

    return mat, pd.DataFrame(records)


def summarize_dir_coverage(coverage: pd.DataFrame) -> dict:
    """覆盖率统计：多少对可算、纳入行数分布、缺失原因分布（协议 §4.3 要求标注缺失）。"""
    n_pairs = len(coverage)
    n_defined = int(coverage["is_defined"].sum())
    rows_hist = coverage["n_included_rows"].value_counts().sort_index().to_dict()

    reason_counts = {"included": 0, "both_below": 0, "ref_below": 0, "tgt_below": 0}
    per_class_excluded: dict[str, dict[str, int]] = {
        c: {"both_below": 0, "ref_below": 0, "tgt_below": 0} for c in CLASS_ORDER
    }
    for _, row in coverage.iterrows():
        reason_counts["included"] += int(row["n_included_rows"])
        if row["exclusion_reasons"]:
            for item in str(row["exclusion_reasons"]).split(";"):
                cls, reason = item.split(":")
                reason_counts[reason] += 1
                per_class_excluded[cls][reason] += 1

    return {
        "n_pairs": n_pairs,
        "n_defined": n_defined,
        "n_undefined": n_pairs - n_defined,
        "n_full_rows": int((coverage["n_included_rows"] == len(CLASS_ORDER)).sum()),
        "included_rows_hist": {int(k): int(v) for k, v in rows_hist.items()},
        "row_slots_total": n_pairs * len(CLASS_ORDER),
        "row_slot_reasons": reason_counts,
        "per_class_excluded": per_class_excluded,
    }


# --------------------------------------------------------------------------
# macro-F1 核对表（由同一批 CM 重算，与 G0 落盘矩阵逐格比对）
# --------------------------------------------------------------------------

def macro_f1_from_cm(cm: np.ndarray) -> float:
    """由混淆矩阵重算 5-class macro-F1（§10 主指标 1）。

    口径与 `robust_iot_research.metric_summary`（第 914-920 行）严格一致：
    `precision_recall_fscore_support(labels=CLASS_ORDER, average='macro', zero_division=0)`。
    这里把计数矩阵展开回 `(y_true, y_pred)` 后调用同一个 sklearn 函数，
    而不是另写一份 F1 公式——避免出现第二份口径。
    """
    counts = np.asarray(cm, dtype=int)
    y_true: list[str] = []
    y_pred: list[str] = []
    for i, true_cls in enumerate(CLASS_ORDER):
        for j, pred_cls in enumerate(CLASS_ORDER):
            n = int(counts[i, j])
            if n:
                y_true.extend([true_cls] * n)
                y_pred.extend([pred_cls] * n)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0,
    )
    return float(macro_f1)


def build_macro_f1_matrix(pair_cms: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """由 30 个 `|S|=1` 混淆矩阵重算的 6×6 macro-F1 核对表。"""
    mat = empty_matrix()
    for (src, tgt), cm in pair_cms.items():
        mat.loc[src, tgt] = macro_f1_from_cm(cm)
    return mat


def crosscheck_macro_f1(recomputed: pd.DataFrame, stored_path: Path,
                        tol: float = CHECK_TOL) -> tuple[float, list[str]]:
    """与 G0 落盘的 `env_topology_matrix_{model}.csv` 逐格比对。

    Returns:
        `(最大绝对偏差, 超差单元格说明列表)`。
    """
    if not stored_path.exists():
        raise FileNotFoundError(f"落盘拓扑矩阵缺失，无法核对：{stored_path}")
    # float_precision="round_trip"：pandas 默认的快速浮点解析会引入约 1 ULP（~2e-16）
    # 的读回误差，会污染 1e-6 容差核对的偏差读数。此处要求逐位还原写入值。
    stored = pd.read_csv(stored_path, index_col=0, encoding="utf-8-sig",
                         float_precision="round_trip")
    stored.index = [str(i) for i in stored.index]
    stored.columns = [str(c) for c in stored.columns]

    if list(stored.index) != ENVIRONMENTS or list(stored.columns) != ENVIRONMENTS:
        raise ValueError(
            f"{stored_path} 的环境轴与约定不符："
            f"index={list(stored.index)} columns={list(stored.columns)}"
        )

    failures: list[str] = []
    max_dev = 0.0
    for src in ENVIRONMENTS:
        for tgt in ENVIRONMENTS:
            got = recomputed.loc[src, tgt]
            want = float(stored.loc[src, tgt])
            if src == tgt:
                # 对角线两侧都必须是缺失：G0 无 g0_R{i}_to_R{i} 任务。
                if not (np.isnan(got) and np.isnan(want)):
                    failures.append(f"[{src},{tgt}] 对角线应为空：stored={want!r} recomputed={got!r}")
                continue
            if np.isnan(want):
                failures.append(f"[{src},{tgt}] 落盘矩阵缺格，无法核对")
                continue
            dev = abs(float(got) - want)
            max_dev = max(max_dev, dev)
            if dev > tol:
                failures.append(
                    f"[{src},{tgt}] |recomputed − stored| = {dev:.3e} > {tol:.0e}"
                    f"（recomputed={got!r} stored={want!r}）"
                )
    return max_dev, failures


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def _print_matrix(title: str, mat: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(mat.round(6).to_string(na_rep="-"))
    row_mean = mat.mean(axis=1, skipna=True)
    col_mean = mat.mean(axis=0, skipna=True)
    print("  行均值（源环境）：" + "  ".join(
        f"{e}={row_mean[e]:.4f}" if np.isfinite(row_mean[e]) else f"{e}=—" for e in ENVIRONMENTS))
    print("  列均值（目标环境）：" + "  ".join(
        f"{e}={col_mean[e]:.4f}" if np.isfinite(col_mean[e]) else f"{e}=—" for e in ENVIRONMENTS))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="同质环境×环境拓扑矩阵（G0 |S|=1，协议 §8.5.5 / §20.2 / D4）",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help="G0 结果根（含 g0_R*_to_R* 与 g0_iid_*），默认 results/g0_environment_grid/raw_all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录，默认 results/g0_environment_grid")
    parser.add_argument("--model", type=str, default="rf",
                        help="模型（rf / xgboost / lightgbm / stacking），D4 用 rf")
    parser.add_argument("--feature-set", type=str, default="all_features",
                        help="特征集，D4 用 all_features")
    parser.add_argument("--stored-matrix", type=Path, default=None,
                        help="用于核对的落盘 macro-F1 拓扑矩阵，默认 <output-dir>/env_topology_matrix_<model>.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算并核对，不写任何文件")
    args = parser.parse_args()

    results_root: Path = args.results_root
    output_dir: Path = args.output_dir
    model: str = args.model
    feature_set: str = args.feature_set
    stored_matrix: Path = args.stored_matrix or (output_dir / f"env_topology_matrix_{model}.csv")

    print("=" * 78)
    print("同质环境×环境拓扑矩阵（G0 |S|=1，30 个有序对）")
    print("=" * 78)
    print(f"  结果根   : {results_root}")
    print(f"  模型/特征: {model} / {feature_set}")
    print(f"  行 = 源环境 i，列 = 目标环境 j；cell = cpd_y(ref=CM_iid(i), tgt=CM_(i→j))")
    print(f"  IID 参照 : primary={PRIMARY_VARIANT}，secondary="
          f"{[v for v in IID_VARIANTS if v != PRIMARY_VARIANT][0]}")
    print(f"  CPD 实现 : cpd_core（协议 §11 唯一实现），min_err={cpd_core.DEFAULT_MIN_ERR}")

    pair_cms = load_pair_cms(results_root, model, feature_set)
    print(f"\n已加载 {len(pair_cms)} 个 |S|=1 混淆矩阵")

    iid_cms = {v: load_iid_cms(results_root, v, model, feature_set) for v in IID_VARIANTS}
    for variant in IID_VARIANTS:
        n_err = {e: off(iid_cms[variant][e]).sum(axis=1).astype(int).tolist() for e in ENVIRONMENTS}
        print(f"已加载 6 个 IID 参照（{variant}）；逐行误分类计数：")
        for e in ENVIRONMENTS:
            print(f"    g0_iid_{e}_{variant:10s} n_err={n_err[e]}")

    # ---- 硬门：macro-F1 核对（超差即停，不写任何输出）--------------------
    f1_mat = build_macro_f1_matrix(pair_cms)
    max_dev, failures = crosscheck_macro_f1(f1_mat, stored_matrix, CHECK_TOL)
    print(f"\nmacro-F1 核对（30 格 vs {stored_matrix.name}，容差 {CHECK_TOL:.0e}）：")
    print(f"  最大绝对偏差 = {max_dev:.3e}")
    if failures:
        print("  核对未通过：")
        for f in failures:
            print(f"    {f}")
        print("\n按 D4 验收标准，核对不通过 → 停，不写任何输出。")
        return 2
    print("  30/30 格一致 ✓")

    # ---- CPD_y 两个变体 -------------------------------------------------
    cpd_y_mats = {v: build_cpd_y_matrix(pair_cms, iid_cms[v]) for v in IID_VARIANTS}
    for variant in IID_VARIANTS:
        tag = "primary" if variant == PRIMARY_VARIANT else "secondary"
        _print_matrix(f"CPD_y（ref = g0_iid_R{{i}}_{variant}，{tag}）", cpd_y_mats[variant])

    # ---- CPD_dir 覆盖率评估（只用 primary 参照，§4.3）--------------------
    dir_mat, coverage = build_cpd_dir_matrix(pair_cms, iid_cms[PRIMARY_VARIANT], PRIMARY_VARIANT)
    cov = summarize_dir_coverage(coverage)
    _print_matrix(f"CPD_dir（ref = g0_iid_R{{i}}_{PRIMARY_VARIANT}，min_err="
                  f"{cpd_core.DEFAULT_MIN_ERR}）", dir_mat)
    print(f"\nCPD_dir 覆盖率（协议 §4.3，不放宽门槛、不强算）：")
    print(f"  可算的有序对：{cov['n_defined']}/{cov['n_pairs']}"
          f"（未定义 {cov['n_undefined']}）")
    print(f"  5 行全部纳入的对：{cov['n_full_rows']}/{cov['n_pairs']}")
    print(f"  纳入行数分布：{cov['included_rows_hist']}")
    print(f"  行槽位（{cov['row_slots_total']} = {cov['n_pairs']}对 × {len(CLASS_ORDER)}类）"
          f"原因分布：{cov['row_slot_reasons']}")
    print("  逐类别剔除原因：")
    for cls in CLASS_ORDER:
        print(f"    {cls:9s} {cov['per_class_excluded'][cls]}")

    if args.dry_run:
        print("\n--dry-run：不写文件。")
        return 0

    # ---- 落盘 ------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for variant in IID_VARIANTS:
        p = output_dir / f"env_topology_cpd_y_ref_{variant}_{model}.csv"
        cpd_y_mats[variant].to_csv(p, encoding="utf-8-sig")
        written.append(p)

    p = output_dir / f"env_topology_macro_f1_from_cm_{model}.csv"
    f1_mat.to_csv(p, encoding="utf-8-sig")
    written.append(p)

    p = output_dir / f"env_topology_cpd_dir_ref_{PRIMARY_VARIANT}_{model}.csv"
    dir_mat.to_csv(p, encoding="utf-8-sig")
    written.append(p)

    p = output_dir / f"env_topology_cpd_dir_coverage_{model}.csv"
    coverage_cols = [
        "source_env", "target_env", "ref_task", "tgt_task", "min_err", "cpd_dir",
        "is_defined", "n_included_rows", "included_classes", "excluded_classes",
        "exclusion_reasons",
    ] + [f"n_err_ref_{c}" for c in CLASS_ORDER] + [f"n_err_tgt_{c}" for c in CLASS_ORDER]
    coverage[coverage_cols].to_csv(p, index=False, encoding="utf-8-sig")
    written.append(p)

    print("\n已写入：")
    for p in written:
        try:
            shown = p.relative_to(REPO_ROOT)
        except ValueError:
            shown = p
        print(f"  {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
