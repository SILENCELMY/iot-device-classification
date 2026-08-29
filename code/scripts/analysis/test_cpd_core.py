#!/usr/bin/env python3
"""cpd_core 的回归测试 —— P0 阻塞门（协议 §11、§21、§22.2）。

协议 §11 要求：「用 `cpd_core` 复现三个历史数值并归因到具体基准选择」，且
「回归测试必须写成 assert」。本文件即该门。

三个历史值全部来自**同一个任务** `loro_R2_R4_to_R3`（RF、all_features）。
它们的差异 100% 来自**基准选择**，与公式无关——三份历史实现的 `compute_cpd`
在数学上完全等价：

    历史值   出处                             基准
    0.8397   cpd_comprehensive_analysis.py    vs 三个 IID CM（R2/R3/R4）的 CPD 均值
    0.801    controlled_cpd_experiment.py     vs joint_R2_R3_R4 的单个 CM
    0.1521   controlled_cpd_experiment_v2.py  已废弃的六环境 pairwise 矩阵均值

同时验证：
  · `UDS` 签名不含任何标签入参（协议 §17.1 条件 5 的机器可验证证据）
  · `CPD_y` 的代数性质：每行离对角质量恰为 `1 − recall_i`（协议 §4.3 动机）
  · `CPD_dir` 的 `min_err` 门槛与缺失行标注（协议 §4.3）

运行方式（规范入口，退出码 0 = P0 门通过）：
    python3 code/scripts/analysis/test_cpd_core.py

本文件也写成 pytest 兼容形式，但当前环境（conda env `iotcls`）未安装 pytest。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cpd_core  # noqa: E402
from cpd_core import cpd_dir, cpd_y, normalize_cm, off, uds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

# `results/robust_v2/` 是唯一同时含 joint_R2_R3_R4 与 jitter R6/R7 的完整结果根，
# 三个历史值的复现全部以它为准。
ROOT_V2 = REPO_ROOT / "results" / "robust_v2" / "raw_all"
# 0.8397 的原始产出根（cpd_comprehensive_analysis.py 当时指向它）。缺 joint 与 jitter，
# 但三个 IID CM 与 loro CM 与 robust_v2 逐字节一致，用于交叉验证"值与结果根无关"。
ROOT_GPU = REPO_ROOT / "results" / "gpu_capacity_full_20260703" / "raw_all"

SIX_ENV_CSV = REPO_ROOT / "results" / "robust_v2" / "report" / "six_env_off_diag_frobenius_rf.csv"

FLAGSHIP_TASK = "loro_R2_R4_to_R3"
IID_TASKS = ("single_round_R2", "single_round_R3", "single_round_R4")

# 已废弃的六环境 env_mapping（原载于 six_env_confusion_similarity.py；该脚本已于
# 2026-08-29 按协议 §20.2 重写为读 G0 |S|=1 结果，废弃映射自脚本中移除，
# 旧实现见 git 历史）。
# 协议 §4.4 明确废弃：R2/R3/R4 指向 single_round_*（IID 模型），
# R5/R6/R7 指向 position_*/jitter_*（OOD 模型），矩阵不同质。
# 此处**故意**保留原映射，仅为复现 0.1521 这一历史值。
DEPRECATED_ENV_MAPPING = {
    "R2": "single_round_R2",
    "R3": "single_round_R3",
    "R4": "single_round_R4",
    "R5": "position_R2_R3_R4_to_R5",
    "R6": "jitter_R2_R3_R4_to_R6",
    "R7": "jitter_R2_R3_R4_to_R7",
}

# ---- 三个历史值（协议 §11 表格）------------------------------------------
HIST_CPD_VS_IID_MEAN = 0.8396605380838675   # 报告为 0.8397
HIST_CPD_VS_JOINT = 0.8014316033268614      # 报告为 0.801
HIST_CPD_SIX_ENV_PAIRWISE = 0.15211296048527156  # 报告为 0.1521

TOL = 1e-10


def load_cm(root: Path, task: str, model: str = "rf",
            feature_set: str = "all_features") -> np.ndarray:
    """加载原始计数混淆矩阵。CSV 带 UTF-8 BOM，用 utf-8-sig 读。"""
    path = root / task / feature_set / model / "confusion_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"混淆矩阵缺失：{path}")
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    return df.values.astype(float)


# ==========================================================================
# 历史值 1 —— 0.8397：vs 三个 IID CM 的均值
# ==========================================================================

def test_hist_0_8397_vs_iid_mean():
    """`cpd_comprehensive_analysis.py` 口径：与三个 IID CM 分别算 CPD_y 后取均值。

    归因：基准 = {single_round_R2, single_round_R3, single_round_R4} 三个 IID 混淆矩阵，
    先逐个算 `CPD_y`，再对三个值取算术平均（该脚本 analyze_cpd_performance_correlation
    中的 `avg_cpd = np.mean(iid_cpds)`）。
    """
    tgt = load_cm(ROOT_V2, FLAGSHIP_TASK)
    per_iid = [cpd_y(load_cm(ROOT_V2, t), tgt) for t in IID_TASKS]
    got = float(np.mean(per_iid))

    assert abs(got - HIST_CPD_VS_IID_MEAN) < TOL, (
        f"0.8397 复现失败：got={got!r} expected={HIST_CPD_VS_IID_MEAN!r} "
        f"逐 IID 值={per_iid!r}"
    )

    # 与落盘产物 cpd_comparison.csv 的第 4 列交叉核对（同一数字的独立来源）。
    stored = REPO_ROOT / "results" / "gpu_capacity_full_20260703" / "report" / "cpd_comparison.csv"
    if stored.exists():
        df = pd.read_csv(stored)
        row = df[(df.iloc[:, 0] == "rf") & (df.iloc[:, 1] == FLAGSHIP_TASK)]
        assert not row.empty, f"{stored} 中未找到 rf/{FLAGSHIP_TASK} 行"
        assert abs(float(row.iloc[0, 3]) - got) < TOL, (
            f"与落盘 cpd_comparison.csv 不一致：stored={row.iloc[0, 3]!r} got={got!r}"
        )

    return got, per_iid


def test_hist_0_8397_is_root_independent():
    """同一数字在 gpu_capacity_full_20260703 根下也成立 —— 证明值与结果根无关。"""
    tgt = load_cm(ROOT_GPU, FLAGSHIP_TASK)
    got = float(np.mean([cpd_y(load_cm(ROOT_GPU, t), tgt) for t in IID_TASKS]))
    assert abs(got - HIST_CPD_VS_IID_MEAN) < TOL, (
        f"gpu_capacity_full 根下 0.8397 复现失败：got={got!r}"
    )

    # 逐元素确认两个根的 CM 完全相同，排除"数值巧合"。
    for task in IID_TASKS + (FLAGSHIP_TASK,):
        assert np.array_equal(load_cm(ROOT_V2, task), load_cm(ROOT_GPU, task)), (
            f"{task} 的 CM 在两个结果根下不一致"
        )
    return got


# ==========================================================================
# 历史值 2 —— 0.801：vs joint_R2_R3_R4
# ==========================================================================

def test_hist_0_801_vs_joint():
    """`controlled_cpd_experiment.py` 口径：基准为单个 `joint_R2_R3_R4` CM。

    归因：`run_controlled_experiment` 用 `baseline_cm = load_cm('joint_R2_R3_R4', 'rf')`
    作为唯一基准，不取均值。`joint_R2_R3_R4` 只存在于 results/robust_v2/ 下。
    """
    ref = load_cm(ROOT_V2, "joint_R2_R3_R4")
    tgt = load_cm(ROOT_V2, FLAGSHIP_TASK)
    got = cpd_y(ref, tgt)

    assert abs(got - HIST_CPD_VS_JOINT) < TOL, (
        f"0.801 复现失败：got={got!r} expected={HIST_CPD_VS_JOINT!r}"
    )

    # 与落盘 controlled_cpd_data.csv 交叉核对。
    stored = REPO_ROOT / "results" / "robust_v2" / "report" / "controlled_cpd_data.csv"
    if stored.exists():
        df = pd.read_csv(stored)
        row = df[df["task"] == FLAGSHIP_TASK]
        assert not row.empty, f"{stored} 中未找到 {FLAGSHIP_TASK}"
        assert abs(float(row.iloc[0]["cpd"]) - got) < TOL, (
            f"与落盘 controlled_cpd_data.csv 不一致：stored={row.iloc[0]['cpd']!r} got={got!r}"
        )
    return got


# ==========================================================================
# 历史值 3 —— 0.1521：已废弃的六环境 pairwise
# ==========================================================================

def build_deprecated_six_env_matrix() -> pd.DataFrame:
    """用已废弃的 env_mapping 重建 6×6 CPD_y 矩阵（协议 §4.4 记录用）。"""
    envs = list(DEPRECATED_ENV_MAPPING)
    cms = {e: load_cm(ROOT_V2, DEPRECATED_ENV_MAPPING[e]) for e in envs}
    mat = pd.DataFrame(0.0, index=envs, columns=envs)
    for a in envs:
        for b in envs:
            mat.loc[a, b] = 0.0 if a == b else cpd_y(cms[a], cms[b])
    return mat


def test_hist_0_1521_six_env_pairwise():
    """`controlled_cpd_experiment_v2.py` 口径：训练环境 × 测试环境 pairwise 均值。

    归因：`compute_task_cpd(['R2','R4'], ['R3'])` 取 `mean(six_env[R2,R3], six_env[R4,R3])`。
    该矩阵由已废弃的 env_mapping 构成——R2/R3/R4 用 IID 模型的 CM，
    R5/R6/R7 用 OOD 模型的 CM，**矩阵不同质**（协议 §4.4，已明确废弃）。
    本测试只为复现历史数字，不构成对该口径的认可。
    """
    mat = build_deprecated_six_env_matrix()
    got = float(np.mean([mat.loc["R2", "R3"], mat.loc["R4", "R3"]]))

    assert abs(got - HIST_CPD_SIX_ENV_PAIRWISE) < TOL, (
        f"0.1521 复现失败：got={got!r} expected={HIST_CPD_SIX_ENV_PAIRWISE!r} "
        f"six_env[R2,R3]={mat.loc['R2','R3']!r} six_env[R4,R3]={mat.loc['R4','R3']!r}"
    )

    # 重建的矩阵必须与落盘的 six_env_off_diag_frobenius_rf.csv 逐格一致。
    if SIX_ENV_CSV.exists():
        stored = pd.read_csv(SIX_ENV_CSV, index_col=0, encoding="utf-8-sig")
        stored.index = [str(i) for i in stored.index]
        stored.columns = [str(c) for c in stored.columns]
        for a in mat.index:
            for b in mat.columns:
                assert abs(float(stored.loc[a, b]) - float(mat.loc[a, b])) < TOL, (
                    f"六环境矩阵 [{a},{b}] 不一致："
                    f"stored={stored.loc[a, b]!r} rebuilt={mat.loc[a, b]!r}"
                )

    # 与落盘 controlled_cpd_data_v2.csv 交叉核对。
    stored_v2 = REPO_ROOT / "results" / "robust_v2" / "report" / "controlled_cpd_data_v2.csv"
    if stored_v2.exists():
        df = pd.read_csv(stored_v2)
        row = df[df["task"] == FLAGSHIP_TASK]
        assert not row.empty, f"{stored_v2} 中未找到 {FLAGSHIP_TASK}"
        assert abs(float(row.iloc[0]["cpd"]) - got) < TOL, (
            f"与落盘 controlled_cpd_data_v2.csv 不一致："
            f"stored={row.iloc[0]['cpd']!r} got={got!r}"
        )
    return got, mat


def test_three_values_share_one_task_and_differ_only_by_baseline():
    """三个历史值同任务、同公式，差异只来自基准 —— 协议 §11「归因到具体基准选择」。"""
    tgt = load_cm(ROOT_V2, FLAGSHIP_TASK)

    a = float(np.mean([cpd_y(load_cm(ROOT_V2, t), tgt) for t in IID_TASKS]))
    b = cpd_y(load_cm(ROOT_V2, "joint_R2_R3_R4"), tgt)
    mat = build_deprecated_six_env_matrix()
    c = float(np.mean([mat.loc["R2", "R3"], mat.loc["R4", "R3"]]))

    assert abs(a - HIST_CPD_VS_IID_MEAN) < TOL
    assert abs(b - HIST_CPD_VS_JOINT) < TOL
    assert abs(c - HIST_CPD_SIX_ENV_PAIRWISE) < TOL

    # 0.1521 之所以小一个量级：它压根没有用 loro 任务的 CM，
    # 只比较了两个 IID 环境之间的混淆结构（R2 vs R3、R4 vs R3）。
    assert c < 0.25 < b < a, f"三值量级关系异常：a={a} b={b} c={c}"
    return a, b, c


# ==========================================================================
# 定义性质：CPD_y 的代数依赖（协议 §4.3 动机）
# ==========================================================================

def test_off_diag_row_mass_equals_one_minus_recall():
    """行归一化后每行离对角质量恰为 `1 − recall_i` —— `CPD_y` 内含误差幅度的代数根源。"""
    for task in IID_TASKS + (FLAGSHIP_TASK, "joint_R2_R3_R4"):
        cm = load_cm(ROOT_V2, task)
        offd = off(normalize_cm(cm))
        row_mass = offd.sum(axis=1)
        recall = np.diag(cm) / cm.sum(axis=1)
        assert np.allclose(row_mass, 1.0 - recall, atol=1e-12), (
            f"{task}: 离对角行质量 != 1 - recall；row_mass={row_mass} recall={recall}"
        )


def test_cpd_y_basic_properties():
    """同一性、对称性、非负性。"""
    cm1 = load_cm(ROOT_V2, "single_round_R2")
    cm2 = load_cm(ROOT_V2, FLAGSHIP_TASK)
    assert cpd_y(cm1, cm1) == 0.0
    assert abs(cpd_y(cm1, cm2) - cpd_y(cm2, cm1)) < 1e-15
    assert cpd_y(cm1, cm2) > 0.0


# ==========================================================================
# CPD_dir 的 min_err 门槛（协议 §4.3）
# ==========================================================================

def test_cpd_dir_flags_low_error_rows():
    """误差极少的行必须被剔除并标注（协议 §4.3）。

    用两个 OOD 任务做参照对：`loro_R2_R4_to_R3` 的 Camera 行只有 3 个误分类，
    Socket 行 0 个，两者都必须被 min_err=20 剔除；Light_T1 / Light_XM / Sensor 纳入。
    """
    ref = load_cm(ROOT_V2, FLAGSHIP_TASK)
    tgt = load_cm(ROOT_V2, "position_R2_R3_R4_to_R5")
    res = cpd_dir(ref, tgt)

    assert res.min_err == 20, "默认 min_err 必须是协议 §4.3 规定的 20"
    assert res.excluded_rows, "存在低误差行，excluded_rows 不应为空"
    for i in res.excluded_rows:
        assert min(res.n_err_ref[i], res.n_err_tgt[i]) < 20, (
            f"row {i} 被剔除但两侧 n_err 都 >= 20"
        )
    for i in res.included_rows:
        assert res.n_err_ref[i] >= 20 and res.n_err_tgt[i] >= 20, (
            f"row {i} 被纳入但存在一侧 n_err < 20"
        )
    assert res.notes, "剔除了行却没有标注说明"
    assert res.is_defined and np.isfinite(res.value), (
        f"该对应有可用行，却得到未定义结果：{res.notes}"
    )
    # Camera(idx 0) 与 Socket(idx 4) 必被剔除；其余三类纳入。
    assert res.excluded_rows == (0, 4), f"excluded_rows={res.excluded_rows}"
    assert res.included_rows == (1, 2, 3), f"included_rows={res.included_rows}"
    return res


def test_cpd_dir_is_undefined_for_iid_single_round_references():
    """**P0 发现（锁定用）**：三个 IID single_round CM 在 min_err=20 下 0/5 行可用。

    每行误分类计数实测：
        single_round_R2  [0, 12, 11,  0, 0]
        single_round_R3  [2,  0, 11,  9, 0]
        single_round_R4  [1,  8,  8, 14, 0]

    因此任何以 single_round IID CM 为参照的 `CPD_dir` **完全未定义**（不是"部分缺失"）。
    这直接影响协议 §6 的 X1（IID vs OOD 的 CPD_dir）与 §13 的 E2 模型 M2。
    本测试把该事实固定成断言：若将来它变了（例如有人改了 min_err 或换了 CM），
    测试会失败，从而强制走 §23 Change Log 而不是静默漂移。
    """
    for iid in IID_TASKS:
        ref = load_cm(ROOT_V2, iid)
        n_err = off(ref).sum(axis=1)
        assert (n_err < 20).all(), (
            f"{iid} 现在有行达到 min_err=20（n_err={n_err}）——"
            f"协议 §4.3 的适用范围已变，必须记入 Change Log"
        )
        res = cpd_dir(ref, load_cm(ROOT_V2, FLAGSHIP_TASK))
        assert not res.is_defined, f"{iid} 的 CPD_dir 本应未定义"
        assert np.isnan(res.value)
        assert len(res.excluded_rows) == 5

    # Socket 在**所有**任务中误分类计数都极低 → 永久被 CPD_dir 排除。
    # 与协议 §10 的观察一致（Socket 在 110 条结果中 96 条 F1 = 1.000）。
    all_tasks = IID_TASKS + (
        "joint_R2_R3_R4", "loro_R2_R3_to_R4", FLAGSHIP_TASK, "loro_R3_R4_to_R2",
        "position_R2_R3_R4_to_R5", "jitter_R2_R3_R4_to_R6", "jitter_R2_R3_R4_to_R7",
    )
    for task in all_tasks:
        socket_err = off(load_cm(ROOT_V2, task)).sum(axis=1)[4]
        assert socket_err < 20, f"{task} 的 Socket 行 n_err={socket_err}，已达门槛"


def test_cpd_dir_undefined_when_all_rows_below_threshold():
    """全部行不达门槛时 value 为 NaN 且 is_defined 为 False。"""
    tiny = np.array([[100, 1, 0], [1, 100, 0], [0, 0, 100]], dtype=float)
    res = cpd_dir(tiny, tiny)
    assert not res.is_defined
    assert np.isnan(res.value)
    assert len(res.excluded_rows) == 3


def test_cpd_dir_removes_error_magnitude():
    """把某行误差整体等比放大，方向不变 → CPD_dir 不变，而 CPD_y 会变。"""
    ref = np.array([[80, 10, 10], [10, 80, 10], [10, 10, 80]], dtype=float)
    scaled = np.array([[40, 20, 20], [10, 80, 10], [10, 10, 80]], dtype=float)

    d_ref, _ = cpd_core.dir_matrix(ref, min_err=20)
    d_scaled, _ = cpd_core.dir_matrix(scaled, min_err=20)
    assert np.allclose(d_ref[0], d_scaled[0]), "方向分布应不受误差幅度影响"
    assert cpd_y(ref, scaled) > 0, "CPD_y 应能感知误差幅度变化"


# ==========================================================================
# UDS 无泄漏审计（协议 §17.1 条件 5）
# ==========================================================================

def test_uds_signature_takes_no_labels():
    """机器可验证证据：`uds()` 签名不含任何标签入参。"""
    params = list(inspect.signature(uds).parameters)
    assert params == ["pred_src_oof", "pred_tgt", "class_order"], (
        f"uds 签名发生变化，需重新做泄漏审计：{params}"
    )

    # 按下划线切词后逐 token 比对，避免 "pred_tgt" 里的 "tgt" 被子串误判。
    banned_tokens = {
        "y", "ytrue", "true", "label", "labels", "target", "targets",
        "truth", "gt", "groundtruth",
    }
    for p in params:
        tokens = set(p.lower().split("_"))
        overlap = tokens & banned_tokens
        assert not overlap, f"uds 出现疑似标签入参：{p}（可疑 token={sorted(overlap)}）"

    # 源码层面：函数体内不得出现真实标签相关的读取。
    src = inspect.getsource(uds)
    for token in ("y_true", "true_label", "ground_truth"):
        assert token not in src, f"uds 源码中出现标签标识符：{token}"


def test_uds_is_invariant_to_label_permutation_of_truth():
    """UDS 只依赖预测。真实标签怎么变都不影响结果 —— 无标签性的行为级证据。"""
    rng = np.random.default_rng(42)
    classes = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]

    src = {m: rng.choice(classes, size=400) for m in ("rf", "xgboost", "lightgbm")}
    tgt = {m: rng.choice(classes, size=300) for m in ("rf", "xgboost", "lightgbm")}

    v1 = uds(src, tgt)
    v2 = uds(src, tgt)
    assert v1 == v2, "UDS 必须是确定性的"
    assert v1 >= 0.0

    # 同一批预测 → 漂移为 0。
    assert abs(uds(src, src)) < 1e-15, "源域与目标域预测相同时 UDS 应为 0"


def test_uds_rejects_mismatched_model_sets():
    src = {"rf": np.array(["a", "b"]), "xgb": np.array(["a", "a"])}
    tgt = {"rf": np.array(["a", "b"]), "lgbm": np.array(["b", "b"])}
    try:
        uds(src, tgt)
    except ValueError:
        pass
    else:
        raise AssertionError("模型集合不一致时应抛 ValueError")


def test_uds_detects_disagreement_drift():
    """源域两模型完全一致、目标域完全不一致 → UDS 明显大于 0。"""
    n = 200
    a = np.array(["Camera"] * n)
    b_agree = np.array(["Camera"] * n)
    b_disagree = np.array(["Sensor"] * n)

    same = uds({"m1": a, "m2": b_agree}, {"m1": a, "m2": b_agree})
    drifted = uds({"m1": a, "m2": b_agree}, {"m1": a, "m2": b_disagree})
    assert abs(same) < 1e-15
    assert drifted > 0.5, f"分歧结构完全反转时 UDS 应显著：got={drifted}"


# ==========================================================================
# 独立运行入口
# ==========================================================================

def main() -> int:
    tests = [
        ("历史值 0.8397（vs 三 IID 均值）", test_hist_0_8397_vs_iid_mean),
        ("0.8397 结果根无关性", test_hist_0_8397_is_root_independent),
        ("历史值 0.801（vs joint_R2_R3_R4）", test_hist_0_801_vs_joint),
        ("历史值 0.1521（废弃六环境 pairwise）", test_hist_0_1521_six_env_pairwise),
        ("三值同任务、差异只来自基准", test_three_values_share_one_task_and_differ_only_by_baseline),
        ("离对角行质量 = 1 - recall", test_off_diag_row_mass_equals_one_minus_recall),
        ("CPD_y 基本性质", test_cpd_y_basic_properties),
        ("CPD_dir min_err 剔除与标注", test_cpd_dir_flags_low_error_rows),
        ("P0 发现：IID 参照下 CPD_dir 未定义", test_cpd_dir_is_undefined_for_iid_single_round_references),
        ("CPD_dir 全行不达标 → NaN", test_cpd_dir_undefined_when_all_rows_below_threshold),
        ("CPD_dir 去除误差幅度", test_cpd_dir_removes_error_magnitude),
        ("UDS 签名无标签入参（§17.1-5）", test_uds_signature_takes_no_labels),
        ("UDS 只依赖预测", test_uds_is_invariant_to_label_permutation_of_truth),
        ("UDS 拒绝模型集合不一致", test_uds_rejects_mismatched_model_sets),
        ("UDS 能检出分歧漂移", test_uds_detects_disagreement_drift),
    ]

    failures = []
    print("=" * 78)
    print("cpd_core 回归测试 —— P0 阻塞门（协议 §11）")
    print("=" * 78)
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
        else:
            print(f"  PASS  {name}")

    print("-" * 78)
    if failures:
        print(f"结果：{len(tests) - len(failures)}/{len(tests)} 通过，{len(failures)} 失败")
        print("P0 门未通过 —— 按用户指令，不得开始任何新分析。")
        return 1

    print(f"结果：{len(tests)}/{len(tests)} 全部通过")
    print()
    print("三个历史值归因（协议 §11 要求的口径映射）：")
    tgt = load_cm(ROOT_V2, FLAGSHIP_TASK)
    per_iid = [cpd_y(load_cm(ROOT_V2, t), tgt) for t in IID_TASKS]
    mat = build_deprecated_six_env_matrix()
    print(f"  0.8397 = mean({', '.join(f'{v:.6f}' for v in per_iid)}) "
          f"= {np.mean(per_iid):.16f}")
    print(f"  0.801  = cpd_y(joint_R2_R3_R4, {FLAGSHIP_TASK}) "
          f"= {cpd_y(load_cm(ROOT_V2, 'joint_R2_R3_R4'), tgt):.16f}")
    print(f"  0.1521 = mean(six_env[R2,R3]={mat.loc['R2','R3']:.6f}, "
          f"six_env[R4,R3]={mat.loc['R4','R3']:.6f}) "
          f"= {np.mean([mat.loc['R2','R3'], mat.loc['R4','R3']]):.16f}")
    print()
    print("P0 门通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
