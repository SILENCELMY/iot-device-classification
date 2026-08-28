#!/usr/bin/env python3
"""r=-0.630 的敏感性分析 —— P0 第 4 项交付（协议 §11、§15、§21、§22.2）。

协议 §11「`r=-0.630` 的处置」：做 leave-one-task-out 与按环境聚类 bootstrap，
结论如实记录，在正文降为探索性结果。

被检验的原始结论出处：
    results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md:22, 140, 303
    「Pearson r = -0.630 (p = 0.0379) 统计显著」
数据来源：
    results/robust_v2/report/controlled_cpd_data.csv（n=11）
    该文件的 `cpd` 列使用 **0.801 口径**（基准 = 单个 joint_R2_R3_R4 CM，见 CPD_DEFINITIONS.md §4）

本脚本只做敏感性分析，不修改任何历史结论文件。输出写入 results/p0_audit/。

运行：
    python3 code/scripts/analysis/r630_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpd_core import cpd_y  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_V2 = REPO_ROOT / "results" / "robust_v2" / "raw_all"
DATA_CSV = REPO_ROOT / "results" / "robust_v2" / "report" / "controlled_cpd_data.csv"
DATA_CSV_V2 = REPO_ROOT / "results" / "robust_v2" / "report" / "controlled_cpd_data_v2.csv"
OUT_DIR = REPO_ROOT / "results" / "p0_audit"

SEED = 42
N_BOOT = 10000

# 每个任务的**测试环境**——协议 §15.1「聚类单位 = 测试环境」。
# 取自 controlled_cpd_experiment_v2.py 的 task_env_mapping（第 47-59 行）。
TASK_TEST_ENV = {
    "single_round_R2": ("R2",),
    "single_round_R3": ("R3",),
    "single_round_R4": ("R4",),
    "loro_R2_R3_to_R4": ("R4",),
    "loro_R2_R4_to_R3": ("R3",),
    "loro_R3_R4_to_R2": ("R2",),
    "joint_R2_R3_R4": ("R2", "R3", "R4"),
    "position_R2_R3_R4_to_R5": ("R5",),
    "jitter_R2_R3_R4_to_R6": ("R6",),
    "jitter_R2_R3_R4_to_R7": ("R7",),
    "jitter_R2_R3_R4_to_R6_R7": ("R6", "R7"),
}


def load_cm(task: str) -> np.ndarray:
    path = ROOT_V2 / task / "all_features" / "rf" / "confusion_matrix.csv"
    return pd.read_csv(path, index_col=0, encoding="utf-8-sig").values.astype(float)


def safe_pearson(x, y):
    """相关系数；退化输入（任一侧常数）返回 nan。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df["test_env"] = df["task"].map(lambda t: "+".join(TASK_TEST_ENV[t]))

    print("=" * 78)
    print("r = -0.630 敏感性分析（P0 第 4 项，协议 §11 / §15）")
    print("=" * 78)

    # ---------------------------------------------------------------
    # 0. 先用 cpd_core 复算 cpd 列，确认与落盘一致（把本分析绑定到唯一实现）
    # ---------------------------------------------------------------
    ref = load_cm("joint_R2_R3_R4")
    recomputed = np.array([cpd_y(ref, load_cm(t)) for t in df["task"]])
    max_dev = float(np.max(np.abs(recomputed - df["cpd"].values)))
    assert max_dev < 1e-10, f"cpd 列无法用 cpd_core 复现，最大偏差 {max_dev}"
    print(f"\n[0] cpd_core 复算 `cpd` 列：最大偏差 {max_dev:.2e} → 一致 ✓")
    print("    （口径 = 0.801 口径，基准为单个 joint_R2_R3_R4 CM）")

    x = df["cpd"].values
    y = df["gain_absolute"].values

    # ---------------------------------------------------------------
    # 1. 原始统计量
    # ---------------------------------------------------------------
    r_full, p_full = safe_pearson(x, y)
    s_full, sp_full = spearmanr(x, y)
    print(f"\n[1] 全样本（n={len(df)}）")
    print(f"    Pearson  r = {r_full:+.4f}  (p = {p_full:.4f})")
    print(f"    Spearman ρ = {s_full:+.4f}  (p = {sp_full:.4f})")

    # 置换检验
    rng = np.random.default_rng(SEED)
    perm = np.array([
        safe_pearson(x, rng.permutation(y))[0] for _ in range(N_BOOT)
    ])
    p_perm = float(np.mean(np.abs(perm) >= abs(r_full)))
    print(f"    置换检验 p = {p_perm:.4f}  (n_perm={N_BOOT})")

    # ---------------------------------------------------------------
    # 2. Leave-one-task-out
    # ---------------------------------------------------------------
    print(f"\n[2] Leave-one-task-out（逐个剔除 1 个任务，n={len(df)}→{len(df)-1}）")
    rows = []
    for i, task in enumerate(df["task"]):
        keep = np.arange(len(df)) != i
        r_i, p_i = safe_pearson(x[keep], y[keep])
        s_i, sp_i = spearmanr(x[keep], y[keep])
        rows.append({
            "removed_task": task,
            "removed_test_env": df["test_env"].iloc[i],
            "removed_cpd": float(x[i]),
            "removed_gain": float(y[i]),
            "n": int(keep.sum()),
            "pearson_r": r_i,
            "pearson_p": p_i,
            "spearman_r": float(s_i),
            "spearman_p": float(sp_i),
            "delta_r_vs_full": r_i - r_full,
            "still_sig_at_0.05": bool(p_i < 0.05),
        })
    lot = pd.DataFrame(rows).sort_values("pearson_r")
    for _, rw in lot.iterrows():
        flag = "显著" if rw["still_sig_at_0.05"] else "不显著"
        print(f"    去掉 {rw['removed_task']:28s} r={rw['pearson_r']:+.4f} "
              f"p={rw['pearson_p']:.4f}  {flag}")
    lot.to_csv(OUT_DIR / "r630_leave_one_task_out.csv", index=False,
               encoding="utf-8-sig")

    n_sig = int(lot["still_sig_at_0.05"].sum())
    worst = lot.iloc[-1]  # r 最接近 0 的那次剔除
    print(f"\n    r 区间 [{lot['pearson_r'].min():+.4f}, {lot['pearson_r'].max():+.4f}]")
    print(f"    11 次剔除中仍达 p<0.05 的：{n_sig}/{len(lot)}")
    print(f"    影响最大的单点：去掉 {worst['removed_task']} → "
          f"r {r_full:+.4f} → {worst['pearson_r']:+.4f} (p={worst['pearson_p']:.4f})")

    # ---------------------------------------------------------------
    # 3. Leave-one-environment-out（协议 §15.5 要求）
    # ---------------------------------------------------------------
    print("\n[3] Leave-one-environment-out（剔除某测试环境涉及的所有任务）")
    envs = sorted({e for tup in TASK_TEST_ENV.values() for e in tup})
    loeo_rows = []
    for env in envs:
        keep = np.array([env not in TASK_TEST_ENV[t] for t in df["task"]])
        r_e, p_e = safe_pearson(x[keep], y[keep])
        loeo_rows.append({
            "removed_env": env,
            "n_tasks_removed": int((~keep).sum()),
            "n": int(keep.sum()),
            "pearson_r": r_e,
            "pearson_p": p_e,
            "delta_r_vs_full": r_e - r_full,
            "still_sig_at_0.05": bool(p_e < 0.05) if np.isfinite(p_e) else False,
        })
        print(f"    去掉 {env}（{(~keep).sum()} 个任务）→ n={keep.sum():2d} "
              f"r={r_e:+.4f} p={p_e:.4f}")
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(OUT_DIR / "r630_leave_one_env_out.csv", index=False,
                encoding="utf-8-sig")
    sign_flips = loeo[np.sign(loeo["pearson_r"]) != np.sign(r_full)]

    # ---------------------------------------------------------------
    # 4. 按测试环境聚类的 bootstrap（协议 §15.1、§15.6）
    # ---------------------------------------------------------------
    print(f"\n[4] 按测试环境聚类的 bootstrap（n_boot={N_BOOT}, seed={SEED}）")
    clusters = df["test_env"].values
    uniq = np.array(sorted(set(clusters)))
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    print(f"    聚类单位 = 测试环境，共 {len(uniq)} 个簇：")
    for c in uniq:
        tasks = ", ".join(df["task"].values[idx_by_cluster[c]])
        print(f"      {c:10s} (n={len(idx_by_cluster[c])})  {tasks}")

    rng = np.random.default_rng(SEED)
    boot = []
    n_degenerate = 0
    for _ in range(N_BOOT):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picked])
        r_b, _ = safe_pearson(x[idx], y[idx])
        if np.isnan(r_b):
            n_degenerate += 1
        else:
            boot.append(r_b)
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5])
    frac_neg = float(np.mean(boot < 0))

    print(f"    有效重抽样 {len(boot)}/{N_BOOT}（退化 {n_degenerate}）")
    print(f"    r 的 95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"    bootstrap 中位数 r = {np.median(boot):+.4f}")
    print(f"    r < 0 的比例 = {frac_neg:.3f}")
    covers_zero = bool(ci[0] <= 0.0 <= ci[1])
    print(f"    区间是否覆盖 0：{'是（不显著）' if covers_zero else '否'}")

    # ---------------------------------------------------------------
    # 5. 口径敏感性：换成 v2 口径同一关系是什么
    # ---------------------------------------------------------------
    dfv2 = pd.read_csv(DATA_CSV_V2)
    r_v2, p_v2 = safe_pearson(dfv2["cpd"].values, dfv2["gain_absolute"].values)
    print("\n[5] 口径敏感性（同样 11 个任务、同样的 gain，只换 CPD 口径）")
    print(f"    0.801 口径（vs joint_R2_R3_R4）  r = {r_full:+.4f} (p={p_full:.4f})")
    print(f"    废弃六环境口径（v2）              r = {r_v2:+.4f} (p={p_v2:.4f})")

    # ---------------------------------------------------------------
    # 汇总 JSON
    # ---------------------------------------------------------------
    summary = {
        "source_claim": "Pearson r = -0.630 (p = 0.0379)，见 CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md",
        "data": str(DATA_CSV.relative_to(REPO_ROOT)),
        "cpd_kou_jing": "0.801 口径（基准 = 单个 joint_R2_R3_R4 CM）",
        "n_tasks": int(len(df)),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "full_sample": {
            "pearson_r": r_full, "pearson_p": p_full,
            "spearman_r": float(s_full), "spearman_p": float(sp_full),
            "permutation_p": p_perm,
        },
        "leave_one_task_out": {
            "r_min": float(lot["pearson_r"].min()),
            "r_max": float(lot["pearson_r"].max()),
            "n_still_significant": n_sig,
            "n_total": int(len(lot)),
            "most_influential_task": worst["removed_task"],
            "r_without_most_influential": float(worst["pearson_r"]),
            "p_without_most_influential": float(worst["pearson_p"]),
        },
        "leave_one_env_out": {
            "r_min": float(loeo["pearson_r"].min()),
            "r_max": float(loeo["pearson_r"].max()),
            "n_sign_flips": int(len(sign_flips)),
            "sign_flip_envs": sign_flips["removed_env"].tolist(),
        },
        "cluster_bootstrap": {
            "cluster_unit": "test environment",
            "n_clusters": int(len(uniq)),
            "clusters": {c: df["task"].values[idx_by_cluster[c]].tolist() for c in uniq},
            "ci95": [float(ci[0]), float(ci[1])],
            "median_r": float(np.median(boot)),
            "frac_negative": frac_neg,
            "covers_zero": covers_zero,
            "n_valid": int(len(boot)),
            "n_degenerate": n_degenerate,
        },
        "kou_jing_sensitivity": {
            "r_0801_baseline": r_full,
            "p_0801_baseline": p_full,
            "r_deprecated_six_env": r_v2,
            "p_deprecated_six_env": p_v2,
        },
    }
    with open(OUT_DIR / "r630_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("输出：")
    for name in ("r630_leave_one_task_out.csv", "r630_leave_one_env_out.csv",
                 "r630_sensitivity.json"):
        print(f"  {(OUT_DIR / name).relative_to(REPO_ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
