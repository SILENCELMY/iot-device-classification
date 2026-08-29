"""S1 深度模型 5 种子聚合（协议 §14 / §22.1 S1）。

只做聚合与标注，不做解读。关键纪律：cnn1d_v3 / cnn1d_v5 在本次运行中是
**复制的单种子参照**（脚本的 copy_existing_result 路径），跨种子 std 恒为 0，
不得报 mean±std；真正的 5 种子候选只有 cnn1d_convnext / cnn1d_inception / cnn1d_tcn。
"""
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = "results/s1_deep_5seed_20260829"
OUT = ROOT
TRUE_MULTISEED = {"cnn1d_convnext", "cnn1d_inception", "cnn1d_tcn"}

rows = []
for sd in sorted(glob.glob(os.path.join(ROOT, "seed*"))):
    seed = int(os.path.basename(sd).replace("seed", ""))
    d = pd.read_csv(os.path.join(sd, "summary_metrics.csv"), encoding="utf-8-sig")
    d["seed"] = seed
    rows.append(d)
a = pd.concat(rows, ignore_index=True)

# 复制行判据：逐任务跨种子 std 恒为 0
per_max = a.groupby(["model", "task"])["macro_f1"].std().groupby("model").max()
copied = sorted(m for m, s in per_max.items() if s == 0.0)
assert set(copied) == (set(a["model"].unique()) - TRUE_MULTISEED), (copied, TRUE_MULTISEED)

agg = (a.groupby(["task", "model"])["macro_f1"]
         .agg(["mean", "std", "min", "max", "count"])
         .reset_index())
agg["seed_status"] = agg["model"].map(
    lambda m: "5seed" if m in TRUE_MULTISEED else "single_seed_reference_copied")
mask_copied = agg["seed_status"] != "5seed"
agg.loc[mask_copied, ["std", "min", "max"]] = pd.NA


def _fmt(r):
    if r["seed_status"] == "5seed":
        return f"{r['mean']:.4f} ± {r['std']:.4f}"
    return f"{r['mean']:.4f} (单种子参照，无 ±std)"


agg["report_as"] = agg.apply(_fmt, axis=1)
agg = agg.sort_values(["task", "model"])
agg.to_csv(os.path.join(OUT, "s1_multiseed_summary.csv"), index=False, encoding="utf-8-sig")
a.to_csv(os.path.join(OUT, "s1_all_seed_rows.csv"), index=False, encoding="utf-8-sig")

meta = {
    "protocol": "§14 多种子规则；§22.1 S1",
    "entry_point": ("cnn_contrast_search_experiment.py --split-source "
                    "results/robustness_scaling_20260706_v2/splits --random-state {42..46}"),
    "seeds": sorted(int(s) for s in a["seed"].unique()),
    "true_multiseed_models": sorted(TRUE_MULTISEED),
    "copied_single_seed_reference_models": copied,
    "copied_row_rule": "逐任务跨种子 std 恒为 0（脚本 copy_existing_result 路径）；不得报 mean±std",
    "n_tasks": int(a["task"].nunique()),
    "rows_per_seed": int(len(a) / a["seed"].nunique()),
    "git_head": subprocess.run(["git", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
}
with open(os.path.join(OUT, "s1_aggregation_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("copied (single-seed reference):", copied)
print()
sub = agg[agg["seed_status"] == "5seed"]
print("真 5 种子候选，逐任务 mean ± std：")
print(sub.pivot(index="task", columns="model", values="report_as").to_string())
print()
print("跨任务平均 std（种子噪声量级）：")
print(sub.groupby("model")["std"].mean().round(5).to_string())
