#!/usr/bin/env python3
"""Generate cross-scenario robustness summary from summary_metrics.csv.

Outputs:
  - cross_scenario_table.csv  : per-(task, feature_set, model) pivot (macro_f1 / accuracy)
  - robustness_pivot.csv      : per-(scenario, model) macro_f1 across all tasks
  - new_feature_importance.md : which of the new (burst/direction) features
                                appear in top-k across tasks
  - summary_report.md         : human-readable executive summary

Usage:
  python3 generate_robustness_report.py \\
    --input /path/to/robust_v2/summary_metrics.csv \\
    --output-dir /path/to/robust_v2/report
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Categorize evaluation tasks (must match configs/research_experiments.json)
SCENARIO_MAP = {
    "single_round_R2": "single_round",
    "single_round_R3": "single_round",
    "single_round_R4": "single_round",
    "loro_R2_R3_to_R4": "loro",
    "loro_R2_R4_to_R3": "loro",
    "loro_R3_R4_to_R2": "loro",
    "joint_R2_R3_R4": "joint",
    "position_R2_R3_R4_to_R5": "cross_position",
    "jitter_R2_R3_R4_to_R6": "cross_jitter",
    "jitter_R2_R3_R4_to_R7": "cross_jitter",
    "jitter_R2_R3_R4_to_R6_R7": "cross_jitter",
    "filtered_R1_single_round": "filtered",
}

# 口径说明块（协议 §9.1 / §12）。summary_report.md 的 stacking 列是随机折叠 OOF
# （E1 A 臂）口径，按 §9.1 偏乐观；分组 OOF（B 臂）结果在 results/e1_oof_arms/。
# 由生成器统一输出，保证重新生成时标注不丢失（EXECUTION_PLAN_20260829.md D5）。
CALIBRATION_NOTE_LINES = [
    "> [!NOTE] **口径说明（2026-08-29）**",
    "> 本文件为自动生成的 110 条结果汇总。其中 `stacking` 行的数值为**随机折叠 OOF**（E1 A 臂）",
    "> 口径，按协议 §9.1 偏乐观。分组 OOF 结果见 `results/e1_oof_arms/`。",
]

# New feature family names added in robust_v2 (burst structure + direction)
NEW_FEATURE_PREFIXES = (
    "burst_count", "burst_size_", "burst_packet_fraction",
    "up_packet_ratio", "down_packet_ratio", "side_packet_ratio",
    "other_packet_ratio", "up_down_ratio", "bssid_known",
    "up_len_", "down_len_", "up_ia_", "down_ia_", "len_up_down_diff",
)


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["scenario"] = df["task"].map(SCENARIO_MAP).fillna("other")
    return df


def cross_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index=["task", "feature_set"],
        columns="model",
        values="macro_f1",
    ).sort_index()


def robustness_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index=["scenario", "model"],
        columns="feature_set",
        values="macro_f1",
    ).sort_index()


def new_feature_importance(results_root: Path) -> pd.DataFrame:
    """Combine per-task feature rankings from feature_rankings_all_tasks.csv
    and count how often new features appear in the top-10."""
    fpath = results_root / "feature_rankings_all_tasks.csv"
    if not fpath.exists():
        return pd.DataFrame(columns=["feature", "appearances_in_top10"])
    fr = pd.read_csv(fpath)
    # We don't have a "rank" column by default; use importance as proxy: top-10 per task
    if "importance" not in fr.columns:
        return pd.DataFrame(columns=["feature", "appearances_in_top10"])
    out_rows = []
    for task, sub in fr.groupby("task"):
        sub = sub.sort_values("importance", ascending=False).head(10)
        for feat in sub["feature"]:
            is_new = any(feat.startswith(p) for p in NEW_FEATURE_PREFIXES)
            if is_new:
                out_rows.append({"task": task, "feature": feat})
    if not out_rows:
        return pd.DataFrame(columns=["feature", "appearances_in_top10"])
    df_top = pd.DataFrame(out_rows)
    counts = (
        df_top.groupby("feature").size().reset_index(name="appearances_in_top10")
        .sort_values("appearances_in_top10", ascending=False)
    )
    return counts


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without tabulate."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join([""] + cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    rows = []
    for idx, row in df.iterrows():
        idx_str = str(idx)
        cells = [f"{row[c]:.4f}" if isinstance(row[c], (int, float, np.floating)) else str(row[c]) for c in df.columns]
        rows.append("| " + idx_str + " | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_markdown_summary(
    cross_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    new_feats: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Robustness V2 — Cross-Scenario Summary")
    lines.append("")
    # 口径说明（协议 §9.1）：本报告的 stacking 列来自 robust_iot_research.py 的历史
    # 随机折叠 OOF 口径（= E1 A 臂），偏乐观。这一行必须随报告一起生成，否则重新生成
    # summary_report.md 时手工加的标注会被覆盖丢失。参见 docs/EXECUTION_PLAN_20260829.md D5。
    lines.extend(CALIBRATION_NOTE_LINES)
    lines.append("")
    lines.append("## 1. Per-task macro-F1")
    lines.append("")
    lines.append(_df_to_md(cross_df.round(4)))
    lines.append("")
    lines.append("## 2. Robustness — mean macro-F1 by scenario and feature_set")
    lines.append("")
    lines.append(_df_to_md(rob_df.round(4)))
    lines.append("")
    lines.append("## 3. Stacking vs. best base learner (delta macro-F1)")
    lines.append("")
    if "stacking" in cross_df.columns:
        base_max = cross_df.drop(columns=["stacking"]).max(axis=1)
        delta = cross_df["stacking"] - base_max
        stacking_delta = (
            pd.DataFrame({"stacking_macro_f1": cross_df["stacking"],
                          "best_base_macro_f1": base_max,
                          "delta": delta})
            .round(4)
        )
        lines.append(_df_to_md(stacking_delta))
    lines.append("")
    lines.append("## 4. New feature family (burst + direction) — top-10 appearance")
    lines.append("")
    if len(new_feats):
        # Render as plain text table
        lines.append("| feature | appearances_in_top10 |")
        lines.append("|---|---|")
        for _, r in new_feats.iterrows():
            lines.append(f"| {r['feature']} | {int(r['appearances_in_top10'])} |")
        lines.append("")
        lines.append(
            f"Total new features reaching top-10 in at least one task: **{len(new_feats)}**"
        )
    else:
        lines.append("(No feature_rankings_all_tasks.csv or no new feature in top-10)")
    lines.append("")
    lines.append("## 5. Key observations")
    lines.append("")
    # Auto-fill observations
    single = cross_df.xs("single_round_R2", level="task") if "single_round_R2" in cross_df.index.get_level_values(0) else None
    loro = rob_df.xs("loro", level="scenario") if "loro" in rob_df.index.get_level_values(0) else None
    cross = rob_df.xs("cross_jitter", level="scenario") if "cross_jitter" in rob_df.index.get_level_values(0) else None
    obs: list[str] = []
    if single is not None and len(single):
        # xs on task gives a DataFrame indexed by feature_set, cols=model.
        best_row = single.max(axis=1)  # best model per feature_set
        overall = best_row.max()
        overall_fs = best_row.idxmax()
        obs.append(
            f"- **Single-round (R2)**: best macro-F1 = **{overall:.4f}** "
            f"(feature_set={overall_fs})."
        )
    if loro is not None and len(loro):
        m = loro.mean(axis=1)
        obs.append(
            f"- **LORO (leave-one-round-out)**: mean macro-F1 = "
            f"**{m.mean():.4f}** (range {m.min():.4f}–{m.max():.4f}). "
            "This is the core cross-session generalization test."
        )
    if cross is not None and len(cross):
        m = cross.mean(axis=1)
        obs.append(
            f"- **Cross-jitter (R6, R7)**: mean macro-F1 = **{m.mean():.4f}** "
            "(train on R2-R4, test on jittered sessions)."
        )
    obs.append("")
    obs.append("- New features (burst + direction) are reported in section 4.")
    lines.extend(obs)
    lines.append("")
    lines.append("## 6. Coverage")
    lines.append("")
    # Identify missing (task, feature_set, model) cells (NaN in cross_df)
    if "macro_f1" not in cross_df.columns and cross_df.shape[1] > 0:
        missing_cells = cross_df.isna().stack()
        missing = missing_cells[missing_cells].index.tolist()
        if missing:
            lines.append("Cells with no completed run (not yet executed):")
            lines.append("")
            lines.append("| task | feature_set | missing_model |")
            lines.append("|---|---|---|")
            for (t, fs), colna in cross_df.isna().iterrows():
                for col, isnan in colna.items():
                    if isnan:
                        lines.append(f"| {t} | {fs} | {col} |")
        else:
            lines.append("All configured cells completed.")
    else:
        lines.append("All configured cells completed.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--results-root", required=True, type=Path,
                    help="Path containing feature_rankings_all_tasks.csv (same as --input dir)")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.input)
    cross_df = cross_scenario_table(df)
    rob_df = robustness_pivot(df)
    new_feats = new_feature_importance(args.results_root)

    cross_df.round(6).to_csv(args.output_dir / "cross_scenario_table.csv")
    rob_df.round(6).to_csv(args.output_dir / "robustness_pivot.csv")
    new_feats.to_csv(args.output_dir / "new_feature_importance.csv", index=False)
    write_markdown_summary(cross_df, rob_df, new_feats,
                           args.output_dir / "summary_report.md")
    print(f"Generated: cross_scenario_table.csv ({len(cross_df)} rows), "
          f"robustness_pivot.csv ({len(rob_df)} rows), "
          f"new_feature_importance.csv ({len(new_feats)} rows), "
          f"summary_report.md")


if __name__ == "__main__":
    main()
