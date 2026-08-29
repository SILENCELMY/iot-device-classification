#!/usr/bin/env python3
"""Plot figures for the lightweight deep robustness validation outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LABELS = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]
CONFUSION_MODELS = [
    "rf",
    "cnn1d",
    "cnn1d_v2",
    "cnn1d_v25",
    "cnn1d_v3",
    "cnn1d_v4",
    "cnn1d_v5",
    "transformer",
    "transformer_v2",
]


def load_confusion_matrix(result_root: Path, task: str, model: str) -> pd.DataFrame | None:
    path = result_root / "raw_all" / task / "all_features" / model / "confusion_matrix.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def plot_degradation(result_root: Path, report_dir: Path) -> None:
    perf = pd.read_csv(report_dir / "performance_comparison.csv")
    data = perf[perf["scenario"] != "IID"].copy()
    data["task_short"] = data["task"].str.replace("jitter_R2_R3_R4_to_", "jitter_", regex=False)
    data["task_short"] = data["task_short"].str.replace("position_R2_R3_R4_to_", "position_", regex=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="task_short", y="ood_drop", hue="model")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("OOD drop = IID mean macro-F1 - task macro-F1")
    plt.xlabel("OOD task")
    plt.title("IID/OOD performance degradation")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "iid_ood_degradation.png", dpi=220)
    plt.close()


def plot_cpd(report_dir: Path) -> None:
    cpd = pd.read_csv(report_dir / "cpd_comparison.csv")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=cpd, x="model", y="cpd_vs_iid_mean", errorbar="sd")
    plt.ylabel("Mean CPD vs IID")
    plt.xlabel("Model")
    plt.title("Confusion topology drift comparison")
    plt.tight_layout()
    plt.savefig(report_dir / "cpd_comparison.png", dpi=220)
    plt.close()


def plot_confusion_topology(result_root: Path, report_dir: Path) -> None:
    tasks = [
        "single_round_R3",
        "loro_R2_R4_to_R3",
        "position_R2_R3_R4_to_R5",
        "jitter_R2_R3_R4_to_R6_R7",
    ]
    models = CONFUSION_MODELS
    fig, axes = plt.subplots(len(models), len(tasks), figsize=(16, 14))
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            cm = load_confusion_matrix(result_root, task, model)
            if cm is None:
                ax.axis("off")
                continue
            cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
            sns.heatmap(
                cm_norm,
                ax=ax,
                cmap="Blues",
                vmin=0,
                vmax=1,
                cbar=False,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 7},
            )
            ax.set_title(f"{model} / {task}", fontsize=9)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_topology_comparison.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot deep robustness validation figures")
    parser.add_argument("result_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.result_root / "report"
    plot_degradation(args.result_root, report_dir)
    plot_cpd(report_dir)
    plot_confusion_topology(args.result_root, report_dir)
    print(f"Saved plots to: {report_dir}")


if __name__ == "__main__":
    main()
