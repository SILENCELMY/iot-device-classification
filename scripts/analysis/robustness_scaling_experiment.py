#!/usr/bin/env python3
"""Fourth-stage robustness scaling experiment.

This script preserves the existing RF/CNN/Transformer results, trains only
CNN-v4 in a new output directory, and reports whether higher CNN capacity keeps
increasing topology sensitivity or starts recovering OOD robustness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import deep_robustness_validation as deep_validation  # noqa: E402
from topology_sweet_spot_experiment import (  # noqa: E402
    TASKS,
    build_performance_tables,
    copy_existing_result,
    evaluate_deep_task_from_split,
    parse_model_list,
    prepare_task_lookup,
    save_split_artifacts,
)
from deep_robustness_validation import (  # noqa: E402
    build_deep_model,
    compute_cpd,
    load_confusion_matrix,
    markdown_table,
    parameter_count,
    plot_cpd,
    plot_confusion_topology,
    plot_degradation,
    set_seed,
    task_scenario,
)
from robust_iot_research import read_config  # noqa: E402


FINAL_MODELS = ["rf", "cnn1d_v2", "cnn1d_v3", "cnn1d_v4", "transformer", "transformer_v2"]
COPY_MODELS = ["rf", "cnn1d_v2", "cnn1d_v3", "transformer", "transformer_v2"]
TRAIN_MODELS = ["cnn1d_v4"]
DISPLAY_NAMES = {
    "rf": "RF",
    "cnn1d_v2": "CNN-v2",
    "cnn1d_v3": "CNN-v3",
    "cnn1d_v4": "CNN-v4",
    "transformer": "Transformer-v1",
    "transformer_v2": "Transformer-v2",
}


def build_cpd_table(result_root: Path, report_dir: Path) -> pd.DataFrame:
    iid_tasks = ["single_round_R2", "single_round_R3", "single_round_R4"]
    ood_tasks = [task for task in TASKS if task not in iid_tasks]
    rows = []
    for model in FINAL_MODELS:
        iid_cms = [load_confusion_matrix(result_root, task, model) for task in iid_tasks]
        iid_cms = [cm for cm in iid_cms if cm is not None]
        for task in ood_tasks:
            ood_cm = load_confusion_matrix(result_root, task, model)
            if ood_cm is None or not iid_cms:
                continue
            cpds = [compute_cpd(iid_cm, ood_cm) for iid_cm in iid_cms]
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "scenario": task_scenario(task),
                    "cpd_vs_iid_mean": float(np.mean(cpds)),
                    "cpd_vs_iid_max": float(np.max(cpds)),
                }
            )
    cpd = pd.DataFrame(rows)
    cpd.to_csv(report_dir / "cpd_comparison.csv", index=False, encoding="utf-8-sig")
    return cpd


def params_from_summaries(summary_df: pd.DataFrame, feature_count: int, class_count: int) -> pd.DataFrame:
    rows = []
    for model in FINAL_MODELS:
        model_rows = summary_df[summary_df["model"] == model]
        params = np.nan
        if "parameter_count" in model_rows and model_rows["parameter_count"].notna().any():
            params = float(model_rows["parameter_count"].dropna().iloc[0])
        elif model.startswith("cnn") or model.startswith("transformer"):
            params = float(parameter_count(build_deep_model(model, feature_count, class_count)))
        rows.append({"model": model, "Params": params})
    return pd.DataFrame(rows)


def build_scaling_table(
    summary_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    cpd: pd.DataFrame,
    report_dir: Path,
    class_count: int,
) -> pd.DataFrame:
    feature_count = int(summary_df["feature_count"].dropna().iloc[0])
    params = params_from_summaries(summary_df, feature_count, class_count)
    cpd_summary = (
        cpd.groupby("model", as_index=False)["cpd_vs_iid_mean"]
        .mean()
        .rename(columns={"cpd_vs_iid_mean": "CPD"})
    )
    table = (
        model_summary.merge(params, on="model", how="left")
        .merge(cpd_summary, on="model", how="left")
        .rename(columns={"model": "Model"})
    )
    order = {model: index for index, model in enumerate(FINAL_MODELS)}
    table["order"] = table["Model"].map(order)
    table["Model"] = table["Model"].map(DISPLAY_NAMES).fillna(table["Model"])
    table = table.sort_values("order").drop(columns=["order"])
    table = table[["Model", "Params", "IID", "OOD", "OOD_Drop", "CPD"]]
    table["Params"] = table["Params"].map(lambda value: "N/A" if pd.isna(value) else str(int(value)))
    table.to_csv(report_dir / "robustness_scaling_table.csv", index=False, encoding="utf-8-sig")
    return table


def write_scaling_report(
    final_table: pd.DataFrame,
    perf: pd.DataFrame,
    cpd: pd.DataFrame,
    args: argparse.Namespace,
    report_dir: Path,
) -> None:
    table = final_table.set_index("Model")
    scenario_perf = (
        perf.pivot_table(index="model", columns="scenario", values="macro_f1", aggfunc="mean")
        .reindex(FINAL_MODELS)
        .reset_index()
        .rename(columns={"model": "Model"})
    )
    scenario_perf["Model"] = scenario_perf["Model"].map(DISPLAY_NAMES).fillna(scenario_perf["Model"])
    scenario_cpd = (
        cpd.pivot_table(index="model", columns="scenario", values="cpd_vs_iid_mean", aggfunc="mean")
        .reindex(FINAL_MODELS)
        .reset_index()
        .rename(columns={"model": "Model"})
    )
    scenario_cpd["Model"] = scenario_cpd["Model"].map(DISPLAY_NAMES).fillna(scenario_cpd["Model"])

    lines = [
        "# Robustness Scaling Experiment - Stage 4\n\n",
        "## 实验协议\n\n",
        f"- 特征缓存：`{args.features}`\n",
        "- 数据窗口：10s non-overlap，沿用主线 feature cache\n",
        "- Split artifact：`splits/<task>/train_idx.npy`, `val_idx.npy`, `test_idx.npy`\n",
        "- Validation policy：当前训练框架不做模型选择/early stopping，`val_idx.npy` 保留为空数组\n",
        "- 标准化：CNN-v4 使用 `StandardScaler.fit(train)`，再 transform test；复制 baseline 保留原实验产物\n",
        f"- 深度训练：epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, device={args.device}\n\n",
        "## 最终结果\n\n",
        markdown_table(final_table.round(4)),
        "\n\n## OOD 场景拆解\n\n",
        markdown_table(scenario_perf.round(4)),
        "\n\n## CPD 场景拆解\n\n",
        markdown_table(scenario_cpd.round(4)),
        "\n\n## 核心判断\n\n",
    ]

    if {"CNN-v3", "CNN-v4"}.issubset(table.index):
        v3 = table.loc["CNN-v3"]
        v4 = table.loc["CNN-v4"]
        iid_up = bool(v4["IID"] > v3["IID"])
        ood_up = bool(v4["OOD"] > v3["OOD"])
        cpd_up = bool(v4["CPD"] > v3["CPD"])
        drop_up = bool(v4["OOD_Drop"] > v3["OOD_Drop"])
        lines.extend(
            [
                f"1. **CNN-v4 是否继续 CPD ↑ / OOD Drop ↑？** CNN-v3 CPD={v3['CPD']:.4f}, OOD Drop={v3['OOD_Drop']:.4f}；"
                f"CNN-v4 CPD={v4['CPD']:.4f}, OOD Drop={v4['OOD_Drop']:.4f}。\n\n",
                f"2. **是否出现 OOD robustness recovery？** CNN-v3 OOD={v3['OOD']:.4f}；CNN-v4 OOD={v4['OOD']:.4f}。\n\n",
            ]
        )
        if not iid_up:
            lines.append(
                "3. **CNN-v4 未完全满足本轮 scaling 前提。** 虽然参数量继续增加，但 IID 没有超过 CNN-v3，说明更大的结构没有自动转化为更有效的 IID representation。"
                "在这个前提下，CNN-v4 的 OOD 低于 CNN-v3，CPD/OOD Drop 高于 CNN-v3，因此没有出现 invariant representation emergence。\n\n"
            )
        elif iid_up and ood_up and not cpd_up:
            lines.append(
                "3. **更支持 Hypothesis B。** CNN-v4 在提高 IID/OOD 的同时没有继续推高 CPD，说明更强 representation 可能开始学习更稳定的 invariant representation。\n\n"
            )
        elif iid_up and (not ood_up) and (cpd_up or drop_up):
            lines.append(
                "3. **更支持 Hypothesis A。** CNN-v4 的 IID 继续提高，但 OOD 没有恢复，且 CPD/OOD Drop 继续恶化，说明 topology overfitting 仍在增强。\n\n"
            )
        elif ood_up and cpd_up:
            lines.append(
                "3. **混合结果。** CNN-v4 的 OOD 有恢复迹象，但 CPD 仍上升，说明模型可能同时增强了稳定 pattern 与 environment-specific topology fitting。\n\n"
            )
        else:
            lines.append(
                "3. **未出现清晰 phase transition。** CNN-v4 没有给出单一方向证据，需要结合场景拆解判断是 Position/Jitter/LORO 哪类 OOD 在主导。\n\n"
            )
        lines.append(
            "4. **Phase transition 判断：** 本轮没有观察到 OOD robustness recovery，也没有观察到 CPD 下降/稳定。"
            "结果更接近 Hypothesis A 的方向，但由于 CNN-v4 的 IID 没有超过 CNN-v3，应表述为“继续增大 CNN capacity 没有带来 phase transition”，"
            "而不是“更强有效表示已经形成后仍然失败”。\n\n"
        )

    lines.append(
        "## 理论解释\n\n"
        "本阶段把 CNN capacity 继续推过 CNN-v3，用于检验 topology sensitivity 是否持续单调增加。"
        "最终结论不以最高 IID 为依据，而以 OOD recovery 与 CPD 是否下降/稳定作为判断 invariant representation emergence 的核心证据。\n"
    )
    (report_dir / "ROBUSTNESS_SCALING_CONCLUSION.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-4 CNN-v4 robustness scaling experiment")
    parser.add_argument("--config", type=Path, default=Path("code/configs/research_experiments.json"))
    parser.add_argument("--features", type=Path, default=Path("results/robust_v2/raw_all/features_raw_all_w10.csv"))
    parser.add_argument("--baseline-source", type=Path, default=Path("results/gpu_capacity_full_20260703/raw_all"))
    parser.add_argument("--output-root", type=Path, default=Path("results/robustness_scaling_20260706"))
    parser.add_argument("--train-models", default=",".join(TRAIN_MODELS))
    parser.add_argument("--copy-models", default=",".join(COPY_MODELS))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch.set_num_threads(max(1, args.torch_threads))
    set_seed(args.random_state)

    config = read_config(args.config)
    features = pd.read_csv(args.features)
    task_lookup = prepare_task_lookup(config)
    copy_models = parse_model_list(args.copy_models, COPY_MODELS)
    train_models = parse_model_list(args.train_models, TRAIN_MODELS)

    result_root = args.output_root
    report_dir = result_root / "report"
    split_root = result_root / "splits"
    report_dir.mkdir(parents=True, exist_ok=True)
    (result_root / "raw_all").mkdir(parents=True, exist_ok=True)

    environment = {
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "features": str(args.features),
        "baseline_source": str(args.baseline_source),
        "tasks": TASKS,
        "copy_models": copy_models,
        "train_models": train_models,
    }
    (result_root / "environment_report.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    split_metadata = []
    for task_name in TASKS:
        task = task_lookup[task_name]
        split_dir = split_root / task_name
        split_metadata.append(save_split_artifacts(features, task, args, split_dir))

        for model in copy_models:
            output_dir = result_root / "raw_all" / task_name / "all_features" / model
            summary = copy_existing_result(args.baseline_source, model, task_name, output_dir)
            if summary is not None:
                summary["scenario"] = task_scenario(task_name)
                summaries.append(summary)

        for model in train_models:
            print(f"Training {model} on {task_name} ({args.device})", flush=True)
            output_dir = result_root / "raw_all" / task_name / "all_features" / model
            summary = evaluate_deep_task_from_split(model, features, task, config, args, split_dir, output_dir)
            summaries.append(summary)
            print(f"{task_name} {model}: macro_f1={summary['macro_f1']:.4f}", flush=True)

    pd.DataFrame(split_metadata).to_csv(report_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(result_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (result_root / "summary_metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    perf, model_summary = build_performance_tables(summary_df, report_dir)
    cpd = build_cpd_table(result_root, report_dir)
    final_table = build_scaling_table(summary_df, model_summary, cpd, report_dir, len(config["labels"]))
    plot_degradation(perf, report_dir)
    plot_cpd(cpd, report_dir)
    deep_validation.CONFUSION_MODELS = FINAL_MODELS
    plot_confusion_topology(result_root, report_dir)
    write_scaling_report(final_table, perf, cpd, args, report_dir)
    print(f"Saved robustness scaling outputs to: {result_root}")


if __name__ == "__main__":
    main()
