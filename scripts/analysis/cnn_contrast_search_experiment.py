#!/usr/bin/env python3
"""Search for CNN architectures that invert IID/OOD ordering relative to CNN-v3."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import deep_robustness_validation as deep_validation  # noqa: E402
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
from topology_sweet_spot_experiment import (  # noqa: E402
    TASKS,
    build_performance_tables,
    copy_existing_result,
    evaluate_deep_task_from_split,
    parse_model_list,
    prepare_task_lookup,
)


BASELINE_MODELS = ["cnn1d_v3", "cnn1d_v5"]
CANDIDATE_MODELS = ["cnn1d_inception", "cnn1d_tcn", "cnn1d_convnext"]
FINAL_MODELS = BASELINE_MODELS + CANDIDATE_MODELS
DISPLAY_NAMES = {
    "cnn1d_v3": "CNN-v3",
    "cnn1d_v5": "CNN-v5",
    "cnn1d_v3_sharp": "CNN-v3-sharp",
    "cnn1d_v3_smooth": "CNN-v3-smooth",
    "cnn1d_v3_hybrid": "CNN-v3-hybrid",
    "cnn1d_inception": "CNN-Inception",
    "cnn1d_tcn": "CNN-TCN",
    "cnn1d_convnext": "CNN-ConvNeXt",
}


def copy_split_artifacts(split_source: Path, split_target: Path) -> list[dict[str, object]]:
    metadata = []
    for task_name in TASKS:
        src = split_source / task_name
        dst = split_target / task_name
        dst.mkdir(parents=True, exist_ok=True)
        for filename in ["train_idx.npy", "val_idx.npy", "test_idx.npy", "split_metadata.json"]:
            source_file = src / filename
            if not source_file.exists():
                raise FileNotFoundError(f"Missing split artifact: {source_file}")
            shutil.copy2(source_file, dst / filename)
        task_meta = json.loads((dst / "split_metadata.json").read_text(encoding="utf-8"))
        task_meta["split_source"] = str(src)
        metadata.append(task_meta)
    return metadata


def load_existing_result(output_dir: Path, model: str, task_name: str) -> dict | None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = dict(metrics)
    metrics["model"] = model
    metrics.setdefault("task", task_name)
    metrics["source"] = metrics.get("source", "strict_split_trained") + "_reused"
    return metrics


def copy_baseline(args: argparse.Namespace, model: str, task_name: str, output_dir: Path) -> dict | None:
    source_root = args.cnn_v5_source if model == "cnn1d_v5" else args.baseline_source
    return copy_existing_result(source_root, model, task_name, output_dir)


def build_cpd_table(result_root: Path, report_dir: Path, models: list[str]) -> pd.DataFrame:
    iid_tasks = ["single_round_R2", "single_round_R3", "single_round_R4"]
    ood_tasks = [task for task in TASKS if task not in iid_tasks]
    rows = []
    for model in models:
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


def build_contrast_table(
    summary_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    cpd: pd.DataFrame,
    report_dir: Path,
    class_count: int,
) -> pd.DataFrame:
    feature_count = int(summary_df["feature_count"].dropna().iloc[0])
    params = []
    for model in FINAL_MODELS:
        rows = summary_df[summary_df["model"] == model]
        if "parameter_count" in rows and rows["parameter_count"].notna().any():
            count = float(rows["parameter_count"].dropna().iloc[0])
        else:
            count = float(parameter_count(build_deep_model(model, feature_count, class_count)))
        params.append({"model": model, "Params": count})
    cpd_summary = (
        cpd.groupby("model", as_index=False)["cpd_vs_iid_mean"]
        .mean()
        .rename(columns={"cpd_vs_iid_mean": "CPD"})
    )
    table = (
        model_summary.merge(pd.DataFrame(params), on="model", how="left")
        .merge(cpd_summary, on="model", how="left")
    )
    order = {model: index for index, model in enumerate(FINAL_MODELS)}
    table["order"] = table["model"].map(order)
    table["Display"] = table["model"].map(DISPLAY_NAMES).fillna(table["model"])
    table = table.sort_values("order").drop(columns=["order"])
    table = table[["model", "Display", "Params", "IID", "OOD", "OOD_Drop", "CPD"]]

    v3 = table[table["model"] == "cnn1d_v3"].iloc[0]
    table["delta_iid_vs_v3"] = table["IID"] - float(v3["IID"])
    table["delta_ood_vs_v3"] = table["OOD"] - float(v3["OOD"])
    table["relation_vs_v3"] = np.select(
        [
            (table["delta_iid_vs_v3"] > 0) & (table["delta_ood_vs_v3"] < 0),
            (table["delta_iid_vs_v3"] < 0) & (table["delta_ood_vs_v3"] > 0),
        ],
        ["IID_higher_OOD_lower", "IID_lower_OOD_higher"],
        default="same_direction_or_tie",
    )
    table.to_csv(report_dir / "cnn_contrast_table.csv", index=False, encoding="utf-8-sig")
    return table


def write_report(
    table: pd.DataFrame,
    perf: pd.DataFrame,
    cpd: pd.DataFrame,
    args: argparse.Namespace,
    report_dir: Path,
    models: list[str],
) -> None:
    scenario_perf = (
        perf.pivot_table(index="model", columns="scenario", values="macro_f1", aggfunc="mean")
        .reindex(models)
        .reset_index()
    )
    scenario_perf["Display"] = scenario_perf["model"].map(DISPLAY_NAMES).fillna(scenario_perf["model"])
    scenario_cpd = (
        cpd.pivot_table(index="model", columns="scenario", values="cpd_vs_iid_mean", aggfunc="mean")
        .reindex(models)
        .reset_index()
    )
    scenario_cpd["Display"] = scenario_cpd["model"].map(DISPLAY_NAMES).fillna(scenario_cpd["model"])
    hits = table[table["relation_vs_v3"] != "same_direction_or_tie"].copy()

    lines = [
        "# CNN Architecture IID/OOD Contrast Search\n\n",
        "## Protocol\n\n",
        f"- Features: `{args.features}`\n",
        f"- Split source: `{args.split_source}`\n",
        "- All candidates reuse the same train/test indices and train-only StandardScaler.\n",
        f"- Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}, device={args.device}\n\n",
        "## Main Table\n\n",
        markdown_table(table.drop(columns=["model"]).round(4)),
        "\n\n## Scenario Performance\n\n",
        markdown_table(scenario_perf.drop(columns=["model"]).round(4)),
        "\n\n## Scenario CPD\n\n",
        markdown_table(scenario_cpd.drop(columns=["model"]).round(4)),
        "\n\n## Contrast Hits\n\n",
    ]
    if hits.empty:
        lines.append("No candidate produced opposite IID/OOD ordering relative to CNN-v3 in this run.\n")
    else:
        lines.append(markdown_table(hits.drop(columns=["model"]).round(4)))
        lines.append("\n")
    (report_dir / "CNN_CONTRAST_SEARCH_CONCLUSION.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search CNN architectures with opposite IID/OOD ordering against CNN-v3")
    parser.add_argument("--config", type=Path, default=Path("code/configs/research_experiments.json"))
    parser.add_argument("--features", type=Path, default=Path("results/robust_v2/raw_all/features_raw_all_w10.csv"))
    parser.add_argument("--baseline-source", type=Path, default=Path("results/gpu_capacity_full_20260703/raw_all"))
    parser.add_argument("--cnn-v5-source", type=Path, default=Path("results/extreme_capacity_1p2m_20260706/raw_all"))
    parser.add_argument("--split-source", type=Path, default=Path("results/robustness_scaling_20260706_v2/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cnn_contrast_search_20260707"))
    parser.add_argument("--candidate-models", default=",".join(CANDIDATE_MODELS))
    parser.add_argument("--copy-models", default=",".join(BASELINE_MODELS))
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
    global FINAL_MODELS

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
    copy_models = parse_model_list(args.copy_models, BASELINE_MODELS)
    candidate_models = parse_model_list(args.candidate_models, CANDIDATE_MODELS)
    models = copy_models + candidate_models
    FINAL_MODELS = models

    result_root = args.output_root
    report_dir = result_root / "report"
    split_root = result_root / "splits"
    report_dir.mkdir(parents=True, exist_ok=True)
    (result_root / "raw_all").mkdir(parents=True, exist_ok=True)

    split_metadata = copy_split_artifacts(args.split_source, split_root)
    pd.DataFrame(split_metadata).to_csv(report_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    environment = {
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "features": str(args.features),
        "baseline_source": str(args.baseline_source),
        "cnn_v5_source": str(args.cnn_v5_source),
        "split_source": str(args.split_source),
        "tasks": TASKS,
        "copy_models": copy_models,
        "candidate_models": candidate_models,
    }
    (result_root / "environment_report.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summaries: list[dict] = []
    for task_name in TASKS:
        task = task_lookup[task_name]
        split_dir = split_root / task_name

        for model in copy_models:
            output_dir = result_root / "raw_all" / task_name / "all_features" / model
            summary = copy_baseline(args, model, task_name, output_dir)
            if summary is not None:
                summary["scenario"] = task_scenario(task_name)
                summaries.append(summary)

        for model in candidate_models:
            output_dir = result_root / "raw_all" / task_name / "all_features" / model
            print(f"Training {model} on {task_name} ({args.device})", flush=True)
            summary = load_existing_result(output_dir, model, task_name)
            if summary is None:
                summary = evaluate_deep_task_from_split(model, features, task, config, args, split_dir, output_dir)
            summaries.append(summary)
            print(
                f"{task_name} {model}: macro_f1={summary['macro_f1']:.4f}, "
                f"params={summary.get('parameter_count', 'N/A')}",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(result_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (result_root / "summary_metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    perf, model_summary = build_performance_tables(summary_df, report_dir)
    cpd = build_cpd_table(result_root, report_dir, models)
    table = build_contrast_table(summary_df, model_summary, cpd, report_dir, len(config["labels"]))
    plot_degradation(perf, report_dir)
    plot_cpd(cpd, report_dir)
    deep_validation.CONFUSION_MODELS = models
    plot_confusion_topology(result_root, report_dir)
    write_report(table, perf, cpd, args, report_dir, models)
    print(f"Saved CNN contrast search outputs to: {result_root}")


if __name__ == "__main__":
    main()
