#!/usr/bin/env python3
"""Final high-capacity robustness experiment.

This final scaling round trains CNN-v5 (~1.2M parameters) while reusing the
fixed split artifacts from the stage-4 robustness scaling run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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


FINAL_MODELS = ["rf", "cnn1d_v2", "cnn1d_v3", "cnn1d_v4", "cnn1d_v5", "transformer_v2"]
COPY_MODELS = ["rf", "cnn1d_v2", "cnn1d_v3", "cnn1d_v4", "transformer_v2"]
TRAIN_MODELS = ["cnn1d_v5"]
DISPLAY_NAMES = {
    "rf": "RF",
    "cnn1d_v2": "CNN-v2",
    "cnn1d_v3": "CNN-v3",
    "cnn1d_v4": "CNN-v4",
    "cnn1d_v5": "CNN-v5",
    "transformer_v2": "Transformer-v2",
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


def copy_baseline_result(args: argparse.Namespace, model: str, task_name: str, output_dir: Path) -> dict | None:
    if model == "cnn1d_v4":
        source_root = args.cnn_v4_source
    else:
        source_root = args.baseline_source
    return copy_existing_result(source_root, model, task_name, output_dir)


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


def build_extreme_table(
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
    table.to_csv(report_dir / "extreme_capacity_table.csv", index=False, encoding="utf-8-sig")
    return table


def write_extreme_report(
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
        "# Final 1.2M-Capacity Robustness Experiment\n\n",
        "## 实验协议\n\n",
        f"- 特征缓存：`{args.features}`\n",
        f"- Split source：`{args.split_source}`，本轮直接复制并复用 `train_idx.npy/val_idx.npy/test_idx.npy`\n",
        "- 数据窗口：10s non-overlap，沿用主线 feature cache\n",
        "- Validation policy：当前训练框架不做模型选择/early stopping，`val_idx.npy` 保留为空数组\n",
        "- 标准化：CNN-v5 使用 `StandardScaler.fit(train)`，再 transform test；复制 baseline 保留原实验产物\n",
        f"- 深度训练：epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, device={args.device}\n\n",
        "## 最终结果\n\n",
        markdown_table(final_table.round(4)),
        "\n\n## OOD 场景拆解\n\n",
        markdown_table(scenario_perf.round(4)),
        "\n\n## CPD 场景拆解\n\n",
        markdown_table(scenario_cpd.round(4)),
        "\n\n## 核心判断\n\n",
    ]

    if {"CNN-v4", "CNN-v5"}.issubset(table.index):
        v4 = table.loc["CNN-v4"]
        v5 = table.loc["CNN-v5"]
        ood_recovery = bool(v5["OOD"] > v4["OOD"])
        cpd_recovery = bool(v5["CPD"] <= v4["CPD"])
        lines.extend(
            [
                f"1. **CNN-v5 是否继续 CPD ↑ / OOD ↓？** CNN-v4 OOD={v4['OOD']:.4f}, CPD={v4['CPD']:.4f}；"
                f"CNN-v5 OOD={v5['OOD']:.4f}, CPD={v5['CPD']:.4f}。\n\n",
                f"2. **是否出现 OOD robustness recovery？** CNN-v5 OOD {'高于' if ood_recovery else '未高于'} CNN-v4，"
                f"CPD {'未升高' if cpd_recovery else '继续升高'}。\n\n",
            ]
        )
        if ood_recovery and cpd_recovery:
            lines.append(
                "3. **更支持 invariant representation emergence。** 1.2M-capacity CNN 同时改善 OOD 并稳定/降低 CPD，说明模型可能开始学习跨环境稳定 representation。\n\n"
            )
        elif (not ood_recovery) and (not cpd_recovery):
            lines.append(
                "3. **更支持 capacity-induced topology overfitting。** 1.2M-capacity CNN 没有恢复 OOD，且 CPD 继续升高，说明 environment-specific topology fitting 进一步增强。\n\n"
            )
        else:
            lines.append(
                "3. **混合结果。** OOD 与 CPD 没有同向恢复，需要结合场景拆解判断是 LORO、Position 还是 Jitter 主导。\n\n"
            )
        lines.append(
            "4. **是否记忆 environment-specific relation topology？** 若 CNN-v5 在 IID/LORO 局部场景较强但 Position/Jitter 或 CPD 显著恶化，"
            "则说明 1.2M-capacity CNN 更可能记忆训练环境中的 relation topology，而不是学到稳定不变量。\n\n"
        )

    lines.append(
        "## 理论解释\n\n"
        "本轮不是模型优化实验，而是 representation scaling vs topology robustness dynamics 的终局验证。"
        "判断标准不是最高准确率，而是 CNN-v5 相对 CNN-v4 是否出现 OOD recovery 与 CPD 下降/稳定。\n"
    )
    (report_dir / "EXTREME_CAPACITY_CONCLUSION.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final CNN-v5 high-capacity robustness experiment")
    parser.add_argument("--config", type=Path, default=Path("code/configs/research_experiments.json"))
    parser.add_argument("--features", type=Path, default=Path("results/robust_v2/raw_all/features_raw_all_w10.csv"))
    parser.add_argument("--baseline-source", type=Path, default=Path("results/gpu_capacity_full_20260703/raw_all"))
    parser.add_argument("--cnn-v4-source", type=Path, default=Path("results/robustness_scaling_20260706_v2/raw_all"))
    parser.add_argument("--split-source", type=Path, default=Path("results/robustness_scaling_20260706_v2/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("results/extreme_capacity_1p2m_20260706"))
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

    split_metadata = copy_split_artifacts(args.split_source, split_root)
    pd.DataFrame(split_metadata).to_csv(report_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")

    environment = {
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "features": str(args.features),
        "baseline_source": str(args.baseline_source),
        "cnn_v4_source": str(args.cnn_v4_source),
        "split_source": str(args.split_source),
        "tasks": TASKS,
        "copy_models": copy_models,
        "train_models": train_models,
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
            summary = copy_baseline_result(args, model, task_name, output_dir)
            if summary is not None:
                summary["scenario"] = task_scenario(task_name)
                summaries.append(summary)

        for model in train_models:
            print(f"Training {model} on {task_name} ({args.device})", flush=True)
            output_dir = result_root / "raw_all" / task_name / "all_features" / model
            summary = load_existing_result(output_dir, model, task_name)
            if summary is None:
                summary = evaluate_deep_task_from_split(model, features, task, config, args, split_dir, output_dir)
            summaries.append(summary)
            print(f"{task_name} {model}: macro_f1={summary['macro_f1']:.4f}", flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(result_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (result_root / "summary_metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    perf, model_summary = build_performance_tables(summary_df, report_dir)
    cpd = build_cpd_table(result_root, report_dir)
    final_table = build_extreme_table(summary_df, model_summary, cpd, report_dir, len(config["labels"]))
    plot_degradation(perf, report_dir)
    plot_cpd(cpd, report_dir)
    deep_validation.CONFUSION_MODELS = FINAL_MODELS
    plot_confusion_topology(result_root, report_dir)
    write_extreme_report(final_table, perf, cpd, args, report_dir)
    print(f"Saved 1.2M-capacity outputs to: {result_root}")


if __name__ == "__main__":
    main()
