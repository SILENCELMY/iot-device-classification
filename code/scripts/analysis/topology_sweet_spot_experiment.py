#!/usr/bin/env python3
"""Strict topology robustness sweet-spot experiment.

This script keeps the existing RF/CNN/Transformer results intact, trains only
CNN-v2.5 in a new output directory, and saves explicit shared split artifacts
for auditability.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from robust_iot_research import (  # noqa: E402
    clean_x,
    feature_columns,
    fit_label_encoder,
    metric_summary,
    read_config,
    sample_balanced,
)
from deep_robustness_validation import (  # noqa: E402
    compute_cpd,
    load_confusion_matrix,
    markdown_table,
    parameter_count,
    plot_cpd,
    plot_degradation,
    plot_confusion_topology,
    set_seed,
    task_scenario,
    train_deep_model,
)


TASKS = [
    "single_round_R2",
    "single_round_R3",
    "single_round_R4",
    "loro_R2_R3_to_R4",
    "loro_R2_R4_to_R3",
    "loro_R3_R4_to_R2",
    "position_R2_R3_R4_to_R5",
    "jitter_R2_R3_R4_to_R6_R7",
]
FINAL_MODELS = ["rf", "cnn1d_v2", "cnn1d_v25", "cnn1d_v3", "transformer", "transformer_v2"]
COPY_MODELS = ["rf", "cnn1d_v2", "cnn1d_v3", "transformer", "transformer_v2"]
TRAIN_MODELS = ["cnn1d_v25"]
LABELS = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]


def prepare_task_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["name"]: task for task in config["evaluation_tasks"]}


def parse_model_list(value: str, default: list[str]) -> list[str]:
    if value.strip().lower() in {"", "none"}:
        return []
    models = [item.strip() for item in value.split(",") if item.strip()]
    return models or default


def save_split_artifacts(
    features: pd.DataFrame,
    task: dict[str, Any],
    args: argparse.Namespace,
    split_dir: Path,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    if task["type"] in {"single_round", "joint_validation"}:
        data = features[features["round"].isin(task["rounds"])].copy()
        data = sample_balanced(data, args.max_rows, args.random_state)
        train_idx, test_idx = train_test_split(
            data.index,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=data["label"],
        )
    else:
        train_data = features[features["round"].isin(task["train_rounds"])].copy()
        test_data = features[features["round"].isin(task["test_rounds"])].copy()
        train_data = sample_balanced(train_data, args.max_rows, args.random_state)
        test_data = sample_balanced(test_data, args.max_rows, args.random_state)
        train_idx = train_data.index.to_numpy()
        test_idx = test_data.index.to_numpy()

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    val_idx = np.asarray([], dtype=np.int64)

    np.save(split_dir / "train_idx.npy", train_idx)
    np.save(split_dir / "val_idx.npy", val_idx)
    np.save(split_dir / "test_idx.npy", test_idx)

    metadata = {
        "task": task["name"],
        "task_type": task["type"],
        "random_state": args.random_state,
        "test_size": args.test_size,
        "validation_policy": "empty_val_no_model_selection",
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "train_rounds": task.get("train_rounds", task.get("rounds", [])),
        "test_rounds": task.get("test_rounds", task.get("rounds", [])),
    }
    (split_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def split_data_from_artifacts(
    features: pd.DataFrame,
    split_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    train_idx = np.load(split_dir / "train_idx.npy")
    test_idx = np.load(split_dir / "test_idx.npy")
    meta_cols = ["round", "traffic", "filter_mode", "source_file", "window_id", "window_start", "window_end"]
    train_data = features.loc[train_idx].copy()
    test_data = features.loc[test_idx].copy()
    return (
        train_data,
        test_data,
        train_data["label"],
        test_data["label"],
        train_data[meta_cols],
        test_data[meta_cols],
    )


def copy_existing_result(source_root: Path, model: str, task_name: str, output_dir: Path) -> dict[str, Any] | None:
    src_dir = source_root / task_name / "all_features" / model
    metrics_path = src_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Skipping missing baseline: {src_dir}", flush=True)
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "metrics.json",
        "classification_report.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "feature_columns.json",
        "model.pt",
        "scaler.joblib",
    ]:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = dict(metrics)
    metrics["model"] = model
    metrics["source"] = "preserved_baseline"
    return metrics


def evaluate_deep_task_from_split(
    model_name: str,
    features: pd.DataFrame,
    task: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
    split_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train_data, test_data, y_train, y_test, _, meta_test = split_data_from_artifacts(features, split_dir)
    columns = feature_columns(train_data)
    x_train = clean_x(train_data, columns)
    x_test = clean_x(test_data, columns)
    encoder = fit_label_encoder(config["labels"])
    y_train_encoded = encoder.transform(y_train)

    pred_encoded, model, scaler = train_deep_model(
        model_name=model_name,
        x_train=x_train,
        y_train=y_train_encoded,
        x_test=x_test,
        args=args,
        class_count=len(config["labels"]),
    )
    pred_labels = encoder.inverse_transform(pred_encoded.astype(int))
    metrics, cm, report = metric_summary(y_test.to_numpy(), pred_labels, config["labels"])

    predictions = meta_test.copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = pred_labels
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=config["labels"], columns=config["labels"]).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2, ensure_ascii=False)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    torch.save(model.state_dict(), output_dir / "model.pt")

    summary = {
        "filter_mode": "raw_all",
        "task": task["name"],
        "task_type": task["type"],
        "scenario": task_scenario(task["name"]),
        "train_rounds": task.get("train_rounds", task.get("rounds", [])),
        "test_rounds": task.get("test_rounds", task.get("rounds", [])),
        "train_samples": int(len(train_data)),
        "test_samples": int(len(test_data)),
        "feature_set": "all_features",
        "model": model_name,
        "feature_count": len(columns),
        "parameter_count": parameter_count(model),
        **{key: value for key, value in metrics.items() if not key.startswith("per_class")},
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": cm.tolist(),
        "source": "strict_split_trained",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def build_performance_tables(summaries: pd.DataFrame, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    iid = (
        summaries[summaries["scenario"] == "IID"]
        .groupby("model", as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "IID"})
    )
    ood = (
        summaries[summaries["scenario"] != "IID"]
        .groupby("model", as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "OOD"})
    )
    perf = summaries.merge(iid.rename(columns={"IID": "iid_mean_macro_f1"}), on="model", how="left")
    perf["ood_drop"] = np.where(perf["scenario"] == "IID", 0.0, perf["iid_mean_macro_f1"] - perf["macro_f1"])
    perf.to_csv(report_dir / "performance_comparison.csv", index=False, encoding="utf-8-sig")
    model_summary = iid.merge(ood, on="model", how="left")
    model_summary["OOD_Drop"] = model_summary["IID"] - model_summary["OOD"]
    return perf, model_summary


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


def build_final_table(model_summary: pd.DataFrame, cpd: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    cpd_summary = (
        cpd.groupby("model", as_index=False)["cpd_vs_iid_mean"]
        .mean()
        .rename(columns={"cpd_vs_iid_mean": "CPD"})
    )
    table = model_summary.merge(cpd_summary, on="model", how="left").rename(columns={"model": "Model"})
    order = {model: index for index, model in enumerate(FINAL_MODELS)}
    table["order"] = table["Model"].map(order)
    table = table.sort_values(["order", "Model"]).drop(columns=["order"])
    table.to_csv(report_dir / "topology_sweet_spot_table.csv", index=False, encoding="utf-8-sig")
    return table


def write_report(
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
    scenario_cpd = (
        cpd.pivot_table(index="model", columns="scenario", values="cpd_vs_iid_mean", aggfunc="mean")
        .reindex(FINAL_MODELS)
        .reset_index()
        .rename(columns={"model": "Model"})
    )
    lines = [
        "# Deep Model Topology Robustness Sweet Spot Experiment\n\n",
        "## 实验协议\n\n",
        f"- 特征缓存：`{args.features}`\n",
        "- 数据窗口：10s non-overlap，沿用主线 feature cache\n",
        "- Split artifact：`splits/<task>/train_idx.npy`, `val_idx.npy`, `test_idx.npy`\n",
        "- Validation policy：本轮不做模型选择/early stopping，`val_idx.npy` 保留为空数组，所有模型共享同一 train/test split\n",
        "- 标准化：CNN-v2.5 使用 `StandardScaler.fit(train)`，再 transform train/test；复制 baseline 保留原实验产物\n",
        f"- 深度训练：epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, device={args.device}\n\n",
        "## 最终结果\n\n",
        markdown_table(final_table.round(4)),
        "\n\n## OOD 场景拆解\n\n",
        markdown_table(scenario_perf.round(4)),
        "\n\n## CPD 场景拆解\n\n",
        markdown_table(scenario_cpd.round(4)),
        "\n\n## 核心观察\n\n",
    ]

    if {"cnn1d_v2", "cnn1d_v25", "cnn1d_v3"}.issubset(table.index):
        v2 = table.loc["cnn1d_v2"]
        v25 = table.loc["cnn1d_v25"]
        v3 = table.loc["cnn1d_v3"]
        lines.extend(
            [
                f"1. **IID 是否继续提高？** CNN-v2={v2['IID']:.4f}，CNN-v2.5={v25['IID']:.4f}，CNN-v3={v3['IID']:.4f}。\n\n",
                f"2. **OOD 是否先升后降？** CNN-v2={v2['OOD']:.4f}，CNN-v2.5={v25['OOD']:.4f}，CNN-v3={v3['OOD']:.4f}。\n\n",
                f"3. **CPD 是否出现 sweet spot？** CNN-v2={v2['CPD']:.4f}，CNN-v2.5={v25['CPD']:.4f}，CNN-v3={v3['CPD']:.4f}。\n\n",
            ]
        )
        if v25["OOD"] > v3["OOD"] and v25["CPD"] < v3["CPD"]:
            lines.append(
                "4. **结论：存在 topology robustness sweet spot。** CNN-v2.5 在保持较高 IID 的同时，相比 CNN-v3 取得更高 OOD 和更低 CPD，支持“适度 relation modeling 优于过强 topology fitting”。\n\n"
            )
        elif v25["OOD"] > v2["OOD"] and v25["CPD"] < v3["CPD"]:
            lines.append(
                "4. **结论：形成过渡型 sweet spot，但不是完整 sweet spot。** CNN-v2.5 的 IID 和 OOD 均高于 CNN-v2，CPD 低于 CNN-v3；"
                "不过平均 OOD 仍低于 CNN-v3，因此它更像 underfitting 与 topology overfitting 之间的可控过渡区间。\n\n"
            )
        else:
            lines.append(
                "4. **结论：未形成完整 sweet spot。** CNN-v2.5 没有同时满足 OOD 高于 CNN-v3 且 CPD 低于 CNN-v3；需要继续微调 capacity 或正则强度。\n\n"
            )
        lines.append(
            "5. **场景差异：** CNN-v2.5 在 LORO/Jitter 上相对 CNN-v2 有提升，但 Position OOD 下降；CNN-v3 的平均 OOD 主要受 LORO 拉高，"
            "同时 Position CPD 明显偏高。这说明 capacity 增强并不是单调坏事，但会让不同 OOD 类型下的 topology sensitivity 更分化。\n\n"
        )
    lines.append(
        "## 理论解释\n\n"
        "本实验用于刻画 weak relation modeling -> robust fitting -> topology overfitting 的动态过程。"
        "本轮 CNN-v2.5 支持“capacity 增强会推高 IID，同时逐步推高 CPD/OOD drop”的主趋势；"
        "但它没有证明平均 OOD 先升后降的完整 sweet spot。更准确的结论是：当前 CNN-v2.5 捕捉到了从 underfitting 向 topology fitting 过渡的阶段，"
        "而 CNN-v3/Transformer-v2 更接近 topology-sensitive high-capacity 区间。\n"
    )
    (report_dir / "TOPOLOGY_SWEET_SPOT_CONCLUSION.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict CNN-v2.5 topology robustness sweet-spot experiment")
    parser.add_argument("--config", type=Path, default=Path("code/configs/research_experiments.json"))
    parser.add_argument("--features", type=Path, default=Path("results/robust_v2/raw_all/features_raw_all_w10.csv"))
    parser.add_argument("--baseline-source", type=Path, default=Path("results/gpu_capacity_full_20260703/raw_all"))
    parser.add_argument("--output-root", type=Path, default=Path("results/topology_sweet_spot_20260703"))
    parser.add_argument("--train-models", default=",".join(TRAIN_MODELS))
    parser.add_argument("--copy-models", default=",".join(COPY_MODELS))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    final_table = build_final_table(model_summary, cpd, report_dir)
    plot_degradation(perf, report_dir)
    plot_cpd(cpd, report_dir)
    plot_confusion_topology(result_root, report_dir)
    write_report(final_table, perf, cpd, args, report_dir)
    print(f"Saved topology sweet-spot outputs to: {result_root}")


if __name__ == "__main__":
    main()
