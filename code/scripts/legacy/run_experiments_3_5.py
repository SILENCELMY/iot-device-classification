import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from run_experiment1 import DEVICE_DIRS, FEATURE_COLUMNS, LABELS, extract_features_for_file


ROUND_DIRS = {
    "R5": "round5_positionB",
    "R6": "round6_jitter",
    "R7": "round7_jitter",
}
FEATURE_SETS = {
    "FULL": FEATURE_COLUMNS,
    "NO_RSSI": [feature for feature in FEATURE_COLUMNS if not feature.startswith("rssi_")],
}
TRAIN_ROUNDS = ["R2", "R3", "R4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint, position, and jitter generalization experiments.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root directory.")
    parser.add_argument("--output", default="results/experiments_3_5_full_vs_no_rssi", help="Output directory.")
    parser.add_argument(
        "--base-feature-cache",
        default="results/experiment1_single_round/window_features.csv",
        help="Existing R2/R3/R4 feature CSV.",
    )
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Window size for feature aggregation.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Baseline stratified validation split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract R5/R6/R7 features.")
    return parser.parse_args()


def extract_round_features(dataset_root: Path, rounds: list[str], window_seconds: float) -> pd.DataFrame:
    frames = []
    for round_name in rounds:
        round_dir = ROUND_DIRS[round_name]
        for label, device_dir in DEVICE_DIRS.items():
            pcap_path = dataset_root / device_dir / round_dir / f"{device_dir}_r{round_name[-1]}.pcapng"
            if not pcap_path.exists():
                raise FileNotFoundError(f"Missing input file: {pcap_path}")
            print(f"Extracting {round_name} {label}: {pcap_path}", flush=True)
            frames.append(extract_features_for_file(pcap_path, label, round_name, window_seconds))
    return pd.concat(frames, ignore_index=True)


def load_features(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    base_cache = Path(args.base_feature_cache)
    if not base_cache.exists():
        raise FileNotFoundError(f"Missing base feature cache: {base_cache}")
    base_features = pd.read_csv(base_cache)

    extra_cache = output_dir / "window_features_R5_R6_R7.csv"
    if extra_cache.exists() and not args.force_extract:
        extra_features = pd.read_csv(extra_cache)
    else:
        extra_features = extract_round_features(Path(args.dataset), ["R5", "R6", "R7"], args.window_seconds)
        extra_features.to_csv(extra_cache, index=False, encoding="utf-8-sig")

    features = pd.concat([base_features, extra_features], ignore_index=True)
    features.to_csv(output_dir / "window_features_used.csv", index=False, encoding="utf-8-sig")
    return features


def clean_x(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return data[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )


def metric_summary(y_true: pd.Series, y_pred: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
    }, cm, report


def save_evaluation(
    experiment_dir: Path,
    mode: str,
    summary: dict,
    cm: np.ndarray,
    report: dict,
    predictions: pd.DataFrame,
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> None:
    mode_dir = experiment_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(mode_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).transpose().to_csv(mode_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(mode_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    importances = (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(mode_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    with (mode_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (mode_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2, ensure_ascii=False)
    joblib.dump(model, mode_dir / "model.joblib")


def save_additional_model(
    mode_dir: Path,
    model: RandomForestClassifier,
    feature_columns: list[str],
    stem: str,
) -> None:
    joblib.dump(model, mode_dir / f"{stem}.joblib")
    importances = (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(mode_dir / f"{stem}_feature_importance.csv", index=False, encoding="utf-8-sig")


def evaluate_baseline(
    features: pd.DataFrame,
    output_dir: Path,
    mode: str,
    feature_columns: list[str],
    test_size: float,
    random_state: int,
) -> dict:
    data = features[features["round"].isin(TRAIN_ROUNDS)].copy()
    x = clean_x(data, feature_columns)
    y = data["label"]
    meta = data[["round", "source_file", "window_id", "window_start", "window_end"]]

    x_train, x_test, y_train, y_test, _, meta_test = train_test_split(
        x,
        y,
        meta,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    eval_model = build_model(random_state)
    eval_model.fit(x_train, y_train)
    y_pred = eval_model.predict(x_test)
    metrics, cm, report = metric_summary(y_test, y_pred)

    predictions = meta_test.copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = y_pred
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    final_model = build_model(random_state)
    final_model.fit(x, y)

    summary = {
        "experiment": "joint_training_baseline",
        "mode": mode,
        "train_rounds": TRAIN_ROUNDS,
        "evaluation": "stratified_70_30_split_within_R2_R3_R4",
        "model_joblib": "final model trained on all R2+R3+R4",
        "validation_model_joblib": "model used for the reported validation metrics",
        "samples": int(len(data)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        **metrics,
        "confusion_matrix": cm.tolist(),
    }
    save_evaluation(
        output_dir / "experiment3_joint_training_baseline",
        mode,
        summary,
        cm,
        report,
        predictions,
        final_model,
        feature_columns,
    )
    save_additional_model(
        output_dir / "experiment3_joint_training_baseline" / mode,
        eval_model,
        feature_columns,
        "validation_model",
    )
    return summary


def evaluate_generalization(
    features: pd.DataFrame,
    output_dir: Path,
    experiment_name: str,
    mode: str,
    feature_columns: list[str],
    test_rounds: list[str],
    random_state: int,
) -> dict:
    train_data = features[features["round"].isin(TRAIN_ROUNDS)].copy()
    test_data = features[features["round"].isin(test_rounds)].copy()

    x_train = clean_x(train_data, feature_columns)
    y_train = train_data["label"]
    x_test = clean_x(test_data, feature_columns)
    y_test = test_data["label"]

    model = build_model(random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics, cm, report = metric_summary(y_test, y_pred)

    predictions = test_data[["round", "source_file", "window_id", "window_start", "window_end"]].copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = y_pred
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    summary = {
        "experiment": experiment_name,
        "mode": mode,
        "train_rounds": TRAIN_ROUNDS,
        "test_rounds": test_rounds,
        "train_samples": int(len(train_data)),
        "test_samples": int(len(test_data)),
        **metrics,
        "confusion_matrix": cm.tolist(),
    }
    save_evaluation(output_dir / experiment_name, mode, summary, cm, report, predictions, model, feature_columns)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args, output_dir)
    summaries = []

    for mode, feature_columns in FEATURE_SETS.items():
        summaries.append(
            evaluate_baseline(
                features=features,
                output_dir=output_dir,
                mode=mode,
                feature_columns=feature_columns,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        )
        summaries.append(
            evaluate_generalization(
                features=features,
                output_dir=output_dir,
                experiment_name="experiment4_position_generalization",
                mode=mode,
                feature_columns=feature_columns,
                test_rounds=["R5"],
                random_state=args.random_state,
            )
        )
        summaries.append(
            evaluate_generalization(
                features=features,
                output_dir=output_dir,
                experiment_name="experiment5_jitter_generalization",
                mode=mode,
                feature_columns=feature_columns,
                test_rounds=["R6", "R7"],
                random_state=args.random_state,
            )
        )

    summary_df = pd.DataFrame([
        {
            "experiment": item["experiment"],
            "mode": item["mode"],
            "train_rounds": "+".join(item["train_rounds"]),
            "test_rounds": "+".join(item.get("test_rounds", [])),
            "samples": item.get("samples", ""),
            "train_samples": item.get("train_samples", ""),
            "test_samples": item.get("test_samples", ""),
            "accuracy": item["accuracy"],
            "precision": item["precision"],
            "recall": item["recall"],
            "macro_f1": item["macro_f1"],
        }
        for item in summaries
    ])
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\nExperiments 3-5 summary")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
