import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from run_experiment1 import FEATURE_COLUMNS, LABELS, build_feature_table


EXPERIMENTS = {
    "experiment_A_train_R2_R3_test_R4": {
        "train_rounds": ["R2", "R3"],
        "test_round": "R4",
    },
    "experiment_B_train_R2_R4_test_R3": {
        "train_rounds": ["R2", "R4"],
        "test_round": "R3",
    },
    "experiment_C_train_R3_R4_test_R2": {
        "train_rounds": ["R3", "R4"],
        "test_round": "R2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment 2: Leave-One-Round-Out classification.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root directory.")
    parser.add_argument("--output", default="results/experiment2_leave_one_round_out", help="Output directory.")
    parser.add_argument(
        "--feature-cache",
        default="results/experiment1_single_round/window_features.csv",
        help="Existing window feature CSV to reuse.",
    )
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Window size if features must be extracted.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_features(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    feature_cache = Path(args.feature_cache)
    if feature_cache.exists():
        features = pd.read_csv(feature_cache)
        copied_cache = output_dir / "window_features_used.csv"
        features.to_csv(copied_cache, index=False, encoding="utf-8-sig")
        return features
    return build_feature_table(Path(args.dataset), output_dir, args.window_seconds, force_extract=False)


def evaluate_split(
    features: pd.DataFrame,
    experiment_name: str,
    train_rounds: list[str],
    test_round: str,
    output_dir: Path,
    random_state: int,
) -> dict:
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_data = features[features["round"].isin(train_rounds)].copy()
    test_data = features[features["round"] == test_round].copy()

    x_train = train_data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_data["label"]
    x_test = test_data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test_data["label"]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0, output_dict=True)

    predictions = test_data[["round", "source_file", "window_id", "window_start", "window_end"]].copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = y_pred
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    predictions.to_csv(experiment_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(report).transpose().to_csv(experiment_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        experiment_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    importances = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(experiment_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": experiment_name,
        "train_rounds": train_rounds,
        "test_round": test_round,
        "train_samples": int(len(train_data)),
        "test_samples": int(len(test_data)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args, output_dir)
    summaries = [
        evaluate_split(
            features=features,
            experiment_name=experiment_name,
            train_rounds=config["train_rounds"],
            test_round=config["test_round"],
            output_dir=output_dir,
            random_state=args.random_state,
        )
        for experiment_name, config in EXPERIMENTS.items()
    ]

    summary_df = pd.DataFrame([
        {
            "experiment": item["experiment"],
            "train_rounds": "+".join(item["train_rounds"]),
            "test_round": item["test_round"],
            "train_samples": item["train_samples"],
            "test_samples": item["test_samples"],
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

    print("\nExperiment 2 summary")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
