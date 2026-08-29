import argparse
import json
import math
import subprocess
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
from sklearn.model_selection import train_test_split


LABELS = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]
DEVICE_DIRS = {
    "Camera": "camera",
    "Light_T1": "light_T1",
    "Light_XM": "light_xm",
    "Sensor": "sensor",
    "Socket": "socket",
}
ROUND_DIRS = {
    "R2": "round2_normal",
    "R3": "round3_normal",
    "R4": "round4_normal",
}
TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.fc.retry",
    "wlan.fc.type_subtype",
    "radiotap.dbm_antsignal",
    "wlan.sa",
    "wlan.da",
]
FEATURE_COLUMNS = [
    "packet_count",
    "byte_count",
    "len_mean",
    "len_std",
    "len_min",
    "len_max",
    "len_p25",
    "len_p50",
    "len_p75",
    "interarrival_mean",
    "interarrival_std",
    "interarrival_min",
    "interarrival_max",
    "interarrival_p50",
    "data_ratio",
    "mgmt_ratio",
    "ctrl_ratio",
    "retry_ratio",
    "unique_sa_count",
    "unique_da_count",
    "rssi_mean",
    "rssi_std",
    "rssi_min",
    "rssi_max",
    "rssi_missing_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment 1: per-round 5-class classification.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root directory.")
    parser.add_argument("--output", default="results/experiment1_single_round", help="Output directory.")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Window size for feature aggregation.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Per-round stratified test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract features even if cached CSV exists.")
    return parser.parse_args()


def run_tshark(pcap_path: Path) -> pd.DataFrame:
    command = ["tshark", "-r", str(pcap_path), "-T", "fields"]
    for field in TSHARK_FIELDS:
        command.extend(["-e", field])
    command.extend(["-E", "header=n", "-E", "separator=\t", "-E", "occurrence=a"])

    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < len(TSHARK_FIELDS):
            parts.extend([""] * (len(TSHARK_FIELDS) - len(parts)))
        rows.append(parts[: len(TSHARK_FIELDS)])
    return pd.DataFrame(rows, columns=[
        "time_epoch",
        "length",
        "fc_type",
        "fc_subtype",
        "retry",
        "type_subtype",
        "rssi",
        "sa",
        "da",
    ])


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_rssi(value: object) -> float:
    if not isinstance(value, str) or not value:
        return math.nan
    values = [to_float(item) for item in value.split(",")]
    values = [item for item in values if not math.isnan(item)]
    if not values:
        return math.nan
    return float(np.mean(values))


def summarize_window(group: pd.DataFrame, label: str, round_name: str, source_file: Path, window_id: int) -> dict:
    lengths = group["length"].astype(float)
    times = group["time_epoch"].astype(float).sort_values()
    interarrival = times.diff().dropna()
    rssi = group["rssi_value"].dropna().astype(float)
    fc_type = group["fc_type"].astype("string")
    retry = group["retry"].astype("string").str.lower()

    row = {
        "label": label,
        "round": round_name,
        "source_file": str(source_file),
        "window_id": window_id,
        "window_start": float(group["relative_time"].min()),
        "window_end": float(group["relative_time"].max()),
        "packet_count": int(len(group)),
        "byte_count": float(lengths.sum()),
        "len_mean": float(lengths.mean()),
        "len_std": float(lengths.std(ddof=0)) if len(lengths) > 1 else 0.0,
        "len_min": float(lengths.min()),
        "len_max": float(lengths.max()),
        "len_p25": float(lengths.quantile(0.25)),
        "len_p50": float(lengths.quantile(0.50)),
        "len_p75": float(lengths.quantile(0.75)),
        "interarrival_mean": float(interarrival.mean()) if len(interarrival) else 0.0,
        "interarrival_std": float(interarrival.std(ddof=0)) if len(interarrival) > 1 else 0.0,
        "interarrival_min": float(interarrival.min()) if len(interarrival) else 0.0,
        "interarrival_max": float(interarrival.max()) if len(interarrival) else 0.0,
        "interarrival_p50": float(interarrival.quantile(0.50)) if len(interarrival) else 0.0,
        "data_ratio": float((fc_type == "2").mean()),
        "mgmt_ratio": float((fc_type == "0").mean()),
        "ctrl_ratio": float((fc_type == "1").mean()),
        "retry_ratio": float(retry.isin(["true", "1"]).mean()),
        "unique_sa_count": int(group["sa"].replace("", np.nan).nunique(dropna=True)),
        "unique_da_count": int(group["da"].replace("", np.nan).nunique(dropna=True)),
        "rssi_mean": float(rssi.mean()) if len(rssi) else 0.0,
        "rssi_std": float(rssi.std(ddof=0)) if len(rssi) > 1 else 0.0,
        "rssi_min": float(rssi.min()) if len(rssi) else 0.0,
        "rssi_max": float(rssi.max()) if len(rssi) else 0.0,
        "rssi_missing_ratio": float(group["rssi_value"].isna().mean()),
    }
    return row


def extract_features_for_file(
    pcap_path: Path,
    label: str,
    round_name: str,
    window_seconds: float,
) -> pd.DataFrame:
    packets = run_tshark(pcap_path)
    packets["time_epoch"] = pd.to_numeric(packets["time_epoch"], errors="coerce")
    packets["length"] = pd.to_numeric(packets["length"], errors="coerce")
    packets = packets.dropna(subset=["time_epoch", "length"]).copy()
    packets = packets[packets["length"] > 0].copy()
    packets["rssi_value"] = packets["rssi"].map(parse_rssi)
    packets["relative_time"] = packets["time_epoch"] - packets["time_epoch"].min()
    packets["window_id"] = np.floor(packets["relative_time"] / window_seconds).astype(int)

    rows = [
        summarize_window(group, label, round_name, pcap_path, int(window_id))
        for window_id, group in packets.groupby("window_id", sort=True)
        if len(group) >= 2
    ]
    return pd.DataFrame(rows)


def build_feature_table(dataset_root: Path, output_dir: Path, window_seconds: float, force_extract: bool) -> pd.DataFrame:
    feature_path = output_dir / "window_features.csv"
    if feature_path.exists() and not force_extract:
        return pd.read_csv(feature_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for round_name, round_dir in ROUND_DIRS.items():
        for label, device_dir in DEVICE_DIRS.items():
            pcap_path = dataset_root / device_dir / round_dir / f"{device_dir}_r{round_name[-1]}.pcapng"
            if not pcap_path.exists():
                raise FileNotFoundError(f"Missing input file: {pcap_path}")
            print(f"Extracting {round_name} {label}: {pcap_path}", flush=True)
            frames.append(extract_features_for_file(pcap_path, label, round_name, window_seconds))

    features = pd.concat(frames, ignore_index=True)
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    return features


def evaluate_round(features: pd.DataFrame, round_name: str, output_dir: Path, test_size: float, random_state: int) -> dict:
    round_dir = output_dir / round_name
    round_dir.mkdir(parents=True, exist_ok=True)
    data = features[features["round"] == round_name].copy()
    x = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = data["label"]

    x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
        x,
        y,
        data[["round", "source_file", "window_id", "window_start", "window_end"]],
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
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

    predictions = meta_test.copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = y_pred
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    predictions.to_csv(round_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(report).transpose().to_csv(round_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(round_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    importances = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(round_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "round": round_name,
        "samples": int(len(data)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }
    with (round_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(dataset_root, output_dir, args.window_seconds, args.force_extract)
    summaries = [
        evaluate_round(features, round_name, output_dir, args.test_size, args.random_state)
        for round_name in ROUND_DIRS
    ]
    summary_df = pd.DataFrame([
        {
            "round": item["round"],
            "samples": item["samples"],
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

    print("\nExperiment 1 summary")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
