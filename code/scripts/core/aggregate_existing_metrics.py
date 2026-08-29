#!/usr/bin/env python3
"""Aggregate existing per-task metrics.json into summary_metrics.csv.

Used as a fallback when the main run is interrupted before save_summary().
Scans <output_root>/raw_all/<task>/<feature_set>/<model>/metrics.json and
writes the same flat CSV format as save_summary().

Usage:
  python3 aggregate_existing_metrics.py \\
    --output-root /path/to/robust_v2 \\
    --out /path/to/robust_v2/summary_metrics.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

# Mirror of META_COLUMNS in robust_iot_research.py
META_COLUMNS = {
    "round", "label", "traffic", "filter_mode", "source_file",
    "window_id", "window_start", "window_end",
}

META_KEYS_DROP = {"per_class_f1", "confusion_matrix"}


def flatten(metrics: dict) -> dict:
    row = {k: v for k, v in metrics.items() if k not in META_KEYS_DROP}
    for label, value in metrics.get("per_class_f1", {}).items():
        row[f"f1_{label}"] = value
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows: list[dict] = []
    for fm_dir in sorted(args.output_root.glob("*")):
        if not fm_dir.is_dir():
            continue
        for task_dir in sorted(fm_dir.glob("*")):
            if not task_dir.is_dir():
                continue
            for fs_dir in sorted(task_dir.glob("*")):
                if not fs_dir.is_dir():
                    continue
                for model_dir in sorted(fs_dir.glob("*")):
                    metrics_file = model_dir / "metrics.json"
                    if not metrics_file.exists():
                        continue
                    metrics = json.loads(metrics_file.read_text())
                    row = flatten(metrics)
                    row.setdefault("filter_mode", fm_dir.name)
                    row.setdefault("task", task_dir.name)
                    row.setdefault("feature_set", fs_dir.name)
                    row.setdefault("model", model_dir.name)
                    rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["filter_mode", "task", "feature_set", "model"]
    ).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows to {args.out}")
    if len(df):
        print(f"  tasks: {sorted(df['task'].unique())}")
        print(f"  models: {sorted(df['model'].unique())}")
        print(f"  feature_sets: {sorted(df['feature_set'].unique())}")


if __name__ == "__main__":
    main()
