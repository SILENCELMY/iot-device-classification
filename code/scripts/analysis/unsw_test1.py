#!/usr/bin/env python3
"""Candidate implementation for D12/D13 Test 1 (UNSW, implementation stage).

This module deliberately contains the implementation only.  It does not run at
import time and it refuses a non-empty output directory.  The formal entry
point is intended for an already-authorized staging run; the implementation
stage uses :mod:`test_unsw_test1` with synthetic data only.

The three scientific primitives below are intentionally imported from the
mainline modules.  There is no local RF, balancing, or CPD implementation:

* ``build_model`` -- ``robust_iot_research``
* ``sample_balanced`` -- ``robust_iot_research``
* ``cpd_y`` -- ``cpd_core``

The code keeps the target label in a separate evaluation path.  A test label
can be present in ``split_task`` because D13 explicitly requires balancing the
test side with the mainline ``sample_balanced``.  It is never an argument to a
fit/OOF function, threshold, or model-selection routine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

# The candidate lives beside cpd_core.py and one directory above the mainline
# core module.  Explicit paths make both direct execution and test import
# deterministic, without changing any existing file.
ANALYSIS_DIR = Path(__file__).resolve().parent
CORE_DIR = ANALYSIS_DIR.parent / "core"
REPO_ROOT = ANALYSIS_DIR.parents[2]
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from robust_iot_research import (  # noqa: E402
    SimpleStackingClassifier,
    build_model,
    sample_balanced,
)
from cpd_core import cpd_y  # noqa: E402


# ---------------------------------------------------------------------------
# Frozen D12/D13 constants
# ---------------------------------------------------------------------------

CATEGORY_ORDER: tuple[str, ...] = (
    "appliance",
    "camera",
    "hub",
    "sensor",
    "speaker",
    "switch",
)

STABLE_DEVICES: tuple[str, ...] = (
    "AmazonEcho",
    "BelkinWemoMotion",
    "BelkinWemoSwitch",
    "Dropcam",
    "HPPrinter",
    "NetatmoWeather",
    "NetatmoWelcome",
    "SamsungSmartCam",
    "SmartThings",
    "TribySpeaker",
)

MODEL_ORDER: tuple[str, ...] = ("rf", "xgboost", "lightgbm", "stacking")
BASE_MODEL_ORDER: tuple[str, ...] = ("rf", "xgboost", "lightgbm")

RANDOM_STATE = 42
MIN_WINDOWS = 100
MAX_ROWS = 20_000
IID_TRAIN_FRACTION = 0.70
OOF_BLOCKS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_Q = (0.025, 0.975)
STACKING_N_JOBS = 1
STACKING_BASE_MODEL_ORDER: tuple[str, ...] = ("rf", "xgboost", "lightgbm")
STACKING_LIGHTGBM_DETERMINISTIC_PARAMS: dict[str, bool] = {
    "deterministic": True,
    "force_col_wise": True,
}

CANONICAL_PYTHON = Path("/home/lmy/anaconda3/envs/iotcls/bin/python")
DISCUSSION_PATH = REPO_ROOT / "docs" / "CROSS_LINE_DISCUSSION_20260830.md"
IMPLEMENTATION_PATH = Path("code/scripts/analysis/unsw_test1.py")
SYNTHETIC_TEST_PATH = Path("code/scripts/analysis/test_unsw_test1.py")
DETERMINISM_PROBE_PATH = Path(
    "code/scripts/analysis/test_unsw_test1_determinism_probe.py"
)
CPD_REGRESSION_PATH = Path("code/scripts/analysis/test_cpd_core.py")
CPD_CORE_PATH = Path("code/scripts/analysis/cpd_core.py")
MAINLINE_MODEL_PATH = Path("code/scripts/core/robust_iot_research.py")
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CANONICAL_MPLCONFIGDIR = Path("/tmp/iotcls-unsw-test1-mpl")
CANONICAL_PYTHON_VERSION = "3.11.15"
CANONICAL_PACKAGE_VERSIONS: dict[str, str] = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scikit-learn": "1.9.0",
    "xgboost": "3.2.0",
    "lightgbm": "4.6.0",
    "threadpoolctl": "3.6.0",
}

EXPECTED_TASK_DEFINITIONS = 74
EXPECTED_TASK_PANEL_CELLS = 148
PROBABILITY_AUDITS_PER_CELL = 1 + len(MODEL_ORDER)
EXPECTED_PROBABILITY_AUDITS = EXPECTED_TASK_PANEL_CELLS * PROBABILITY_AUDITS_PER_CELL
PROBABILITY_ROW_SUM_ATOL = 1e-9

SHARD_PACKET_FILES: tuple[str, ...] = (
    "task_detail.csv",
    "cpd_table.csv",
    "gain_table.csv",
    "oof_folds.csv",
    "stage_audit.json",
    "input_manifest.json",
)
DETERMINISTIC_PACKET_FILES: tuple[str, ...] = (
    "task_detail.csv",
    "cpd_table.csv",
    "gain_table.csv",
    "oof_folds.csv",
    "passline.csv",
    "bootstrap_replicates.csv",
    "acceptance.json",
    "provenance.json",
    "TEST1_RESULTS_NOTE.md",
)
COMPLETE_PACKET_FILES: tuple[str, ...] = DETERMINISTIC_PACKET_FILES + ("manifest.md5",)

UNSW_META_COLUMNS = frozenset(
    {
        "device",
        "day",
        "label",
        "category",
        "source_file",
        "window_id",
        "window_start",
        "window_end",
        "window_start_epoch",
    }
)
AUDIT_COLUMNS = frozenset({"side_packet_ratio", "other_packet_ratio"})
REQUIRED_COLUMNS = frozenset(
    {
        "device",
        "day",
        "label",
        "window_id",
        "window_start_epoch",
        "side_packet_ratio",
        "other_packet_ratio",
    }
)

_DAY_FILE_RE = re.compile(r"^features_day_(?P<day>.+)\.csv$")


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def canonical_json(value: Any) -> str:
    """Return one stable JSON representation for CSV cells and manifests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json(path: Path, value: Any) -> None:
    """Write JSON with stable key order, indentation, and a final newline."""

    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish_directory(output_root: Path, writer: Callable[[Path], None]) -> None:
    """Build a complete packet in a sibling temporary directory, then rename it.

    In particular, a failed local gate cannot leave a partially populated
    ``results/unsw_test1`` directory behind.
    """

    if output_root.exists():
        raise FileExistsError(f"refusing to reuse existing output path: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        writer(temporary)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_stable_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    """Write CSV with explicit column order, UTF-8, and LF line endings."""

    out = frame.copy()
    if columns is not None:
        missing = [column for column in columns if column not in out.columns]
        if missing:
            raise ValueError(f"cannot serialize missing columns {missing} to {path}")
        out = out.loc[:, list(columns)]
    out.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


# ---------------------------------------------------------------------------
# Task catalogue: 54 OOD + 20 IID = 74 definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskDefinition:
    name: str
    kind: str
    train_days: tuple[str, ...]
    test_day: str
    k: int | None

    @property
    def task_days(self) -> tuple[str, ...]:
        return self.train_days + (self.test_day,)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "train_days": list(self.train_days),
            "test_day": self.test_day,
            "k": self.k,
        }


def build_task_catalog(days: Sequence[str]) -> list[TaskDefinition]:
    """Construct D12's fixed task order from the 20 chronological days.

    OOD tasks are emitted by ``k=1,2,3`` and then by test-day order.  IID
    tasks follow them in day order.  The ordering is part of the shard and
    serialization contract.
    """

    ordered_days = tuple(sorted({str(day) for day in days}))
    if len(ordered_days) != len(days):
        raise ValueError("task days must be unique")
    if len(ordered_days) != 20:
        raise ValueError(f"D12 requires exactly 20 days, got {len(ordered_days)}")

    tasks: list[TaskDefinition] = []
    for k in (1, 2, 3):
        for test_index in range(k, len(ordered_days)):
            train_days = ordered_days[test_index - k : test_index]
            test_day = ordered_days[test_index]
            tasks.append(
                TaskDefinition(
                    name=f"ood_k{k}_{train_days[0]}_to_{test_day}",
                    kind="ood",
                    train_days=train_days,
                    test_day=test_day,
                    k=k,
                )
            )
    for day in ordered_days:
        tasks.append(
            TaskDefinition(
                name=f"iid_{day}",
                kind="iid",
                train_days=(day,),
                test_day=day,
                k=None,
            )
        )
    validate_task_catalog(tasks, ordered_days)
    return tasks


# Short alias used by the CLI and convenient for audit code.
build_tasks = build_task_catalog


def validate_task_catalog(tasks: Sequence[TaskDefinition], days: Sequence[str]) -> dict[str, int]:
    """Assert the mechanical 54/20/74 task count before any model fit."""

    expected_days = tuple(sorted(str(day) for day in days))
    if len(expected_days) != 20 or len(set(expected_days)) != 20:
        raise AssertionError("task-count gate requires 20 unique days")
    if len(tasks) != 74:
        raise AssertionError(f"expected 74 task definitions, got {len(tasks)}")

    names = [task.name for task in tasks]
    if len(names) != len(set(names)):
        raise AssertionError("task names are not unique")
    counts = {
        "ood": sum(task.kind == "ood" for task in tasks),
        "iid": sum(task.kind == "iid" for task in tasks),
        "paired_day_iid": sum(
            task.kind == "iid" and task.test_day in {ood.test_day for ood in tasks if ood.kind == "ood"}
            for task in tasks
        ),
        "task_definitions": len(tasks),
        "panel_arms": 2,
        "task_panel_cells": len(tasks) * 2,
    }
    if counts["ood"] != 54 or counts["iid"] != 20 or counts["paired_day_iid"] != 19:
        raise AssertionError(f"D12/D13 task count mismatch: {counts}")
    for task in tasks:
        if task.kind == "ood":
            if task.k not in (1, 2, 3) or len(task.train_days) != task.k:
                raise AssertionError(f"invalid OOD task: {task}")
            if task.train_days[-1] >= task.test_day:
                raise AssertionError(f"OOD train/test days are not ordered: {task}")
        elif task.kind == "iid":
            if task.train_days != (task.test_day,) or task.k is not None:
                raise AssertionError(f"invalid IID task: {task}")
        else:
            raise AssertionError(f"unknown task kind: {task.kind}")
    return counts


def tasks_for_shard(
    tasks: Sequence[TaskDefinition], shard_index: int, shard_count: int
) -> list[TaskDefinition]:
    """Select tasks by fixed ordinal modulo for deterministic six-way sharding."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    return [task for index, task in enumerate(tasks) if index % shard_count == shard_index]


# ---------------------------------------------------------------------------
# Input and label preparation
# ---------------------------------------------------------------------------

def load_category_map(path: Path) -> dict[str, str]:
    """Load ``device_id -> category`` from the UNSW MAC map.

    Only IoT entries are accepted as model labels.  The map may contain other
    IoT categories (for example ``health`` or ``light``); those are mapped and
    then excluded by the frozen six-class panel rule.
    """

    table = pd.read_csv(path, dtype=str).fillna("")
    required = {"device_id", "category"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"MAC map missing columns: {missing}")
    if "is_iot" in table.columns:
        table = table[table["is_iot"].astype(str).str.strip() == "1"]
    table["device_id"] = table["device_id"].astype(str).str.strip()
    table["category"] = table["category"].astype(str).str.strip()
    table = table[table["device_id"] != ""]
    if table["device_id"].duplicated().any():
        dup = table.loc[table["device_id"].duplicated(), "device_id"].tolist()
        raise ValueError(f"duplicate IoT device_id in MAC map: {dup}")
    return dict(zip(table["device_id"], table["category"], strict=True))


def attach_categories(
    features: pd.DataFrame, category_by_device: Mapping[str, str]
) -> pd.DataFrame:
    """Replace the raw device-identity label with the frozen type label."""

    missing = sorted(set(features["device"].astype(str)) - set(category_by_device))
    if missing:
        raise ValueError(f"feature rows have no category mapping: {missing[:10]}")
    out = features.copy()
    out["device"] = out["device"].astype(str)
    out["day"] = out["day"].astype(str)
    out["category"] = out["device"].map(category_by_device).astype(str)
    # Mainline sample_balanced consumes the column named label.  Its value is
    # the category, never the device identity.
    out["label"] = out["category"]
    return out


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the frozen 61 numeric data features and exclude metadata only.

    ``side_packet_ratio`` and ``other_packet_ratio`` are two of the 61 frozen
    columns.  They are identically zero on Ethernet UNSW input and therefore
    remain harmless constant placeholders.  D12 forbids an *importance
    comparison* for them; it does not authorize silently changing 61D to 59D.
    """

    excluded = UNSW_META_COLUMNS
    return [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]


def validate_feature_table(
    features: pd.DataFrame,
    *,
    require_61_features: bool = False,
) -> dict[str, Any]:
    """Validate finite features and the Ethernet side/other audit columns."""

    missing = sorted(REQUIRED_COLUMNS - set(features.columns))
    if missing:
        raise ValueError(f"feature table missing required columns: {missing}")
    columns = numeric_feature_columns(features)
    if require_61_features and len(columns) != 61:
        raise AssertionError(f"expected 61 numeric data features, got {len(columns)}")
    if not AUDIT_COLUMNS.issubset(columns):
        raise AssertionError("side/other audit columns must be numeric members of the frozen 61D set")
    if not columns:
        raise ValueError("no numeric data features remain after metadata exclusion")

    finite = np.isfinite(features[columns].to_numpy(dtype=float)).all()
    if not finite:
        raise AssertionError("data features contain NaN or inf")
    side_zero = bool(np.all(features["side_packet_ratio"].to_numpy(dtype=float) == 0.0))
    other_zero = bool(np.all(features["other_packet_ratio"].to_numpy(dtype=float) == 0.0))
    if not side_zero or not other_zero:
        raise AssertionError("side_packet_ratio/other_packet_ratio must be identically zero")
    if not pd.api.types.is_numeric_dtype(features["window_id"]):
        raise ValueError("window_id must be numeric")
    if not pd.api.types.is_numeric_dtype(features["window_start_epoch"]):
        raise ValueError("window_start_epoch must be numeric")
    if features["device"].isna().any() or features["day"].isna().any():
        raise ValueError("device/day metadata cannot be missing")
    return {
        "n_rows": int(len(features)),
        "n_numeric_features": int(len(columns)),
        "numeric_features": list(columns),
        "finite": bool(finite),
        "side_packet_ratio_zero": side_zero,
        "other_packet_ratio_zero": other_zero,
        "zero_audit_columns_in_model_features": sorted(AUDIT_COLUMNS),
        "feature_importance_comparison_performed": False,
    }


def discover_input_files(feature_root: Path) -> list[tuple[str, Path, Path]]:
    """Discover exactly 20 day CSVs and their 20 run_meta sidecars."""

    found: list[tuple[str, Path, Path]] = []
    for path in sorted(feature_root.glob("features_day_*.csv"), key=lambda item: item.name):
        match = _DAY_FILE_RE.match(path.name)
        if match is None:
            continue
        day = match.group("day")
        meta = path.with_suffix(".run_meta.json")
        if not meta.exists():
            raise FileNotFoundError(f"missing run_meta sidecar for {path.name}: {meta.name}")
        found.append((day, path, meta))
    if len(found) != 20:
        raise ValueError(f"D12 requires 20 feature CSVs, found {len(found)} in {feature_root}")
    days = [item[0] for item in found]
    if len(set(days)) != 20:
        raise ValueError("duplicate day in feature CSV names")
    return found


def load_unsw_features(
    feature_root: Path, mac_map_path: Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load the formal input; this function is never called by synthetic tests."""

    files = discover_input_files(feature_root)
    category_map = load_category_map(mac_map_path)
    frames: list[pd.DataFrame] = []
    input_manifest: list[dict[str, Any]] = []
    for day, csv_path, meta_path in files:
        # Parsing the sidecar is an integrity check only; its scientific values
        # are not used as labels, features, or model-selection inputs.
        try:
            json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid run_meta JSON: {meta_path}") from exc
        frame = pd.read_csv(csv_path)
        if "day" not in frame or set(frame["day"].astype(str)) != {day}:
            raise ValueError(f"day column does not match filename in {csv_path.name}")
        frames.append(frame)
        input_manifest.extend(
            [
                {"file": csv_path.name, "sha256": sha256_file(csv_path)},
                {"file": meta_path.name, "sha256": sha256_file(meta_path)},
            ]
        )
    features = attach_categories(pd.concat(frames, ignore_index=True), category_map)
    audit = validate_feature_table(features, require_61_features=True)
    manifest = {
        "feature_root_relative": "results/unsw_features_full",
        "mac_map": {
            "file": "dataset/unsw/device_mac_map.csv",
            "sha256": sha256_file(mac_map_path),
        },
        "files": input_manifest,
        "days": sorted(day for day, _, _ in files),
        "feature_audit": audit,
    }
    return features, manifest, audit


# ---------------------------------------------------------------------------
# Device panels and deterministic splits
# ---------------------------------------------------------------------------

def device_day_window_counts(features: pd.DataFrame) -> pd.DataFrame:
    counts = (
        features.groupby(["device", "day"], sort=True)["window_id"]
        .nunique()
        .rename("n_windows")
        .reset_index()
    )
    return counts.sort_values(["device", "day"], kind="stable").reset_index(drop=True)


def _count_lookup(counts: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (str(row.device), str(row.day)): int(row.n_windows)
        for row in counts.itertuples(index=False)
    }


def panel_devices(
    features: pd.DataFrame,
    task: TaskDefinition,
    panel: str,
    *,
    min_windows: int = MIN_WINDOWS,
) -> tuple[list[str], pd.DataFrame]:
    """Return the task's device panel and the full device/day count table."""

    if panel not in {"primary", "stable"}:
        raise ValueError(f"unknown panel arm: {panel}")
    counts = device_day_window_counts(features)
    lookup = _count_lookup(counts)
    required_days = (task.test_day,) if task.kind == "iid" else task.task_days
    all_devices = sorted(str(device) for device in features["device"].unique())

    def has_threshold(device: str) -> bool:
        return all(lookup.get((device, day), 0) >= min_windows for day in required_days)

    if panel == "stable":
        all_days = tuple(sorted(str(day) for day in features["day"].unique()))
        missing = [
            f"{device}/{day}"
            for device in STABLE_DEVICES
            for day in all_days
            if lookup.get((device, day), 0) < min_windows
        ]
        if missing:
            raise AssertionError(
                f"stable panel is not full across all input days for {task.name}: {missing[:10]}"
            )
        selected = list(STABLE_DEVICES)
    else:
        selected = [device for device in all_devices if has_threshold(device)]

    # The panel is a six-class support restriction, not a post-result choice.
    categories = features.set_index("device")["category"].to_dict()
    selected = [
        device for device in selected if str(categories.get(device, "")) in CATEGORY_ORDER
    ]
    if panel == "stable" and selected != list(STABLE_DEVICES):
        raise AssertionError("stable panel contains a device outside the frozen six-class support")
    if not selected:
        raise AssertionError(f"empty {panel} panel for {task.name}")
    return selected, counts


def _sort_for_iid(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(
        ["window_start_epoch", "window_id"],
        kind="mergesort",
    )


def iid_time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each device's stable time order into the first 70% / last 30%."""

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in data.groupby("device", sort=True):
        ordered = _sort_for_iid(group)
        n = len(ordered)
        if n < 2:
            raise ValueError("IID time split requires at least two windows per device")
        n_train = int(np.floor(n * IID_TRAIN_FRACTION))
        n_train = max(1, min(n - 1, n_train))
        train_parts.append(ordered.iloc[:n_train])
        test_parts.append(ordered.iloc[n_train:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def sample_key_records(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return deterministic, label-free sample keys for the audit packet."""

    required = {"device", "day", "window_id"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"cannot form sample keys; missing {missing}")
    records: list[dict[str, Any]] = []
    for row in data[["device", "day", "window_id"]].itertuples(index=False, name=None):
        records.append(
            {
                "day": str(row[1]),
                "device": str(row[0]),
                "window_id": int(row[2]),
            }
        )
    return records


@dataclass
class PreparedTask:
    task: TaskDefinition
    panel: str
    train: pd.DataFrame
    test: pd.DataFrame
    detail: dict[str, Any]


def split_task(
    features: pd.DataFrame,
    task: TaskDefinition,
    panel: str,
    *,
    max_rows: int = MAX_ROWS,
    random_state: int = RANDOM_STATE,
    min_windows: int = MIN_WINDOWS,
) -> PreparedTask:
    """Apply panel filtering, split, and the prescribed two balancing calls."""

    selected_devices, counts = panel_devices(
        features,
        task,
        panel,
        min_windows=min_windows,
    )
    relevant_days = set(task.task_days)
    panel_data = features.loc[
        features["device"].isin(selected_devices)
        & features["day"].isin(relevant_days)
        & features["category"].isin(CATEGORY_ORDER)
    ].copy()
    panel_data = panel_data.reset_index(drop=True)
    if task.kind == "ood":
        train_pre = panel_data[panel_data["day"].isin(task.train_days)].copy()
        test_pre = panel_data[panel_data["day"] == task.test_day].copy()
    else:
        train_pre, test_pre = iid_time_split(panel_data)

    if train_pre.empty or test_pre.empty:
        raise AssertionError(f"empty split for {task.name}/{panel}")

    # These are the only two calls to the mainline balancing implementation.
    # In particular, do not replace this with a local sampler.
    train = sample_balanced(train_pre, max_rows=max_rows, random_state=random_state)
    test = sample_balanced(test_pre, max_rows=max_rows, random_state=random_state)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    expected_categories = tuple(sorted(CATEGORY_ORDER))
    train_categories = tuple(sorted(set(train["category"].astype(str))))
    test_categories = tuple(sorted(set(test["category"].astype(str))))
    if train_categories != expected_categories or test_categories != expected_categories:
        raise AssertionError(
            f"{task.name}/{panel} support differs by split: "
            f"train={train_categories}, test={test_categories}"
        )
    if not train["label"].astype(str).equals(train["category"].astype(str)):
        raise AssertionError(f"training label/category mismatch in {task.name}/{panel}")
    if not test["label"].astype(str).equals(test["category"].astype(str)):
        raise AssertionError(f"test label/category mismatch in {task.name}/{panel}")
    train_keys = sample_key_records(train)
    test_keys = sample_key_records(test)
    if len({(item["device"], item["day"], item["window_id"]) for item in train_keys}) != len(train_keys):
        raise AssertionError(f"duplicate training sample keys in {task.name}/{panel}")
    if len({(item["device"], item["day"], item["window_id"]) for item in test_keys}) != len(test_keys):
        raise AssertionError(f"duplicate test sample keys in {task.name}/{panel}")
    overlap = set((item["device"], item["day"], item["window_id"]) for item in train_keys) & set(
        (item["device"], item["day"], item["window_id"]) for item in test_keys
    )
    if overlap:
        raise AssertionError(f"train/test sample-key overlap in {task.name}/{panel}")

    task_counts = counts[counts["day"].isin(relevant_days) & counts["device"].isin(selected_devices)]
    device_day = {
        str(device): {
            str(day): int(n_windows)
            for day, n_windows in sorted(
                zip(group["day"], group["n_windows"], strict=True),
                key=lambda item: item[0],
            )
        }
        for device, group in task_counts.groupby("device", sort=True)
    }
    detail = {
        "task": task.name,
        "kind": task.kind,
        "k": task.k,
        "train_days": list(task.train_days),
        "test_day": task.test_day,
        "panel": panel,
        "panel_device_count": len(selected_devices),
        "panel_devices": list(selected_devices),
        "device_day_windows": device_day,
        "category_order": list(CATEGORY_ORDER),
        "category_count": len(CATEGORY_ORDER),
        "train_category_order": list(train_categories),
        "train_category_count": len(train_categories),
        "test_category_order": list(test_categories),
        "test_category_count": len(test_categories),
        "train_rows_before_sampling": int(len(train_pre)),
        "test_rows_before_sampling": int(len(test_pre)),
        "train_rows_after_sampling": int(len(train)),
        "test_rows_after_sampling": int(len(test)),
        "max_rows": int(max_rows),
        "random_state": int(random_state),
        "iid_split": (
            "device_internal_sort(window_start_epoch,window_id),first_70pct_train,last_30pct_test"
            if task.kind == "iid"
            else "consecutive_complete_days"
        ),
        "train_sample_keys": train_keys,
        "test_sample_keys": test_keys,
    }
    return PreparedTask(task=task, panel=panel, train=train, test=test, detail=detail)


# ---------------------------------------------------------------------------
# D13 OOF folds and model paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OofFold:
    fold: int
    mode: str
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    train_days: tuple[str, ...]
    val_days: tuple[str, ...]
    train_time_min: float | None
    train_time_max: float | None
    val_time_min: float | None
    val_time_max: float | None


def make_oof_folds(train: pd.DataFrame, task: TaskDefinition) -> list[OofFold]:
    """Make the exact D13 day-grouped or five-block OOF folds."""

    if train.empty:
        raise ValueError("cannot make OOF folds for empty training data")
    y_placeholder = np.zeros(len(train), dtype=int)
    x_placeholder = np.zeros((len(train), 1), dtype=float)
    days = train["day"].astype(str).to_numpy()
    unique_days = np.unique(days)
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

    if task.kind == "ood" and (task.k or 0) >= 2:
        if len(unique_days) < 2:
            raise AssertionError(f"{task.name} k>=2 has fewer than two training days")
        n_splits = max(2, min(OOF_BLOCKS, len(unique_days)))
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in splitter.split(x_placeholder, y_placeholder, groups=days):
            pairs.append(("grouped_day", np.asarray(train_idx), np.asarray(val_idx)))
    else:
        # This is the same quantile/searchsorted construction used by the
        # mainline SimpleStackingClassifier for one-group grouped OOF.
        ws = train["window_start_epoch"].to_numpy(dtype=float)
        n_blocks = max(2, min(OOF_BLOCKS, len(ws)))
        edges = np.quantile(ws, np.linspace(0, 1, n_blocks + 1)[1:-1])
        block_id = np.searchsorted(edges, ws, side="right")
        for block in range(n_blocks):
            val_idx = np.flatnonzero(block_id == block)
            if len(val_idx) == 0:
                continue
            train_idx = np.flatnonzero(block_id != block)
            pairs.append(("time_block", train_idx, val_idx))

    folds: list[OofFold] = []
    for fold_no, (mode, train_idx, val_idx) in enumerate(pairs):
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise AssertionError(f"empty OOF side in fold {fold_no} for {task.name}")
        if set(train_idx) & set(val_idx):
            raise AssertionError(f"OOF train/validation overlap in fold {fold_no}")
        train_times = train.iloc[train_idx]["window_start_epoch"].to_numpy(dtype=float)
        val_times = train.iloc[val_idx]["window_start_epoch"].to_numpy(dtype=float)
        folds.append(
            OofFold(
                fold=fold_no,
                mode=mode,
                train_idx=tuple(int(index) for index in train_idx),
                val_idx=tuple(int(index) for index in val_idx),
                train_days=tuple(sorted(set(days[train_idx]))),
                val_days=tuple(sorted(set(days[val_idx]))),
                train_time_min=float(train_times.min()),
                train_time_max=float(train_times.max()),
                val_time_min=float(val_times.min()),
                val_time_max=float(val_times.max()),
            )
        )

    validation_indices = [index for fold in folds for index in fold.val_idx]
    if sorted(validation_indices) != list(range(len(train))):
        raise AssertionError(f"OOF validation folds do not cover training rows for {task.name}")
    expected_folds = len(unique_days) if task.kind == "ood" and (task.k or 0) >= 2 else OOF_BLOCKS
    if len(folds) != expected_folds:
        raise AssertionError(
            f"{task.name} requires exactly {expected_folds} OOF folds, got {len(folds)}"
        )
    return folds


def assert_mainline_stacking_fold_semantics(
    train: pd.DataFrame, task: TaskDefinition, folds: Sequence[OofFold]
) -> None:
    """Compare our persisted fold partition with the mainline splitter."""

    model = SimpleStackingClassifier(
        estimators=[],
        final_estimator=None,
        cv=OOF_BLOCKS,
        random_state=RANDOM_STATE,
        oof_mode="grouped",
    )
    actual = list(
        model._splitter(
            train,
            np.zeros(len(train), dtype=int),
            train["day"].astype(str).to_numpy(),
            train["window_start_epoch"].to_numpy(dtype=float),
        )
    )
    expected = [(np.asarray(fold.train_idx), np.asarray(fold.val_idx)) for fold in folds]
    if len(actual) != len(expected) or any(
        not np.array_equal(a[0], e[0]) or not np.array_equal(a[1], e[1])
        for a, e in zip(actual, expected, strict=True)
    ):
        raise AssertionError("persisted OOF folds differ from mainline stacking splitter")


def _encode_categories(labels: pd.Series) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(CATEGORY_ORDER)}
    unknown = sorted(set(labels.astype(str)) - set(mapping))
    if unknown:
        raise ValueError(f"unknown category labels: {unknown}")
    return labels.astype(str).map(mapping).to_numpy(dtype=int)


def clean_model_features(data: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    return data.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _expand_probabilities(
    probabilities: np.ndarray,
    classes: Sequence[Any],
    n_classes: int,
    *,
    context: str = "model",
) -> np.ndarray:
    raw = np.asarray(probabilities)
    if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] == 0:
        raise AssertionError(f"{context} returned an empty/non-matrix probability array")
    if not np.issubdtype(raw.dtype, np.floating):
        raise AssertionError(f"{context} returned non-floating probabilities")
    class_indices = np.asarray(classes, dtype=int)
    if raw.shape[1] != len(class_indices):
        raise AssertionError(f"{context} probability/class column count mismatch")

    full = np.zeros((len(raw), n_classes), dtype=np.float64)
    for source_index, cls in enumerate(class_indices):
        if not 0 <= int(cls) < n_classes:
            raise AssertionError(f"model returned class outside fixed support: {cls}")
        full[:, int(cls)] = raw[:, source_index]
    return _normalize_probability_rows(full, context, source_dtype=raw.dtype)


def _normalize_probability_rows(
    probabilities: np.ndarray,
    context: str,
    *,
    source_dtype: Any | None = None,
) -> np.ndarray:
    """Normalize representation-level probability drift without hiding malformed output."""

    raw = np.asarray(probabilities)
    if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] == 0:
        raise AssertionError(f"{context} returned an empty/non-matrix probability array")
    dtype = np.dtype(raw.dtype if source_dtype is None else source_dtype)
    if not np.issubdtype(raw.dtype, np.floating) or not np.issubdtype(dtype, np.floating):
        raise AssertionError(f"{context} returned non-floating probabilities")

    values = np.asarray(raw, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    lower_bound = float(np.min(values))
    upper_bound = float(np.max(values))
    bounded = bool(lower_bound >= -1e-12 and upper_bound <= 1.0 + 1e-12)
    if not finite or not bounded:
        raise AssertionError(
            f"{context} pre-normalization probability bounds failed: "
            f"finite={finite}, min={lower_bound}, max={upper_bound}"
        )

    row_sums = values.sum(axis=1)
    if not bool(np.all(row_sums > 0.0)):
        raise AssertionError(f"{context} returned a non-positive probability row sum")
    max_error = float(np.max(np.abs(row_sums - 1.0)))
    rounding_tolerance = max(
        1e-12,
        values.shape[1] * float(np.finfo(dtype).eps),
    )
    if max_error > rounding_tolerance:
        raise AssertionError(
            f"{context} probability row sum exceeds dtype rounding allowance: "
            f"max_error={max_error}, allowance={rounding_tolerance}"
        )

    normalized = values / row_sums[:, np.newaxis]
    normalized_error = float(
        np.max(np.abs(normalized.sum(axis=1) - 1.0))
    )
    if normalized_error > PROBABILITY_ROW_SUM_ATOL:
        raise AssertionError(
            f"{context} normalized probability row sum exceeds "
            f"{PROBABILITY_ROW_SUM_ATOL}: max_error={normalized_error}"
        )
    return normalized


def probability_row_audit(probabilities: np.ndarray, context: str) -> dict[str, Any]:
    if probabilities.ndim != 2 or len(probabilities) == 0 or probabilities.shape[1] == 0:
        raise AssertionError(f"{context} returned an empty/non-matrix probability array")
    finite = bool(np.isfinite(probabilities).all())
    lower_bound = float(np.min(probabilities))
    upper_bound = float(np.max(probabilities))
    row_sums = probabilities.sum(axis=1)
    max_error = float(np.max(np.abs(row_sums - 1.0)))
    bounded = bool(lower_bound >= -1e-12 and upper_bound <= 1.0 + 1e-12)
    rows_one = bool(max_error <= PROBABILITY_ROW_SUM_ATOL)
    if not finite or not bounded or not rows_one:
        raise AssertionError(
            f"{context} probability audit failed: finite={finite}, bounded={bounded}, "
            f"max_row_sum_error={max_error}"
        )
    return {
        "n_rows": int(len(probabilities)),
        "finite": finite,
        "bounded_0_1": bounded,
        "min_probability": lower_bound,
        "max_probability": upper_bound,
        "max_row_sum_error": max_error,
        "row_sums_one": rows_one,
    }


def rf_oof_reference_cm(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    task: TaskDefinition,
    folds: Sequence[OofFold],
    *,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train mainline RF fold models and form the no-leakage reference CM."""

    x = clean_model_features(train, feature_columns)
    y = _encode_categories(train["label"])
    probabilities = np.zeros((len(train), len(CATEGORY_ORDER)), dtype=float)
    for fold in folds:
        # Every fold model is constructed by the mainline factory.
        model = build_model("rf", RANDOM_STATE, n_jobs, len(CATEGORY_ORDER))
        if model is None:
            raise RuntimeError("mainline RF factory unexpectedly returned None")
        model.fit(x.iloc[list(fold.train_idx)], y[list(fold.train_idx)])
        fold_proba = _expand_probabilities(
            model.predict_proba(x.iloc[list(fold.val_idx)]),
            model.classes_,
            len(CATEGORY_ORDER),
            context="RF OOF",
        )
        probabilities[list(fold.val_idx), :] = fold_proba

    audit = probability_row_audit(probabilities, "RF OOF")
    predictions = np.argmax(probabilities, axis=1)
    cm = confusion_matrix(y, predictions, labels=np.arange(len(CATEGORY_ORDER)))
    return cm, audit


@dataclass
class ModelPrediction:
    name: str
    predictions: np.ndarray
    probabilities: np.ndarray


def build_test1_model(model_name: str, *, n_jobs: int):
    """Build one mainline model with the frozen R4 execution constraints.

    Standalone models retain the formal caller's thread count.  Stacking is
    numerically isolated to one thread and its required LightGBM member is put
    in explicit deterministic column-wise mode.  The model family and all
    scientific hyperparameters continue to come from the mainline factory.
    """

    effective_n_jobs = STACKING_N_JOBS if model_name == "stacking" else n_jobs
    model = build_model(
        model_name,
        RANDOM_STATE,
        effective_n_jobs,
        len(CATEGORY_ORDER),
    )
    if model is None:
        raise RuntimeError(f"required model is unavailable: {model_name}")
    if model_name != "stacking":
        if isinstance(model, SimpleStackingClassifier):
            raise AssertionError(f"unexpected stacking instance for model {model_name}")
        return model
    if not isinstance(model, SimpleStackingClassifier):
        raise AssertionError("mainline stacking factory returned an unexpected model type")

    estimator_names = tuple(name for name, _estimator in model.estimators)
    if estimator_names != STACKING_BASE_MODEL_ORDER:
        raise AssertionError(
            "formal stacking base models changed: "
            f"expected {STACKING_BASE_MODEL_ORDER}, got {estimator_names}"
        )
    for name, estimator in model.estimators:
        params = estimator.get_params(deep=False)
        if params.get("n_jobs") != STACKING_N_JOBS:
            raise AssertionError(f"stacking base model {name} is not single-threaded")
        if name == "lightgbm":
            estimator.set_params(**STACKING_LIGHTGBM_DETERMINISTIC_PARAMS)
            configured = estimator.get_params(deep=False)
            if any(
                configured.get(key) is not expected
                for key, expected in STACKING_LIGHTGBM_DETERMINISTIC_PARAMS.items()
            ):
                raise AssertionError("LightGBM deterministic parameters were not applied")
    return model


def fit_model_predictions(
    model_name: str,
    train: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    n_jobs: int,
) -> ModelPrediction:
    """Fit one fixed model and predict test features.

    ``test_features`` must be an already-separated feature-only frame.  The
    fitting boundary therefore cannot access a test label even accidentally;
    evaluation labels are consumed by the caller only after this returns.
    """

    x_train = clean_model_features(train, feature_columns)
    if "label" in test_features.columns or "category" in test_features.columns:
        raise AssertionError("fit boundary received test labels/categories")
    if list(test_features.columns) != list(feature_columns):
        raise AssertionError("fit boundary requires the exact ordered feature-only columns")
    x_test = test_features.loc[:, list(feature_columns)].copy()
    y_train = _encode_categories(train["label"])
    model = build_test1_model(model_name, n_jobs=n_jobs)

    if isinstance(model, SimpleStackingClassifier):
        # k>=2: day groups; k=1 and IID: one group, mainline falls back to
        # five continuous time blocks.  Both metadata arrays are supplied so
        # the mainline splitter makes the D13 decision mechanically.
        with threadpool_limits(limits=STACKING_N_JOBS):
            model.fit(
                x_train,
                y_train,
                train_round=train["day"].astype(str).to_numpy(),
                window_start=train["window_start_epoch"].to_numpy(dtype=float),
            )
            raw_probabilities = model.predict_proba(x_test)
            raw_predictions = model.predict(x_test)
    else:
        model.fit(x_train, y_train)
        raw_probabilities = model.predict_proba(x_test)
        raw_predictions = model.predict(x_test)
    probabilities = _expand_probabilities(
        raw_probabilities,
        getattr(model, "classes_", np.arange(len(CATEGORY_ORDER))),
        len(CATEGORY_ORDER),
        context=model_name,
    )
    probability_row_audit(probabilities, model_name)
    predictions = np.asarray(raw_predictions, dtype=int)
    if not np.isin(predictions, np.arange(len(CATEGORY_ORDER))).all():
        raise AssertionError(f"{model_name} predicted outside fixed class support")
    return ModelPrediction(model_name, predictions, probabilities)


def _fold_rows(
    prepared: PreparedTask, folds: Sequence[OofFold], oof_model: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        rows.append(
            {
                "task": prepared.task.name,
                "panel": prepared.panel,
                "oof_model": oof_model,
                "fold": fold.fold,
                "mode": fold.mode,
                "group_field": "day" if fold.mode == "grouped_day" else "none",
                "train_indices": list(fold.train_idx),
                "validation_indices": list(fold.val_idx),
                "train_days": list(fold.train_days),
                "validation_days": list(fold.val_days),
                "train_time_min": fold.train_time_min,
                "train_time_max": fold.train_time_max,
                "validation_time_min": fold.val_time_min,
                "validation_time_max": fold.val_time_max,
            }
        )
    return rows


@dataclass
class TaskResult:
    detail: dict[str, Any]
    cpd: dict[str, Any]
    gain: dict[str, Any]
    models: list[dict[str, Any]]
    oof_folds: list[dict[str, Any]]
    probability_audit: dict[str, Any]


def run_prepared_task(
    prepared: PreparedTask,
    feature_columns: Sequence[str],
    *,
    n_jobs: int,
    model_names: Sequence[str] = MODEL_ORDER,
) -> TaskResult:
    """Run one task/panel cell; no file or canonical result is created here."""

    requested = tuple(model_names)
    if requested != MODEL_ORDER:
        raise ValueError(f"formal model set must be exactly {MODEL_ORDER}, got {requested}")
    folds = make_oof_folds(prepared.train, prepared.task)
    assert_mainline_stacking_fold_semantics(prepared.train, prepared.task, folds)
    ref_cm, ref_audit = rf_oof_reference_cm(
        prepared.train,
        feature_columns,
        prepared.task,
        folds,
        n_jobs=n_jobs,
    )

    predictions: dict[str, ModelPrediction] = {}
    model_rows: list[dict[str, Any]] = []
    probability_audit: dict[str, Any] = {"rf_oof": ref_audit, "test_models": {}}
    x_test = clean_model_features(prepared.test, feature_columns)
    for model_name in MODEL_ORDER:
        result = fit_model_predictions(
            model_name,
            prepared.train,
            x_test,
            feature_columns,
            n_jobs=n_jobs,
        )
        predictions[model_name] = result
        probability_audit["test_models"][model_name] = probability_row_audit(
            result.probabilities, model_name
        )

    # Target labels are first read only after every fixed model has completed
    # fit and prediction.  Nothing below retrains, calibrates or routes a model.
    y_test = _encode_categories(prepared.test["label"])
    for model_name in MODEL_ORDER:
        result = predictions[model_name]
        score = float(
            f1_score(
                y_test,
                result.predictions,
                labels=np.arange(len(CATEGORY_ORDER)),
                average="macro",
                zero_division=0,
            )
        )
        cm = confusion_matrix(y_test, result.predictions, labels=np.arange(len(CATEGORY_ORDER)))
        model_rows.append(
            {
                "task": prepared.task.name,
                "kind": prepared.task.kind,
                "test_day": prepared.task.test_day,
                "panel": prepared.panel,
                "model": model_name,
                "class_count": len(CATEGORY_ORDER),
                "macro_f1": score,
                "confusion_matrix": cm.tolist(),
            }
        )

    target_cm = np.asarray(
        next(row["confusion_matrix"] for row in model_rows if row["model"] == "rf"),
        dtype=float,
    )
    cpd_value = cpd_y(ref_cm, target_cm)
    if not np.isfinite(cpd_value):
        raise AssertionError(f"non-finite CPD_y for {prepared.task.name}/{prepared.panel}")
    cpd_row = {
        "task": prepared.task.name,
        "kind": prepared.task.kind,
        "test_day": prepared.task.test_day,
        "panel": prepared.panel,
        "class_count": len(CATEGORY_ORDER),
        "category_order": list(CATEGORY_ORDER),
        "cpd_y": float(cpd_value),
        "cm_ref_rf_oof": ref_cm.tolist(),
        "cm_tgt_rf": target_cm.tolist(),
    }

    scores = {row["model"]: float(row["macro_f1"]) for row in model_rows}
    best_base = max(BASE_MODEL_ORDER, key=lambda name: (scores[name], -BASE_MODEL_ORDER.index(name)))
    gain_row = {
        "task": prepared.task.name,
        "kind": prepared.task.kind,
        "test_day": prepared.task.test_day,
        "panel": prepared.panel,
        "class_count": len(CATEGORY_ORDER),
        "rf_macro_f1": scores["rf"],
        "xgboost_macro_f1": scores["xgboost"],
        "lightgbm_macro_f1": scores["lightgbm"],
        "best_base_model": best_base,
        "best_base_macro_f1": scores[best_base],
        "stacking_macro_f1": scores["stacking"],
        "stacking_gain": scores["stacking"] - scores[best_base],
    }

    detail = dict(prepared.detail)
    detail.update(
        {
            "oof_fold_count": len(folds),
            "oof_mode": folds[0].mode,
            "oof_group_field": "day" if folds[0].mode == "grouped_day" else "none",
            "oof_semantics": (
                "k>=2: GroupKFold(group=training_day)"
                if folds[0].mode == "grouped_day"
                else "k=1/IID: five continuous window_start_epoch blocks"
            ),
        }
    )
    return TaskResult(
        detail=detail,
        cpd=cpd_row,
        gain=gain_row,
        models=model_rows,
        oof_folds=_fold_rows(prepared, folds, "rf_reference")
        + _fold_rows(prepared, folds, "stacking"),
        probability_audit=probability_audit,
    )


# ---------------------------------------------------------------------------
# Bootstrap pass lines (D12/D13 frozen, deterministic)
# ---------------------------------------------------------------------------

def _ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    lower, upper = np.quantile(array, BOOTSTRAP_Q, method="linear")
    return float(lower), float(upper)


def _cpd_clusters(
    cpd_table: pd.DataFrame,
    *,
    panel: str,
) -> tuple[dict[str, np.ndarray], dict[str, float], float, float]:
    primary = cpd_table[cpd_table["panel"] == panel].copy()
    ood = primary[primary["kind"] == "ood"]
    iid = primary[primary["kind"] == "iid"]
    if len(ood) != 54 or len(iid) != 20:
        raise AssertionError("primary CPD table does not contain 54 OOD + 20 IID rows")
    first_day = min(str(day) for day in iid["test_day"])
    paired_days = sorted(set(ood["test_day"].astype(str)) & set(iid["test_day"].astype(str)))
    if len(paired_days) != 19 or first_day in paired_days:
        raise AssertionError(f"expected 19 paired test days, got {paired_days}")
    ood_clusters = {
        day: ood.loc[ood["test_day"].astype(str) == day, "cpd_y"].to_numpy(dtype=float)
        for day in paired_days
    }
    iid_by_day = {
        day: float(iid.loc[iid["test_day"].astype(str) == day, "cpd_y"].iloc[0])
        for day in paired_days
    }
    ood_values = np.concatenate([ood_clusters[day] for day in paired_days])
    iid_values = np.asarray([iid_by_day[day] for day in paired_days], dtype=float)
    return ood_clusters, iid_by_day, float(np.mean(ood_values)), float(np.mean(iid_values))


def bootstrap_cpd_difference(
    cpd_table: pd.DataFrame,
    *,
    panel: str = "primary",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Cluster-bootstrap OOD mean minus paired IID mean by test day."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if panel not in {"primary", "stable"}:
        raise ValueError(f"unknown panel: {panel}")
    ood_clusters, iid_by_day, ood_mean, iid_mean = _cpd_clusters(cpd_table, panel=panel)
    days = tuple(sorted(ood_clusters))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    while len(values) < replicates:
        attempts += 1
        sampled = rng.integers(0, len(days), size=len(days))
        sampled_ood = np.concatenate([ood_clusters[days[index]] for index in sampled])
        sampled_iid = np.asarray([iid_by_day[days[index]] for index in sampled], dtype=float)
        if len(sampled_ood) == 0 or len(sampled_iid) == 0:
            continue
        statistic = float(np.mean(sampled_ood) - np.mean(sampled_iid))
        if np.isfinite(statistic):
            values.append(statistic)
    lower, upper = _ci(values)
    summary = {
        "criterion": "criterion_1_cpd_ood_minus_iid",
        "panel": panel,
        "ood_mean": ood_mean,
        "paired_iid_mean": iid_mean,
        "point_estimate": ood_mean - iid_mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_replicates": int(replicates),
        "bootstrap_attempts": int(attempts),
        "bootstrap_seed": int(seed),
        "cluster_field": "test_day",
        "paired_day_count": len(days),
        "passed": (
            bool((ood_mean - iid_mean) > 0 and lower > 0)
            if panel == "primary"
            else None
        ),
        "applicability": "primary_decision" if panel == "primary" else "sensitivity_only",
    }
    replicate_table = pd.DataFrame(
        {
            "criterion": "criterion_1_cpd_ood_minus_iid",
            "panel": panel,
            "replicate": np.arange(replicates, dtype=int),
            "statistic": values,
        }
    )
    return summary, replicate_table


def bootstrap_stacking_gain(
    cpd_table: pd.DataFrame,
    gain_table: pd.DataFrame,
    *,
    panel: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Bootstrap criterion 2 by test-day clusters with a fixed OOD median."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    cpd = cpd_table[(cpd_table["panel"] == panel) & (cpd_table["kind"] == "ood")].copy()
    gain = gain_table[(gain_table["panel"] == panel) & (gain_table["kind"] == "ood")].copy()
    merged = cpd[["task", "test_day", "cpd_y"]].merge(
        gain[["task", "test_day", "stacking_gain"]],
        on=["task", "test_day"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 54:
        raise AssertionError(f"criterion 2 requires 54 OOD rows, got {len(merged)}")
    median = float(np.median(merged["cpd_y"].to_numpy(dtype=float)))
    merged["cpd_group"] = np.where(merged["cpd_y"] > median, "high", "low")
    high = merged[merged["cpd_group"] == "high"]
    low = merged[merged["cpd_group"] == "low"]
    if high.empty or low.empty:
        raise AssertionError("fixed CPD median produced an empty group")
    high_mean = float(high["stacking_gain"].mean())
    low_mean = float(low["stacking_gain"].mean())
    diff_mean = high_mean - low_mean

    clusters = {
        str(day): group for day, group in merged.groupby("test_day", sort=True)
    }
    days = tuple(sorted(clusters))
    rng = np.random.default_rng(seed)
    high_values: list[float] = []
    low_values: list[float] = []
    diff_values: list[float] = []
    attempts = 0
    while len(high_values) < replicates:
        attempts += 1
        sampled = rng.integers(0, len(days), size=len(days))
        sampled_frame = pd.concat([clusters[days[index]] for index in sampled], ignore_index=True)
        sampled_high = sampled_frame[sampled_frame["cpd_group"] == "high"]["stacking_gain"]
        sampled_low = sampled_frame[sampled_frame["cpd_group"] == "low"]["stacking_gain"]
        if sampled_high.empty or sampled_low.empty:
            continue
        high_stat = float(sampled_high.mean())
        low_stat = float(sampled_low.mean())
        diff_stat = high_stat - low_stat
        if all(np.isfinite(value) for value in (high_stat, low_stat, diff_stat)):
            high_values.append(high_stat)
            low_values.append(low_stat)
            diff_values.append(diff_stat)

    high_lower, high_upper = _ci(high_values)
    low_lower, low_upper = _ci(low_values)
    diff_lower, diff_upper = _ci(diff_values)
    summary = {
        "criterion": "criterion_2_high_cpd_stacking_gain",
        "panel": panel,
        "cpd_median": median,
        "high_task_count": int(len(high)),
        "low_task_count": int(len(low)),
        "high_mean_gain": high_mean,
        "high_ci_lower": high_lower,
        "high_ci_upper": high_upper,
        "low_mean_gain": low_mean,
        "low_ci_lower": low_lower,
        "low_ci_upper": low_upper,
        "high_minus_low_mean": diff_mean,
        "high_minus_low_ci_lower": diff_lower,
        "high_minus_low_ci_upper": diff_upper,
        "bootstrap_replicates": int(replicates),
        "bootstrap_attempts": int(attempts),
        "bootstrap_seed": int(seed),
        "cluster_field": "test_day",
        # Stable-device is repeated as sensitivity but is not a PASS/FAIL arm.
        "passed": (
            bool(high_mean < 0 and high_upper < 0)
            if panel == "primary"
            else None
        ),
        "applicability": "primary_decision" if panel == "primary" else "sensitivity_only",
        "high_task_ids": sorted(high["task"].astype(str).tolist()),
        "low_task_ids": sorted(low["task"].astype(str).tolist()),
    }
    replicate_table = pd.DataFrame(
        {
            "criterion": "criterion_2_high_cpd_stacking_gain",
            "panel": panel,
            "replicate": np.arange(replicates, dtype=int),
            "high_mean_gain": high_values,
            "low_mean_gain": low_values,
            "high_minus_low": diff_values,
        }
    )
    return summary, replicate_table


def build_passline_tables(
    cpd_table: pd.DataFrame,
    gain_table: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build criterion summaries and all finite bootstrap replicate rows."""

    summaries: list[dict[str, Any]] = []
    replicate_frames: list[pd.DataFrame] = []
    for panel in ("primary", "stable"):
        c1, r1 = bootstrap_cpd_difference(
            cpd_table,
            panel=panel,
            replicates=replicates,
        )
        summaries.append(c1)
        replicate_frames.append(r1)
    for panel in ("primary", "stable"):
        summary, replicas = bootstrap_stacking_gain(
            cpd_table,
            gain_table,
            panel=panel,
            replicates=replicates,
        )
        summaries.append(summary)
        replicate_frames.append(replicas)
    passline = pd.DataFrame(summaries)
    replicas = pd.concat(replicate_frames, ignore_index=True)
    # Three-branch overall decision is fixed and uses primary only.
    criterion_1_pass = bool(
        next(
            item["passed"]
            for item in summaries
            if item["criterion"] == "criterion_1_cpd_ood_minus_iid"
            and item["panel"] == "primary"
        )
    )
    criterion_2_pass = bool(next(item for item in summaries if item["criterion"].startswith("criterion_2") and item["panel"] == "primary")["passed"])
    if criterion_1_pass and criterion_2_pass:
        overall = "both_pass"
    elif criterion_1_pass or criterion_2_pass:
        overall = "partial_default_not_pass"
    else:
        overall = "both_fail"
    decision = {
        "criterion_1_pass": criterion_1_pass,
        "criterion_2_pass": criterion_2_pass,
        "overall_three_branch": overall,
    }
    return passline, replicas, decision


# ---------------------------------------------------------------------------
# Audits, output packets, and deterministic staging
# ---------------------------------------------------------------------------

def function_level_leakage_audit() -> dict[str, Any]:
    """Machine-check the label boundary required by §18.2 item 3."""

    fit_signature = list(inspect.signature(fit_model_predictions).parameters)
    oof_signature = list(inspect.signature(rf_oof_reference_cm).parameters)
    forbidden_test_tokens = {"y_test", "test_label", "test_labels", "target_label"}
    fit_bad = sorted(forbidden_test_tokens.intersection(fit_signature))
    oof_bad = sorted(forbidden_test_tokens.intersection(oof_signature))
    source = inspect.getsource(fit_model_predictions)
    feature_only_boundary = (
        "test_features" in fit_signature
        and "fit boundary received test labels/categories" in source
        and 'test_features["label"]' not in source
        and "test_features['label']" not in source
    )
    return {
        "pass": not fit_bad and not oof_bad and feature_only_boundary and "model.fit" in source,
        "fit_function": "fit_model_predictions",
        "fit_parameters": fit_signature,
        "feature_only_test_boundary": feature_only_boundary,
        "oof_function": "rf_oof_reference_cm",
        "oof_parameters": oof_signature,
        "test_label_parameters_in_fit": fit_bad,
        "test_label_parameters_in_oof": oof_bad,
        "test_labels_appear_in": {
            "split_task": "label column is consumed only by the prescribed test-side sample_balanced call",
            "fit_model_predictions": "no test label/category column is accepted; only an exact feature frame",
            "run_prepared_task": "y_test is read after all models return for CM and gain reporting",
        },
        "forbidden_flow": [
            "test labels do not enter model.fit",
            "test labels do not enter OOF fold construction",
            "test labels do not select a model for fitting or deployment; best_base is an evaluation denominator only",
        ],
    }


def _json_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, list) else None
    return None


def _validate_oof_records(
    task_details: Sequence[Mapping[str, Any]],
    oof_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the persisted OOF coverage gate from serialized records."""

    detail_by_key = {
        (str(row.get("task")), str(row.get("panel"))): row for row in task_details
    }
    if len(detail_by_key) != EXPECTED_TASK_PANEL_CELLS:
        return {"pass": False, "reason": "task_detail_key_count"}
    rows_by_key: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in oof_rows:
        key = (str(row.get("task")), str(row.get("panel")), str(row.get("oof_model")))
        rows_by_key.setdefault(key, []).append(row)

    expected_models = {"rf_reference", "stacking"}
    expected_row_count = 0
    for detail_key, detail in detail_by_key.items():
        expected_fold_count = int(detail.get("oof_fold_count", -1))
        n_train = int(detail.get("train_rows_after_sampling", -1))
        expected_mode = str(detail.get("oof_mode"))
        if expected_fold_count < 2 or n_train < 2:
            return {"pass": False, "reason": "invalid_detail_fold_metadata"}
        expected_row_count += expected_fold_count * len(expected_models)
        signatures: dict[str, list[tuple[Any, ...]]] = {}
        for model_name in expected_models:
            records = sorted(
                rows_by_key.get((*detail_key, model_name), []),
                key=lambda row: int(row.get("fold", -1)),
            )
            if len(records) != expected_fold_count:
                return {"pass": False, "reason": "fold_count_mismatch"}
            if [int(row.get("fold", -1)) for row in records] != list(range(expected_fold_count)):
                return {"pass": False, "reason": "fold_number_mismatch"}
            validation_seen: list[int] = []
            model_signature: list[tuple[Any, ...]] = []
            for row in records:
                train_indices_raw = _json_list(row.get("train_indices"))
                validation_indices_raw = _json_list(row.get("validation_indices"))
                train_days = _json_list(row.get("train_days"))
                validation_days = _json_list(row.get("validation_days"))
                if train_indices_raw is None or validation_indices_raw is None:
                    return {"pass": False, "reason": "unparseable_fold_indices"}
                train_indices = tuple(int(index) for index in train_indices_raw)
                validation_indices = tuple(int(index) for index in validation_indices_raw)
                if not train_indices or not validation_indices:
                    return {"pass": False, "reason": "empty_fold_side"}
                if set(train_indices) & set(validation_indices):
                    return {"pass": False, "reason": "fold_overlap"}
                if set(train_indices) | set(validation_indices) != set(range(n_train)):
                    return {"pass": False, "reason": "fold_not_partition"}
                if str(row.get("mode")) != expected_mode:
                    return {"pass": False, "reason": "fold_mode_mismatch"}
                if expected_mode == "grouped_day":
                    if train_days is None or validation_days is None:
                        return {"pass": False, "reason": "missing_group_days"}
                    if set(map(str, train_days)) & set(map(str, validation_days)):
                        return {"pass": False, "reason": "group_day_overlap"}
                validation_seen.extend(validation_indices)
                model_signature.append(
                    (str(row.get("mode")), train_indices, validation_indices)
                )
            if sorted(validation_seen) != list(range(n_train)):
                return {"pass": False, "reason": "validation_coverage"}
            signatures[model_name] = model_signature
        if signatures["rf_reference"] != signatures["stacking"]:
            return {"pass": False, "reason": "rf_stacking_fold_mismatch"}

    if set(rows_by_key) != {
        (*detail_key, model_name)
        for detail_key in detail_by_key
        for model_name in expected_models
    }:
        return {"pass": False, "reason": "unexpected_oof_task_model_key"}
    if len(oof_rows) != expected_row_count:
        return {"pass": False, "reason": "oof_row_count"}
    return {
        "pass": True,
        "reason": "ok",
        "task_model_groups": len(rows_by_key),
        "oof_rows": len(oof_rows),
    }


def collect_static_acceptance(
    feature_audit: Mapping[str, Any],
    task_details: Sequence[Mapping[str, Any]],
    oof_rows: Sequence[Mapping[str, Any]],
    probability_audits: Sequence[Mapping[str, Any]],
    task_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Collect local D12 gates; dual-run and cpd-core gates are external."""

    def category_order_of(row: Mapping[str, Any]) -> Any:
        value = row.get("category_order")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    detail_keys = [(str(row.get("task")), str(row.get("panel"))) for row in task_details]
    support_gate = (
        len(task_details) == EXPECTED_TASK_PANEL_CELLS
        and len(set(detail_keys)) == len(detail_keys)
        and all(
            row.get("category_count") == 6
            and category_order_of(row) == list(CATEGORY_ORDER)
            and row.get("train_category_count") == 6
            and category_order_of(
                {"category_order": row.get("train_category_order")}
            ) == list(CATEGORY_ORDER)
            and row.get("test_category_count") == 6
            and category_order_of(
                {"category_order": row.get("test_category_order")}
            ) == list(CATEGORY_ORDER)
            and row.get("panel_device_count", 0) > 0
            for row in task_details
        )
    )
    panel_multiplicity_gate = all(
        {panel for task, panel in detail_keys if task == task_name} == {"primary", "stable"}
        for task_name in {task for task, _ in detail_keys}
    ) and len({task for task, _ in detail_keys}) == EXPECTED_TASK_DEFINITIONS
    oof_audit = _validate_oof_records(task_details, oof_rows)
    expected_probability_keys = {
        (task, panel, source)
        for task, panel in detail_keys
        for source in ("rf_oof", *[f"test_{model}" for model in MODEL_ORDER])
    }
    probability_keys = [
        (str(audit.get("task")), str(audit.get("panel")), str(audit.get("source")))
        for audit in probability_audits
        if isinstance(audit, Mapping)
    ]
    probability_gate = (
        len(probability_audits) == EXPECTED_PROBABILITY_AUDITS
        and len(probability_keys) == len(probability_audits)
        and len(set(probability_keys)) == len(probability_keys)
        and set(probability_keys) == expected_probability_keys
        and all(
            isinstance(audit, Mapping)
            and int(audit.get("n_rows", 0)) > 0
            and bool(audit.get("finite"))
            and bool(audit.get("bounded_0_1"))
            and bool(audit.get("row_sums_one"))
            and float(audit.get("max_row_sum_error", np.inf))
            <= PROBABILITY_ROW_SUM_ATOL
            for audit in probability_audits
        )
    )
    leakage = function_level_leakage_audit()
    local = {
        "task_definition_count": task_counts.get("task_definitions") == 74,
        "task_panel_count": task_counts.get("task_panel_cells") == 148,
        "support_and_device_panel": support_gate and panel_multiplicity_gate,
        "oof_fold_records": bool(oof_audit["pass"]),
        "probability_row_sums": probability_gate,
        "feature_finite_61": bool(feature_audit.get("finite"))
        and int(feature_audit.get("n_numeric_features", -1)) == 61,
        "side_other_zero_no_importance_comparison": bool(
            feature_audit.get("side_packet_ratio_zero")
            and feature_audit.get("other_packet_ratio_zero")
            and set(feature_audit.get("zero_audit_columns_in_model_features", []))
            == set(AUDIT_COLUMNS)
            and feature_audit.get("feature_importance_comparison_performed") is False
        ),
        "function_level_no_label_leakage": bool(leakage["pass"]),
    }
    return {
        "local_gates": local,
        "local_all_pass": bool(all(local.values())),
        "external_gates": {
            "deterministic_double_run": "required_before_canonical_output",
            "test_cpd_core": "required_before_canonical_output",
        },
        "constants": {
            "category_order": list(CATEGORY_ORDER),
            "stable_devices": list(STABLE_DEVICES),
            "max_rows": MAX_ROWS,
            "random_state": RANDOM_STATE,
            "iid_train_fraction": IID_TRAIN_FRACTION,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "leakage_audit": leakage,
        "oof_audit": oof_audit,
        "probability_audit": {
            "expected": EXPECTED_PROBABILITY_AUDITS,
            "observed": len(probability_audits),
        },
    }


def _serialize_detail_frame(details: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for detail in details:
        row = dict(detail)
        for column in (
            "train_days",
            "panel_devices",
            "device_day_windows",
            "train_category_order",
            "test_category_order",
            "train_sample_keys",
            "test_sample_keys",
        ):
            row[column] = canonical_json(row[column])
        rows.append(row)
    columns = [
        "task", "kind", "k", "train_days", "test_day", "panel",
        "panel_device_count", "panel_devices", "device_day_windows",
        "category_order", "category_count", "train_category_order",
        "train_category_count", "test_category_order", "test_category_count",
        "train_rows_before_sampling",
        "test_rows_before_sampling", "train_rows_after_sampling",
        "test_rows_after_sampling", "max_rows", "random_state", "iid_split",
        "oof_fold_count", "oof_mode", "oof_group_field", "oof_semantics",
        "train_sample_keys", "test_sample_keys",
    ]
    for row in rows:
        row["category_order"] = canonical_json(row["category_order"])
    return pd.DataFrame(rows, columns=columns)


def _serialize_nested_frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    serialized: list[dict[str, Any]] = []
    nested_columns = {
        "category_order",
        "cm_ref_rf_oof",
        "cm_tgt_rf",
        "confusion_matrix",
        "train_indices",
        "validation_indices",
        "train_days",
        "validation_days",
        "high_task_ids",
        "low_task_ids",
    }
    for item in rows:
        row = dict(item)
        for column in nested_columns:
            if column in row:
                row[column] = canonical_json(row[column])
        serialized.append(row)
    return pd.DataFrame(serialized, columns=list(columns))


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def _git_text(*args: str) -> str:
    return _git_bytes(*args).decode("utf-8").strip()


def validate_run_authorization(authorized_commit: str) -> dict[str, Any]:
    """Verify the exact pre-result authorization and committed candidate bytes."""

    if not re.fullmatch(r"[0-9a-f]{40}", authorized_commit):
        raise PermissionError("--authorized-commit must be one full lowercase 40-character hash")
    expected_block = (
        f"RUN_AUTHORIZED\ncommit: {authorized_commit}\n"
        f"canonical_python: {CANONICAL_PYTHON}"
    )
    discussion = DISCUSSION_PATH.read_text(encoding="utf-8")
    if expected_block not in discussion:
        raise PermissionError("discussion document does not contain the exact RUN_AUTHORIZED block")
    if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
        raise RuntimeError(
            f"formal D12 execution requires {CANONICAL_PYTHON}, got {sys.executable}"
        )
    bad_thread_env = {
        name: os.environ.get(name) for name in THREAD_ENV_VARS if os.environ.get(name) != "4"
    }
    if bad_thread_env:
        raise RuntimeError(f"formal D12 thread caps must all equal 4: {bad_thread_env}")
    if Path(os.environ.get("MPLCONFIGDIR", "")).resolve() != CANONICAL_MPLCONFIGDIR.resolve():
        raise RuntimeError(f"formal D12 MPLCONFIGDIR must equal {CANONICAL_MPLCONFIGDIR}")

    committed_hashes: dict[str, str] = {}
    current_hashes: dict[str, str] = {}
    for relative in (
        IMPLEMENTATION_PATH,
        SYNTHETIC_TEST_PATH,
        DETERMINISM_PROBE_PATH,
        CPD_REGRESSION_PATH,
        CPD_CORE_PATH,
        MAINLINE_MODEL_PATH,
    ):
        committed = _git_bytes("show", f"{authorized_commit}:{relative.as_posix()}")
        committed_hashes[relative.as_posix()] = hashlib.sha256(committed).hexdigest()
        current_hashes[relative.as_posix()] = sha256_file(REPO_ROOT / relative)
        if committed_hashes[relative.as_posix()] != current_hashes[relative.as_posix()]:
            raise PermissionError(
                f"working copy of {relative} differs from authorized commit {authorized_commit}"
            )
    return {
        "authorized_implementation_commit": authorized_commit,
        "execution_head_commit": _git_text("rev-parse", "HEAD"),
        "script_sha256": current_hashes,
    }


def runtime_provenance(authorization: Mapping[str, Any]) -> dict[str, Any]:
    packages = {
        distribution: importlib.metadata.version(distribution)
        for distribution in CANONICAL_PACKAGE_VERSIONS
    }
    python_version = platform.python_version()
    if python_version != CANONICAL_PYTHON_VERSION or packages != CANONICAL_PACKAGE_VERSIONS:
        raise RuntimeError(
            f"canonical environment mismatch: python={python_version}, packages={packages}"
        )
    return {
        "authorization": dict(authorization),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": python_version,
        "packages": packages,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "thread_environment": {name: os.environ[name] for name in THREAD_ENV_VARS},
        "mplconfigdir": str(CANONICAL_MPLCONFIGDIR),
    }


def normalized_shard_command(
    *, shard_index: int, shard_count: int, n_jobs: int, authorized_commit: str
) -> list[str]:
    return [
        *[f"{name}=4" for name in THREAD_ENV_VARS],
        f"MPLCONFIGDIR={CANONICAL_MPLCONFIGDIR}",
        str(CANONICAL_PYTHON),
        IMPLEMENTATION_PATH.as_posix(),
        "--feature-root", "results/unsw_features_full",
        "--mac-map", "dataset/unsw/device_mac_map.csv",
        "--output-root", f"<STAGING_ROOT>/shard-{shard_index}",
        "--n-jobs", str(n_jobs),
        "--shard-index", str(shard_index),
        "--shard-count", str(shard_count),
        "--authorized-commit", authorized_commit,
    ]


def _all_task_results(
    features: pd.DataFrame,
    tasks: Sequence[TaskDefinition],
    feature_columns: Sequence[str],
    *,
    n_jobs: int,
) -> tuple[list[TaskResult], dict[str, Any]]:
    results: list[TaskResult] = []
    for task in tasks:
        for panel in ("primary", "stable"):
            print(f"task_start task={task.name} panel={panel}", flush=True)
            prepared = split_task(features, task, panel)
            results.append(run_prepared_task(prepared, feature_columns, n_jobs=n_jobs))
            print(f"task_complete task={task.name} panel={panel}", flush=True)
    probability_audits: list[dict[str, Any]] = []
    for result in results:
        identity = {"task": result.detail["task"], "panel": result.detail["panel"]}
        probability_audits.append(
            {**identity, "source": "rf_oof", **result.probability_audit["rf_oof"]}
        )
        probability_audits.extend(
            {
                **identity,
                "source": f"test_{model_name}",
                **result.probability_audit["test_models"][model_name],
            }
            for model_name in MODEL_ORDER
        )
    return results, {
        "probability_audits": probability_audits,
        "n_task_results": len(results),
    }


def write_shard_packet(
    output_root: Path,
    results: Sequence[TaskResult],
    feature_audit: Mapping[str, Any],
    task_counts: Mapping[str, int],
    probability_audits: Sequence[Mapping[str, Any]],
    shard_manifest: Mapping[str, Any],
) -> None:
    """Write deterministic raw tables for one task shard."""

    if len(probability_audits) != len(results) * PROBABILITY_AUDITS_PER_CELL:
        raise AssertionError("shard probability audit count does not match result cells")
    detail = _serialize_detail_frame([result.detail for result in results])
    cpd_rows = [result.cpd for result in results]
    gain_rows = [result.gain for result in results]
    fold_rows = [row for result in results for row in result.oof_folds]
    cpd_frame = _serialize_nested_frame(
        cpd_rows,
        ["task", "kind", "test_day", "panel", "class_count", "category_order", "cpd_y", "cm_ref_rf_oof", "cm_tgt_rf"],
    )
    gain_frame = _serialize_nested_frame(
        gain_rows,
        [
            "task", "kind", "test_day", "panel", "class_count", "rf_macro_f1",
            "xgboost_macro_f1", "lightgbm_macro_f1", "best_base_model",
            "best_base_macro_f1", "stacking_macro_f1", "stacking_gain",
        ],
    )
    fold_frame = _serialize_nested_frame(
        fold_rows,
        [
            "task", "panel", "oof_model", "fold", "mode", "group_field",
            "train_indices", "validation_indices", "train_days", "validation_days",
            "train_time_min", "train_time_max", "validation_time_min", "validation_time_max",
        ],
    )
    stage_audit = {
        "feature_audit": dict(feature_audit),
        "task_counts": dict(task_counts),
        "result_cells": len(results),
        "probability_audit_count": len(probability_audits),
        "probability_audits": [dict(audit) for audit in probability_audits],
    }

    def write_packet(root: Path) -> None:
        write_stable_csv(detail, root / "task_detail.csv")
        write_stable_csv(cpd_frame, root / "cpd_table.csv")
        write_stable_csv(gain_frame, root / "gain_table.csv")
        write_stable_csv(fold_frame, root / "oof_folds.csv")
        write_json(root / "stage_audit.json", stage_audit)
        write_json(root / "input_manifest.json", dict(shard_manifest))

    atomic_publish_directory(output_root, write_packet)


def _read_nested_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _table_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["task"].astype(str), frame["panel"].astype(str), strict=True))


def _validated_shards(shard_roots: Sequence[Path]) -> dict[str, Any]:
    """Load exactly six complete, mutually consistent and correctly indexed shards."""

    if len(shard_roots) != 6:
        raise ValueError(f"D13 shard plan requires six shard roots, got {len(shard_roots)}")
    resolved = [root.resolve() for root in shard_roots]
    if len(set(resolved)) != 6:
        raise ValueError("six distinct shard roots are required")

    loaded: list[dict[str, Any]] = []
    for root in shard_roots:
        missing = [name for name in SHARD_PACKET_FILES if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete shard {root}: missing {missing}")
        unexpected = sorted(
            path.name for path in root.iterdir()
            if path.is_file() and path.name not in SHARD_PACKET_FILES
        )
        if unexpected:
            raise AssertionError(f"shard {root} contains unexpected packet files: {unexpected}")
        manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
        stage_audit = json.loads((root / "stage_audit.json").read_text(encoding="utf-8"))
        loaded.append(
            {
                "root": root,
                "manifest": manifest,
                "stage_audit": stage_audit,
                "detail": _read_nested_csv(root / "task_detail.csv"),
                "cpd": _read_nested_csv(root / "cpd_table.csv"),
                "gain": _read_nested_csv(root / "gain_table.csv"),
                "fold": _read_nested_csv(root / "oof_folds.csv"),
            }
        )

    indices = [int(item["manifest"].get("shard_index", -1)) for item in loaded]
    if sorted(indices) != list(range(6)) or len(set(indices)) != 6:
        raise AssertionError(f"shard indices must be exactly 0..5, got {indices}")
    loaded.sort(key=lambda item: int(item["manifest"]["shard_index"]))

    first = loaded[0]["manifest"]
    required_manifest_keys = {
        "input_manifest", "feature_audit", "run_audit", "task_counts",
        "shard_index", "shard_count", "n_jobs", "runtime_provenance",
        "normalized_command",
    }
    for item in loaded:
        manifest = item["manifest"]
        missing_keys = sorted(required_manifest_keys - set(manifest))
        if missing_keys:
            raise AssertionError(f"shard manifest missing fields: {missing_keys}")
        if int(manifest["shard_count"]) != 6:
            raise AssertionError("every shard must record shard_count=6")

    for field in ("input_manifest", "feature_audit", "task_counts", "runtime_provenance", "n_jobs"):
        expected = canonical_json(first[field])
        if any(canonical_json(item["manifest"][field]) != expected for item in loaded[1:]):
            raise AssertionError(f"{field} differs across task shards")

    input_manifest = first["input_manifest"]
    days = list(input_manifest.get("days", []))
    files = list(input_manifest.get("files", []))
    if len(days) != 20 or len(set(map(str, days))) != 20 or len(files) != 40:
        raise AssertionError("input manifest must contain 20 days and 40 day/sidecar files")
    input_names = [str(item.get("file", "")) for item in files]
    input_hashes = [str(item.get("sha256", "")) for item in files]
    if len(set(input_names)) != 40 or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in input_hashes):
        raise AssertionError("input file names/hashes are incomplete or duplicated")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(input_manifest.get("mac_map", {}).get("sha256", ""))
    ):
        raise AssertionError("input manifest is missing the MAC/category map hash")

    feature_audit = first["feature_audit"]
    task_counts = first["task_counts"]
    if int(feature_audit.get("n_numeric_features", -1)) != 61:
        raise AssertionError("shard feature audit is not the frozen 61D input")
    if canonical_json(input_manifest.get("feature_audit")) != canonical_json(feature_audit):
        raise AssertionError("input and shard-level feature audits differ")
    if task_counts.get("task_definitions") != 74 or task_counts.get("task_panel_cells") != 148:
        raise AssertionError("shard task-count manifest is not 74 x 2")

    runtime = first["runtime_provenance"]
    authorized_commit = str(
        runtime.get("authorization", {}).get("authorized_implementation_commit", "")
    )
    current_authorization = validate_run_authorization(authorized_commit)
    if canonical_json(runtime) != canonical_json(runtime_provenance(current_authorization)):
        raise AssertionError("shard runtime provenance differs from current code/environment")
    n_jobs = int(first["n_jobs"])
    if not 1 <= n_jobs <= 4:
        raise AssertionError("formal six-shard run requires 1 <= n_jobs <= 4")

    tasks = build_task_catalog(days)
    all_probability_audits: list[Mapping[str, Any]] = []
    normalized_commands: list[list[str]] = []
    for item in loaded:
        manifest = item["manifest"]
        stage_audit = item["stage_audit"]
        shard_index = int(manifest["shard_index"])
        expected_command = normalized_shard_command(
            shard_index=shard_index,
            shard_count=6,
            n_jobs=n_jobs,
            authorized_commit=authorized_commit,
        )
        if manifest["normalized_command"] != expected_command:
            raise AssertionError(f"shard {shard_index} normalized command mismatch")
        normalized_commands.append(expected_command)

        expected_tasks = tasks_for_shard(tasks, shard_index, 6)
        expected_keys = {
            (task.name, panel) for task in expected_tasks for panel in ("primary", "stable")
        }
        for table_name in ("detail", "cpd", "gain"):
            frame = item[table_name]
            if frame[["task", "panel"]].duplicated().any() or _table_keys(frame) != expected_keys:
                raise AssertionError(f"shard {shard_index} {table_name} task coverage mismatch")
        if _table_keys(item["fold"]) != expected_keys:
            raise AssertionError(f"shard {shard_index} OOF task coverage mismatch")
        if set(item["fold"]["oof_model"].astype(str)) != {"rf_reference", "stacking"}:
            raise AssertionError(f"shard {shard_index} OOF model coverage mismatch")

        expected_cells = len(expected_keys)
        audits = stage_audit.get("probability_audits", [])
        if (
            int(stage_audit.get("result_cells", -1)) != expected_cells
            or int(stage_audit.get("probability_audit_count", -1))
            != expected_cells * PROBABILITY_AUDITS_PER_CELL
            or len(audits) != expected_cells * PROBABILITY_AUDITS_PER_CELL
            or int(manifest["run_audit"].get("n_task_results", -1)) != expected_cells
            or len(manifest["run_audit"].get("probability_audits", [])) != len(audits)
        ):
            raise AssertionError(f"shard {shard_index} result/audit count mismatch")
        if canonical_json(stage_audit.get("feature_audit")) != canonical_json(feature_audit):
            raise AssertionError(f"shard {shard_index} stage feature audit mismatch")
        if canonical_json(stage_audit.get("task_counts")) != canonical_json(task_counts):
            raise AssertionError(f"shard {shard_index} stage task counts mismatch")
        if canonical_json(manifest["run_audit"].get("probability_audits")) != canonical_json(audits):
            raise AssertionError(f"shard {shard_index} duplicated probability audits differ")
        all_probability_audits.extend(audits)

    return {
        "loaded": loaded,
        "input_manifest": input_manifest,
        "feature_audit": feature_audit,
        "task_counts": task_counts,
        "runtime_provenance": runtime,
        "normalized_commands": normalized_commands,
        "probability_audits": all_probability_audits,
        "n_jobs": n_jobs,
    }


def merge_shard_packets(shard_roots: Sequence[Path], output_root: Path) -> None:
    """Merge six shards; all local gates pass before an atomic staging publish."""

    if output_root.resolve() == (REPO_ROOT / "results" / "unsw_test1").resolve():
        raise PermissionError("canonical output can only be created by --publish-canonical")
    bundle = _validated_shards(shard_roots)
    loaded = bundle["loaded"]
    details = pd.concat([item["detail"] for item in loaded], ignore_index=True).sort_values(
        ["task", "panel"], kind="stable"
    )
    cpd = pd.concat([item["cpd"] for item in loaded], ignore_index=True).sort_values(
        ["task", "panel"], kind="stable"
    )
    gain = pd.concat([item["gain"] for item in loaded], ignore_index=True).sort_values(
        ["task", "panel"], kind="stable"
    )
    folds = pd.concat([item["fold"] for item in loaded], ignore_index=True).sort_values(
        ["task", "panel", "oof_model", "fold"], kind="stable"
    )
    if len(details) != 148 or len(cpd) != 148 or len(gain) != 148:
        raise AssertionError("merged shards do not contain 74 tasks x 2 panels")
    for name, frame in (("task_detail", details), ("cpd", cpd), ("gain", gain)):
        if frame[["task", "panel"]].duplicated().any():
            raise AssertionError(f"duplicate task/panel rows in merged {name} table")
        if set(frame["panel"]) != {"primary", "stable"}:
            raise AssertionError(f"merged {name} table does not contain both panel arms")

    passline, replicas, decision = build_passline_tables(cpd, gain)
    acceptance = collect_static_acceptance(
        bundle["feature_audit"],
        details.to_dict(orient="records"),
        folds.to_dict(orient="records"),
        bundle["probability_audits"],
        bundle["task_counts"],
    )
    acceptance["external_gates"] = {
        "deterministic_double_run": "pending",
        "test_cpd_core": "pending",
    }
    acceptance["overall_local_and_external"] = None
    acceptance["decision_summary"] = decision
    if not acceptance["local_all_pass"]:
        failed = [name for name, passed in acceptance["local_gates"].items() if not passed]
        raise AssertionError(f"local hard gates failed before packet publication: {failed}")

    provenance = {
        "protocol": "D12+D13",
        "implementation": IMPLEMENTATION_PATH.as_posix(),
        "input_manifest": bundle["input_manifest"],
        "runtime": bundle["runtime_provenance"],
        "normalized_shard_commands": bundle["normalized_commands"],
        "normalized_merge_command": [
            *[f"{name}=4" for name in THREAD_ENV_VARS],
            f"MPLCONFIGDIR={CANONICAL_MPLCONFIGDIR}",
            str(CANONICAL_PYTHON), IMPLEMENTATION_PATH.as_posix(),
            "--merge-shards", *[f"<STAGING_ROOT>/shard-{index}" for index in range(6)],
            "--output-root", "<STAGING_ROOT>/packet",
        ],
        "category_order": list(CATEGORY_ORDER),
        "stable_devices": list(STABLE_DEVICES),
        "model_order": list(MODEL_ORDER),
        "model_feature_columns": bundle["feature_audit"]["numeric_features"],
        "max_rows": MAX_ROWS,
        "random_state": RANDOM_STATE,
        "stacking_execution": {
            "n_jobs": STACKING_N_JOBS,
            "threadpool_limits": STACKING_N_JOBS,
            "base_model_order": list(STACKING_BASE_MODEL_ORDER),
            "lightgbm": dict(STACKING_LIGHTGBM_DETERMINISTIC_PARAMS),
        },
        "iid_split": "per-device stable (window_start_epoch, window_id), floor(70%) train",
        "oof": "GroupKFold by training day for k>=2; five time blocks for k=1/IID",
        "cpd": "cpd_core.cpd_y(CM_ref_RF_OOF, CM_tgt_RF)",
        "bootstrap": "numpy.default_rng(42), linear quantiles, 10000 finite replicates",
        "no_absolute_staging_path": True,
        "no_wall_clock_or_completion_order": True,
    }
    note = (
        "# TEST1_RESULTS_NOTE.md\n\n"
        "This deterministic packet transcribes D12/D13 tables and execution "
        "definitions only. It contains no scientific interpretation or PASS/FAIL "
        "claim beyond the frozen mechanical pass lines.\n"
    )

    def write_packet(root: Path) -> None:
        write_stable_csv(details, root / "task_detail.csv")
        write_stable_csv(cpd, root / "cpd_table.csv")
        write_stable_csv(gain, root / "gain_table.csv")
        write_stable_csv(folds, root / "oof_folds.csv")
        write_stable_csv(passline, root / "passline.csv")
        write_stable_csv(replicas, root / "bootstrap_replicates.csv")
        write_json(root / "acceptance.json", acceptance)
        write_json(root / "provenance.json", provenance)
        (root / "TEST1_RESULTS_NOTE.md").write_text(note, encoding="utf-8")
        write_md5_manifest(root)

    atomic_publish_directory(output_root, write_packet)


def _md5_bytes(payload: bytes) -> str:
    # MD5 is the frozen byte-reproducibility format, not a security primitive.
    return hashlib.md5(payload).hexdigest()


def verify_md5_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.md5"
    if not manifest_path.is_file():
        return {"pass": False, "reason": "missing_manifest"}
    expected_lines = [
        f"{_md5_bytes((root / name).read_bytes())}  {name}"
        for name in DETERMINISTIC_PACKET_FILES
        if (root / name).is_file()
    ]
    actual_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    complete = all((root / name).is_file() for name in DETERMINISTIC_PACKET_FILES)
    return {
        "pass": bool(complete and actual_lines == expected_lines),
        "reason": "ok" if complete and actual_lines == expected_lines else "content_mismatch",
        "entries": len(actual_lines),
    }


def compare_deterministic_packets(first: Path, second: Path) -> dict[str, Any]:
    """Compare two staging packets byte-for-byte without reading scientific values."""

    names = list(COMPLETE_PACKET_FILES)
    missing_first = [name for name in names if not (first / name).is_file()]
    missing_second = [name for name in names if not (second / name).is_file()]
    unexpected_first = sorted(
        path.name for path in first.iterdir()
        if path.is_file() and path.name not in COMPLETE_PACKET_FILES
    ) if first.is_dir() else []
    unexpected_second = sorted(
        path.name for path in second.iterdir()
        if path.is_file() and path.name not in COMPLETE_PACKET_FILES
    ) if second.is_dir() else []
    differing = [
        name
        for name in names
        if (first / name).is_file()
        and (second / name).is_file()
        and (first / name).read_bytes() != (second / name).read_bytes()
    ]
    first_manifest = verify_md5_manifest(first)
    second_manifest = verify_md5_manifest(second)
    return {
        "files": names,
        "missing_first": missing_first,
        "missing_second": missing_second,
        "unexpected_first": unexpected_first,
        "unexpected_second": unexpected_second,
        "differing": differing,
        "first_manifest_valid": bool(first_manifest["pass"]),
        "second_manifest_valid": bool(second_manifest["pass"]),
        "pass": bool(
            not missing_first
            and not missing_second
            and not unexpected_first
            and not unexpected_second
            and not differing
            and first_manifest["pass"]
            and second_manifest["pass"]
        ),
    }


def write_md5_manifest(root: Path) -> None:
    """Write a stable md5 list after all deterministic files are present."""

    missing = [name for name in DETERMINISTIC_PACKET_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot write manifest; packet is incomplete: {missing}")
    lines = [f"{_md5_bytes((root / name).read_bytes())}  {name}" for name in DETERMINISTIC_PACKET_FILES]
    (root / "manifest.md5").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cpd_regression_gate() -> dict[str, Any]:
    """Run the frozen 15-test CPD regression gate in the canonical interpreter."""

    if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
        raise RuntimeError("CPD regression finalizer is not running in the canonical interpreter")
    completed = subprocess.run(
        [str(CANONICAL_PYTHON), CPD_REGRESSION_PATH.as_posix()],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout
    summary_present = "结果：15/15 全部通过".encode("utf-8") in output
    return {
        "pass": bool(completed.returncode == 0 and summary_present),
        "exit_code": int(completed.returncode),
        "expected": "15/15",
        "summary_present": summary_present,
        "stdout_sha256": hashlib.sha256(output).hexdigest(),
        "test_sha256": sha256_file(REPO_ROOT / CPD_REGRESSION_PATH),
    }


def publish_canonical_packets(first: Path, second: Path, output_root: Path) -> None:
    """Verify both runs and CPD regression, then atomically publish canonical output."""

    canonical_root = (REPO_ROOT / "results" / "unsw_test1").resolve()
    if output_root.resolve() != canonical_root:
        raise PermissionError("--publish-canonical may target only results/unsw_test1")
    if first.resolve() == second.resolve():
        raise AssertionError("double-run gate requires two distinct staging packet paths")
    if output_root.exists():
        raise FileExistsError(f"canonical output already exists: {output_root}")
    comparison = compare_deterministic_packets(first, second)
    if not comparison["pass"]:
        raise AssertionError(f"deterministic double-run gate failed: {comparison}")

    acceptance = json.loads((first / "acceptance.json").read_text(encoding="utf-8"))
    provenance = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
    if acceptance.get("local_all_pass") is not True:
        raise AssertionError("staging packet local gates are not all PASS")
    runtime = provenance.get("runtime", {})
    authorized_commit = str(
        runtime.get("authorization", {}).get("authorized_implementation_commit", "")
    )
    current_authorization = validate_run_authorization(authorized_commit)
    if canonical_json(runtime) != canonical_json(runtime_provenance(current_authorization)):
        raise AssertionError("staging packet runtime does not match current code/environment")

    cpd_gate = run_cpd_regression_gate()
    if not cpd_gate["pass"]:
        raise AssertionError(f"test_cpd_core hard gate failed: {cpd_gate}")

    comparison_evidence = {
        "pass": True,
        "normative_file_count": len(COMPLETE_PACKET_FILES),
        "packet_manifest_sha256": hashlib.sha256(
            (first / "manifest.md5").read_bytes()
        ).hexdigest(),
    }
    acceptance["external_gates"] = {
        "deterministic_double_run": comparison_evidence,
        "test_cpd_core": cpd_gate,
    }
    acceptance["overall_local_and_external"] = True
    provenance["canonical_publication"] = {
        "deterministic_double_run": comparison_evidence,
        "test_cpd_core": cpd_gate,
        "normalized_command": [
            *[f"{name}=4" for name in THREAD_ENV_VARS],
            f"MPLCONFIGDIR={CANONICAL_MPLCONFIGDIR}",
            str(CANONICAL_PYTHON), IMPLEMENTATION_PATH.as_posix(),
            "--publish-canonical", "<RUN_A_PACKET>", "<RUN_B_PACKET>",
            "--output-root", "results/unsw_test1",
        ],
        "no_absolute_staging_path": True,
    }

    def write_packet(root: Path) -> None:
        for name in DETERMINISTIC_PACKET_FILES:
            if name in {"acceptance.json", "provenance.json"}:
                continue
            shutil.copyfile(first / name, root / name)
        write_json(root / "acceptance.json", acceptance)
        write_json(root / "provenance.json", provenance)
        write_md5_manifest(root)

    atomic_publish_directory(output_root, write_packet)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D12/D13 UNSW Test 1 candidate; use only with an authorized staging root."
    )
    parser.add_argument("--feature-root", type=Path, default=Path("results/unsw_features_full"))
    parser.add_argument("--mac-map", type=Path, default=Path("dataset/unsw/device_mac_map.csv"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument(
        "--authorized-commit",
        type=str,
        default=None,
        help="Full implementation commit named by the exact RUN_AUTHORIZED block.",
    )
    parser.add_argument(
        "--merge-shards",
        nargs=6,
        type=Path,
        metavar=("SHARD0", "SHARD1", "SHARD2", "SHARD3", "SHARD4", "SHARD5"),
        help="Merge six completed shards into one locally accepted staging packet.",
    )
    parser.add_argument(
        "--compare-runs",
        nargs=2,
        type=Path,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two complete deterministic staging packets and exit.",
    )
    parser.add_argument(
        "--publish-canonical",
        nargs=2,
        type=Path,
        metavar=("RUN_A", "RUN_B"),
        help="Verify both packets plus test_cpd_core, then atomically publish canonical output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    modes = sum(option is not None for option in (args.merge_shards, args.compare_runs, args.publish_canonical))
    if modes > 1:
        raise SystemExit("--merge-shards, --compare-runs and --publish-canonical are mutually exclusive")
    if args.compare_runs:
        comparison = compare_deterministic_packets(args.compare_runs[0], args.compare_runs[1])
        print(json.dumps(comparison, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if comparison["pass"] else 1
    if args.output_root is None:
        raise SystemExit("--output-root is required for staging, shard merge or publication")

    if args.publish_canonical:
        if any(
            not packet.resolve().is_relative_to(Path("/tmp").resolve())
            for packet in args.publish_canonical
        ):
            raise SystemExit("canonical publication requires two /tmp staging packets")
        publish_canonical_packets(
            args.publish_canonical[0], args.publish_canonical[1], args.output_root
        )
        print("canonical D12/D13 packet published after all hard gates", flush=True)
        return 0

    if args.merge_shards:
        if not args.output_root.resolve().is_relative_to(Path("/tmp").resolve()):
            raise SystemExit("merged staging packet must remain under /tmp")
        if any(not root.resolve().is_relative_to(Path("/tmp").resolve()) for root in args.merge_shards):
            raise SystemExit("all shard roots must be under /tmp")
        merge_shard_packets(args.merge_shards, args.output_root)
        print(f"merged deterministic packet: output={args.output_root}", flush=True)
        return 0

    if args.authorized_commit is None:
        raise SystemExit("formal shard execution requires --authorized-commit")
    if not 1 <= args.n_jobs <= 4:
        raise SystemExit("formal six-shard execution requires 1 <= --n-jobs <= 4")
    if args.shard_count != 6 or not 0 <= args.shard_index < 6:
        raise SystemExit("formal D13 execution requires --shard-count 6 and index 0..5")
    if args.output_root.resolve() == (REPO_ROOT / "results" / "unsw_test1").resolve():
        raise SystemExit("canonical results can only be created by --publish-canonical")
    if not args.output_root.resolve().is_relative_to(Path("/tmp").resolve()):
        raise SystemExit("formal shard output must be under a /tmp staging root")
    expected_feature_root = (REPO_ROOT / "results" / "unsw_features_full").resolve()
    expected_mac_map = (REPO_ROOT / "dataset" / "unsw" / "device_mac_map.csv").resolve()
    if args.feature_root.resolve() != expected_feature_root or args.mac_map.resolve() != expected_mac_map:
        raise SystemExit("formal run must use the frozen UNSW feature root and MAC/category map")

    authorization = validate_run_authorization(args.authorized_commit)
    runtime = runtime_provenance(authorization)

    features, manifest, feature_audit = load_unsw_features(args.feature_root, args.mac_map)
    days = sorted(features["day"].astype(str).unique())
    tasks = build_task_catalog(days)
    task_counts = validate_task_catalog(tasks, days)
    selected = tasks_for_shard(tasks, args.shard_index, args.shard_count)

    results, run_audit = _all_task_results(
        features,
        selected,
        feature_audit["numeric_features"],
        n_jobs=args.n_jobs,
    )
    shard_audit = {
        "input_manifest": manifest,
        "feature_audit": feature_audit,
        "run_audit": run_audit,
        "task_counts": task_counts,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "n_jobs": args.n_jobs,
        "runtime_provenance": runtime,
        "normalized_command": normalized_shard_command(
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            n_jobs=args.n_jobs,
            authorized_commit=args.authorized_commit,
        ),
    }
    write_shard_packet(
        args.output_root,
        results,
        feature_audit,
        task_counts,
        probability_audits=run_audit["probability_audits"],
        shard_manifest=shard_audit,
    )
    print(
        f"staging shard complete: index={args.shard_index}/{args.shard_count} "
        f"task_panel_cells={len(results)} output={args.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
