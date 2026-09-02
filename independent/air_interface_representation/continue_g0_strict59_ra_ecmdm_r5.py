#!/usr/bin/env python3
"""Recover the completed R5 G0 repeats and continue the frozen oracle pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_g0_strict59_ra_ecmdm as base  # noqa: E402


CONTINUATION_PROTOCOL = (
    HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_R5_BOUNDED_CONTINUATION_20260903.md"
)
CONTINUATION_PROTOCOL_FREEZE = (
    HERE / "G0_STRICT59_RA_ECMDM_R5_CONTINUATION_PROTOCOL_FREEZE.json"
)
CONTINUATION_R2_PROTOCOL = (
    HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_R5_CONTINUATION_PREFLIGHT_REPAIR_R2_20260903.md"
)
CONTINUATION_R2_PROTOCOL_FREEZE = (
    HERE / "G0_STRICT59_RA_ECMDM_R5_CONTINUATION_R2_PROTOCOL_FREEZE.json"
)
CONTINUATION_IMPLEMENTATION_FREEZE = (
    HERE / "G0_STRICT59_RA_ECMDM_R5_CONTINUATION_IMPLEMENTATION_FREEZE.json"
)
CONTINUATION_TEST_FILE = HERE / "test_continue_g0_strict59_ra_ecmdm_r5.py"

G0_ROOT_B = Path("/tmp/strict59_ra_ecmdm_ge7nrsdf/g0_b")
TEMP_ROOT = G0_ROOT_B.parent
ORIGINAL_FAILED = base.AUDIT_ROOT / "FAILED.json"
RECOVERED_FAILURE = base.AUDIT_ROOT / "RECOVERED_S2_G0_VERIFY_FAILURE.json"
CONTINUATION_AUDIT = base.AUDIT_ROOT / "continuation_repair.json"
SYSTEMD_LOG = (
    REPO_ROOT
    / "results/air_interface_representation_audit/strict59_ra_ecmdm_r5_20260902.systemd.log"
)

PARENT_PROTOCOL_SHA256 = "d3c0c19821effee9ce1c9370fe6dc0046742c0f2038a887c34b8952445bfee5a"
R2_REPAIR_PROTOCOL_SHA256 = "ad0644d3f396fcbe44caac8e4c4398fb6eda23dd872c18cddcd50b1cb6047269"
R3_RECOVERY_PROTOCOL_SHA256 = "e98aef970698827e531a685f6d5f87a30ce68a1f6360138556a4621fc710ea01"
R4_REPAIR_PROTOCOL_SHA256 = "607686217f0d18a71b8df6d5ff63e03d8599f7b714df8c10035e12baa29adcbf"
R5_ISOLATION_PROTOCOL_SHA256 = "6cd5161dca8a7c8c5e0fa47a2a294b1d0b07ae6bd68e6cdca73f315620d62288"
R5_IMPLEMENTATION_FREEZE_SHA256 = "6b6a137d2675f9c931ec7b8cfcacdf7258f0e3e244711efde21be79ab8421f93"

EXPECTED_FILE_COUNT = 5835
EXPECTED_COUNTS = {
    "metrics.json": 648,
    "predictions.csv": 648,
    "pred_proba.csv": 648,
    "feature_columns.json": 648,
    "oof_meta.csv": 162,
}
PROBABILITY_MAX_ABS_DELTA = 0.05
STACKING_GLOBAL_LABEL_DISAGREEMENT_MAX = 1e-4
STACKING_CELL_LABEL_DISAGREEMENT_MAX = 0.002
STACKING_METRIC_ABS_DELTA_MAX = 0.002
PROBABILITY_ROW_SUM_TOLERANCE = 1e-6
METRIC_NAMES = ("accuracy", "precision", "recall", "macro_f1")
MODELS = ("rf", "xgboost", "lightgbm", "stacking")
BASE_MODELS = ("rf", "xgboost", "lightgbm")


class ContinuationError(RuntimeError):
    """Raised when the bounded continuation contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_counts(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    return {
        "all": len(files),
        **{
            name: sum(path.name == name for path in files)
            for name in EXPECTED_COUNTS
        },
    }


def _validate_counts(root: Path) -> dict[str, int]:
    counts = _file_counts(root)
    expected = {"all": EXPECTED_FILE_COUNT, **EXPECTED_COUNTS}
    if counts != expected:
        raise ContinuationError(f"R5 G0 staging count mismatch at {root}: {counts} != {expected}")
    cache_link = root / "raw_all/features_raw_all_w10.csv"
    if not cache_link.is_symlink() or cache_link.resolve() != (
        base.AUDIT_ROOT / base.STRICT_CACHE_NAME
    ).resolve():
        raise ContinuationError(f"R5 G0 feature-cache symlink mismatch at {root}")
    return counts


def _validate_isolation_audit() -> dict[str, Any]:
    audit = read_json(base.AUDIT_ROOT / "model_process_isolation_audit.json")
    for repeat in ("repeat_a", "repeat_b"):
        record = audit.get(repeat, {})
        required = {
            "model_calls": 648,
            "child_attempts": 648,
            "child_successes": 648,
            "first_attempt_successes": 648,
            "recovered_model_calls": 0,
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ContinuationError(f"model isolation audit incomplete for {repeat}")
        if record.get("native_signal_retries") or record.get("nonretry_failures"):
            raise ContinuationError(f"model isolation audit records a failure for {repeat}")
    return audit


def validate_static(
    expected_continuation_protocol_sha256: str,
    expected_continuation_protocol_freeze_sha256: str,
    expected_continuation_r2_protocol_sha256: str,
    expected_continuation_r2_protocol_freeze_sha256: str,
    expected_continuation_implementation_freeze_sha256: str,
) -> dict[str, Any]:
    base_static = base.validate_static(
        PARENT_PROTOCOL_SHA256,
        R2_REPAIR_PROTOCOL_SHA256,
        R3_RECOVERY_PROTOCOL_SHA256,
        R4_REPAIR_PROTOCOL_SHA256,
        R5_ISOLATION_PROTOCOL_SHA256,
        R5_IMPLEMENTATION_FREEZE_SHA256,
        require_output_absence=False,
    )
    protocol_hash = sha256_file(CONTINUATION_PROTOCOL)
    protocol_freeze_hash = sha256_file(CONTINUATION_PROTOCOL_FREEZE)
    r2_protocol_hash = sha256_file(CONTINUATION_R2_PROTOCOL)
    r2_protocol_freeze_hash = sha256_file(CONTINUATION_R2_PROTOCOL_FREEZE)
    implementation_freeze_hash = sha256_file(CONTINUATION_IMPLEMENTATION_FREEZE)
    if protocol_hash != expected_continuation_protocol_sha256:
        raise ContinuationError("CLI continuation protocol SHA-256 mismatch")
    if protocol_freeze_hash != expected_continuation_protocol_freeze_sha256:
        raise ContinuationError("CLI continuation protocol-freeze SHA-256 mismatch")
    if r2_protocol_hash != expected_continuation_r2_protocol_sha256:
        raise ContinuationError("CLI continuation R2 protocol SHA-256 mismatch")
    if r2_protocol_freeze_hash != expected_continuation_r2_protocol_freeze_sha256:
        raise ContinuationError("CLI continuation R2 protocol-freeze SHA-256 mismatch")
    if implementation_freeze_hash != expected_continuation_implementation_freeze_sha256:
        raise ContinuationError("CLI continuation implementation-freeze SHA-256 mismatch")

    protocol_freeze = read_json(CONTINUATION_PROTOCOL_FREEZE)
    if protocol_freeze["continuation_protocol"]["sha256"] != protocol_hash:
        raise ContinuationError("continuation protocol freeze record mismatch")
    if protocol_freeze["parent_implementation_freeze_sha256"] != R5_IMPLEMENTATION_FREEZE_SHA256:
        raise ContinuationError("continuation parent implementation mismatch")
    r2_protocol_freeze = read_json(CONTINUATION_R2_PROTOCOL_FREEZE)
    if r2_protocol_freeze["repair_protocol"]["sha256"] != r2_protocol_hash:
        raise ContinuationError("continuation R2 protocol freeze record mismatch")
    if r2_protocol_freeze["parent_continuation_protocol_sha256"] != protocol_hash:
        raise ContinuationError("continuation R2 parent protocol mismatch")
    if (
        r2_protocol_freeze["parent_continuation_protocol_freeze_sha256"]
        != protocol_freeze_hash
    ):
        raise ContinuationError("continuation R2 parent freeze mismatch")
    implementation_freeze = read_json(CONTINUATION_IMPLEMENTATION_FREEZE)
    if implementation_freeze["continuation_protocol_sha256"] != protocol_hash:
        raise ContinuationError("continuation implementation parent protocol mismatch")
    if implementation_freeze["continuation_r2_protocol_sha256"] != r2_protocol_hash:
        raise ContinuationError("continuation implementation R2 protocol mismatch")
    if (
        implementation_freeze["continuation_r2_protocol_freeze_sha256"]
        != r2_protocol_freeze_hash
    ):
        raise ContinuationError("continuation implementation R2 freeze mismatch")
    for relative, expected in implementation_freeze["implementation_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ContinuationError(f"continuation implementation anchor mismatch: {relative}")

    frozen = protocol_freeze["frozen_r5_staging"]
    if not ORIGINAL_FAILED.is_file() or RECOVERED_FAILURE.exists():
        raise ContinuationError("R5 stop marker is not in the pre-continuation state")
    if base.SCIENCE_ROOT_A.exists():
        raise ContinuationError("R5 science root already exists")
    for path in (TEMP_ROOT / "m1_b", TEMP_ROOT / "m1r_b", TEMP_ROOT / "m2_b"):
        if path.exists():
            raise ContinuationError(f"R5 temporary downstream root already exists: {path}")
    anchors = {
        "failed_json_sha256": sha256_file(ORIGINAL_FAILED),
        "model_process_isolation_audit_sha256": sha256_file(
            base.AUDIT_ROOT / "model_process_isolation_audit.json"
        ),
        "pre_continuation_systemd_log_sha256": sha256_file(SYSTEMD_LOG),
    }
    for key, actual in anchors.items():
        if actual != frozen[key]:
            raise ContinuationError(f"frozen R5 staging anchor mismatch: {key}")
    for label, root in (("g0_a", base.G0_ROOT_A), ("g0_b", G0_ROOT_B)):
        record = frozen[label]
        if not root.is_dir():
            raise ContinuationError(f"frozen R5 staging is absent: {root}")
        _validate_counts(root)
        for name in ("summary_metrics.csv", "summary_metrics.json"):
            key = f"{name.replace('.', '_')}_sha256"
            if sha256_file(root / name) != record[key]:
                raise ContinuationError(f"frozen R5 summary anchor mismatch: {label}/{name}")
    isolation = _validate_isolation_audit()
    return {
        "parent_static": base_static,
        "continuation_protocol_sha256": protocol_hash,
        "continuation_protocol_freeze_sha256": protocol_freeze_hash,
        "continuation_r2_protocol_sha256": r2_protocol_hash,
        "continuation_r2_protocol_freeze_sha256": r2_protocol_freeze_hash,
        "continuation_implementation_freeze_sha256": implementation_freeze_hash,
        "frozen_staging_anchors": anchors,
        "isolation_audit": isolation,
    }


def _assert_metadata_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
    relative: str,
) -> None:
    if len(left) != len(right) or list(left.columns) != list(right.columns):
        raise ContinuationError(f"shape/column mismatch: {relative}")
    if not left[list(columns)].equals(right[list(columns)]):
        raise ContinuationError(f"sample metadata mismatch: {relative}")


def _validate_probability_matrix(values: np.ndarray, relative: str) -> float:
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ContinuationError(f"invalid probability matrix: {relative}")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ContinuationError(f"probability outside [0,1]: {relative}")
    error = float(np.max(np.abs(values.sum(axis=1) - 1.0), initial=0.0))
    if error > PROBABILITY_ROW_SUM_TOLERANCE:
        raise ContinuationError(f"probability row-sum error {error}: {relative}")
    return error


def enforce_stability_limits(report: Mapping[str, Any]) -> None:
    predictions = report["predictions"]
    probabilities = report["probabilities"]
    for model in BASE_MODELS:
        if int(predictions[model]["different_labels"]) != 0:
            raise ContinuationError(f"{model} predictions are not exactly stable")
        if int(report["metrics"][model]["non_exact_files"]) != 0:
            raise ContinuationError(f"{model} metrics are not byte exact")
    for model in ("rf", "xgboost"):
        if int(probabilities[model]["non_exact_files"]) != 0:
            raise ContinuationError(f"{model} probabilities are not byte exact")
    for model in ("lightgbm", "stacking"):
        if float(probabilities[model]["max_abs_delta"]) > PROBABILITY_MAX_ABS_DELTA:
            raise ContinuationError(f"{model} probability stability bound exceeded")
    stack = predictions["stacking"]
    if float(stack["global_disagreement_rate"]) > STACKING_GLOBAL_LABEL_DISAGREEMENT_MAX:
        raise ContinuationError("stacking global label stability bound exceeded")
    if float(stack["max_cell_disagreement_rate"]) > STACKING_CELL_LABEL_DISAGREEMENT_MAX:
        raise ContinuationError("stacking per-cell label stability bound exceeded")
    if float(report["metrics"]["stacking"]["max_abs_delta"]) > STACKING_METRIC_ABS_DELTA_MAX:
        raise ContinuationError("stacking metric stability bound exceeded")
    if float(report["oof"]["max_abs_delta"]) > PROBABILITY_MAX_ABS_DELTA:
        raise ContinuationError("OOF probability stability bound exceeded")


def audit_g0_bounded_stability(root_a: Path, root_b: Path) -> dict[str, Any]:
    files_a = {
        path.relative_to(root_a).as_posix(): path
        for path in sorted(root_a.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    files_b = {
        path.relative_to(root_b).as_posix(): path
        for path in sorted(root_b.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if set(files_a) != set(files_b):
        raise ContinuationError("G0-A/G0-B relative file sets differ")

    report: dict[str, Any] = {
        "status": "G0_BOUNDED_REPLICATE_STABILITY_PASS",
        "byte_identical": False,
        "root_a": str(root_a),
        "root_b": str(root_b),
        "file_counts": {"a": _validate_counts(root_a), "b": _validate_counts(root_b)},
        "thresholds": {
            "probability_max_abs_delta": PROBABILITY_MAX_ABS_DELTA,
            "probability_row_sum_tolerance": PROBABILITY_ROW_SUM_TOLERANCE,
            "stacking_global_label_disagreement_max": STACKING_GLOBAL_LABEL_DISAGREEMENT_MAX,
            "stacking_cell_label_disagreement_max": STACKING_CELL_LABEL_DISAGREEMENT_MAX,
            "stacking_metric_abs_delta_max": STACKING_METRIC_ABS_DELTA_MAX,
        },
        "predictions": {},
        "probabilities": {},
        "metrics": {},
        "oof": {},
        "non_byte_exact_files": [],
        "model_joblib_byte_comparison": "NOT_REQUIRED_MODEL_SERIALIZATION_EXCLUDED",
    }

    strict_columns = base.strict59_ra_columns()
    for relative, path_a in files_a.items():
        if path_a.name == "feature_columns.json":
            path_b = files_b[relative]
            if sha256_file(path_a) != sha256_file(path_b):
                raise ContinuationError(f"feature-column byte mismatch: {relative}")
            if read_json(path_a) != strict_columns:
                raise ContinuationError(f"feature-column content mismatch: {relative}")

    for model in MODELS:
        prediction_files = sorted(root_a.rglob(f"all_features/{model}/predictions.csv"))
        probability_files = sorted(root_a.rglob(f"all_features/{model}/pred_proba.csv"))
        metric_files = sorted(root_a.rglob(f"all_features/{model}/metrics.json"))
        if not (len(prediction_files) == len(probability_files) == len(metric_files) == 162):
            raise ContinuationError(f"incomplete per-model file set: {model}")

        prediction_total = 0
        prediction_differences = 0
        changed_prediction_files = 0
        max_cell_disagreement = 0.0
        probability_non_exact = 0
        probability_max_delta = 0.0
        probability_different_values = 0
        probability_value_count = 0
        row_sum_error = 0.0
        metric_non_exact = 0
        metric_max_delta = 0.0

        for prediction_a, probability_a, metric_a in zip(
            prediction_files, probability_files, metric_files, strict=True
        ):
            prediction_relative = prediction_a.relative_to(root_a).as_posix()
            probability_relative = probability_a.relative_to(root_a).as_posix()
            metric_relative = metric_a.relative_to(root_a).as_posix()
            prediction_b = root_b / prediction_relative
            probability_b = root_b / probability_relative
            metric_b = root_b / metric_relative

            pred_a = pd.read_csv(prediction_a, encoding="utf-8-sig")
            pred_b = pd.read_csv(prediction_b, encoding="utf-8-sig")
            pred_meta = ("round", "source_file", "window_id", "true_label")
            _assert_metadata_equal(pred_a, pred_b, pred_meta, prediction_relative)
            labels_a = pred_a["predicted_label"].astype(str).to_numpy()
            labels_b = pred_b["predicted_label"].astype(str).to_numpy()
            different = int(np.count_nonzero(labels_a != labels_b))
            prediction_total += len(pred_a)
            prediction_differences += different
            if different:
                changed_prediction_files += 1
                max_cell_disagreement = max(max_cell_disagreement, different / len(pred_a))
                report["non_byte_exact_files"].append(prediction_relative)
            elif sha256_file(prediction_a) != sha256_file(prediction_b):
                raise ContinuationError(f"metadata-only prediction byte mismatch: {prediction_relative}")

            proba_a = pd.read_csv(probability_a, encoding="utf-8-sig")
            proba_b = pd.read_csv(probability_b, encoding="utf-8-sig")
            proba_meta = ("source_file", "round", "window_id", "window_start", "true_label")
            _assert_metadata_equal(proba_a, proba_b, proba_meta, probability_relative)
            if not pred_a[list(pred_meta)].equals(
                proba_a[["round", "source_file", "window_id", "true_label"]]
            ):
                raise ContinuationError(f"prediction/probability alignment mismatch: {prediction_relative}")
            probability_columns = [column for column in proba_a if column.startswith("proba_")]
            classes = np.asarray([column.removeprefix("proba_") for column in probability_columns])
            values_a = proba_a[probability_columns].to_numpy(dtype=float)
            values_b = proba_b[probability_columns].to_numpy(dtype=float)
            row_sum_error = max(
                row_sum_error,
                _validate_probability_matrix(values_a, probability_relative),
                _validate_probability_matrix(values_b, probability_relative),
            )
            if not np.array_equal(classes[np.argmax(values_a, axis=1)], labels_a):
                raise ContinuationError(f"G0-A probability argmax mismatch: {probability_relative}")
            if not np.array_equal(classes[np.argmax(values_b, axis=1)], labels_b):
                raise ContinuationError(f"G0-B probability argmax mismatch: {probability_relative}")
            delta = np.abs(values_a - values_b)
            probability_value_count += delta.size
            probability_different_values += int(np.count_nonzero(delta))
            probability_max_delta = max(
                probability_max_delta, float(np.max(delta, initial=0.0))
            )
            if sha256_file(probability_a) != sha256_file(probability_b):
                probability_non_exact += 1
                report["non_byte_exact_files"].append(probability_relative)

            metrics_a = read_json(metric_a)
            metrics_b = read_json(metric_b)
            if sha256_file(metric_a) != sha256_file(metric_b):
                metric_non_exact += 1
                report["non_byte_exact_files"].append(metric_relative)
            metric_max_delta = max(
                metric_max_delta,
                *(abs(float(metrics_a[name]) - float(metrics_b[name])) for name in METRIC_NAMES),
            )

        report["predictions"][model] = {
            "files": len(prediction_files),
            "total_rows": prediction_total,
            "different_labels": prediction_differences,
            "changed_files": changed_prediction_files,
            "global_disagreement_rate": prediction_differences / prediction_total,
            "max_cell_disagreement_rate": max_cell_disagreement,
        }
        report["probabilities"][model] = {
            "files": len(probability_files),
            "non_exact_files": probability_non_exact,
            "different_values": probability_different_values,
            "total_values": probability_value_count,
            "max_abs_delta": probability_max_delta,
            "max_row_sum_error": row_sum_error,
        }
        report["metrics"][model] = {
            "files": len(metric_files),
            "non_exact_files": metric_non_exact,
            "max_abs_delta": metric_max_delta,
        }

    oof_files = sorted(root_a.rglob("all_features/stacking/oof_meta.csv"))
    if len(oof_files) != 162:
        raise ContinuationError("incomplete OOF file set")
    oof_non_exact = 0
    oof_max_delta = 0.0
    oof_different_values = 0
    oof_value_count = 0
    oof_row_sum_error = 0.0
    for path_a in oof_files:
        relative = path_a.relative_to(root_a).as_posix()
        path_b = root_b / relative
        frame_a = pd.read_csv(path_a, encoding="utf-8-sig")
        frame_b = pd.read_csv(path_b, encoding="utf-8-sig")
        _assert_metadata_equal(
            frame_a, frame_b, ("window_start", "window_id", "round", "true_label"), relative
        )
        all_columns: list[str] = []
        for model in BASE_MODELS:
            columns = [column for column in frame_a if column.startswith(f"oof_{model}_")]
            if len(columns) != 5:
                raise ContinuationError(f"invalid OOF probability block: {relative}/{model}")
            all_columns.extend(columns)
            oof_row_sum_error = max(
                oof_row_sum_error,
                _validate_probability_matrix(frame_a[columns].to_numpy(dtype=float), relative),
                _validate_probability_matrix(frame_b[columns].to_numpy(dtype=float), relative),
            )
        values_a = frame_a[all_columns].to_numpy(dtype=float)
        values_b = frame_b[all_columns].to_numpy(dtype=float)
        delta = np.abs(values_a - values_b)
        oof_value_count += delta.size
        oof_different_values += int(np.count_nonzero(delta))
        oof_max_delta = max(oof_max_delta, float(np.max(delta, initial=0.0)))
        if sha256_file(path_a) != sha256_file(path_b):
            oof_non_exact += 1
            report["non_byte_exact_files"].append(relative)
    report["oof"] = {
        "files": len(oof_files),
        "non_exact_files": oof_non_exact,
        "different_values": oof_different_values,
        "total_values": oof_value_count,
        "max_abs_delta": oof_max_delta,
        "max_row_sum_error": oof_row_sum_error,
    }

    topology_a = root_a / "env_topology_matrix_rf.csv"
    topology_b = root_b / "env_topology_matrix_rf.csv"
    if sha256_file(topology_a) != sha256_file(topology_b):
        raise ContinuationError("RF topology matrix is not byte exact")
    summary_a = pd.read_csv(root_a / "summary_metrics.csv", encoding="utf-8-sig")
    summary_b = pd.read_csv(root_b / "summary_metrics.csv", encoding="utf-8-sig")
    summary_keys = ["filter_mode", "task", "feature_set", "model"]
    if len(summary_a) != 648 or not summary_a[summary_keys].equals(summary_b[summary_keys]):
        raise ContinuationError("G0 summary row/key mismatch")
    summary_report: dict[str, Any] = {}
    for model in MODELS:
        mask = summary_a["model"] == model
        deltas = np.column_stack(
            [
                np.abs(
                    summary_a.loc[mask, metric].to_numpy(dtype=float)
                    - summary_b.loc[mask, metric].to_numpy(dtype=float)
                )
                for metric in METRIC_NAMES
            ]
        )
        summary_report[model] = {
            "rows": int(mask.sum()),
            "changed_rows": int(np.count_nonzero(np.any(deltas != 0.0, axis=1))),
            "max_abs_delta": float(np.max(deltas, initial=0.0)),
        }
    report["summary"] = summary_report
    for name in ("summary_metrics.csv", "summary_metrics.json"):
        if sha256_file(root_a / name) != sha256_file(root_b / name):
            report["non_byte_exact_files"].append(name)
    enforce_stability_limits(report)
    report["non_byte_exact_files"] = sorted(set(report["non_byte_exact_files"]))
    report["non_byte_exact_file_count"] = len(report["non_byte_exact_files"])
    report["canonical_science_input"] = str(root_a / "raw_all")
    report["replicate_role"] = "BOUNDED_STABILITY_AUDIT_ONLY"
    return report


def _build_acceptance(passline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": passline["status"],
        "engineering": {
            "static_anchors": True,
            "full94_reproduction": True,
            "materialization": True,
            "g0_bounded_replicate_stability": True,
            "g0_byte_identical": False,
            "g0_a_model_cells": 648,
            "g0_b_model_cells": 648,
            "canonical_g0_input": "G0-A",
            "m1_double_run_from_canonical_input": True,
            "m1r_double_run_from_canonical_input": True,
            "m2_double_run_from_canonical_input": True,
            "recovered_in_place_after_s2_stop": True,
            "m1_runs": 162,
            "m1r_runs": 156,
            "m2_runs": 156,
        },
        "scientific_gates": {
            "oracle_recoverability": passline["oracle_recoverability"]["passed"],
            "materiality": passline["materiality"]["passed"],
            "c_structure": passline["c_structure"]["passed"],
        },
        "observable_estimability": "NOT_EVALUATED",
        "deployability": "NOT_ESTABLISHED",
        "cpd_relation": "CANDIDATE_SUPERORDINATE_CONSTRUCT_HYPOTHESIS_ONLY",
    }


def continue_after_g0(
    expected_continuation_protocol_sha256: str,
    expected_continuation_protocol_freeze_sha256: str,
    expected_continuation_r2_protocol_sha256: str,
    expected_continuation_r2_protocol_freeze_sha256: str,
    expected_continuation_implementation_freeze_sha256: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    started_wall = time.time()
    started_utc = datetime.now(timezone.utc)
    current_phase = "R5C_STATIC_AND_G0_BOUNDED_AUDIT"
    static = validate_static(
        expected_continuation_protocol_sha256,
        expected_continuation_protocol_freeze_sha256,
        expected_continuation_r2_protocol_sha256,
        expected_continuation_r2_protocol_freeze_sha256,
        expected_continuation_implementation_freeze_sha256,
    )
    g0_verification = audit_g0_bounded_stability(base.G0_ROOT_A, G0_ROOT_B)
    original_failure_sha256 = sha256_file(ORIGINAL_FAILED)
    os.replace(ORIGINAL_FAILED, RECOVERED_FAILURE)
    try:
        base.stable_json(base.AUDIT_ROOT / "g0_double_run_verification.json", g0_verification)
        base.stable_json(
            CONTINUATION_AUDIT,
            {
                "status": "R5_IN_PLACE_CONTINUATION_STARTED",
                "original_failure_preserved": str(RECOVERED_FAILURE),
                "original_failure_sha256": original_failure_sha256,
                "g0_stability_status": g0_verification["status"],
                "canonical_science_input": str(base.G0_ROOT_A / "raw_all"),
                "replicate_g0_role": "BOUNDED_STABILITY_AUDIT_ONLY",
            },
        )

        current_phase = "R5C_S3_M1"
        base.SCIENCE_ROOT_A.mkdir(parents=True, exist_ok=False)
        links = base.AUDIT_ROOT / "_active_inputs"
        logical_g0 = links / "g0_raw_all"
        logical_m1 = links / "m1"
        logical_m1r = links / "m1r"
        canonical_g0 = base.G0_ROOT_A / "raw_all"
        m1_a_root, m1_b_root = base.SCIENCE_ROOT_A / "m1", TEMP_ROOT / "m1_b"
        m1_gate_a = base.run_m1(canonical_g0, m1_a_root, logical_g0)
        m1_gate_b = base.run_m1(canonical_g0, m1_b_root, logical_g0)
        m1_hashes = base.compare_directories(m1_a_root, m1_b_root, {"provenance.json"})
        if m1_gate_a != m1_gate_b:
            raise ContinuationError("M1 gates differ for repeated canonical input")

        current_phase = "R5C_S4_M1R"
        m1r_a_root, m1r_b_root = base.SCIENCE_ROOT_A / "m1r", TEMP_ROOT / "m1r_b"
        base.run_m1r(canonical_g0, m1_a_root, m1r_a_root, logical_g0, logical_m1)
        base.run_m1r(canonical_g0, m1_b_root, m1r_b_root, logical_g0, logical_m1)
        m1r_gate_final = base.m1r.verify_double_run(m1r_a_root, m1r_b_root)

        current_phase = "R5C_S5_M2"
        m2_a_root, m2_b_root = base.SCIENCE_ROOT_A / "m2", TEMP_ROOT / "m2_b"
        base.run_m2(canonical_g0, m1r_a_root, m2_a_root, logical_g0, logical_m1r)
        base.run_m2(canonical_g0, m1r_b_root, m2_b_root, logical_g0, logical_m1r)
        m2_gate_final = base.m2.verify(m2_a_root, m2_b_root)
        base.switch_symlink(logical_g0, canonical_g0)
        base.switch_symlink(logical_m1, m1_a_root)
        base.switch_symlink(logical_m1r, m1r_a_root)

        current_phase = "R5C_S5_PIPELINE_VERIFY"
        m1r_hashes = base.compare_directories(m1r_a_root, m1r_b_root, {"provenance.json"})
        m2_hashes = base.compare_directories(m2_a_root, m2_b_root, {"provenance.json"})
        pipeline_verification = {
            "consistent": True,
            "algorithm": "sha256",
            "canonical_g0_input_for_both_repeats": str(canonical_g0),
            "g0_replicate_verification": "G0_BOUNDED_REPLICATE_STABILITY_PASS",
            "g0_byte_identical": False,
            "m1_deterministic_file_count": len(m1_hashes),
            "m1r_deterministic_file_count": len(m1r_hashes),
            "m2_deterministic_file_count": len(m2_hashes),
            "m1_files": m1_hashes,
            "m1r_files": m1r_hashes,
            "m2_files": m2_hashes,
        }
        base.stable_json(
            base.AUDIT_ROOT / "pipeline_double_run_verification.json", pipeline_verification
        )

        current_phase = "R5C_S6_ADJUDICATION"
        passline, per_environment = base.adjudicate(m1_gate_a, m1r_gate_final, m2_gate_final)
        base.stable_json(base.AUDIT_ROOT / "oracle_passline.json", passline)
        base.stable_csv(per_environment, base.AUDIT_ROOT / "per_environment.csv")
        acceptance = _build_acceptance(passline)
        base.stable_json(base.AUDIT_ROOT / "acceptance.json", acceptance)
        full94 = read_json(base.AUDIT_ROOT / "full94_reproduction_gate.json")
        extraction = read_json(base.AUDIT_ROOT / "extraction_audit.json")
        verdict = base.build_verdict(passline, full94, extraction).replace(
            "- G0：162 次运行、648 模型单元；G0/M1/M1-R/M2 双跑判定性产物一致。",
            "- G0：A/B 各 162 次运行、648 模型单元；G0 通过有界重复稳定性门但不逐字节一致；"
            "M1/M1-R/M2 在 canonical G0-A 输入上双跑逐字节一致。",
        )
        verdict += "\n".join(
            [
                "",
                "## R5 原地恢复披露",
                "",
                "R5 曾在 S2 的 G0 逐字节哈希门停止。本次按续跑前冻结的有界稳定性协议原地恢复；",
                "原停止记录保存在 `RECOVERED_S2_G0_VERIFY_FAILURE.json`。G0-A/G0-B 不是逐字节一致，",
                "G0-B 仅用于独立拟合稳定性审计；正式科学计算及其双跑均使用 canonical G0-A。",
                "",
            ]
        )
        (base.AUDIT_ROOT / "VERDICT.md").write_text(verdict, encoding="utf-8", newline="\n")

        versions: dict[str, str | None] = {}
        for package in (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "joblib",
            "xgboost",
            "lightgbm",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        provenance = {
            "argv": list(argv),
            "original_r5_started_utc": "2026-09-02T15:42:05+00:00",
            "continuation_started_utc": started_utc.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "continuation_elapsed_seconds": time.time() - started_wall,
            "interpreter": str(Path(sys.executable).resolve()),
            "python": platform.python_version(),
            "package_versions": versions,
            "git_head": base._git_value(["rev-parse", "HEAD"]),
            "git_status_porcelain": base._git_value(["status", "--porcelain"]),
            "parent_protocol_sha256": sha256_file(base.PROTOCOL),
            "parent_r5_isolation_protocol_sha256": sha256_file(base.ISOLATION_PROTOCOL),
            "parent_r5_implementation_freeze_sha256": sha256_file(base.IMPLEMENTATION_FREEZE),
            "continuation_protocol_sha256": sha256_file(CONTINUATION_PROTOCOL),
            "continuation_protocol_freeze_sha256": sha256_file(CONTINUATION_PROTOCOL_FREEZE),
            "continuation_r2_protocol_sha256": sha256_file(CONTINUATION_R2_PROTOCOL),
            "continuation_r2_protocol_freeze_sha256": sha256_file(
                CONTINUATION_R2_PROTOCOL_FREEZE
            ),
            "continuation_implementation_freeze_sha256": sha256_file(
                CONTINUATION_IMPLEMENTATION_FREEZE
            ),
            "continuation_runner_sha256": sha256_file(Path(__file__).resolve()),
            "continuation_tests_sha256": sha256_file(CONTINUATION_TEST_FILE),
            "original_failure_sha256": original_failure_sha256,
            "original_failure_preserved": str(RECOVERED_FAILURE),
            "static_audit": static,
            "g0_bounded_stability": g0_verification,
            "temporary_root_preserved": str(TEMP_ROOT),
            "network_access_attempted": False,
            "proxy_variables_empty": len(base.check_proxy_environment()) == 0,
            "resource_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }
        base.stable_json(base.AUDIT_ROOT / "provenance.json", provenance)
        base.stable_json(
            CONTINUATION_AUDIT,
            {
                "status": "R5_IN_PLACE_CONTINUATION_COMPLETED",
                "original_failure_preserved": str(RECOVERED_FAILURE),
                "original_failure_sha256": original_failure_sha256,
                "g0_stability_status": g0_verification["status"],
                "g0_byte_identical": False,
                "canonical_science_input": str(canonical_g0),
                "final_status": acceptance["status"],
            },
        )
        base.write_manifest(base.AUDIT_ROOT)
        print(f"completed continuation: {acceptance['status']} -> {base.AUDIT_ROOT}", flush=True)
        return acceptance
    except BaseException as error:
        failure = {
            "status": "INVALID_RUN_STOP",
            "phase": current_phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "original_failure_preserved": str(RECOVERED_FAILURE),
            "temporary_root_preserved": str(TEMP_ROOT),
            "formal_staging_preserved": {
                "audit_root": str(base.AUDIT_ROOT),
                "g0_root": str(base.G0_ROOT_A),
                "science_root": str(base.SCIENCE_ROOT_A),
            },
        }
        base.stable_json(ORIGINAL_FAILED, failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-continuation-protocol-sha256", required=True)
    parser.add_argument("--expected-continuation-protocol-freeze-sha256", required=True)
    parser.add_argument("--expected-continuation-r2-protocol-sha256", required=True)
    parser.add_argument("--expected-continuation-r2-protocol-freeze-sha256", required=True)
    parser.add_argument("--expected-continuation-implementation-freeze-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-no-fit", action="store_true")
    mode.add_argument("--continue-after-g0", action="store_true")
    args = parser.parse_args(argv)
    effective_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    try:
        if args.preflight_no_fit:
            static = validate_static(
                args.expected_continuation_protocol_sha256,
                args.expected_continuation_protocol_freeze_sha256,
                args.expected_continuation_r2_protocol_sha256,
                args.expected_continuation_r2_protocol_freeze_sha256,
                args.expected_continuation_implementation_freeze_sha256,
            )
            stability = audit_g0_bounded_stability(base.G0_ROOT_A, G0_ROOT_B)
            print(
                json.dumps(
                    {
                        "status": "R5_CONTINUATION_PREFLIGHT_PASS",
                        "static": static,
                        "g0_stability": stability,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        continue_after_g0(
            args.expected_continuation_protocol_sha256,
            args.expected_continuation_protocol_freeze_sha256,
            args.expected_continuation_r2_protocol_sha256,
            args.expected_continuation_r2_protocol_freeze_sha256,
            args.expected_continuation_implementation_freeze_sha256,
            effective_argv,
        )
        return 0
    except BaseException as error:
        print(f"R5 CONTINUATION STOPPED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
