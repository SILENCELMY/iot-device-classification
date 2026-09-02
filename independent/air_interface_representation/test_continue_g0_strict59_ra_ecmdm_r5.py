#!/usr/bin/env python3
"""Tests for the frozen R5 bounded-stability continuation."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continue_g0_strict59_ra_ecmdm_r5 as recovery


def passing_report() -> dict:
    return {
        "predictions": {
            "rf": {"different_labels": 0},
            "xgboost": {"different_labels": 0},
            "lightgbm": {"different_labels": 0},
            "stacking": {
                "global_disagreement_rate": 2e-5,
                "max_cell_disagreement_rate": 0.0011,
            },
        },
        "probabilities": {
            "rf": {"non_exact_files": 0, "max_abs_delta": 0.0},
            "xgboost": {"non_exact_files": 0, "max_abs_delta": 0.0},
            "lightgbm": {"non_exact_files": 125, "max_abs_delta": 0.029},
            "stacking": {"non_exact_files": 125, "max_abs_delta": 0.014},
        },
        "metrics": {
            "rf": {"non_exact_files": 0},
            "xgboost": {"non_exact_files": 0},
            "lightgbm": {"non_exact_files": 0},
            "stacking": {"non_exact_files": 4, "max_abs_delta": 0.0012},
        },
        "oof": {"max_abs_delta": 0.016},
    }


class TestContinuationProtocol(unittest.TestCase):
    def test_protocol_hash_matches_freeze(self) -> None:
        freeze = json.loads(recovery.CONTINUATION_PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(
            recovery.sha256_file(recovery.CONTINUATION_PROTOCOL),
            freeze["continuation_protocol"]["sha256"],
        )

    def test_parent_runner_remains_frozen(self) -> None:
        self.assertEqual(
            recovery.sha256_file(recovery.base.HERE / "run_g0_strict59_ra_ecmdm.py"),
            "ead0fe1e3e0920129fd546be54bec4d6ac0d2ede5f2cd1e071810ca558402bdf",
        )

    def test_r2_protocol_hash_matches_freeze(self) -> None:
        freeze = json.loads(
            recovery.CONTINUATION_R2_PROTOCOL_FREEZE.read_text(encoding="utf-8")
        )
        self.assertEqual(
            recovery.sha256_file(recovery.CONTINUATION_R2_PROTOCOL),
            freeze["repair_protocol"]["sha256"],
        )

    def test_continuation_never_calls_g0_training(self) -> None:
        source = inspect.getsource(recovery.continue_after_g0)
        self.assertNotIn("run_g0(", source)
        self.assertEqual(source.count("base.run_m1(canonical_g0"), 2)
        self.assertIn("os.replace(ORIGINAL_FAILED, RECOVERED_FAILURE)", source)


class TestProbabilityGate(unittest.TestCase):
    def test_valid_probability_matrix(self) -> None:
        values = np.asarray([[0.1, 0.2, 0.3, 0.15, 0.25], [1.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertLessEqual(recovery._validate_probability_matrix(values, "synthetic"), 1e-15)

    def test_rejects_nonfinite_probability(self) -> None:
        values = np.asarray([[np.nan, 0.2, 0.3, 0.2, 0.3]])
        with self.assertRaises(recovery.ContinuationError):
            recovery._validate_probability_matrix(values, "synthetic")

    def test_accepts_float32_probability_row_sum_error(self) -> None:
        values = np.asarray([[0.2, 0.2, 0.2, 0.2, 0.20000009]])
        error = recovery._validate_probability_matrix(values, "synthetic")
        self.assertGreater(error, 1e-10)
        self.assertLess(error, recovery.PROBABILITY_ROW_SUM_TOLERANCE)

    def test_rejects_probability_row_sum(self) -> None:
        values = np.asarray([[0.1, 0.2, 0.3, 0.2, 0.3]])
        with self.assertRaises(recovery.ContinuationError):
            recovery._validate_probability_matrix(values, "synthetic")


class TestBoundedStabilityGate(unittest.TestCase):
    def test_accepts_disclosed_profile(self) -> None:
        recovery.enforce_stability_limits(passing_report())

    def test_rejects_base_prediction_difference(self) -> None:
        report = passing_report()
        report["predictions"]["lightgbm"]["different_labels"] = 1
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_base_metric_difference(self) -> None:
        report = passing_report()
        report["metrics"]["xgboost"]["non_exact_files"] = 1
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_exact_model_probability_difference(self) -> None:
        report = passing_report()
        report["probabilities"]["rf"]["non_exact_files"] = 1
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_probability_bound(self) -> None:
        report = passing_report()
        report["probabilities"]["lightgbm"]["max_abs_delta"] = 0.0500001
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_stacking_global_label_bound(self) -> None:
        report = passing_report()
        report["predictions"]["stacking"]["global_disagreement_rate"] = 0.0001001
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_stacking_cell_label_bound(self) -> None:
        report = passing_report()
        report["predictions"]["stacking"]["max_cell_disagreement_rate"] = 0.002001
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_stacking_metric_bound(self) -> None:
        report = passing_report()
        report["metrics"]["stacking"]["max_abs_delta"] = 0.002001
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)

    def test_rejects_oof_bound(self) -> None:
        report = passing_report()
        report["oof"]["max_abs_delta"] = 0.050001
        with self.assertRaises(recovery.ContinuationError):
            recovery.enforce_stability_limits(report)


class TestFailurePreservation(unittest.TestCase):
    def test_atomic_rename_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "FAILED.json"
            recovered = root / "RECOVERED.json"
            original.write_bytes(b'{"status":"stop"}\n')
            expected = recovery.sha256_file(original)
            original.replace(recovered)
            self.assertFalse(original.exists())
            self.assertEqual(recovery.sha256_file(recovered), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
