#!/usr/bin/env python3
"""Tests for the R5 M2 transient optimizer retry recovery."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import resume_g0_strict59_ra_ecmdm_r5_m2 as recovery


class TestRetryProtocol(unittest.TestCase):
    def test_protocol_hash_matches_freeze(self) -> None:
        freeze = json.loads(recovery.RETRY_PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(
            recovery.continuation.sha256_file(recovery.RETRY_PROTOCOL),
            freeze["retry_protocol"]["sha256"],
        )

    def test_retry_scope_is_exact(self) -> None:
        self.assertEqual(recovery.MAX_M2_ATTEMPTS, 3)
        self.assertIsNotNone(recovery.RETRYABLE_ERROR.match("I: optimizer failed: ABNORMAL: "))
        self.assertIsNotNone(recovery.RETRYABLE_ERROR.match("G: optimizer failed: ABNORMAL: x"))
        self.assertIsNone(recovery.RETRYABLE_ERROR.match("I: optimizer failed: iteration limit"))
        self.assertIsNone(recovery.RETRYABLE_ERROR.match("other failure"))

    def test_resume_does_not_call_upstream_runners(self) -> None:
        source = inspect.getsource(recovery.resume_m2)
        self.assertNotIn("run_g0(", source)
        self.assertNotIn("run_m1(", source)
        self.assertNotIn("run_m1r(", source)


class TestM2Retry(unittest.TestCase):
    def test_two_retryable_failures_then_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m2"
            output.mkdir()
            calls = []

            def runner(_g0, _m1r, root, _logical_g0, _logical_m1r):
                calls.append(len(calls) + 1)
                root.mkdir()
                if len(calls) < 3:
                    raise recovery.base.m2.M2Error("I: optimizer failed: ABNORMAL: ")
                (root / "done.json").write_text("{}\n", encoding="utf-8")
                return {"status": "ok"}

            records = []
            gate = recovery.run_m2_repeat_with_retry(
                "A", output, Path("g0"), Path("m1r"), records, runner
            )
            self.assertEqual(gate, {"status": "ok"})
            self.assertEqual(calls, [1, 2, 3])
            self.assertEqual(
                [record["status"] for record in records],
                ["RETRYABLE_FAILURE", "RETRYABLE_FAILURE", "SUCCESS"],
            )
            self.assertTrue((output / "done.json").is_file())

    def test_nonretry_error_stops_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m2"
            output.mkdir()

            def runner(_g0, _m1r, root, _logical_g0, _logical_m1r):
                root.mkdir()
                raise recovery.base.m2.M2Error("I: optimizer failed: iteration limit")

            records = []
            with self.assertRaises(recovery.base.m2.M2Error):
                recovery.run_m2_repeat_with_retry(
                    "A", output, Path("g0"), Path("m1r"), records, runner
                )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "NONRETRY_FAILURE")

    def test_retry_refuses_nonempty_failed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m2"
            output.mkdir()

            def runner(_g0, _m1r, root, _logical_g0, _logical_m1r):
                root.mkdir()
                (root / "partial.txt").write_text("preserve", encoding="utf-8")
                raise recovery.base.m2.M2Error("C: optimizer failed: ABNORMAL: ")

            records = []
            with self.assertRaises(recovery.M2RetryRecoveryError):
                recovery.run_m2_repeat_with_retry(
                    "A", output, Path("g0"), Path("m1r"), records, runner
                )
            self.assertEqual(len(records), 1)
            self.assertTrue((output / "partial.txt").is_file())

    def test_three_retryable_failures_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m2"
            output.mkdir()

            def runner(_g0, _m1r, root, _logical_g0, _logical_m1r):
                root.mkdir()
                raise recovery.base.m2.M2Error("G: optimizer failed: ABNORMAL: ")

            records = []
            with self.assertRaises(recovery.base.m2.M2Error):
                recovery.run_m2_repeat_with_retry(
                    "B", output, Path("g0"), Path("m1r"), records, runner
                )
            self.assertEqual(len(records), 3)
            self.assertTrue(all(record["status"] == "RETRYABLE_FAILURE" for record in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
