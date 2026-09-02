from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_g0_strict59_ra_ecmdm as runner


def synthetic_gates(
    *,
    matched: float = 0.08,
    matched_positive: int = 6,
    excess: float = 0.06,
    excess_positive: int = 6,
    first: str = "CLASS_CONDITIONAL_BLOCK_REWEIGHTING_SUFFICIENT",
    er_i: float = 0.35,
    er_g: float = 0.65,
    er_c: float = 0.88,
    c_count: int = 6,
) -> tuple[dict, dict, dict]:
    m1_gate = {
        "engineering_status": "PASS",
        "h_m1_status": "SUPPORTED_FOR_DECOMPOSITION",
    }
    matched_env = {env: matched for env in runner.ROUNDS}
    excess_env = {env: excess for env in runner.ROUNDS}
    m1r_gate = {
        "engineering_status": "PASS",
        "m1r": {
            "raw_comparison_quantities": {
                "matched_ood_mmg_equal_weight_mean": matched,
                "matched_ood_environments_with_positive_mean_mmg": matched_positive,
                "matched_ood_minus_iid_mmg_equal_weight_mean": excess,
                "environments_with_positive_matched_ood_minus_iid_mmg": excess_positive,
            },
            "environment_matched_ood_mmg": matched_env,
            "environment_ood_minus_iid_mmg": excess_env,
        },
    }
    m2_environment = {
        env: {
            "excess_I": excess * er_i,
            "ER_I": er_i,
            "excess_G": excess * er_g,
            "ER_G": er_g,
            "excess_C": excess * er_c,
            "ER_C": er_c,
        }
        for env in runner.ROUNDS
    }
    m2_gate = {
        "engineering_status": "PASS",
        "first_sufficient_stage": first,
        "sufficiency": {
            "I": {"ER_equal": er_i, "environments_ER_ge_0.50": 0, "sufficient": False},
            "G": {"ER_equal": er_g, "environments_ER_ge_0.50": 6, "sufficient": False},
            "C": {"ER_equal": er_c, "environments_ER_ge_0.50": c_count, "sufficient": c_count >= 4 and er_c >= 0.8},
        },
        "environment_excess_ER": m2_environment,
    }
    return m1_gate, m1r_gate, m2_gate


class FrozenStateTreeTests(unittest.TestCase):
    def test_candidate_supported_only_when_all_three_gates_pass(self) -> None:
        passline, per_environment = runner.adjudicate(*synthetic_gates())
        self.assertEqual(
            passline["status"], "EC_MDM_ORACLE_CANDIDATE_SUPPORTED_STRICT59_RA"
        )
        self.assertTrue(passline["oracle_recoverability"]["passed"])
        self.assertTrue(passline["materiality"]["passed"])
        self.assertTrue(passline["c_structure"]["passed"])
        self.assertEqual(len(per_environment), 6)
        self.assertEqual(passline["epistemic_status"]["observable_estimability"], "NOT_EVALUATED")
        self.assertEqual(passline["epistemic_status"]["deployability"], "NOT_ESTABLISHED")

    def test_recoverability_failure_has_priority(self) -> None:
        gates = synthetic_gates(excess=0.004, excess_positive=3)
        passline, _ = runner.adjudicate(*gates)
        self.assertEqual(
            passline["status"],
            "EC_MDM_ORACLE_RECOVERABILITY_NOT_SUPPORTED_STRICT59_RA_STOP",
        )

    def test_low_materiality_stops_before_structure_claim(self) -> None:
        gates = synthetic_gates(excess=0.019)
        passline, _ = runner.adjudicate(*gates)
        self.assertEqual(
            passline["status"],
            "EC_MDM_ORACLE_SIGNAL_BELOW_MATERIALITY_STRICT59_RA_STOP",
        )

    def test_non_c_first_stage_fails_structure(self) -> None:
        gates = synthetic_gates(first="GLOBAL_BLOCK_REWEIGHTING_SUFFICIENT")
        gates[2]["sufficiency"]["G"]["sufficient"] = True
        passline, _ = runner.adjudicate(*gates)
        self.assertEqual(
            passline["status"],
            "EC_MDM_ORACLE_STRUCTURE_NOT_REPLICATED_STRICT59_RA_STOP",
        )


class EngineeringHelperTests(unittest.TestCase):
    def test_proxy_gate_reports_names_only(self) -> None:
        self.assertEqual(runner.check_proxy_environment({}), [])
        self.assertEqual(
            runner.check_proxy_environment({"HTTPS_PROXY": "secret-value"}),
            ["HTTPS_PROXY"],
        )

    def test_compare_directories_is_byte_exact_and_excludes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left", root / "right"
            left.mkdir()
            right.mkdir()
            (left / "value.json").write_text('{"x":1}\n', encoding="utf-8")
            (right / "value.json").write_text('{"x":1}\n', encoding="utf-8")
            (left / "provenance.json").write_text("a", encoding="utf-8")
            (right / "provenance.json").write_text("b", encoding="utf-8")
            hashes = runner.compare_directories(left, right, {"provenance.json"})
            self.assertEqual(set(hashes), {"value.json"})
            (right / "value.json").write_text('{"x":2}\n', encoding="utf-8")
            with self.assertRaises(runner.PipelineError):
                runner.compare_directories(left, right, {"provenance.json"})

    def test_switch_symlink_never_replaces_a_real_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_a, target_b = root / "a", root / "b"
            target_a.mkdir()
            target_b.mkdir()
            link = root / "active"
            runner.switch_symlink(link, target_a)
            self.assertEqual(link.resolve(), target_a.resolve())
            runner.switch_symlink(link, target_b)
            self.assertEqual(link.resolve(), target_b.resolve())
            link.unlink()
            link.mkdir()
            with self.assertRaises(runner.PipelineError):
                runner.switch_symlink(link, target_a)

    def test_stable_json_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.json"
            with self.assertRaises(ValueError):
                runner.stable_json(path, {"bad": float("nan")})

    def test_runner_does_not_reimplement_frozen_scientific_functions(self) -> None:
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        function_names = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forbidden = {
            "build_task_grid",
            "load_task_input",
            "evaluate_task",
            "adapter_objective",
            "build_gate",
            "aggregate_environment",
        }
        self.assertTrue(function_names.isdisjoint(forbidden))

    def test_protocol_freeze_record_matches_protocol(self) -> None:
        freeze = json.loads(runner.PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(freeze["protocol"]["sha256"], runner.sha256_file(runner.PROTOCOL))
        self.assertEqual(freeze["protocol"]["bytes"], runner.PROTOCOL.stat().st_size)
        repair_freeze = json.loads(runner.R2_PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(
            repair_freeze["repair_protocol"]["sha256"],
            runner.sha256_file(runner.REPAIR_PROTOCOL),
        )
        self.assertEqual(repair_freeze["parent_protocol_sha256"], freeze["protocol"]["sha256"])

    def test_relative_and_absolute_source_paths_must_resolve_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = root / "expected.pcapng"
            different = root / "different.pcapng"
            expected.touch()
            different.touch()
            relative = Path(os.path.relpath(expected, runner.REPO_ROOT))
            self.assertEqual(
                runner.resolve_recorded_source(relative),
                runner.resolve_recorded_source(expected),
            )
            self.assertNotEqual(
                runner.resolve_recorded_source(different),
                runner.resolve_recorded_source(expected),
            )


if __name__ == "__main__":
    unittest.main()
