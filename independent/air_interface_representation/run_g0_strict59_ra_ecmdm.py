#!/usr/bin/env python3
"""Frozen strict59_ra G0 -> M1/M1-R/M2 oracle recalibration pipeline."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Resource variables are fixed before importing numerical libraries.  Proxy
# variables are only inspected; this runner never sets or uses them.
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
META_DIR = REPO_ROOT / "independent/meta_mismatch"
CORE_DIR = REPO_ROOT / "code/scripts/core"
for _import_dir in (HERE, META_DIR, CORE_DIR):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

import run_strict59_ra_direction_repair as direction_repair  # noqa: E402
import environment_grid_experiment as g0  # noqa: E402
import m1_meta_counterfactual as m1  # noqa: E402
import m1r_matched_control as m1r  # noqa: E402
import m2_meta_mechanism as m2  # noqa: E402


PROTOCOL = HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_RECALIBRATION_20260902.md"
PROTOCOL_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_PROTOCOL_FREEZE.json"
REPAIR_PROTOCOL = HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_PATH_REPAIR_R2_20260902.md"
R2_PROTOCOL_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R2_PROTOCOL_FREEZE.json"
R2_IMPLEMENTATION_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R2_IMPLEMENTATION_FREEZE.json"
RECOVERY_PROTOCOL = HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_TERMINAL_RECOVERY_R3_20260902.md"
R3_PROTOCOL_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R3_PROTOCOL_FREEZE.json"
R3_IMPLEMENTATION_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R3_IMPLEMENTATION_FREEZE.json"
TMP_REPAIR_PROTOCOL = HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_TMP_DISPLAY_REPAIR_R4_20260902.md"
R4_PROTOCOL_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R4_PROTOCOL_FREEZE.json"
IMPLEMENTATION_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R4_IMPLEMENTATION_FREEZE.json"
TEST_FILE = HERE / "test_g0_strict59_ra_ecmdm.py"

AUDIT_ROOT = (
    REPO_ROOT
    / "results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r4"
)
G0_ROOT_A = REPO_ROOT / "results/g0_environment_grid_strict59_ra_r4"
SCIENCE_ROOT_A = REPO_ROOT / "results/meta_mismatch_exploratory/strict59_ra_ecmdm_r4"
SOURCE_CACHE = REPO_ROOT / "results/robust_v2/raw_all/features_raw_all_w10.csv"
ACCEPTED_RA_ROOT = (
    REPO_ROOT
    / "results/air_interface_representation_audit/strict59_ra_run_20260902_r2"
)
FULL94_M1_ROOT = REPO_ROOT / "results/meta_mismatch_exploratory/m1"
FULL94_M1R_ROOT = REPO_ROOT / "results/meta_mismatch_exploratory/m1r_matched_control"

STRICT_CACHE_NAME = "strict59_ra_features_raw_all_w10.csv"
META_COLUMNS = [
    "label",
    "round",
    "traffic",
    "filter_mode",
    "source_file",
    "window_id",
    "window_start",
    "window_end",
]
KEY_COLUMNS = ["label", "round", "source_file", "window_id"]
ROUNDS = ("R2", "R3", "R4", "R5", "R6", "R7")
ROUND_COUNTS = {"R2": 1816, "R3": 1837, "R4": 1853, "R5": 1816, "R6": 1988, "R7": 1993}
EXPECTED_ROWS = 11_303
EXPECTED_R2_R4_ROWS = 5_506
EXPECTED_G0_RUNS = 162
EXPECTED_G0_MODEL_CELLS = 648
FULL94_TOLERANCE = 1e-12

PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

ANCHOR_SHA256 = {
    "results/robust_v2/raw_all/features_raw_all_w10.csv":
        "9bf191f0fb74d66463c829bbc39de73d752265163bf6dc1729a668e3d1c6ca41",
    "code/scripts/core/environment_grid_experiment.py":
        "eeb7cda6c71b14f5ee6b725faef5c0bc056a703f956f9acb93b09243d87499e9",
    "code/scripts/core/robust_iot_research.py":
        "1d29434570d35422ce2b7cd9d485259f5520761aa11999342ca9b0cc2f36a5e3",
    "code/configs/research_experiments.json":
        "78425ac8be43065b642345608bedd82522e46f60df42cda0da3dcebe06c4e28c",
    "independent/air_interface_representation/run_strict59_ra_direction_repair.py":
        "b32cb504055360a5f381fd39aa1ab650532098f3cc443aa0b23cd867a6d2289d",
    "independent/meta_mismatch/PROTOCOL_M1.md":
        "b055dca8c430ef2e42d817d5e16f9bedf963cdd4104a8cb0e34618c10cb386c8",
    "independent/meta_mismatch/PROTOCOL_M1R.md":
        "0c6195bc17ea972f7cc18ad343cf7939e1b35814f4bfdd4ac0771409dc0ee2b6",
    "independent/meta_mismatch/PROTOCOL_M2.md":
        "4b23bfa1509ef1fa26b6afe945f4191a4099a6bc7c0446ffbc8bd2cdee53cfb5",
    "independent/meta_mismatch/LUNA_HANDOFF_M2.md":
        "4aeacdc0680d466586db6deba345718f44e97bde92a215b84ccd162dd1798069",
    "independent/meta_mismatch/m1_meta_counterfactual.py":
        "96f5419608b66e77617420f381e149da4738b8780205a6cc994e36fcab330dea",
    "independent/meta_mismatch/m1r_matched_control.py":
        "07110482c458a256c4582762d8217a52a835b0bb239ab88a913291ce4ac17884",
    "independent/meta_mismatch/m2_meta_mechanism.py":
        "2ca7e4a7e9a516910b038e8d16529611f75e443182385414c8794205609d2e51",
    "results/air_interface_representation_audit/strict59_ra_run_20260902_r2/VERDICT.md":
        "279eefa6358a7cf132fb623a82283f3cd0b7e7b8a49f3e1c4fbe1a638865b53a",
    "results/air_interface_representation_audit/strict59_ra_run_20260902_r2/acceptance.json":
        "67a510f732a726752e5840c2d865fd176dd0a715dc4907e62b9c52777758941e",
    "results/air_interface_representation_audit/strict59_ra_run_20260902_r2/feature_arms.json":
        "5a44b90e9a7cf30588c348addb10ad3d0292abd3adb9ef5c7227ee9247910005",
    "results/air_interface_representation_audit/strict59_ra_run_20260902_r2/corrected_direction_features.csv":
        "732751066390e615b3f3dbf20d3be808e1f90982ae73773ca3186e94cba7f552",
    "results/air_interface_representation_audit/strict59_ra_run_20260902_r2/split_audit.csv":
        "6b3cd823cb74d2e847dc629eefb31cecc293509ecaab16e9f20b7ccd9195dafb",
    "results/air_interface_representation_audit/strict59_ra_run_20260902_r2/manifest.json":
        "f57975a459eabb3ccd1c1a6a12db5042d3ecfb3c4f49e095f9c4bdd5b0c26bd0",
}

FULL94_REFERENCE_SHA256 = {
    "results/meta_mismatch_exploratory/m1/m1_gate.json":
        "41ace5c45382a6d7219f2969018d1f01e9ab3142a25d6b231bbbeb85c8b567ef",
    "results/meta_mismatch_exploratory/m1/input_manifest.json":
        "8ca6e95d79c73a17442094bab8e2b01f8d4a0928321a85ee03f79835d7407a36",
    "results/meta_mismatch_exploratory/m1r_matched_control/m1r_gate.json":
        "a33e89029d33dcedf9c31591f73196729bfaa8d715ced3454578442c74fedcd5",
    "results/meta_mismatch_exploratory/m1r_matched_control/input_manifest.json":
        "b86339909292298f8462bc09cd47afc29c31874a1a41224714df2c11e7fbe9cd",
    "results/meta_mismatch_exploratory/m1r_matched_control/double_run_verification.json":
        "625341027624f82ac22b3786b287c2ad927265af490f5c12a03e7abedaa839e7",
}

PCAP_SHA256 = {
    "dataset/camera/round2_normal/camera_r2.pcapng": "9e8dab77b439fa97865a2705a3d91b99c3d567fe0282f8e85b7c081b3080cad4",
    "dataset/light_T1/round2_normal/light_T1_r2.pcapng": "9c62775af38207f128941da2de98ae7b91ef12b798ed9f68b79cfecf49275ce6",
    "dataset/light_xm/round2_normal/light_xm_r2.pcapng": "88b46b2b2c68b6725e70a4820173f580bdaf96042253e34e90b62c8af822d5c2",
    "dataset/sensor/round2_normal/sensor_r2.pcapng": "068d027f5d1a06324f84d38aadddbdde7bba70453b12a4654e9c58a7a2668e57",
    "dataset/socket/round2_normal/socket_r2.pcapng": "133fb099ccb33a3b1bc4e817c3ebe1c408c593681248584c00ebab5690769760",
    "dataset/camera/round3_normal/camera_r3.pcapng": "b86bc442465ab53ae1d0850f0096f6d5d3e4a92f2f7067a968b82669e77c3202",
    "dataset/light_T1/round3_normal/light_T1_r3.pcapng": "ebfa62cbc5d25855a4b47c80bd48644665a5a604e1f78ad658e35c1eea6cfa9c",
    "dataset/light_xm/round3_normal/light_xm_r3.pcapng": "410d415615646d322a78f72c0112c8d1db1ac99949831fe9eb5b38fbac95026d",
    "dataset/sensor/round3_normal/sensor_r3.pcapng": "4665629a665ce062ed4682d476b72e8e896ec9e1e7bc19287579d37e759deedc",
    "dataset/socket/round3_normal/socket_r3.pcapng": "15ca818ea3a927f46fb057667c8fcce9a76fb1709985af00dbdba84719859990",
    "dataset/camera/round4_normal/camera_r4.pcapng": "ad68d87c6c705b38bdac901e9fb5038824a84f9e2ea43a87866486c04af52974",
    "dataset/light_T1/round4_normal/light_T1_r4.pcapng": "50b7968226ccb6ea254b704795befe89bb6d4e648b9f3bd0ff22c3de528bdb99",
    "dataset/light_xm/round4_normal/light_xm_r4.pcapng": "5cff4efe8862525051f5a97362dd8b6c18c04fec4108d82186a4c5df45315c97",
    "dataset/sensor/round4_normal/sensor_r4.pcapng": "4ce712b7ab9c05d181cd0819bf31f94e82b80e311e0389f44d1a7ad2a5caf7a6",
    "dataset/socket/round4_normal/socket_r4.pcapng": "901b25b1af1f4c3baef527f88a3f602b8606a3f3f7aaac378f261bd716267ea6",
    "dataset/camera/round5_positionB/camera_r5.pcapng": "fb5280ae19283f38abbf0ea0ac8f1295f06b7f83d5ecef47e0efb5b2fbe1fbf5",
    "dataset/light_T1/round5_positionB/light_T1_r5.pcapng": "fb70d9ef4ec7c277e8287ed6f213b5cfa9bf7949b69cc7e3400fcfe1fb5a8ad2",
    "dataset/light_xm/round5_positionB/light_xm_r5.pcapng": "e93120f0707bd741dfaff22cd26add7b00ba4c132d2871254b9a0c3fbf6d9fb8",
    "dataset/sensor/round5_positionB/sensor_r5.pcapng": "70a78607d215ba4af46f605ab9630860c34a7421bace60b9b6193abe61d02b57",
    "dataset/socket/round5_positionB/socket_r5.pcapng": "de622b7be00643a1ef97f244eee803b4cd353ba66b7ae6ecc76629f79ff02fd5",
    "dataset/camera/round6_jitter/camera_r6.pcapng": "55df60693bd517ffbfc63c89df00c6a134f9cedca911cb70f95b3f1d664d45e6",
    "dataset/light_T1/round6_jitter/light_T1_r6.pcapng": "9fab293b83490fe8e58b3de1007cadb30511475aef7143dd2204844d2fd5f5c0",
    "dataset/light_xm/round6_jitter/light_xm_r6.pcapng": "17fe8d7f0fa0e17ffcb00f9fdbb13109f2b4d313b9618ebb19baa40259571c8e",
    "dataset/sensor/round6_jitter/sensor_r6.pcapng": "c2ab988b677f5cd436aa25f8f05055041a6b9cd89e2cfe872cebcb7d866e38d7",
    "dataset/socket/round6_jitter/socket_r6.pcapng": "62ee7a337a5adf43d852922a0c5e20b9993e4f481db3e6f83b3d9e6b416a890b",
    "dataset/camera/round7_jitter/camera_r7.pcapng": "a3e9a0282268ad3fca1a3c25f13b03dc6c0ed5c374d07e96cbf8f19c16a44ca3",
    "dataset/light_T1/round7_jitter/light_T1_r7.pcapng": "36af506bd89947e7e9c693b8fbfc6bc70fe46cefd58120e19f113c7a14f2bdc3",
    "dataset/light_xm/round7_jitter/light_xm_r7.pcapng": "e752598f7fd7ad472a0c58dea9232f71341601451cdaf1301187f6dbb39427dd",
    "dataset/sensor/round7_jitter/sensor_r7.pcapng": "8645dfd930733d104035d294ff76f9ad3d1510b2687baf43d6ed52c638c86d8b",
    "dataset/socket/round7_jitter/socket_r7.pcapng": "d2dec476114dc99845affd5332fa6c78a52605788230e2c876e38f32104fc3ed",
}


class PipelineError(RuntimeError):
    """A frozen input, implementation, execution, or adjudication failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def stable_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g", lineterminator="\n")


def check_proxy_environment(environ: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    return [name for name in PROXY_VARIABLES if str(env.get(name, "")) != ""]


def ensure_proxy_gate() -> None:
    bad = check_proxy_environment()
    if bad:
        raise PipelineError(f"forbidden proxy variables are non-empty: {bad}")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return Path(path).relative_to(REPO_ROOT).as_posix()


def _verify_hash_map(expected: Mapping[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise PipelineError(f"missing frozen anchor: {relative}")
        actual[relative] = sha256_file(path)
        if actual[relative] != expected_hash:
            raise PipelineError(
                f"frozen anchor SHA-256 mismatch: {relative}: "
                f"{actual[relative]} != {expected_hash}"
            )
    return actual


def _verify_accepted_manifest() -> None:
    manifest = _read_json(ACCEPTED_RA_ROOT / "manifest.json")
    for name, expected in manifest.items():
        path = ACCEPTED_RA_ROOT / name
        if not path.is_file():
            raise PipelineError(f"accepted strict59_ra manifest file missing: {name}")
        if path.stat().st_size != int(expected["bytes"]):
            raise PipelineError(f"accepted strict59_ra manifest byte mismatch: {name}")
        if sha256_file(path) != str(expected["sha256"]):
            raise PipelineError(f"accepted strict59_ra manifest hash mismatch: {name}")
    acceptance = _read_json(ACCEPTED_RA_ROOT / "acceptance.json")
    if acceptance.get("status") != "STRICT59_RA_DIRECTION_REPAIR_ACCEPTED":
        raise PipelineError("accepted strict59_ra status mismatch")


def strict59_ra_columns() -> list[str]:
    arms = _read_json(ACCEPTED_RA_ROOT / "feature_arms.json")
    columns = list(arms.get("strict59_ra", []))
    if len(columns) != 59 or len(set(columns)) != 59:
        raise PipelineError("strict59_ra feature list is not 59 unique columns")
    if tuple(columns) != tuple(direction_repair.STRICT59_RA):
        raise PipelineError("accepted strict59_ra feature order differs from frozen implementation")
    return columns


def validate_static(
    expected_protocol_sha256: str,
    expected_repair_protocol_sha256: str,
    expected_recovery_protocol_sha256: str,
    expected_tmp_repair_protocol_sha256: str,
    expected_implementation_freeze_sha256: str,
    require_output_absence: bool,
) -> dict[str, Any]:
    ensure_proxy_gate()
    freeze = _read_json(PROTOCOL_FREEZE)
    protocol_hash = sha256_file(PROTOCOL)
    if protocol_hash != expected_protocol_sha256:
        raise PipelineError("CLI expected protocol SHA-256 mismatch")
    if protocol_hash != freeze["protocol"]["sha256"]:
        raise PipelineError("protocol freeze record mismatch")
    repair_freeze = _read_json(R2_PROTOCOL_FREEZE)
    repair_protocol_hash = sha256_file(REPAIR_PROTOCOL)
    if repair_protocol_hash != expected_repair_protocol_sha256:
        raise PipelineError("CLI expected R2 repair protocol SHA-256 mismatch")
    if repair_protocol_hash != repair_freeze["repair_protocol"]["sha256"]:
        raise PipelineError("R2 repair protocol freeze record mismatch")
    if repair_freeze["parent_protocol_sha256"] != protocol_hash:
        raise PipelineError("R2 repair protocol parent hash mismatch")
    recovery_freeze = _read_json(R3_PROTOCOL_FREEZE)
    recovery_protocol_hash = sha256_file(RECOVERY_PROTOCOL)
    if recovery_protocol_hash != expected_recovery_protocol_sha256:
        raise PipelineError("CLI expected R3 recovery protocol SHA-256 mismatch")
    if recovery_protocol_hash != recovery_freeze["recovery_protocol"]["sha256"]:
        raise PipelineError("R3 recovery protocol freeze record mismatch")
    if recovery_freeze["parent_protocol_sha256"] != protocol_hash:
        raise PipelineError("R3 recovery protocol parent hash mismatch")
    if recovery_freeze["r2_repair_protocol_sha256"] != repair_protocol_hash:
        raise PipelineError("R3 recovery protocol R2 repair hash mismatch")
    r2_implementation_hash = sha256_file(R2_IMPLEMENTATION_FREEZE)
    if recovery_freeze["r2_implementation_freeze_sha256"] != r2_implementation_hash:
        raise PipelineError("R3 recovery protocol R2 implementation hash mismatch")
    r4_freeze = _read_json(R4_PROTOCOL_FREEZE)
    tmp_repair_protocol_hash = sha256_file(TMP_REPAIR_PROTOCOL)
    if tmp_repair_protocol_hash != expected_tmp_repair_protocol_sha256:
        raise PipelineError("CLI expected R4 temporary display repair protocol SHA-256 mismatch")
    if tmp_repair_protocol_hash != r4_freeze["repair_protocol"]["sha256"]:
        raise PipelineError("R4 temporary display repair protocol freeze record mismatch")
    if r4_freeze["parent_protocol_sha256"] != protocol_hash:
        raise PipelineError("R4 temporary display repair parent protocol hash mismatch")
    if r4_freeze["r2_repair_protocol_sha256"] != repair_protocol_hash:
        raise PipelineError("R4 temporary display repair R2 protocol hash mismatch")
    if r4_freeze["r3_recovery_protocol_sha256"] != recovery_protocol_hash:
        raise PipelineError("R4 temporary display repair R3 protocol hash mismatch")
    r3_implementation_hash = sha256_file(R3_IMPLEMENTATION_FREEZE)
    if r4_freeze["r3_implementation_freeze_sha256"] != r3_implementation_hash:
        raise PipelineError("R4 temporary display repair R3 implementation hash mismatch")
    if sha256_file(IMPLEMENTATION_FREEZE) != expected_implementation_freeze_sha256:
        raise PipelineError("implementation freeze SHA-256 mismatch")
    implementation = _read_json(IMPLEMENTATION_FREEZE)
    implementation_files = implementation.get("implementation_sha256", {})
    for relative, expected in implementation_files.items():
        path = REPO_ROOT / relative
        if sha256_file(path) != expected:
            raise PipelineError(f"implementation anchor mismatch: {relative}")
    anchors = _verify_hash_map(ANCHOR_SHA256)
    full94 = _verify_hash_map(FULL94_REFERENCE_SHA256)
    pcap = _verify_hash_map(PCAP_SHA256)
    _verify_accepted_manifest()
    columns = strict59_ra_columns()
    grid = g0.build_task_grid()
    if len(grid) != EXPECTED_G0_RUNS:
        raise PipelineError("G0 task-grid run count mismatch")
    if sum(task["grid_kind"] == "ood" for task in grid) != 150:
        raise PipelineError("G0 OOD count mismatch")
    if sum(task["grid_kind"] == "iid_time_block" for task in grid) != 6:
        raise PipelineError("G0 IID time-block count mismatch")
    if sum(task["grid_kind"] == "iid_random" for task in grid) != 6:
        raise PipelineError("G0 IID random count mismatch")
    output_absence = {
        _relative(path): not path.exists()
        for path in (AUDIT_ROOT, G0_ROOT_A, SCIENCE_ROOT_A)
    }
    if require_output_absence and not all(output_absence.values()):
        raise PipelineError(f"formal output root already exists: {output_absence}")
    return {
        "protocol_sha256": protocol_hash,
        "protocol_freeze_sha256": sha256_file(PROTOCOL_FREEZE),
        "repair_protocol_sha256": repair_protocol_hash,
        "r2_protocol_freeze_sha256": sha256_file(R2_PROTOCOL_FREEZE),
        "r2_implementation_freeze_sha256": r2_implementation_hash,
        "recovery_protocol_sha256": recovery_protocol_hash,
        "r3_protocol_freeze_sha256": sha256_file(R3_PROTOCOL_FREEZE),
        "r3_implementation_freeze_sha256": r3_implementation_hash,
        "tmp_repair_protocol_sha256": tmp_repair_protocol_hash,
        "r4_protocol_freeze_sha256": sha256_file(R4_PROTOCOL_FREEZE),
        "implementation_freeze_sha256": sha256_file(IMPLEMENTATION_FREEZE),
        "anchor_sha256": anchors,
        "full94_reference_sha256": full94,
        "pcap_sha256": pcap,
        "strict59_ra_feature_count": len(columns),
        "g0_run_count": len(grid),
        "formal_output_absence": output_absence,
    }


def _close(actual: float, expected: float, tolerance: float = FULL94_TOLERANCE) -> bool:
    return bool(abs(float(actual) - float(expected)) <= tolerance)


def reproduce_full94(temp_root: Path) -> dict[str, Any]:
    """Re-run frozen M2 twice on frozen full94 inputs before strict59_ra fitting."""
    ensure_proxy_gate()
    m1_gate = _read_json(FULL94_M1_ROOT / "m1_gate.json")
    m1r_gate = _read_json(FULL94_M1R_ROOT / "m1r_gate.json")
    out_a = temp_root / "full94_m2_a"
    out_b = temp_root / "full94_m2_b"
    original_m1_input = m1.INPUT_ROOT
    original_m1r_root = m2.M1R_ROOT
    original_g0_root = m2.G0_ROOT
    try:
        m1.INPUT_ROOT = REPO_ROOT / "results/g0_environment_grid/raw_all"
        m2.M1R_ROOT = FULL94_M1R_ROOT
        m2.G0_ROOT = m1.INPUT_ROOT
        m2.run("full", out_a, [str(Path(m2.__file__)), "--scope", "full", "--out-dir", str(out_a)])
        m2.run("full", out_b, [str(Path(m2.__file__)), "--scope", "full", "--out-dir", str(out_b)])
        reproduced_gate = m2.verify(out_a, out_b)
    finally:
        m1.INPUT_ROOT = original_m1_input
        m2.M1R_ROOT = original_m1r_root
        m2.G0_ROOT = original_g0_root

    m1_values = m1_gate["h_m1"]["raw_comparison_quantities"]
    m1r_values = m1r_gate["m1r"]["raw_comparison_quantities"]
    checks = {
        "m1_mmg_ood_equal": _close(m1_values["ood_mmg_equal_weight_mean"], 0.20292761623438427),
        "m1_positive_environment_count": int(m1_values["ood_environments_with_positive_mean_mmg"]) == 6,
        "m1r_matched_ood_mmg_equal": _close(
            m1r_values["matched_ood_mmg_equal_weight_mean"], 0.1919156647423589
        ),
        "m1r_excess_f_equal": _close(
            m1r_values["matched_ood_minus_iid_mmg_equal_weight_mean"], 0.18336639758796638
        ),
        "m2_er_c_equal": _close(
            reproduced_gate["sufficiency"]["C"]["ER_equal"], 0.8909627931962164
        ),
        "m2_er_c_environment_count": int(
            reproduced_gate["sufficiency"]["C"]["environments_ER_ge_0.50"]
        ) == 6,
        "m2_engineering_pass": reproduced_gate.get("engineering_status") == "PASS",
        "m2_double_run_pass": bool(
            reproduced_gate.get("engineering", {}).get("double_run_hash_consistent")
        ),
    }
    if not all(checks.values()):
        raise PipelineError(f"FULL94_REPRODUCTION_INVALID_STOP: {checks}")
    return {
        "status": "FULL94_REPRODUCTION_PASS",
        "tolerance": FULL94_TOLERANCE,
        "checks": checks,
        "values": {
            "m1_mmg_ood_equal": float(m1_values["ood_mmg_equal_weight_mean"]),
            "m1_positive_environment_count": int(m1_values["ood_environments_with_positive_mean_mmg"]),
            "m1r_matched_ood_mmg_equal": float(m1r_values["matched_ood_mmg_equal_weight_mean"]),
            "m1r_excess_f_equal": float(m1r_values["matched_ood_minus_iid_mmg_equal_weight_mean"]),
            "m2_er_c_equal": float(reproduced_gate["sufficiency"]["C"]["ER_equal"]),
            "m2_er_c_environment_count": int(
                reproduced_gate["sufficiency"]["C"]["environments_ER_ge_0.50"]
            ),
        },
        "temporary_outputs": [str(out_a), str(out_b)],
    }


def _expected_pcap_paths(config: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for round_name in ROUNDS:
        round_info = config["rounds"][round_name]
        for _label, device_dir in config["device_dirs"].items():
            paths.append(
                Path("dataset")
                / str(device_dir)
                / str(round_info["dir"])
                / f"{device_dir}_r{round_name[-1]}.pcapng"
            )
    if {path.as_posix() for path in paths} != set(PCAP_SHA256):
        raise PipelineError("config-derived pcap paths differ from frozen 30-pcap set")
    return paths


def resolve_recorded_source(value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _numeric_max_abs(left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]) -> float:
    if not columns:
        return 0.0
    a = left[list(columns)].to_numpy(dtype=np.float64)
    b = right[list(columns)].to_numpy(dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def materialize_strict59_ra() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Re-extract all six rounds, verify old-cache reproduction, and write 67 columns."""
    cached, config, prior_input_audit = direction_repair.prior.load_and_validate_inputs()
    _expected_pcap_paths(config)
    if len(cached) != EXPECTED_ROWS or list(cached.columns[:8]) != META_COLUMNS:
        raise PipelineError("source cache row count or metadata order mismatch")
    if cached.duplicated(KEY_COLUMNS).any():
        raise PipelineError("source cache has duplicate row keys")
    observed_round_counts = {
        str(key): int(value) for key, value in cached["round"].value_counts().sort_index().items()
    }
    if observed_round_counts != ROUND_COUNTS:
        raise PipelineError(f"source cache round counts mismatch: {observed_round_counts}")

    reconstructed_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for round_name in ROUNDS:
        round_info = config["rounds"][round_name]
        for label, device_dir in config["device_dirs"].items():
            relative_path = (
                Path("dataset")
                / str(device_dir)
                / str(round_info["dir"])
                / f"{device_dir}_r{round_name[-1]}.pcapng"
            )
            pcap_path = resolve_recorded_source(relative_path)
            packets = direction_repair.preprocess_packets(pcap_path)
            kept = 0
            for window_id, group in packets.groupby("window_id", sort=True):
                if len(group) < 2:
                    continue
                kept += 1
                original = direction_repair.core.summarize_window(
                    group,
                    label,
                    round_name,
                    round_info["traffic"],
                    "raw_all",
                    pcap_path,
                    int(window_id),
                    10.0,
                )
                patched = group.copy()
                patched["da"] = patched["ra"]
                corrected = direction_repair.core.summarize_window(
                    patched,
                    label,
                    round_name,
                    round_info["traffic"],
                    "raw_all",
                    pcap_path,
                    int(window_id),
                    10.0,
                )
                reconstructed_rows.append(original)
                row: dict[str, Any] = {key: original[key] for key in KEY_COLUMNS}
                for feature in direction_repair.DIRECTION14:
                    row[f"old_{feature}"] = original[feature]
                    row[f"ra_{feature}"] = corrected[feature]
                for feature in ("side_packet_ratio", "other_packet_ratio"):
                    row[f"old_{feature}"] = original[feature]
                    row[f"ra_{feature}"] = corrected[feature]
                direction_rows.append(row)
            print(f"re-extracted {round_name}/{label}: {kept} windows", flush=True)

    reconstructed = pd.DataFrame(reconstructed_rows).sort_values(KEY_COLUMNS).reset_index(drop=True)
    direction = pd.DataFrame(direction_rows).sort_values(KEY_COLUMNS).reset_index(drop=True)
    cached_sorted = cached.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if len(reconstructed) != EXPECTED_ROWS or len(direction) != EXPECTED_ROWS:
        raise PipelineError(
            f"full pcap window count mismatch: reconstructed={len(reconstructed)} direction={len(direction)}"
        )
    if not reconstructed[KEY_COLUMNS].equals(cached_sorted[KEY_COLUMNS]):
        raise PipelineError("pcap reconstructed keys differ from source cache")
    full94 = list(direction_repair.prior.FULL94)
    full94_error = _numeric_max_abs(reconstructed, cached_sorted, full94)
    if full94_error > 1e-9:
        raise PipelineError(f"pcap-to-cache full94 error {full94_error} exceeds 1e-9")
    for column in ("label", "round", "traffic", "filter_mode", "source_file", "window_id"):
        if not reconstructed[column].equals(cached_sorted[column]):
            raise PipelineError(f"pcap-to-cache metadata mismatch: {column}")
    meta_time_error = _numeric_max_abs(
        reconstructed, cached_sorted, ("window_start", "window_end")
    )
    if meta_time_error > 1e-9:
        raise PipelineError("pcap-to-cache window time mismatch")

    corrected_full = cached.copy()
    replacement = direction.set_index(KEY_COLUMNS)
    target_keys = pd.MultiIndex.from_frame(corrected_full[KEY_COLUMNS])
    for feature in direction_repair.DIRECTION14:
        values = replacement[f"ra_{feature}"].reindex(target_keys)
        if values.isna().any():
            raise PipelineError(f"missing RA replacement: {feature}")
        corrected_full[feature] = values.to_numpy(dtype=np.float64)
    common45 = list(direction_repair.COMMON45)
    if not np.array_equal(corrected_full[common45].to_numpy(), cached[common45].to_numpy()):
        raise PipelineError("common45 fields changed during materialization")

    accepted = pd.read_csv(
        ACCEPTED_RA_ROOT / "corrected_direction_features.csv", float_precision="round_trip"
    )
    accepted["resolved_source_file"] = accepted["source_file"].map(
        lambda value: str(resolve_recorded_source(value))
    )
    accepted_keys = ["label", "round", "window_id"]
    accepted = accepted.sort_values(accepted_keys).reset_index(drop=True)
    current_r2_r4 = (
        direction[direction["round"].isin(("R2", "R3", "R4"))]
        .sort_values(accepted_keys)
        .reset_index(drop=True)
    )
    if len(accepted) != EXPECTED_R2_R4_ROWS or not current_r2_r4[accepted_keys].equals(accepted[accepted_keys]):
        raise PipelineError("R2-R4 accepted direction row-key reproduction failed")
    if not current_r2_r4["source_file"].astype(str).equals(accepted["resolved_source_file"]):
        raise PipelineError("R2-R4 accepted relative/absolute source-file resolution failed")
    accepted_columns = [f"ra_{feature}" for feature in direction_repair.DIRECTION14]
    accepted_ra_error = _numeric_max_abs(current_r2_r4, accepted, accepted_columns)
    if accepted_ra_error > 1e-9:
        raise PipelineError(f"R2-R4 accepted RA feature error {accepted_ra_error} exceeds 1e-9")

    closure = (
        direction["ra_up_packet_ratio"]
        + direction["ra_down_packet_ratio"]
        + direction["ra_side_packet_ratio"]
        + direction["ra_other_packet_ratio"]
    )
    closure_error = float(np.max(np.abs(closure.to_numpy(dtype=float) - 1.0)))
    strict_columns = strict59_ra_columns()
    strict_values = corrected_full[strict_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(strict_values).all():
        raise PipelineError("materialized strict59_ra values are non-finite")

    per_round: dict[str, dict[str, float | int]] = {}
    for round_name in ROUNDS:
        subset = direction[direction["round"] == round_name]
        old_up = subset["old_up_packet_ratio"].to_numpy(dtype=float)
        ra_up = subset["ra_up_packet_ratio"].to_numpy(dtype=float)
        old_std = subset["old_up_len_std"].to_numpy(dtype=float)
        ra_std = subset["ra_up_len_std"].to_numpy(dtype=float)
        per_round[round_name] = {
            "windows": int(len(subset)),
            "old_up_packet_ratio_zero_fraction": float((old_up == 0).mean()),
            "ra_up_packet_ratio_zero_fraction": float((ra_up == 0).mean()),
            "up_packet_ratio_changed_fraction": float((np.abs(old_up - ra_up) > 1e-12).mean()),
            "old_up_len_std_zero_fraction": float((old_std == 0).mean()),
            "ra_up_len_std_zero_fraction": float((ra_std == 0).mean()),
        }
    old_up_all = direction["old_up_packet_ratio"].to_numpy(dtype=float)
    ra_up_all = direction["ra_up_packet_ratio"].to_numpy(dtype=float)
    ra_std_all = direction["ra_up_len_std"].to_numpy(dtype=float)
    overall = {
        "old_up_packet_ratio_zero_fraction": float((old_up_all == 0).mean()),
        "ra_up_packet_ratio_zero_fraction": float((ra_up_all == 0).mean()),
        "up_packet_ratio_changed_fraction": float((np.abs(old_up_all - ra_up_all) > 1e-12).mean()),
        "ra_up_len_std_zero_fraction": float((ra_std_all == 0).mean()),
    }
    gates = {
        "window_count_exact": len(direction) == EXPECTED_ROWS,
        "pcap_to_cached_full94_max_abs_le_1e_9": full94_error <= 1e-9,
        "pcap_to_cached_meta_time_max_abs_le_1e_9": meta_time_error <= 1e-9,
        "common45_bitwise_unchanged": True,
        "direction_ratio_closure_max_abs_le_1e_12": closure_error <= 1e-12,
        "accepted_r2_r4_ra_max_abs_le_1e_9": accepted_ra_error <= 1e-9,
        "overall_ra_up_zero_le_0_50": overall["ra_up_packet_ratio_zero_fraction"] <= 0.50,
        "overall_up_changed_ge_0_50": overall["up_packet_ratio_changed_fraction"] >= 0.50,
        "overall_ra_up_len_std_not_all_zero": overall["ra_up_len_std_zero_fraction"] < 1.0,
        "strict59_ra_all_finite": bool(np.isfinite(strict_values).all()),
    }
    if not all(gates.values()):
        raise PipelineError(f"strict59_ra full materialization gate failed: {gates}")

    strict_cache = AUDIT_ROOT / STRICT_CACHE_NAME
    materialized = corrected_full[META_COLUMNS + strict_columns]
    if materialized.shape != (EXPECTED_ROWS, 67):
        raise PipelineError(f"materialized cache shape mismatch: {materialized.shape}")
    stable_csv(materialized, strict_cache)
    roundtrip = pd.read_csv(strict_cache, encoding="utf-8", float_precision="round_trip")
    if list(roundtrip.columns) != META_COLUMNS + strict_columns or len(roundtrip) != EXPECTED_ROWS:
        raise PipelineError("materialized cache round-trip schema mismatch")
    if not np.isfinite(roundtrip[strict_columns].to_numpy(dtype=np.float64)).all():
        raise PipelineError("materialized cache round-trip finite-value gate failed")

    input_audit = {
        "source_cache_sha256": sha256_file(SOURCE_CACHE),
        "source_rows": int(len(cached)),
        "source_columns": int(len(cached.columns)),
        "meta_columns": META_COLUMNS,
        "model_feature_count": len(strict_columns),
        "materialized_columns": int(materialized.shape[1]),
        "round_counts": observed_round_counts,
        "row_key_unique": not cached.duplicated(KEY_COLUMNS).any(),
        "prior_input_audit": prior_input_audit,
    }
    extraction_audit = {
        "status": "STRICT59_RA_R2_R7_MATERIALIZATION_PASS",
        "windows": int(len(direction)),
        "pcap_to_cached_full94_max_abs_error": full94_error,
        "pcap_to_cached_meta_time_max_abs_error": meta_time_error,
        "accepted_r2_r4_ra_max_abs_error": accepted_ra_error,
        "direction_ratio_closure_max_abs_error": closure_error,
        "overall": overall,
        "per_round": per_round,
        "gates": gates,
        "materialized_cache_sha256": sha256_file(strict_cache),
    }
    return strict_cache, input_audit, extraction_audit


@contextlib.contextmanager
def patched_argv(argv: Sequence[str]) -> Iterator[None]:
    old = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


def g0_display_root(output_root: Path, repository_root: Path | None = None) -> Path:
    repo = Path(g0.REPO_ROOT if repository_root is None else repository_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(repo)
        return repo
    except ValueError:
        return Path(os.path.commonpath((str(repo), str(output))))


def run_g0(cache: Path, output_root: Path) -> None:
    if output_root.exists():
        raise PipelineError(f"G0 output root already exists: {output_root}")
    old_cache, old_output, old_repo = g0.CACHE_SRC, g0.OUT_ROOT, g0.REPO_ROOT
    g0.CACHE_SRC, g0.OUT_ROOT = cache, output_root
    g0.REPO_ROOT = g0_display_root(output_root, old_repo)
    argv = [
        str(Path(g0.__file__)),
        "--models",
        "rf,xgboost,lightgbm,stacking",
        "--n-jobs",
        "4",
        "--max-rows",
        str(10**9),
    ]
    try:
        with patched_argv(argv):
            try:
                result = g0.main()
            except SystemExit as error:
                raise PipelineError(f"G0 exited early: {error}") from error
        if result != 0:
            raise PipelineError(f"G0 returned nonzero: {result}")
    finally:
        g0.CACHE_SRC, g0.OUT_ROOT, g0.REPO_ROOT = old_cache, old_output, old_repo


def _deterministic_files(root: Path, excluded_names: set[str] | None = None) -> dict[str, Path]:
    excluded = excluded_names or set()
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def compare_directories(
    root_a: Path,
    root_b: Path,
    excluded_names: set[str] | None = None,
) -> dict[str, str]:
    files_a = _deterministic_files(root_a, excluded_names)
    files_b = _deterministic_files(root_b, excluded_names)
    if set(files_a) != set(files_b):
        missing_a = sorted(set(files_b) - set(files_a))
        missing_b = sorted(set(files_a) - set(files_b))
        raise PipelineError(
            f"deterministic file sets differ; missing_a={missing_a[:5]} missing_b={missing_b[:5]}"
        )
    hashes: dict[str, str] = {}
    for relative in sorted(files_a):
        hash_a = sha256_file(files_a[relative])
        hash_b = sha256_file(files_b[relative])
        if hash_a != hash_b:
            raise PipelineError(f"deterministic file hash mismatch: {relative}")
        hashes[relative] = hash_a
    return hashes


def compare_g0(root_a: Path, root_b: Path, strict_columns: Sequence[str]) -> dict[str, Any]:
    required_names = {"metrics.json", "predictions.csv", "pred_proba.csv", "feature_columns.json"}
    selected_a: dict[str, Path] = {}
    selected_b: dict[str, Path] = {}
    for root, selected in ((root_a, selected_a), (root_b, selected_b)):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if (
                path.name in required_names
                or path.name == "oof_meta.csv"
                or relative in {"summary_metrics.csv", "summary_metrics.json", "env_topology_matrix_rf.csv"}
            ):
                selected[relative] = path
    if set(selected_a) != set(selected_b):
        raise PipelineError("G0 deterministic file sets differ")
    counts = {
        name: sum(path.name == name for path in selected_a.values())
        for name in required_names | {"oof_meta.csv"}
    }
    expected_counts = {
        "metrics.json": EXPECTED_G0_MODEL_CELLS,
        "predictions.csv": EXPECTED_G0_MODEL_CELLS,
        "pred_proba.csv": EXPECTED_G0_MODEL_CELLS,
        "feature_columns.json": EXPECTED_G0_MODEL_CELLS,
        "oof_meta.csv": EXPECTED_G0_RUNS,
    }
    if counts != expected_counts:
        raise PipelineError(f"G0 file-count gate failed: {counts} != {expected_counts}")
    hashes: dict[str, str] = {}
    for relative in sorted(selected_a):
        hash_a = sha256_file(selected_a[relative])
        hash_b = sha256_file(selected_b[relative])
        if hash_a != hash_b:
            raise PipelineError(f"G0 deterministic hash mismatch: {relative}")
        hashes[relative] = hash_a
        if selected_a[relative].name == "feature_columns.json":
            if _read_json(selected_a[relative]) != list(strict_columns):
                raise PipelineError(f"G0 feature-column mismatch: {relative}")
    summary = pd.read_csv(root_a / "summary_metrics.csv", encoding="utf-8-sig")
    if len(summary) != EXPECTED_G0_MODEL_CELLS or set(summary["feature_count"]) != {59}:
        raise PipelineError("G0 summary cell/feature-count gate failed")
    return {
        "consistent": True,
        "algorithm": "sha256",
        "counts": counts,
        "deterministic_file_count": len(hashes),
        "files": hashes,
    }


def switch_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise PipelineError(f"refusing to replace non-symlink logical input: {link}")
    if link.is_symlink():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=True)


def run_m1(input_root: Path, output_root: Path, logical_input: Path) -> dict[str, Any]:
    switch_symlink(logical_input, input_root)
    old_input = m1.INPUT_ROOT
    try:
        m1.INPUT_ROOT = logical_input
        return m1.run(
            "full",
            output_root,
            [str(Path(m1.__file__)), "--scope", "full", "--out-dir", str(output_root)],
        )
    finally:
        m1.INPUT_ROOT = old_input


def run_m1r(
    g0_input: Path,
    m1_input: Path,
    output_root: Path,
    logical_g0: Path,
    logical_m1: Path,
) -> dict[str, Any]:
    switch_symlink(logical_g0, g0_input)
    switch_symlink(logical_m1, m1_input)
    old_g0, old_m1_result = m1.INPUT_ROOT, m1r.M1_RESULT_ROOT
    try:
        m1.INPUT_ROOT = logical_g0
        m1r.M1_RESULT_ROOT = logical_m1
        return m1r.run(
            "full",
            output_root,
            [str(Path(m1r.__file__)), "--scope", "full", "--out-dir", str(output_root)],
        )
    finally:
        m1.INPUT_ROOT, m1r.M1_RESULT_ROOT = old_g0, old_m1_result


def run_m2(
    g0_input: Path,
    m1r_input: Path,
    output_root: Path,
    logical_g0: Path,
    logical_m1r: Path,
) -> dict[str, Any]:
    switch_symlink(logical_g0, g0_input)
    switch_symlink(logical_m1r, m1r_input)
    old_m1_input, old_g0, old_m1r = m1.INPUT_ROOT, m2.G0_ROOT, m2.M1R_ROOT
    try:
        m1.INPUT_ROOT = logical_g0
        m2.G0_ROOT = logical_g0
        m2.M1R_ROOT = logical_m1r
        return m2.run(
            "full",
            output_root,
            [str(Path(m2.__file__)), "--scope", "full", "--out-dir", str(output_root)],
        )
    finally:
        m1.INPUT_ROOT, m2.G0_ROOT, m2.M1R_ROOT = old_m1_input, old_g0, old_m1r


def adjudicate(m1_gate: Mapping[str, Any], m1r_gate: Mapping[str, Any], m2_gate: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    m1_pass = (
        m1_gate.get("engineering_status") == "PASS"
        and m1_gate.get("h_m1_status") == "SUPPORTED_FOR_DECOMPOSITION"
    )
    m1r_values = m1r_gate["m1r"]["raw_comparison_quantities"]
    recoverability_conditions = {
        "m1_pass": m1_pass,
        "matched_ood_mmg_equal_ge_0_010": float(
            m1r_values["matched_ood_mmg_equal_weight_mean"]
        ) >= 0.010,
        "matched_ood_positive_environments_ge_4": int(
            m1r_values["matched_ood_environments_with_positive_mean_mmg"]
        ) >= 4,
        "excess_f_equal_ge_0_005": float(
            m1r_values["matched_ood_minus_iid_mmg_equal_weight_mean"]
        ) >= 0.005,
        "excess_f_positive_environments_ge_4": int(
            m1r_values["environments_with_positive_matched_ood_minus_iid_mmg"]
        ) >= 4,
    }
    recoverability = all(recoverability_conditions.values())
    excess_f = float(m1r_values["matched_ood_minus_iid_mmg_equal_weight_mean"])
    materiality = excess_f >= 0.020
    sufficiency = m2_gate["sufficiency"]
    structure_conditions = {
        "m2_engineering_pass": m2_gate.get("engineering_status") == "PASS",
        "c_equal_ge_0_80": float(sufficiency["C"]["ER_equal"]) >= 0.80,
        "c_environment_count_ge_4": int(sufficiency["C"]["environments_ER_ge_0.50"]) >= 4,
        "i_not_sufficient": not bool(sufficiency["I"]["sufficient"]),
        "g_not_sufficient": not bool(sufficiency["G"]["sufficient"]),
        "c_first_sufficient_status": (
            m2_gate.get("first_sufficient_stage")
            == "CLASS_CONDITIONAL_BLOCK_REWEIGHTING_SUFFICIENT"
        ),
    }
    structure = all(structure_conditions.values())
    if not recoverability:
        status = "EC_MDM_ORACLE_RECOVERABILITY_NOT_SUPPORTED_STRICT59_RA_STOP"
    elif not materiality:
        status = "EC_MDM_ORACLE_SIGNAL_BELOW_MATERIALITY_STRICT59_RA_STOP"
    elif not structure:
        status = "EC_MDM_ORACLE_STRUCTURE_NOT_REPLICATED_STRICT59_RA_STOP"
    else:
        status = "EC_MDM_ORACLE_CANDIDATE_SUPPORTED_STRICT59_RA"

    matched_by_env = m1r_gate["m1r"]["environment_matched_ood_mmg"]
    excess_by_env = m1r_gate["m1r"]["environment_ood_minus_iid_mmg"]
    rows: list[dict[str, Any]] = []
    for environment in ROUNDS:
        m2_env = m2_gate["environment_excess_ER"][environment]
        rows.append(
            {
                "target_env": environment,
                "matched_ood_mmg": float(matched_by_env[environment]),
                "excess_F": float(excess_by_env[environment]),
                "excess_I": float(m2_env["excess_I"]),
                "ER_I": float(m2_env["ER_I"]),
                "excess_G": float(m2_env["excess_G"]),
                "ER_G": float(m2_env["ER_G"]),
                "excess_C": float(m2_env["excess_C"]),
                "ER_C": float(m2_env["ER_C"]),
            }
        )
    passline = {
        "status": status,
        "oracle_recoverability": {
            "passed": recoverability,
            "conditions": recoverability_conditions,
            "matched_ood_mmg_equal": float(m1r_values["matched_ood_mmg_equal_weight_mean"]),
            "excess_f_equal": excess_f,
        },
        "materiality": {"passed": materiality, "threshold": 0.020, "excess_f_equal": excess_f},
        "c_structure": {
            "passed": structure,
            "conditions": structure_conditions,
            "first_sufficient_stage": m2_gate.get("first_sufficient_stage"),
            "ER_I_equal": float(sufficiency["I"]["ER_equal"]),
            "ER_G_equal": float(sufficiency["G"]["ER_equal"]),
            "ER_C_equal": float(sufficiency["C"]["ER_equal"]),
        },
        "epistemic_status": {
            "oracle_recoverability_structure": "SUPPORTED" if status == "EC_MDM_ORACLE_CANDIDATE_SUPPORTED_STRICT59_RA" else "NOT_FULLY_SUPPORTED",
            "observable_estimability": "NOT_EVALUATED",
            "deployability": "NOT_ESTABLISHED",
            "cpd_relation": "CANDIDATE_SUPERORDINATE_CONSTRUCT_HYPOTHESIS_ONLY",
        },
    }
    return passline, pd.DataFrame(rows)


def build_verdict(passline: Mapping[str, Any], full94: Mapping[str, Any], extraction: Mapping[str, Any]) -> str:
    oracle = passline["oracle_recoverability"]
    structure = passline["c_structure"]
    return "\n".join(
        [
            "# strict59_ra 下 EC-MDM oracle 重裁定",
            "",
            f"**状态**：`{passline['status']}`",
            "",
            "## 工程门",
            "",
            f"- full94 判否复现：`{full94['status']}`。",
            f"- R2--R7 RA 物化：{extraction['windows']}/11303 窗口；旧 full94 复现最大误差 "
            f"{extraction['pcap_to_cached_full94_max_abs_error']:.3e}；R2--R4 已接受 RA 最大误差 "
            f"{extraction['accepted_r2_r4_ra_max_abs_error']:.3e}。",
            "- G0：162 次运行、648 模型单元；G0/M1/M1-R/M2 双跑判定性产物一致。",
            "",
            "## oracle recoverability 与结构",
            "",
            f"- matched OOD MMG 等权：{oracle['matched_ood_mmg_equal']:.10f}。",
            f"- OOD-IID `excess_F_equal`：{oracle['excess_f_equal']:.10f}；实质量级门 "
            f"{'PASS' if passline['materiality']['passed'] else 'FAIL'}（门槛 0.020）。",
            f"- `ER_I/G/C_equal`：{structure['ER_I_equal']:.7f} / "
            f"{structure['ER_G_equal']:.7f} / {structure['ER_C_equal']:.7f}。",
            f"- 首个充分阶段：`{structure['first_sufficient_stage']}`；C 结构门 "
            f"{'PASS' if structure['passed'] else 'FAIL'}。",
            "",
            "## 三层证据边界",
            "",
            "本结果只裁定有目标标签的 oracle recoverability/structure。observable estimability 未在本协议中检验，"
            "deployability 未建立。即使状态通过，EC-MDM 也只能称为 CPD 的候选上位构念；本实验未计算 CPD，"
            "不证明因果机制、无标签可估计性或部署收益。",
            "",
            "若 oracle 三门全过，下一步只允许另冻 strict59_ra observable estimability 协议；若该无标签估计仍失败，"
            "commissioning 必须另冻并战胜最强相同目标标签预算基线。旧 full94 与本臂差值同时混有方向修复与字段删除，"
            "不得解释为纯表示消融。独立线不并入主线。",
            "",
        ]
    )


def _git_value(arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_manifest(root: Path) -> None:
    entries = {
        path.relative_to(root).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    stable_json(root / "manifest.json", entries)


def run_all(
    expected_protocol_sha256: str,
    expected_repair_protocol_sha256: str,
    expected_recovery_protocol_sha256: str,
    expected_tmp_repair_protocol_sha256: str,
    expected_implementation_freeze_sha256: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    started_wall = time.time()
    started_utc = datetime.now(timezone.utc)
    static = validate_static(
        expected_protocol_sha256,
        expected_repair_protocol_sha256,
        expected_recovery_protocol_sha256,
        expected_tmp_repair_protocol_sha256,
        expected_implementation_freeze_sha256,
        require_output_absence=True,
    )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=False)
    temp_root = Path(tempfile.mkdtemp(prefix="strict59_ra_ecmdm_", dir="/tmp"))
    current_phase = "F2_FULL94_REPRODUCTION"
    try:
        full94 = reproduce_full94(temp_root)
        stable_json(AUDIT_ROOT / "full94_reproduction_gate.json", full94)

        current_phase = "S1_MATERIALIZATION"
        strict_cache, input_audit, extraction_audit = materialize_strict59_ra()
        stable_json(AUDIT_ROOT / "input_audit.json", input_audit)
        stable_json(AUDIT_ROOT / "pcap_manifest.json", {"algorithm": "sha256", "files": PCAP_SHA256})
        stable_json(AUDIT_ROOT / "extraction_audit.json", extraction_audit)

        current_phase = "S2A_G0"
        run_g0(strict_cache, G0_ROOT_A)
        current_phase = "S2B_G0"
        g0_root_b = temp_root / "g0_b"
        run_g0(strict_cache, g0_root_b)
        current_phase = "S2_G0_VERIFY"
        strict_columns = strict59_ra_columns()
        g0_verification = compare_g0(G0_ROOT_A, g0_root_b, strict_columns)
        stable_json(AUDIT_ROOT / "g0_double_run_verification.json", g0_verification)

        current_phase = "S3_M1"
        SCIENCE_ROOT_A.mkdir(parents=True, exist_ok=False)
        links = AUDIT_ROOT / "_active_inputs"
        logical_g0 = links / "g0_raw_all"
        logical_m1 = links / "m1"
        logical_m1r = links / "m1r"
        m1_a_root, m1_b_root = SCIENCE_ROOT_A / "m1", temp_root / "m1_b"
        m1_gate_a = run_m1(G0_ROOT_A / "raw_all", m1_a_root, logical_g0)
        m1_gate_b = run_m1(g0_root_b / "raw_all", m1_b_root, logical_g0)
        m1_hashes = compare_directories(m1_a_root, m1_b_root, {"provenance.json"})
        if m1_gate_a != m1_gate_b:
            raise PipelineError("M1 gates differ between pipeline repeats")

        current_phase = "S4_M1R"
        m1r_a_root, m1r_b_root = SCIENCE_ROOT_A / "m1r", temp_root / "m1r_b"
        run_m1r(G0_ROOT_A / "raw_all", m1_a_root, m1r_a_root, logical_g0, logical_m1)
        run_m1r(g0_root_b / "raw_all", m1_b_root, m1r_b_root, logical_g0, logical_m1)
        m1r_gate_final = m1r.verify_double_run(m1r_a_root, m1r_b_root)

        current_phase = "S5_M2"
        m2_a_root, m2_b_root = SCIENCE_ROOT_A / "m2", temp_root / "m2_b"
        run_m2(G0_ROOT_A / "raw_all", m1r_a_root, m2_a_root, logical_g0, logical_m1r)
        run_m2(g0_root_b / "raw_all", m1r_b_root, m2_b_root, logical_g0, logical_m1r)
        m2_gate_final = m2.verify(m2_a_root, m2_b_root)
        switch_symlink(logical_g0, G0_ROOT_A / "raw_all")
        switch_symlink(logical_m1, m1_a_root)
        switch_symlink(logical_m1r, m1r_a_root)

        current_phase = "S5_PIPELINE_VERIFY"
        m1r_hashes = compare_directories(m1r_a_root, m1r_b_root, {"provenance.json"})
        m2_hashes = compare_directories(m2_a_root, m2_b_root, {"provenance.json"})
        pipeline_verification = {
            "consistent": True,
            "algorithm": "sha256",
            "g0_deterministic_file_count": g0_verification["deterministic_file_count"],
            "m1_deterministic_file_count": len(m1_hashes),
            "m1r_deterministic_file_count": len(m1r_hashes),
            "m2_deterministic_file_count": len(m2_hashes),
            "m1_files": m1_hashes,
            "m1r_files": m1r_hashes,
            "m2_files": m2_hashes,
        }
        stable_json(AUDIT_ROOT / "pipeline_double_run_verification.json", pipeline_verification)

        current_phase = "S6_ADJUDICATION"
        passline, per_environment = adjudicate(m1_gate_a, m1r_gate_final, m2_gate_final)
        stable_json(AUDIT_ROOT / "oracle_passline.json", passline)
        stable_csv(per_environment, AUDIT_ROOT / "per_environment.csv")
        acceptance = {
            "status": passline["status"],
            "engineering": {
                "static_anchors": True,
                "full94_reproduction": True,
                "materialization": True,
                "g0_double_run": True,
                "m1_double_run": True,
                "m1r_double_run": True,
                "m2_double_run": True,
                "g0_model_cells": EXPECTED_G0_MODEL_CELLS,
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
        stable_json(AUDIT_ROOT / "acceptance.json", acceptance)
        (AUDIT_ROOT / "VERDICT.md").write_text(
            build_verdict(passline, full94, extraction_audit), encoding="utf-8", newline="\n"
        )
        versions: dict[str, str | None] = {}
        for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost", "lightgbm"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        provenance = {
            "argv": list(argv),
            "started_utc": started_utc.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started_wall,
            "interpreter": str(Path(sys.executable).resolve()),
            "python": platform.python_version(),
            "package_versions": versions,
            "git_head": _git_value(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git_value(["status", "--porcelain"]),
            "protocol_sha256": sha256_file(PROTOCOL),
            "protocol_freeze_sha256": sha256_file(PROTOCOL_FREEZE),
            "repair_protocol_sha256": sha256_file(REPAIR_PROTOCOL),
            "r2_protocol_freeze_sha256": sha256_file(R2_PROTOCOL_FREEZE),
            "r2_implementation_freeze_sha256": sha256_file(R2_IMPLEMENTATION_FREEZE),
            "recovery_protocol_sha256": sha256_file(RECOVERY_PROTOCOL),
            "r3_protocol_freeze_sha256": sha256_file(R3_PROTOCOL_FREEZE),
            "r3_implementation_freeze_sha256": sha256_file(R3_IMPLEMENTATION_FREEZE),
            "tmp_repair_protocol_sha256": sha256_file(TMP_REPAIR_PROTOCOL),
            "r4_protocol_freeze_sha256": sha256_file(R4_PROTOCOL_FREEZE),
            "implementation_freeze_sha256": sha256_file(IMPLEMENTATION_FREEZE),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "tests_sha256": sha256_file(TEST_FILE),
            "static_audit": static,
            "temporary_root_preserved": str(temp_root),
            "network_access_attempted": False,
            "proxy_variables_empty": len(check_proxy_environment()) == 0,
            "resource_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "g0_n_jobs": 4,
        }
        stable_json(AUDIT_ROOT / "provenance.json", provenance)
        write_manifest(AUDIT_ROOT)
        print(f"completed: {acceptance['status']} -> {AUDIT_ROOT}", flush=True)
        return acceptance
    except BaseException as error:
        failure = {
            "status": "INVALID_RUN_STOP",
            "phase": current_phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "temporary_root_preserved": str(temp_root),
            "formal_staging_preserved": {
                "audit_root": str(AUDIT_ROOT),
                "g0_root": str(G0_ROOT_A),
                "science_root": str(SCIENCE_ROOT_A),
            },
        }
        stable_json(AUDIT_ROOT / "FAILED.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-repair-protocol-sha256", required=True)
    parser.add_argument("--expected-recovery-protocol-sha256", required=True)
    parser.add_argument("--expected-tmp-repair-protocol-sha256", required=True)
    parser.add_argument("--expected-implementation-freeze-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-no-fit", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    args = parser.parse_args(argv)
    effective_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    try:
        if args.preflight_no_fit:
            audit = validate_static(
                args.expected_protocol_sha256,
                args.expected_repair_protocol_sha256,
                args.expected_recovery_protocol_sha256,
                args.expected_tmp_repair_protocol_sha256,
                args.expected_implementation_freeze_sha256,
                require_output_absence=True,
            )
            print(json.dumps({"status": "PREFLIGHT_NO_FIT_PASS", **audit}, ensure_ascii=False, sort_keys=True))
        else:
            run_all(
                args.expected_protocol_sha256,
                args.expected_repair_protocol_sha256,
                args.expected_recovery_protocol_sha256,
                args.expected_tmp_repair_protocol_sha256,
                args.expected_implementation_freeze_sha256,
                effective_argv,
            )
    except (PipelineError, m1.M1Error, m1r.M1RError, m2.M2Error, AssertionError) as error:
        print(f"STRICT59_RA_ECMDM STOPPED: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
