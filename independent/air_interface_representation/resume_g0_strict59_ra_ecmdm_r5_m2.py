#!/usr/bin/env python3
"""Resume R5 at M2 with a bounded retry for transient L-BFGS-B ABNORMAL stops."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continue_g0_strict59_ra_ecmdm_r5 as continuation  # noqa: E402
import run_g0_strict59_ra_ecmdm as base  # noqa: E402


RETRY_PROTOCOL = (
    HERE / "PROTOCOL_G0_STRICT59_RA_ECMDM_R5_CONTINUATION_M2_RETRY_R3_20260903.md"
)
RETRY_PROTOCOL_FREEZE = HERE / "G0_STRICT59_RA_ECMDM_R5_CONTINUATION_R3_PROTOCOL_FREEZE.json"
RETRY_IMPLEMENTATION_FREEZE = (
    HERE / "G0_STRICT59_RA_ECMDM_R5_CONTINUATION_R3_IMPLEMENTATION_FREEZE.json"
)
RETRY_TEST_FILE = HERE / "test_resume_g0_strict59_ra_ecmdm_r5_m2.py"

CURRENT_FAILED = base.AUDIT_ROOT / "FAILED.json"
RECOVERED_M2_FAILURE = base.AUDIT_ROOT / "RECOVERED_M2_OPTIMIZER_FAILURE.json"
M2_RETRY_AUDIT = base.AUDIT_ROOT / "m2_retry_recovery.json"
M1_A_ROOT = base.SCIENCE_ROOT_A / "m1"
M1_B_ROOT = continuation.TEMP_ROOT / "m1_b"
M1R_A_ROOT = base.SCIENCE_ROOT_A / "m1r"
M1R_B_ROOT = continuation.TEMP_ROOT / "m1r_b"
M2_A_ROOT = base.SCIENCE_ROOT_A / "m2"
M2_B_ROOT = continuation.TEMP_ROOT / "m2_b"

MAX_M2_ATTEMPTS = 3
RETRYABLE_ERROR = re.compile(r"^(I|G|C): optimizer failed: ABNORMAL:")


class M2RetryRecoveryError(RuntimeError):
    """Raised when the frozen M2 retry recovery contract is violated."""


def _count_regular_files(root: Path) -> int:
    return sum(path.is_file() and not path.is_symlink() for path in root.rglob("*"))


def _require_empty_directory(path: Path) -> None:
    if not path.is_dir() or any(path.iterdir()):
        raise M2RetryRecoveryError(f"M2 retry directory is absent or non-empty: {path}")


def _remove_empty_directory(path: Path) -> None:
    _require_empty_directory(path)
    path.rmdir()


def _validate_parent_implementation() -> dict[str, Any]:
    record = continuation.read_json(continuation.CONTINUATION_IMPLEMENTATION_FREEZE)
    for relative, expected in record["implementation_sha256"].items():
        if continuation.sha256_file(REPO_ROOT / relative) != expected:
            raise M2RetryRecoveryError(f"parent continuation implementation mismatch: {relative}")
    return record


def validate_static(
    expected_retry_protocol_sha256: str,
    expected_retry_protocol_freeze_sha256: str,
    expected_retry_implementation_freeze_sha256: str,
) -> dict[str, Any]:
    base.ensure_proxy_gate()
    base_static = base.validate_static(
        continuation.PARENT_PROTOCOL_SHA256,
        continuation.R2_REPAIR_PROTOCOL_SHA256,
        continuation.R3_RECOVERY_PROTOCOL_SHA256,
        continuation.R4_REPAIR_PROTOCOL_SHA256,
        continuation.R5_ISOLATION_PROTOCOL_SHA256,
        continuation.R5_IMPLEMENTATION_FREEZE_SHA256,
        require_output_absence=False,
    )
    parent_implementation = _validate_parent_implementation()
    retry_protocol_hash = continuation.sha256_file(RETRY_PROTOCOL)
    retry_protocol_freeze_hash = continuation.sha256_file(RETRY_PROTOCOL_FREEZE)
    retry_implementation_freeze_hash = continuation.sha256_file(RETRY_IMPLEMENTATION_FREEZE)
    if retry_protocol_hash != expected_retry_protocol_sha256:
        raise M2RetryRecoveryError("CLI retry protocol SHA-256 mismatch")
    if retry_protocol_freeze_hash != expected_retry_protocol_freeze_sha256:
        raise M2RetryRecoveryError("CLI retry protocol-freeze SHA-256 mismatch")
    if retry_implementation_freeze_hash != expected_retry_implementation_freeze_sha256:
        raise M2RetryRecoveryError("CLI retry implementation-freeze SHA-256 mismatch")

    protocol_freeze = continuation.read_json(RETRY_PROTOCOL_FREEZE)
    if protocol_freeze["retry_protocol"]["sha256"] != retry_protocol_hash:
        raise M2RetryRecoveryError("retry protocol freeze record mismatch")
    if (
        protocol_freeze["parent_continuation_implementation_freeze_sha256"]
        != continuation.sha256_file(continuation.CONTINUATION_IMPLEMENTATION_FREEZE)
    ):
        raise M2RetryRecoveryError("retry protocol parent implementation mismatch")
    implementation_freeze = continuation.read_json(RETRY_IMPLEMENTATION_FREEZE)
    if implementation_freeze["retry_protocol_sha256"] != retry_protocol_hash:
        raise M2RetryRecoveryError("retry implementation parent protocol mismatch")
    if implementation_freeze["retry_protocol_freeze_sha256"] != retry_protocol_freeze_hash:
        raise M2RetryRecoveryError("retry implementation parent freeze mismatch")
    for relative, expected in implementation_freeze["implementation_sha256"].items():
        if continuation.sha256_file(REPO_ROOT / relative) != expected:
            raise M2RetryRecoveryError(f"retry implementation anchor mismatch: {relative}")

    frozen = protocol_freeze["pre_retry_staging"]
    anchors = {
        "current_failed_json_sha256": continuation.sha256_file(CURRENT_FAILED),
        "original_s2_failure_sha256": continuation.sha256_file(
            continuation.RECOVERED_FAILURE
        ),
        "g0_double_run_verification_sha256": continuation.sha256_file(
            base.AUDIT_ROOT / "g0_double_run_verification.json"
        ),
        "continuation_repair_sha256": continuation.sha256_file(
            continuation.CONTINUATION_AUDIT
        ),
        "m1_gate_sha256": continuation.sha256_file(M1_A_ROOT / "m1_gate.json"),
        "m1r_gate_sha256": continuation.sha256_file(M1R_A_ROOT / "m1r_gate.json"),
        "m1r_double_run_verification_sha256": continuation.sha256_file(
            M1R_A_ROOT / "double_run_verification.json"
        ),
        "pre_retry_systemd_log_sha256": continuation.sha256_file(
            continuation.SYSTEMD_LOG
        ),
    }
    for key, actual in anchors.items():
        if actual != frozen[key]:
            raise M2RetryRecoveryError(f"frozen pre-retry staging mismatch: {key}")
    current_failure = continuation.read_json(CURRENT_FAILED)
    if current_failure.get("phase") != "R5C_S5_M2" or not RETRYABLE_ERROR.match(
        str(current_failure.get("error", ""))
    ):
        raise M2RetryRecoveryError("current failure is not the frozen retryable M2 failure")
    if RECOVERED_M2_FAILURE.exists():
        raise M2RetryRecoveryError("M2 failure has already been recovered")
    if _count_regular_files(M1_A_ROOT) != int(frozen["m1_file_count"]):
        raise M2RetryRecoveryError("M1-A file count mismatch")
    if _count_regular_files(M1R_A_ROOT) != int(frozen["m1r_file_count"]):
        raise M2RetryRecoveryError("M1-R-A file count mismatch")
    base.compare_directories(M1_A_ROOT, M1_B_ROOT, {"provenance.json"})
    base.compare_directories(M1R_A_ROOT, M1R_B_ROOT, {"provenance.json"})
    g0_verification = continuation.read_json(
        base.AUDIT_ROOT / "g0_double_run_verification.json"
    )
    if g0_verification.get("status") != "G0_BOUNDED_REPLICATE_STABILITY_PASS":
        raise M2RetryRecoveryError("G0 bounded stability release is not PASS")
    _require_empty_directory(M2_A_ROOT)
    if M2_B_ROOT.exists():
        raise M2RetryRecoveryError("M2-B root must be absent before retry")
    for name in ("VERDICT.md", "acceptance.json", "provenance.json", "manifest.json"):
        if (base.AUDIT_ROOT / name).exists():
            raise M2RetryRecoveryError(f"final artifact already exists: {name}")
    return {
        "base_static": base_static,
        "parent_continuation_implementation": parent_implementation,
        "retry_protocol_sha256": retry_protocol_hash,
        "retry_protocol_freeze_sha256": retry_protocol_freeze_hash,
        "retry_implementation_freeze_sha256": retry_implementation_freeze_hash,
        "staging_anchors": anchors,
    }


def run_m2_repeat_with_retry(
    label: str,
    output_root: Path,
    logical_g0: Path,
    logical_m1r: Path,
    records: list[dict[str, Any]] | None = None,
    runner: Callable[..., Mapping[str, Any]] = base.run_m2,
) -> Mapping[str, Any]:
    history = records if records is not None else []
    for attempt in range(1, MAX_M2_ATTEMPTS + 1):
        if output_root.exists():
            _remove_empty_directory(output_root)
        started = datetime.now(timezone.utc)
        try:
            gate = runner(
                base.G0_ROOT_A / "raw_all",
                M1R_A_ROOT,
                output_root,
                logical_g0,
                logical_m1r,
            )
        except base.m2.M2Error as error:
            finished = datetime.now(timezone.utc)
            record = {
                "repeat": label,
                "attempt": attempt,
                "status": "RETRYABLE_FAILURE" if RETRYABLE_ERROR.match(str(error)) else "NONRETRY_FAILURE",
                "error": str(error),
                "started_utc": started.isoformat(),
                "finished_utc": finished.isoformat(),
                "elapsed_seconds": (finished - started).total_seconds(),
            }
            history.append(record)
            if not RETRYABLE_ERROR.match(str(error)) or attempt >= MAX_M2_ATTEMPTS:
                raise
            _require_empty_directory(output_root)
            continue
        finished = datetime.now(timezone.utc)
        history.append(
            {
                "repeat": label,
                "attempt": attempt,
                "status": "SUCCESS",
                "started_utc": started.isoformat(),
                "finished_utc": finished.isoformat(),
                "elapsed_seconds": (finished - started).total_seconds(),
            }
        )
        return gate
    raise M2RetryRecoveryError(f"unreachable retry exhaustion for {label}")


def _write_final_results(
    static: Mapping[str, Any],
    retry_records: Sequence[Mapping[str, Any]],
    original_failure_sha256: str,
    started_utc: datetime,
    started_wall: float,
    argv: Sequence[str],
) -> dict[str, Any]:
    m1_gate = continuation.read_json(M1_A_ROOT / "m1_gate.json")
    m1r_gate = continuation.read_json(M1R_A_ROOT / "m1r_gate.json")
    m2_gate = continuation.read_json(M2_A_ROOT / "m2_gate.json")
    m1_hashes = base.compare_directories(M1_A_ROOT, M1_B_ROOT, {"provenance.json"})
    m1r_hashes = base.compare_directories(M1R_A_ROOT, M1R_B_ROOT, {"provenance.json"})
    m2_hashes = base.compare_directories(M2_A_ROOT, M2_B_ROOT, {"provenance.json"})
    canonical_g0 = base.G0_ROOT_A / "raw_all"
    base.stable_json(
        base.AUDIT_ROOT / "pipeline_double_run_verification.json",
        {
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
        },
    )
    passline, per_environment = base.adjudicate(m1_gate, m1r_gate, m2_gate)
    base.stable_json(base.AUDIT_ROOT / "oracle_passline.json", passline)
    base.stable_csv(per_environment, base.AUDIT_ROOT / "per_environment.csv")
    acceptance = continuation._build_acceptance(passline)
    acceptance["engineering"]["m2_transient_retry_recovery"] = True
    acceptance["engineering"]["m2_attempts"] = {
        label: sum(record["repeat"] == label for record in retry_records)
        for label in ("A", "B")
    }
    base.stable_json(base.AUDIT_ROOT / "acceptance.json", acceptance)

    full94 = continuation.read_json(base.AUDIT_ROOT / "full94_reproduction_gate.json")
    extraction = continuation.read_json(base.AUDIT_ROOT / "extraction_audit.json")
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
            "R5 曾在 S2 的 G0 逐字节哈希门停止，后按续跑前冻结的有界稳定性协议恢复；",
            "原记录保存在 `RECOVERED_S2_G0_VERIFY_FAILURE.json`。首次续跑又遇到一次瞬时 M2",
            "L-BFGS-B `ABNORMAL` 返回，记录保存在 `RECOVERED_M2_OPTIMIZER_FAILURE.json`；",
            "随后仅按冻结 R3 重试完整 M2，未改变优化器、目标函数或科学阈值。",
            "G0-B 只用于独立拟合稳定性审计；科学计算及其双跑均使用 canonical G0-A。",
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
        "m2_retry_started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "m2_retry_elapsed_seconds": time.time() - started_wall,
        "interpreter": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "package_versions": versions,
        "git_head": base._git_value(["rev-parse", "HEAD"]),
        "git_status_porcelain": base._git_value(["status", "--porcelain"]),
        "parent_protocol_sha256": continuation.sha256_file(base.PROTOCOL),
        "parent_r5_isolation_protocol_sha256": continuation.sha256_file(
            base.ISOLATION_PROTOCOL
        ),
        "parent_r5_implementation_freeze_sha256": continuation.sha256_file(
            base.IMPLEMENTATION_FREEZE
        ),
        "continuation_protocol_sha256": continuation.sha256_file(
            continuation.CONTINUATION_PROTOCOL
        ),
        "continuation_r2_protocol_sha256": continuation.sha256_file(
            continuation.CONTINUATION_R2_PROTOCOL
        ),
        "continuation_implementation_freeze_sha256": continuation.sha256_file(
            continuation.CONTINUATION_IMPLEMENTATION_FREEZE
        ),
        "m2_retry_protocol_sha256": continuation.sha256_file(RETRY_PROTOCOL),
        "m2_retry_protocol_freeze_sha256": continuation.sha256_file(RETRY_PROTOCOL_FREEZE),
        "m2_retry_implementation_freeze_sha256": continuation.sha256_file(
            RETRY_IMPLEMENTATION_FREEZE
        ),
        "m2_retry_runner_sha256": continuation.sha256_file(Path(__file__).resolve()),
        "m2_retry_tests_sha256": continuation.sha256_file(RETRY_TEST_FILE),
        "original_m2_failure_sha256": original_failure_sha256,
        "original_m2_failure_preserved": str(RECOVERED_M2_FAILURE),
        "m2_retry_records": list(retry_records),
        "static_audit": static,
        "g0_bounded_stability": continuation.read_json(
            base.AUDIT_ROOT / "g0_double_run_verification.json"
        ),
        "temporary_root_preserved": str(continuation.TEMP_ROOT),
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
        continuation.CONTINUATION_AUDIT,
        {
            "status": "R5_IN_PLACE_CONTINUATION_COMPLETED",
            "original_s2_failure_preserved": str(continuation.RECOVERED_FAILURE),
            "original_m2_failure_preserved": str(RECOVERED_M2_FAILURE),
            "g0_stability_status": "G0_BOUNDED_REPLICATE_STABILITY_PASS",
            "g0_byte_identical": False,
            "canonical_science_input": str(canonical_g0),
            "m2_retry_records": list(retry_records),
            "final_status": acceptance["status"],
        },
    )
    base.stable_json(
        M2_RETRY_AUDIT,
        {
            "status": "M2_TRANSIENT_RETRY_RECOVERED",
            "attempts": list(retry_records),
            "optimizer_or_scientific_parameters_changed": False,
            "m2_double_run_verified": True,
        },
    )
    base.write_manifest(base.AUDIT_ROOT)
    print(f"completed M2 retry recovery: {acceptance['status']} -> {base.AUDIT_ROOT}", flush=True)
    return acceptance


def resume_m2(
    expected_retry_protocol_sha256: str,
    expected_retry_protocol_freeze_sha256: str,
    expected_retry_implementation_freeze_sha256: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    started_wall = time.time()
    started_utc = datetime.now(timezone.utc)
    current_phase = "R5R3_STATIC"
    static = validate_static(
        expected_retry_protocol_sha256,
        expected_retry_protocol_freeze_sha256,
        expected_retry_implementation_freeze_sha256,
    )
    original_failure_sha256 = continuation.sha256_file(CURRENT_FAILED)
    os.replace(CURRENT_FAILED, RECOVERED_M2_FAILURE)
    retry_records: list[dict[str, Any]] = []
    try:
        base.stable_json(
            M2_RETRY_AUDIT,
            {
                "status": "M2_TRANSIENT_RETRY_STARTED",
                "attempts": retry_records,
                "original_failure_sha256": original_failure_sha256,
                "original_failure_preserved": str(RECOVERED_M2_FAILURE),
            },
        )
        links = base.AUDIT_ROOT / "_active_inputs"
        logical_g0 = links / "g0_raw_all"
        logical_m1 = links / "m1"
        logical_m1r = links / "m1r"
        base.switch_symlink(logical_g0, base.G0_ROOT_A / "raw_all")
        base.switch_symlink(logical_m1, M1_A_ROOT)
        base.switch_symlink(logical_m1r, M1R_A_ROOT)

        current_phase = "R5R3_M2_A"
        run_m2_repeat_with_retry(
            "A", M2_A_ROOT, logical_g0, logical_m1r, retry_records
        )
        current_phase = "R5R3_M2_B"
        run_m2_repeat_with_retry(
            "B", M2_B_ROOT, logical_g0, logical_m1r, retry_records
        )
        base.stable_json(
            M2_RETRY_AUDIT,
            {
                "status": "M2_REPEATS_COMPLETED_PENDING_VERIFY",
                "attempts": retry_records,
                "original_failure_sha256": original_failure_sha256,
                "original_failure_preserved": str(RECOVERED_M2_FAILURE),
            },
        )
        current_phase = "R5R3_M2_VERIFY"
        base.m2.verify(M2_A_ROOT, M2_B_ROOT)
        current_phase = "R5R3_FINALIZE"
        return _write_final_results(
            static,
            retry_records,
            original_failure_sha256,
            started_utc,
            started_wall,
            argv,
        )
    except BaseException as error:
        base.stable_json(
            CURRENT_FAILED,
            {
                "status": "INVALID_RUN_STOP",
                "phase": current_phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "original_s2_failure_preserved": str(continuation.RECOVERED_FAILURE),
                "original_m2_failure_preserved": str(RECOVERED_M2_FAILURE),
                "retry_records": retry_records,
                "temporary_root_preserved": str(continuation.TEMP_ROOT),
                "formal_staging_preserved": {
                    "audit_root": str(base.AUDIT_ROOT),
                    "g0_root": str(base.G0_ROOT_A),
                    "science_root": str(base.SCIENCE_ROOT_A),
                },
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-retry-protocol-sha256", required=True)
    parser.add_argument("--expected-retry-protocol-freeze-sha256", required=True)
    parser.add_argument("--expected-retry-implementation-freeze-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-no-fit", action="store_true")
    mode.add_argument("--resume-m2", action="store_true")
    args = parser.parse_args(argv)
    effective_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    try:
        if args.preflight_no_fit:
            static = validate_static(
                args.expected_retry_protocol_sha256,
                args.expected_retry_protocol_freeze_sha256,
                args.expected_retry_implementation_freeze_sha256,
            )
            print(
                json.dumps(
                    {"status": "R5_M2_RETRY_PREFLIGHT_PASS", "static": static},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        resume_m2(
            args.expected_retry_protocol_sha256,
            args.expected_retry_protocol_freeze_sha256,
            args.expected_retry_implementation_freeze_sha256,
            effective_argv,
        )
        return 0
    except BaseException as error:
        print(f"R5 M2 RETRY STOPPED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
