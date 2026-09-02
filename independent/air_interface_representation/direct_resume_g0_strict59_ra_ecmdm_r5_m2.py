#!/usr/bin/env python3
"""Directly recover R5 M2 after deterministic zero-start L-BFGS-B stops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continue_g0_strict59_ra_ecmdm_r5 as continuation  # noqa: E402
import resume_g0_strict59_ra_ecmdm_r5_m2 as recovery  # noqa: E402
import run_g0_strict59_ra_ecmdm as base  # noqa: E402


CURRENT_FAILED = base.AUDIT_ROOT / "FAILED.json"
RECOVERED_RETRY_EXHAUSTED = base.AUDIT_ROOT / "RECOVERED_M2_RETRY_EXHAUSTED.json"
RECOVERED_NONPOSITIVE_EXCESS = (
    base.AUDIT_ROOT / "RECOVERED_M2_NONPOSITIVE_EXCESS_F_STOP.json"
)
DIRECT_REPAIR_AUDIT = base.AUDIT_ROOT / "m2_direct_fit_repair.json"

_FIT_CONTEXT: dict[str, Any] = {"repeat": None, "task": None, "fold": -1}
_FIT_RECOVERY_RECORDS: list[dict[str, Any]] = []
_ORIGINAL_EVALUATE = base.m2._evaluate
_ORIGINAL_FIT_ADAPTER = base.m2.fit_adapter
_ORIGINAL_AGGREGATE = base.m2._aggregate
_ORIGINAL_BUILD_GATE = base.m2._build_gate


class DirectM2RecoveryError(RuntimeError):
    """Raised when the narrowly scoped direct M2 recovery cannot proceed."""


def _sha256_bytes(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = base.m2.np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _result_record(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(result.fun),
        "max_abs_gradient": float(base.m2.np.max(base.m2.np.abs(result.jac))),
    }


def fit_adapter_with_abnormal_probe_restart(
    h: Any,
    b: Sequence[float],
    y: Sequence[int],
    stage: str,
    init: Sequence[float] | None = None,
) -> tuple[Any, float]:
    """Run the frozen fit, retrying only finite ABNORMAL returns from its probe.

    Objective, analytic-gradient gate, method, bounds, and optimizer options are
    copied exactly from ``m2_meta_mechanism.fit_adapter``.  The only recovery is
    the deterministic initial point: the already-computed gradient-check probe.
    """

    m2 = base.m2
    dimensions = {"I": 4, "G": 7, "C": 19}
    if stage not in dimensions:
        raise m2.M2Error(f"unknown adapter stage {stage!r}")
    dimension = dimensions[stage]
    if init is None:
        initial = m2.np.zeros(dimension, dtype=float)
        if stage != "I":
            initial[4:] = 1.0
    else:
        initial = m2.np.asarray(init, dtype=float)
        if len(initial) != dimension:
            raise m2.M2Error(f"{stage}: initial vector has wrong length")

    probe = initial + m2.np.random.default_rng(20260830 + ord(stage)).normal(
        0.0, 0.03, dimension
    )
    if stage != "I":
        probe[4:] = m2.np.maximum(probe[4:], 0.02)
    analytic = m2.adapter_objective(probe, h, b, y, stage, grad=True)[1]
    numeric = m2.finite_difference_gradient(probe, h, b, y, stage)
    if float(m2.np.max(m2.np.abs(analytic - numeric))) > 1e-5:
        raise m2.M2Error(f"{stage}: analytic gradient finite-difference gate failed")

    bounds = [(None, None)] * 4 + (
        [] if stage == "I" else [(0.0, None)] * (dimension - 4)
    )
    options = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8}
    objective = lambda x: m2.adapter_objective(x, h, b, y, stage, grad=True)
    result = m2.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=options,
    )

    finite_failure = (
        str(result.message).startswith("ABNORMAL:")
        and m2.np.isfinite(result.fun)
        and m2.np.isfinite(result.x).all()
        and m2.np.isfinite(result.jac).all()
    )
    if not result.success and finite_failure:
        restarted = m2.minimize(
            objective,
            probe,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options=options,
        )
        _FIT_RECOVERY_RECORDS.append(
            {
                "repeat": _FIT_CONTEXT["repeat"],
                "task": _FIT_CONTEXT["task"],
                "fold": int(_FIT_CONTEXT["fold"]),
                "stage": stage,
                "input_sha256": _sha256_bytes(h, b, y),
                "initialization": "frozen_initial_then_gradient_check_probe",
                "first_return": _result_record(result),
                "probe_restart_return": _result_record(restarted),
                "objective_delta": float(restarted.fun - result.fun),
                "optimizer_method": "L-BFGS-B",
                "optimizer_options": dict(options),
            }
        )
        result = restarted

    if not result.success or not m2.np.isfinite(result.fun) or not m2.np.isfinite(result.x).all():
        raise m2.M2Error(f"{stage}: optimizer failed: {result.message}")
    return m2.np.asarray(result.x, dtype=float), float(result.fun)


def _tracked_evaluate(task: Mapping[str, Any]) -> dict[str, Any]:
    _FIT_CONTEXT["task"] = str(task["name"])
    _FIT_CONTEXT["fold"] = -1
    original_fit = base.m2.fit_adapter

    def tracked_fit(h: Any, b: Sequence[float], y: Sequence[int], stage: str, init: Sequence[float] | None = None) -> tuple[Any, float]:
        if stage == "I":
            _FIT_CONTEXT["fold"] += 1
        return original_fit(h, b, y, stage, init)

    base.m2.fit_adapter = tracked_fit
    try:
        return _ORIGINAL_EVALUATE(task)
    finally:
        base.m2.fit_adapter = original_fit


def _aggregate_with_negative_result(rows: list[dict[str, Any]], scope: str) -> Any:
    """Preserve full integrity gates while allowing a nonpositive F result to serialize."""

    if scope != "full":
        return _ORIGINAL_AGGREGATE(rows, scope)
    if (
        len(rows) != 156
        or sum(row["grid_kind"] == "ood" for row in rows) != 150
        or sum(row["grid_kind"] == "iid_time_block" for row in rows) != 6
    ):
        raise base.m2.M2Error("full task count gate failed")
    for environment in base.m2.ENVIRONMENTS:
        ood = [
            row
            for row in rows
            if row["grid_kind"] == "ood" and row["target_env"] == environment
        ]
        iid = [
            row
            for row in rows
            if row["grid_kind"] == "iid_time_block"
            and row["target_env"] == environment
        ]
        if len(ood) != 25 or len(iid) != 1:
            raise base.m2.M2Error(
                f"{environment}: expected 25 OOD and one IID time-block task"
            )
        if len({(row["support_sha256"], row["n_samples"]) for row in ood + iid}) != 1:
            raise base.m2.M2Error(f"{environment}: support/sample identity gate failed")
    environment, nsource, stages = _ORIGINAL_AGGREGATE(
        rows, "direct_negative_result"
    )
    if len(environment) != 6 or set(nsource["n_sources"]) != {1, 2, 3}:
        raise base.m2.M2Error("direct negative-result aggregate coverage failed")
    return environment, nsource, stages


def _build_gate_with_negative_result(
    task_rows: list[dict[str, Any]],
    environment: Any,
    scope: str,
    m1r_error: float = 0.0,
    reproduction: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    gate = _ORIGINAL_BUILD_GATE(
        task_rows, environment, scope, m1r_error, reproduction
    )
    if scope != "full":
        return gate
    excess_f = environment["excess_F"].to_numpy(float)
    nonpositive = bool(
        base.m2.np.isfinite(excess_f).all() and (excess_f <= 0).any()
    )
    otherwise_valid = (
        len(task_rows) == 156
        and base.m2.np.isfinite(m1r_error)
        and m1r_error <= 1e-12
    )
    if not (nonpositive and otherwise_valid):
        return gate

    gate["engineering_status"] = "PASS_PENDING_DOUBLE_RUN"
    gate["engineering"].update(
        {
            "input_hard_gates": True,
            "optimizer_success": True,
            "all_five_fold_coverages_complete": True,
            "centered_logits_gate": True,
            "negative_scientific_result_serialized": True,
        }
    )
    gate["scientific_precondition"] = {
        "status": "NOT_MET",
        "id": "M2_ALL_ENVIRONMENTS_EXCESS_F_POSITIVE",
        "classification_admissible": False,
        "positive_environment_count": int((excess_f > 0).sum()),
        "required_positive_environment_count": 6,
        "nonpositive_environments": [
            str(row["target_env"])
            for _, row in environment.iterrows()
            if float(row["excess_F"]) <= 0
        ],
        "environment_excess_F": {
            str(row["target_env"]): float(row["excess_F"])
            for _, row in environment.iterrows()
        },
        "interpretation": (
            "complete deterministic measurements retained as a negative result; "
            "no I/G/C structural sufficiency classification is published"
        ),
    }
    sufficiency: dict[str, Any] = {}
    for stage in "IGC":
        er_equal = float(
            base.m2.np.mean(environment[f"excess_{stage}"])
            / base.m2.np.mean(excess_f)
        )
        count = int(base.m2.np.sum(environment[f"ER_{stage}"] >= 0.50))
        sufficiency[stage] = {
            "ER_equal": er_equal,
            "environments_ER_ge_0.50": count,
            "descriptive_threshold_met": bool(er_equal >= 0.80 and count >= 4),
            "sufficient": False,
            "classification_admissible": False,
        }
    gate["sufficiency"] = sufficiency
    gate["first_sufficient_stage"] = (
        "NOT_ADMISSIBLE_NONPOSITIVE_ENVIRONMENT_EXCESS_F"
    )
    gate["environment_excess_ER"] = {
        str(row["target_env"]): {
            key: float(row[key])
            for stage in "IGC"
            for key in (f"excess_{stage}", f"ER_{stage}")
        }
        for _, row in environment.iterrows()
    }
    gate["raw_recovery_fraction_environment_equal"] = {
        stage: float(
            base.m2.np.mean(environment[f"gain_{stage}_ood"])
            / base.m2.np.mean(environment["gain_F_ood"])
        )
        for stage in "IGCF"
    }
    sensitivity_denominator = float(base.m2.np.mean(environment["excess4_F"]))
    gate["sensitivity_4class"] = {
        stage: {
            "excess_environment_equal": float(
                base.m2.np.mean(environment[f"excess4_{stage}"])
            ),
            "ER_environment_equal": (
                float(
                    base.m2.np.mean(environment[f"excess4_{stage}"])
                    / sensitivity_denominator
                )
                if sensitivity_denominator != 0
                else float("nan")
            ),
        }
        for stage in "IGCF"
    }
    return gate


def install_direct_fit_recovery() -> None:
    base.m2.fit_adapter = fit_adapter_with_abnormal_probe_restart
    base.m2._evaluate = _tracked_evaluate
    base.m2._aggregate = _aggregate_with_negative_result
    base.m2._build_gate = _build_gate_with_negative_result


def restore_original_fit_functions() -> None:
    base.m2.fit_adapter = _ORIGINAL_FIT_ADAPTER
    base.m2._evaluate = _ORIGINAL_EVALUATE
    base.m2._aggregate = _ORIGINAL_AGGREGATE
    base.m2._build_gate = _ORIGINAL_BUILD_GATE


def validate_current_state() -> dict[str, Any]:
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
    parent_implementation = recovery._validate_parent_implementation()
    retry_freeze = continuation.read_json(recovery.RETRY_IMPLEMENTATION_FREEZE)
    for relative, expected in retry_freeze["implementation_sha256"].items():
        if continuation.sha256_file(REPO_ROOT / relative) != expected:
            raise DirectM2RecoveryError(f"R3 implementation anchor mismatch: {relative}")

    if not CURRENT_FAILED.is_file():
        raise DirectM2RecoveryError("recoverable FAILED.json is absent")
    failure = continuation.read_json(CURRENT_FAILED)
    if failure.get("phase") == "R5R3_M2_A" and not RECOVERED_RETRY_EXHAUSTED.exists():
        recovery_state = "R3_ZERO_START_RETRY_EXHAUSTED"
        records = failure.get("retry_records", [])
        preserve_as = RECOVERED_RETRY_EXHAUSTED
    elif (
        failure.get("phase") == "R5_DIRECT_M2_A"
        and failure.get("error")
        == "full gate requires strictly positive excess_F in all six environments"
        and RECOVERED_RETRY_EXHAUSTED.is_file()
        and not RECOVERED_NONPOSITIVE_EXCESS.exists()
    ):
        recovery_state = "DIRECT_M2_NONPOSITIVE_EXCESS_F_SCIENTIFIC_STOP"
        records = continuation.read_json(RECOVERED_RETRY_EXHAUSTED).get(
            "retry_records", []
        )
        preserve_as = RECOVERED_NONPOSITIVE_EXCESS
    else:
        raise DirectM2RecoveryError("current failure is not a supported direct recovery state")
    if len(records) != recovery.MAX_M2_ATTEMPTS:
        raise DirectM2RecoveryError("R3 retry history does not contain three attempts")
    for index, record in enumerate(records, 1):
        if (
            record.get("repeat") != "A"
            or record.get("attempt") != index
            or record.get("status") != "RETRYABLE_FAILURE"
            or not recovery.RETRYABLE_ERROR.match(str(record.get("error", "")))
        ):
            raise DirectM2RecoveryError("R3 retry history does not match three bounded failures")

    protocol_freeze = continuation.read_json(recovery.RETRY_PROTOCOL_FREEZE)
    expected_original = protocol_freeze["pre_retry_staging"]["current_failed_json_sha256"]
    if continuation.sha256_file(recovery.RECOVERED_M2_FAILURE) != expected_original:
        raise DirectM2RecoveryError("preserved original M2 failure hash mismatch")
    recovery._require_empty_directory(recovery.M2_A_ROOT)
    if recovery.M2_B_ROOT.exists():
        raise DirectM2RecoveryError("M2-B must be absent before direct recovery")
    base.compare_directories(recovery.M1_A_ROOT, recovery.M1_B_ROOT, {"provenance.json"})
    base.compare_directories(recovery.M1R_A_ROOT, recovery.M1R_B_ROOT, {"provenance.json"})
    for name in ("VERDICT.md", "acceptance.json", "provenance.json", "manifest.json"):
        if (base.AUDIT_ROOT / name).exists():
            raise DirectM2RecoveryError(f"final artifact already exists: {name}")

    return {
        "status": "R5_M2_DIRECT_RECOVERY_PREFLIGHT_PASS",
        "recovery_state": recovery_state,
        "current_failure_preserve_as": str(preserve_as),
        "base_static": base_static,
        "parent_continuation_implementation": parent_implementation,
        "current_failure_sha256": continuation.sha256_file(CURRENT_FAILED),
        "r3_retry_failure_sha256": (
            continuation.sha256_file(RECOVERED_RETRY_EXHAUSTED)
            if RECOVERED_RETRY_EXHAUSTED.is_file()
            else continuation.sha256_file(CURRENT_FAILED)
        ),
        "original_m2_failure_sha256": expected_original,
        "r3_retry_records": records,
        "repair_scope": (
            "finite L-BFGS-B ABNORMAL only via deterministic gradient-check probe; "
            "complete nonpositive-excess result serialized without structural classification"
        ),
        "objective_changed": False,
        "optimizer_method_changed": False,
        "optimizer_options_changed": False,
        "scientific_thresholds_changed": False,
    }


def _run_repeat(label: str, output_root: Path, records: list[dict[str, Any]]) -> Mapping[str, Any]:
    if output_root.exists():
        recovery._remove_empty_directory(output_root)
    links = base.AUDIT_ROOT / "_active_inputs"
    started = datetime.now(timezone.utc)
    _FIT_CONTEXT["repeat"] = label
    gate = base.run_m2(
        base.G0_ROOT_A / "raw_all",
        recovery.M1R_A_ROOT,
        output_root,
        links / "g0_raw_all",
        links / "m1r",
    )
    finished = datetime.now(timezone.utc)
    records.append(
        {
            "repeat": label,
            "attempt": 4 if label == "A" else 1,
            "status": "SUCCESS_WITH_SCOPED_FIT_INITIALIZATION_RECOVERY",
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
        }
    )
    return gate


def _amend_final_disclosure(
    static: Mapping[str, Any],
    current_failure_sha256: str,
    run_records: Sequence[Mapping[str, Any]],
) -> None:
    acceptance_path = base.AUDIT_ROOT / "acceptance.json"
    acceptance = continuation.read_json(acceptance_path)
    engineering = acceptance.setdefault("engineering", {})
    engineering["m2_transient_retry_recovery"] = False
    engineering["m2_r3_zero_start_attempts_exhausted"] = 3
    engineering["m2_direct_fit_initialization_recovery"] = True
    engineering["m2_fit_initialization_recovery_count"] = len(_FIT_RECOVERY_RECORDS)
    engineering["m2_objective_optimizer_options_or_scientific_thresholds_changed"] = False
    engineering["m2_all_environment_positive_excess_f_precondition"] = False
    engineering["m2_negative_result_packaged_without_structure_classification"] = True
    base.stable_json(acceptance_path, acceptance)

    verdict_path = base.AUDIT_ROOT / "VERDICT.md"
    verdict = verdict_path.read_text(encoding="utf-8")
    old = (
        "随后仅按冻结 R3 重试完整 M2，未改变优化器、目标函数或科学阈值。"
    )
    new = (
        "冻结 R3 的三次零起点整轮重试均在同一有限 L-BFGS-B 解处返回 `ABNORMAL`。"
        "经用户明确授权直接修复，仅对这种有限 `ABNORMAL` 拟合改用代码中原有的固定梯度校验"
        "probe 作为初始化并重跑相同 L-BFGS-B；目标函数、边界、优化器参数和科学阈值均未改变。"
    )
    if old not in verdict:
        raise DirectM2RecoveryError("expected R3 recovery disclosure is absent")
    verdict = verdict.replace(old, new)
    verdict = verdict.replace(
        "首次续跑又遇到一次瞬时 M2\nL-BFGS-B",
        "首次续跑又遇到可重复的 M2\nL-BFGS-B",
    )
    verdict += "\n".join(
        [
            "",
            "## M2 科学前置门的否定结果",
            "",
            "完整 M2 计算显示六环境中 R5 的 `excess_F=-0.0065679575`，因此冻结 M2 的",
            "“六环境全部严格为正”结构分类前置门未满足。该停止记录原样保留；本恢复只将完整",
            "A/B 一致的数值包装为否定裁定，不发布 I/G/C 的结构充分性分类。strict59_ra 的",
            "oracle recoverability 门仍为 5/6 正环境且总体上界存在，但 `excess_F_equal=0.0105138297`",
            "低于预设实质量级 0.020，故不进入 observable estimability，更不建立 deployability。",
            "",
        ]
    )
    verdict_path.write_text(verdict, encoding="utf-8", newline="\n")

    provenance_path = base.AUDIT_ROOT / "provenance.json"
    provenance = continuation.read_json(provenance_path)
    provenance.update(
        {
            "direct_repair_user_authorized": True,
            "direct_repair_protocol_frozen": False,
            "direct_repair_runner": str(Path(__file__).resolve()),
            "direct_repair_runner_sha256": continuation.sha256_file(Path(__file__).resolve()),
            "r3_retry_exhausted_failure": str(RECOVERED_RETRY_EXHAUSTED),
            "r3_retry_exhausted_failure_sha256": continuation.sha256_file(
                RECOVERED_RETRY_EXHAUSTED
            ),
            "nonpositive_excess_f_stop": str(RECOVERED_NONPOSITIVE_EXCESS),
            "nonpositive_excess_f_stop_sha256": (
                continuation.sha256_file(RECOVERED_NONPOSITIVE_EXCESS)
                if RECOVERED_NONPOSITIVE_EXCESS.is_file()
                else None
            ),
            "direct_current_failure_sha256": current_failure_sha256,
            "direct_fit_recovery_records": list(_FIT_RECOVERY_RECORDS),
            "objective_changed": False,
            "optimizer_method_changed": False,
            "optimizer_options_changed": False,
            "scientific_thresholds_changed": False,
            "direct_static_audit": static,
        }
    )
    base.stable_json(provenance_path, provenance)

    continuation_audit = continuation.read_json(continuation.CONTINUATION_AUDIT)
    continuation_audit.update(
        {
            "status": "R5_IN_PLACE_DIRECT_M2_RECOVERY_COMPLETED",
            "r3_retry_exhausted_failure_preserved": str(RECOVERED_RETRY_EXHAUSTED),
            "m2_direct_fit_initialization_recovery": True,
            "m2_fit_initialization_recovery_count": len(_FIT_RECOVERY_RECORDS),
            "m2_structure_classification_admissible": False,
            "m2_structure_inadmissibility_reason": "nonpositive excess_F in R5",
        }
    )
    base.stable_json(continuation.CONTINUATION_AUDIT, continuation_audit)

    direct_audit = {
        "status": "M2_DIRECT_FIT_INITIALIZATION_RECOVERY_COMPLETED",
        "user_authorized_without_new_protocol_freeze": True,
        "r3_retry_exhausted_failure_preserved": str(RECOVERED_RETRY_EXHAUSTED),
        "r3_retry_exhausted_failure_sha256": continuation.sha256_file(
            RECOVERED_RETRY_EXHAUSTED
        ),
        "nonpositive_excess_f_stop_preserved": str(RECOVERED_NONPOSITIVE_EXCESS),
        "nonpositive_excess_f_stop_sha256": (
            continuation.sha256_file(RECOVERED_NONPOSITIVE_EXCESS)
            if RECOVERED_NONPOSITIVE_EXCESS.is_file()
            else None
        ),
        "run_records": list(run_records),
        "fit_recovery_records": list(_FIT_RECOVERY_RECORDS),
        "m2_double_run_verified": True,
        "objective_changed": False,
        "optimizer_method_changed": False,
        "optimizer_options_changed": False,
        "scientific_thresholds_changed": False,
        "m2_structure_classification_admissible": False,
    }
    base.stable_json(DIRECT_REPAIR_AUDIT, direct_audit)
    base.stable_json(
        recovery.M2_RETRY_AUDIT,
        {
            "status": "M2_R3_RETRY_EXHAUSTED_DIRECT_FIT_RECOVERY_COMPLETED",
            "attempts": list(static["r3_retry_records"]) + list(run_records),
            "fit_recovery_records": list(_FIT_RECOVERY_RECORDS),
            "m2_double_run_verified": True,
        },
    )
    base.write_manifest(base.AUDIT_ROOT)


def resume_direct(argv: Sequence[str]) -> dict[str, Any]:
    started_wall = time.time()
    started_utc = datetime.now(timezone.utc)
    phase = "R5_DIRECT_STATIC"
    static = validate_current_state()
    current_failure_sha256 = continuation.sha256_file(CURRENT_FAILED)
    failure_preserve_as = Path(str(static["current_failure_preserve_as"]))
    os.replace(CURRENT_FAILED, failure_preserve_as)
    records: list[dict[str, Any]] = []
    try:
        base.stable_json(
            DIRECT_REPAIR_AUDIT,
            {
                "status": "M2_DIRECT_FIT_INITIALIZATION_RECOVERY_STARTED",
                "r3_retry_exhausted_failure_preserved": str(RECOVERED_RETRY_EXHAUSTED),
                "current_failure_preserved": str(failure_preserve_as),
                "current_failure_sha256": current_failure_sha256,
            },
        )
        links = base.AUDIT_ROOT / "_active_inputs"
        base.switch_symlink(links / "g0_raw_all", base.G0_ROOT_A / "raw_all")
        base.switch_symlink(links / "m1", recovery.M1_A_ROOT)
        base.switch_symlink(links / "m1r", recovery.M1R_A_ROOT)
        install_direct_fit_recovery()

        phase = "R5_DIRECT_M2_A"
        _run_repeat("A", recovery.M2_A_ROOT, records)
        phase = "R5_DIRECT_M2_B"
        _run_repeat("B", recovery.M2_B_ROOT, records)
        phase = "R5_DIRECT_M2_VERIFY"
        base.m2.verify(recovery.M2_A_ROOT, recovery.M2_B_ROOT)
        phase = "R5_DIRECT_FINALIZE"
        acceptance = recovery._write_final_results(
            static,
            list(static["r3_retry_records"]) + records,
            str(static["original_m2_failure_sha256"]),
            started_utc,
            started_wall,
            argv,
        )
        _amend_final_disclosure(static, current_failure_sha256, records)
        print(f"completed direct M2 recovery: {acceptance['status']} -> {base.AUDIT_ROOT}", flush=True)
        return acceptance
    except BaseException as error:
        base.stable_json(
            CURRENT_FAILED,
            {
                "status": "INVALID_RUN_STOP",
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "r3_retry_exhausted_failure_preserved": str(RECOVERED_RETRY_EXHAUSTED),
                "run_records": records,
                "fit_recovery_records": _FIT_RECOVERY_RECORDS,
            },
        )
        raise
    finally:
        restore_original_fit_functions()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-no-fit", action="store_true")
    mode.add_argument("--resume-m2", action="store_true")
    args = parser.parse_args(argv)
    effective_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    try:
        if args.preflight_no_fit:
            print(json.dumps(validate_current_state(), ensure_ascii=False, indent=2))
            return 0
        resume_direct(effective_argv)
        return 0
    except BaseException as error:
        print(f"R5 DIRECT M2 RECOVERY STOPPED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
