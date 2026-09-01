#!/usr/bin/env python3
"""Pure-synthetic tests for the D12/D13 UNSW Test 1 candidate.

This file intentionally does not discover, read, or execute any UNSW input.
The synthetic checks cover task counts, the two panel definitions, the fixed
label axis, D13's time split and OOF semantics, mainline primitive usage, the
same-shape CPD call, and the function-level label boundary.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unsw_test1 as U  # noqa: E402


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  PASS  {label}")


def synthetic_days() -> list[str]:
    return [f"d{index:02d}" for index in range(20)]


def synthetic_features() -> pd.DataFrame:
    # Ten stable devices, six categories, five windows per device/day.  The
    # panel function receives a lower threshold in this synthetic test; the
    # production constant remains exactly 100.
    device_category = {
        "AmazonEcho": "speaker",
        "BelkinWemoMotion": "sensor",
        "BelkinWemoSwitch": "switch",
        "Dropcam": "camera",
        "HPPrinter": "appliance",
        "NetatmoWeather": "sensor",
        "NetatmoWelcome": "camera",
        "SamsungSmartCam": "camera",
        "SmartThings": "hub",
        "TribySpeaker": "speaker",
    }
    rows: list[dict[str, object]] = []
    row_id = 0
    for day_index, day in enumerate(synthetic_days()):
        for device_index, device in enumerate(U.STABLE_DEVICES):
            for window_id in range(5):
                rows.append(
                    {
                        "device": device,
                        "day": day,
                        "category": device_category[device],
                        "label": device_category[device],
                        "window_id": window_id,
                        "window_start_epoch": float(day_index * 1000 + device_index * 10 + window_id),
                        "side_packet_ratio": 0.0,
                        "other_packet_ratio": 0.0,
                        "source_file": f"{day}.pcap",
                        "f0": float((device_index + window_id) % 7),
                        "f1": float(day_index),
                        "f2": float(row_id % 11),
                    }
                )
                row_id += 1
    return pd.DataFrame(rows)


def test_task_catalogue() -> None:
    tasks = U.build_task_catalog(synthetic_days())
    counts = U.validate_task_catalog(tasks, synthetic_days())
    check("task catalog has 54 OOD + 20 IID + 19 paired IID", counts == {
        "ood": 54,
        "iid": 20,
        "paired_day_iid": 19,
        "task_definitions": 74,
        "panel_arms": 2,
        "task_panel_cells": 148,
    })
    check("six-way shard assignment covers each task exactly once",
          sorted(task.name for i in range(6) for task in U.tasks_for_shard(tasks, i, 6))
          == sorted(task.name for task in tasks))


def test_feature_and_panel_rules() -> None:
    features = synthetic_features()
    audit = U.validate_feature_table(features)
    check("synthetic feature finite audit passes", audit["finite"])
    check("side/other audit columns are zero", audit["side_packet_ratio_zero"] and audit["other_packet_ratio_zero"])
    check(
        "side/other remain members of the frozen model feature schema",
        {"side_packet_ratio", "other_packet_ratio"}.issubset(U.numeric_feature_columns(features)),
    )
    task = U.build_task_catalog(synthetic_days())[2]  # first k=1 task after d00/d01/d02 ordering
    primary, _ = U.panel_devices(features, task, "primary", min_windows=5)
    stable, _ = U.panel_devices(features, task, "stable", min_windows=5)
    check("primary panel selects the ten six-class devices", primary == list(U.STABLE_DEVICES))
    check("stable panel is exactly the frozen ten devices", stable == list(U.STABLE_DEVICES))
    prepared = U.split_task(
        features,
        task,
        "primary",
        max_rows=20,
        random_state=42,
        min_windows=5,
    )
    check("train split separately has fixed six-class order", set(prepared.train.label) == set(U.CATEGORY_ORDER))
    check("test split separately has fixed six-class order", set(prepared.test.label) == set(U.CATEGORY_ORDER))
    check("D13 sample cap is recorded", prepared.detail["max_rows"] == 20)
    check("sample keys are label-free", all(set(key) == {"day", "device", "window_id"} for key in prepared.detail["train_sample_keys"]))


def test_iid_time_order() -> None:
    data = pd.DataFrame(
        {
            "device": ["a"] * 10,
            "window_start_epoch": [9, 1, 8, 0, 7, 2, 6, 3, 5, 4],
            "window_id": [9, 1, 8, 0, 7, 2, 6, 3, 5, 4],
        }
    )
    train, test = U.iid_time_split(data)
    check("IID first 70% is sorted by (epoch, window_id)", train["window_id"].tolist() == list(range(7)))
    check("IID last 30% is the later time block", test["window_id"].tolist() == [7, 8, 9])


def test_oof_semantics() -> None:
    features = synthetic_features()
    tasks = U.build_task_catalog(synthetic_days())
    one_day = U.split_task(features, tasks[0], "primary", max_rows=20, min_windows=5)
    one_day_folds = U.make_oof_folds(one_day.train, tasks[0])
    check("k=1 OOF uses time blocks", all(fold.mode == "time_block" for fold in one_day_folds))
    U.assert_mainline_stacking_fold_semantics(one_day.train, tasks[0], one_day_folds)
    check("k=1 persisted folds equal mainline splitter", True)

    # The first k=2 task is after the 53 k=1 tasks.
    two_day = U.split_task(features, tasks[19], "primary", max_rows=20, min_windows=5)
    two_day_folds = U.make_oof_folds(two_day.train, tasks[19])
    check("k=2 OOF uses training-day groups", all(fold.mode == "grouped_day" for fold in two_day_folds))
    check("grouped OOF validation days do not overlap training days",
          all(not (set(fold.train_days) & set(fold.val_days)) for fold in two_day_folds))
    U.assert_mainline_stacking_fold_semantics(two_day.train, tasks[19], two_day_folds)


def test_synthetic_rf_reference() -> None:
    features = synthetic_features()
    task = U.build_task_catalog(synthetic_days())[0]
    prepared = U.split_task(features, task, "primary", max_rows=20, min_windows=5)
    folds = U.make_oof_folds(prepared.train, task)
    cm, audit = U.rf_oof_reference_cm(
        prepared.train,
        ["f0", "f1", "f2"],
        task,
        folds,
        n_jobs=1,
    )
    check("synthetic RF OOF reference has fixed 6x6 shape", cm.shape == (6, 6))
    check("synthetic RF OOF probability rows sum to one", audit["row_sums_one"])
    check("synthetic RF OOF uses every validation row", int(cm.sum()) == len(prepared.train))


def test_probability_normalization_and_strict_audit() -> None:
    raw_float32 = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    copied_float64 = raw_float32.astype(np.float64)
    copied_error = float(abs(copied_float64.sum(axis=1)[0] - 1.0))
    check(
        "float32-to-float64 fixture reproduces an absolute row-sum drift above 1e-9",
        copied_error > U.PROBABILITY_ROW_SUM_ATOL,
    )

    try:
        U.probability_row_audit(copied_float64, "unnormalized float32 fixture")
    except AssertionError:
        strict_rejected = True
    else:
        strict_rejected = False
    check(
        "strict probability audit rejects default-rtol false positives",
        strict_rejected,
    )

    normalized = U._expand_probabilities(
        raw_float32,
        classes=[0, 1, 2, 3],
        n_classes=6,
        context="float32 regression fixture",
    )
    normalized_error = float(np.max(np.abs(normalized.sum(axis=1) - 1.0)))
    check("probability normalization emits float64", normalized.dtype == np.float64)
    check(
        "probability normalization meets the frozen absolute 1e-9 gate",
        normalized_error <= U.PROBABILITY_ROW_SUM_ATOL,
    )
    normalized_audit = U.probability_row_audit(normalized, "normalized fixture")
    check(
        "normalized fixture passes the same strict shard audit",
        normalized_audit["row_sums_one"]
        and normalized_audit["max_row_sum_error"] <= U.PROBABILITY_ROW_SUM_ATOL,
    )
    check(
        "positive row normalization preserves argmax",
        int(np.argmax(raw_float32[0])) == int(np.argmax(normalized[0])),
    )

    malformed = np.array([[0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    try:
        U._expand_probabilities(
            malformed,
            classes=[0, 1, 2, 3],
            n_classes=6,
            context="malformed fixture",
        )
    except AssertionError:
        malformed_rejected = True
    else:
        malformed_rejected = False
    check(
        "normalization rejects material row-sum errors before repair",
        malformed_rejected,
    )


def test_synthetic_all_fixed_models() -> None:
    features = synthetic_features()
    task = U.build_task_catalog(synthetic_days())[0]
    prepared = U.split_task(features, task, "primary", max_rows=20, min_windows=5)
    x_test = U.clean_model_features(prepared.test, ["f0", "f1", "f2"])
    for model_name in U.MODEL_ORDER:
        result = U.fit_model_predictions(
            model_name,
            prepared.train,
            x_test,
            ["f0", "f1", "f2"],
            n_jobs=1,
        )
        check(f"synthetic {model_name} fit/predict has fixed probability rows", result.probabilities.shape[1] == 6)
        row_sum_error = float(
            np.max(np.abs(result.probabilities.sum(axis=1) - 1.0))
        )
        check(
            f"synthetic {model_name} probability rows meet the absolute 1e-9 gate",
            row_sum_error <= U.PROBABILITY_ROW_SUM_ATOL,
        )
        check(
            f"synthetic {model_name} normalized probabilities preserve predicted classes",
            np.array_equal(result.predictions, np.argmax(result.probabilities, axis=1)),
        )
    try:
        U.fit_model_predictions(
            "rf", prepared.train, prepared.test, ["f0", "f1", "f2"], n_jobs=1
        )
    except AssertionError:
        rejected = True
    else:
        rejected = False
    check("fit boundary rejects a label-bearing test frame", rejected)


def test_synthetic_passlines() -> None:
    tasks = U.build_task_catalog(synthetic_days())
    cpd_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    for panel in ("primary", "stable"):
        for index, task in enumerate(tasks):
            cpd_rows.append(
                {
                    "task": task.name,
                    "kind": task.kind,
                    "test_day": task.test_day,
                    "panel": panel,
                    "cpd_y": float(index + (0.25 if panel == "stable" else 0.0)),
                }
            )
            if task.kind == "ood":
                gain = -1.0 if index % 2 else 1.0
            else:
                gain = 0.0
            gain_rows.append(
                {
                    "task": task.name,
                    "kind": task.kind,
                    "test_day": task.test_day,
                    "panel": panel,
                    "stacking_gain": gain,
                }
            )
    passline, replicas, decision = U.build_passline_tables(
        pd.DataFrame(cpd_rows),
        pd.DataFrame(gain_rows),
        replicates=25,
    )
    check("synthetic passline includes primary and stable criterion summaries", len(passline) == 4)
    check("synthetic passline stores every bootstrap replicate", len(replicas) == 25 * 4)
    check("synthetic overall branch is one of the frozen three", decision["overall_three_branch"] in {
        "both_pass", "partial_default_not_pass", "both_fail"
    })


def test_primitive_imports_and_label_boundary() -> None:
    source = inspect.getsource(U)
    check("candidate imports mainline build_model", "from robust_iot_research import" in source and "build_model" in source)
    check("candidate imports mainline sample_balanced", "sample_balanced" in source)
    check("candidate imports cpd_core.cpd_y", "from cpd_core import cpd_y" in source)
    check("candidate has no private CPD implementation", "def cpd_y" not in source and "def compute_cpd" not in source)
    audit = U.function_level_leakage_audit()
    check("fit/OOF signatures have no test-label parameters", audit["pass"])
    check("fit function has no y_test parameter", "y_test" not in audit["fit_parameters"])
    check("fit function receives feature-only test data", audit["feature_only_test_boundary"])

    cm_ref = np.diag([8, 7, 6, 5, 4, 3]).astype(float)
    cm_tgt = cm_ref.copy()
    cm_tgt[0, 1] = 1.0
    cm_tgt[0, 0] -= 1.0
    got = U.cpd_y(cm_ref, cm_tgt)
    check("CPD is obtained from the imported cpd_core function", got > 0.0)


def test_packet_and_publication_guards() -> None:
    empty_acceptance = U.collect_static_acceptance(
        {
            "finite": True,
            "n_numeric_features": 61,
            "side_packet_ratio_zero": True,
            "other_packet_ratio_zero": True,
            "zero_audit_columns_in_model_features": sorted(U.AUDIT_COLUMNS),
            "feature_importance_comparison_performed": False,
        },
        [],
        [],
        [],
        {"task_definitions": 74, "task_panel_cells": 148},
    )
    check(
        "empty probability audit cannot pass vacuously",
        not empty_acceptance["local_gates"]["probability_row_sums"],
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first.mkdir()
        second.mkdir()
        for packet in (first, second):
            for name in U.DETERMINISTIC_PACKET_FILES:
                (packet / name).write_text(f"fixed:{name}\n", encoding="utf-8")
            U.write_md5_manifest(packet)
        check("two complete fixed packets compare byte-identically", U.compare_deterministic_packets(first, second)["pass"])
        (first / "gain_table.csv").unlink()
        (second / "gain_table.csv").unlink()
        check(
            "same missing required file in both packets still fails",
            not U.compare_deterministic_packets(first, second)["pass"],
        )

        shard_roots = []
        for index in range(6):
            shard = root / f"shard-{index}"
            shard.mkdir()
            shard_roots.append(shard)
        output = root / "packet"
        try:
            U.merge_shard_packets(shard_roots, output)
        except FileNotFoundError:
            rejected = True
        else:
            rejected = False
        check("incomplete six-shard set is rejected", rejected)
        check("failed shard gate leaves output path nonexistent", not output.exists())

    parser_source = inspect.getsource(U.parse_args)
    check("unchecked external-gate boolean was removed", "external-gates-verified" not in parser_source)
    check("canonical finalizer requires two staging packets", "--publish-canonical" in parser_source)


def test_complete_synthetic_local_gate() -> None:
    features = synthetic_features()
    tasks = U.build_task_catalog(synthetic_days())
    details: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    probability_audits: list[dict[str, object]] = []
    for task in tasks:
        for panel in ("primary", "stable"):
            prepared = U.split_task(
                features, task, panel, max_rows=20, min_windows=5
            )
            folds = U.make_oof_folds(prepared.train, task)
            detail = dict(prepared.detail)
            detail.update(
                {
                    "oof_fold_count": len(folds),
                    "oof_mode": folds[0].mode,
                    "oof_group_field": "day" if folds[0].mode == "grouped_day" else "none",
                    "oof_semantics": "synthetic gate fixture",
                }
            )
            details.append(detail)
            fold_rows.extend(U._fold_rows(prepared, folds, "rf_reference"))
            fold_rows.extend(U._fold_rows(prepared, folds, "stacking"))
            for source in ("rf_oof", *[f"test_{model}" for model in U.MODEL_ORDER]):
                probability_audits.append(
                    {
                        "task": task.name,
                        "panel": panel,
                        "source": source,
                        "n_rows": len(prepared.test),
                        "finite": True,
                        "bounded_0_1": True,
                        "min_probability": 0.0,
                        "max_probability": 1.0,
                        "max_row_sum_error": 0.0,
                        "row_sums_one": True,
                    }
                )
    feature_audit = U.validate_feature_table(features)
    feature_audit["n_numeric_features"] = 61
    serialized_details = U._serialize_detail_frame(details).to_dict(orient="records")
    serialized_folds = U._serialize_nested_frame(
        fold_rows,
        [
            "task", "panel", "oof_model", "fold", "mode", "group_field",
            "train_indices", "validation_indices", "train_days", "validation_days",
            "train_time_min", "train_time_max", "validation_time_min", "validation_time_max",
        ],
    ).to_dict(orient="records")
    acceptance = U.collect_static_acceptance(
        feature_audit,
        serialized_details,
        serialized_folds,
        probability_audits,
        U.validate_task_catalog(tasks, synthetic_days()),
    )
    check("complete 148-cell synthetic local hard-gate fixture passes", acceptance["local_all_pass"])
    check("probability gate observes exactly 148 x 5 audits", acceptance["probability_audit"]["observed"] == 740)
    check("OOF gate covers both recorded models", acceptance["oof_audit"]["task_model_groups"] == 296)


def main() -> int:
    print("=" * 78)
    print("D12/D13 UNSW Test 1 candidate — pure synthetic tests")
    print("=" * 78)
    for test in (
        test_task_catalogue,
        test_feature_and_panel_rules,
        test_iid_time_order,
        test_oof_semantics,
        test_synthetic_rf_reference,
        test_probability_normalization_and_strict_audit,
        test_synthetic_all_fixed_models,
        test_synthetic_passlines,
        test_primitive_imports_and_label_boundary,
        test_packet_and_publication_guards,
        test_complete_synthetic_local_gate,
    ):
        print(f"--- {test.__name__}")
        test()
    print("synthetic tests: all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
