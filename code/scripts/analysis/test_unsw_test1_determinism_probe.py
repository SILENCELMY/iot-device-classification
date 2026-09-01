#!/usr/bin/env python3
"""Cross-process, pure-synthetic determinism probe for D12 R4 Stacking.

The worker imports the real mainline Stacking factory and the R4 execution
wrapper.  It never discovers or reads UNSW files and emits only a SHA-256 over
predictions and normalized probabilities.  The coordinator requires two
serial and three concurrent independent worker processes to agree exactly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[2]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import unsw_test1 as U  # noqa: E402


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PROXY_ENV_VARS = (
    "http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "no_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "NO_PROXY",
)
WORKER_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(20260901)
    feature_columns = [f"f{index}" for index in range(8)]
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, float]] = []
    for round_index in range(5):
        for class_index, label in enumerate(U.CATEGORY_ORDER):
            for sample_index in range(6):
                vector = rng.normal(0.0, 0.35, len(feature_columns))
                vector[class_index % len(feature_columns)] += 1.5
                vector[-1] += round_index * 0.05
                row: dict[str, object] = {
                    "label": label,
                    "day": f"d{round_index}",
                    "window_start_epoch": float(round_index * 100 + class_index * 10 + sample_index),
                }
                row.update({name: float(value) for name, value in zip(feature_columns, vector)})
                train_rows.append(row)
    for class_index, _label in enumerate(U.CATEGORY_ORDER):
        for sample_index in range(3):
            vector = rng.normal(0.0, 0.35, len(feature_columns))
            vector[class_index % len(feature_columns)] += 1.5
            vector[-1] += sample_index * 0.02
            test_rows.append(
                {name: float(value) for name, value in zip(feature_columns, vector)}
            )
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows), feature_columns


def worker_hash() -> str:
    train, test_features, feature_columns = synthetic_frames()
    model = U.build_test1_model("stacking", n_jobs=4)
    for _name, estimator in model.estimators:
        params = estimator.get_params(deep=False)
        probe_only: dict[str, object] = {}
        if "n_estimators" in params:
            probe_only["n_estimators"] = 16
        if "max_depth" in params and params.get("max_depth") not in (None, -1):
            probe_only["max_depth"] = min(int(params["max_depth"]), 4)
        estimator.set_params(**probe_only)

    original_builder = U.build_test1_model
    try:
        U.build_test1_model = lambda model_name, *, n_jobs: model
        result = U.fit_model_predictions(
            "stacking",
            train,
            test_features,
            feature_columns,
            n_jobs=4,
        )
    finally:
        U.build_test1_model = original_builder

    digest = hashlib.sha256()
    digest.update(np.asarray(result.predictions, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(result.probabilities, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in PROXY_ENV_VARS:
        environment.pop(name, None)
    for name in THREAD_ENV_VARS:
        environment[name] = "4"
    environment["MPLCONFIGDIR"] = "/tmp/iotcls-unsw-test1-mpl"
    return environment


def launch_worker() -> str:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=REPO_ROOT,
        env=worker_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"synthetic determinism worker failed with exit {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    output = completed.stdout.strip()
    if not WORKER_HASH_RE.fullmatch(output):
        raise RuntimeError(f"worker emitted non-hash output: {output!r}")
    return output


def coordinator() -> int:
    serial = [launch_worker() for _index in range(2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        concurrent_hashes = list(executor.map(lambda _index: launch_worker(), range(3)))
    hashes = serial + concurrent_hashes
    if len(set(hashes)) != 1:
        raise AssertionError("serial/concurrent synthetic Stacking hashes differ")
    print(hashes[0])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        print(worker_hash())
        return 0
    return coordinator()


if __name__ == "__main__":
    raise SystemExit(main())
