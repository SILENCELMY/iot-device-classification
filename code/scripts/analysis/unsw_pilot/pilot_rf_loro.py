#!/usr/bin/env python3
"""UNSW pilot — 最小 RF LORO（Leave-One-Round-Out，此处 round = 采集日）。五问之 5。

协议口径
--------
- §7  RF：`n_estimators=500, class_weight="balanced"` —— **直接 import 主线 `build_model("rf", ...)`**，
      不在本脚本重新写一份 RF 配置（§11「唯一实现」纪律）。
- §16.4 类别不均衡：**复用主线 `sample_balanced`**，不另写平衡逻辑。
- §16.2 结论一律来自 pcap 派生特征表。

做什么
------
对每个有序日对 (train_day → test_day)：整天训练、另一整天测试，报 macro-F1 / accuracy /
per-class F1 / 混淆矩阵，并跑一组**标签错误迹象**的检查（见 `label_sanity`）。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

def _find_repo_root(start: Path) -> Path:
    """向上找含 code/scripts/core/robust_iot_research.py 的目录。"""
    for cand in [start, *start.parents]:
        if (cand / "code" / "scripts" / "core" / "robust_iot_research.py").exists():
            return cand
    raise SystemExit("找不到仓库根（code/scripts/core/robust_iot_research.py）")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

from robust_iot_research import (  # noqa: E402  —— §11 唯一实现：不重写这三个
    build_model,
    clean_x,
    sample_balanced,
)

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# 与 extract_features_generic.py 的 META_COLUMNS 保持一致
META_COLUMNS = {
    "device", "day", "label", "source_file",
    "window_id", "window_start", "window_end", "window_start_epoch",
}


def feature_columns(features: pd.DataFrame) -> list[str]:
    """与主线 robust_iot_research.py L523-530 同逻辑。"""
    return [
        c for c in features.columns
        if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(features[c])
    ]


def label_sanity(cm: np.ndarray, labels: list[str], y_true: np.ndarray,
                 y_pred: np.ndarray) -> dict:
    """标签错误迹象的机械检查（不做解读，只报数）。

    「肉眼可见的标签错误」典型形态：
      A. 某类几乎全部被判成**另一个特定类**（近乎 1:1 的类互换 —— 典型的标签串位）；
      B. 某一对类**双向**大量互判（两类物理上被交换或 MAC 被复用）；
      C. 某类 recall ≈ 0 但 support 充足（该类样本被系统性吸走）。
    """
    n = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = np.divide(cm, n[:, None], where=n[:, None] > 0)
    row_norm = np.nan_to_num(row_norm)

    findings = {"A_dominant_offdiag": [], "B_mutual_swap": [], "C_zero_recall": []}
    k = len(labels)
    for i in range(k):
        if n[i] == 0:
            continue
        off = row_norm[i].copy()
        off[i] = 0.0
        j = int(off.argmax())
        # A: 离对角最大项 > 0.5 且明显高于对角
        if off[j] > 0.5 and off[j] > row_norm[i, i]:
            findings["A_dominant_offdiag"].append(
                {"true": labels[i], "mostly_predicted_as": labels[j],
                 "share": round(float(off[j]), 4),
                 "own_recall": round(float(row_norm[i, i]), 4),
                 "support": int(n[i])}
            )
        # C: recall ~ 0 且 support >= 30
        if row_norm[i, i] < 0.02 and n[i] >= 30:
            findings["C_zero_recall"].append(
                {"true": labels[i], "recall": round(float(row_norm[i, i]), 4),
                 "support": int(n[i]),
                 "mostly_predicted_as": labels[j], "share": round(float(off[j]), 4)}
            )
    for i in range(k):
        for j in range(i + 1, k):
            if n[i] == 0 or n[j] == 0:
                continue
            # B: 双向互判都 > 0.3
            if row_norm[i, j] > 0.30 and row_norm[j, i] > 0.30:
                findings["B_mutual_swap"].append(
                    {"pair": [labels[i], labels[j]],
                     "i_to_j": round(float(row_norm[i, j]), 4),
                     "j_to_i": round(float(row_norm[j, i]), 4)}
                )
    findings["any_flag"] = bool(
        findings["A_dominant_offdiag"] or findings["B_mutual_swap"] or findings["C_zero_recall"]
    )
    return findings


def run_pair(feats: pd.DataFrame, train_day: str, test_day: str, cols: list[str],
             min_windows: int, max_rows: int, random_state: int, n_jobs: int,
             out_dir: Path) -> dict:
    tr = feats[feats["day"] == train_day]
    te = feats[feats["day"] == test_day]

    # 入选门槛：两天都 >= min_windows 的设备才进任务（§16.4「入选门槛写入正文」）
    ok_tr = set(tr["label"].value_counts()[lambda s: s >= min_windows].index)
    ok_te = set(te["label"].value_counts()[lambda s: s >= min_windows].index)
    keep = sorted(ok_tr & ok_te)
    tr = tr[tr["label"].isin(keep)].reset_index(drop=True)
    te = te[te["label"].isin(keep)].reset_index(drop=True)

    # §16.4：类别不均衡复用主线 sample_balanced（按 label 分组等量抽样）
    tr_bal = sample_balanced(tr, max_rows=max_rows, random_state=random_state)
    te_bal = sample_balanced(te, max_rows=max_rows, random_state=random_state)

    x_tr, y_tr = clean_x(tr_bal, cols), tr_bal["label"].to_numpy()
    x_te, y_te = clean_x(te_bal, cols), te_bal["label"].to_numpy()

    model = build_model("rf", random_state=random_state, n_jobs=n_jobs,
                        class_count=len(keep))
    t0 = time.time()
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    fit_seconds = round(time.time() - t0, 1)

    labels = sorted(keep)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0))
    acc = float(accuracy_score(y_te, y_pred))

    tag = f"{train_day}__to__{test_day}"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / f"cm_{tag}.csv", encoding="utf-8-sig")
    rep = classification_report(y_te, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_dir / f"per_class_{tag}.csv", encoding="utf-8-sig")

    sanity = label_sanity(cm, labels, y_te, y_pred)

    result = {
        "train_day": train_day,
        "test_day": test_day,
        "n_classes": len(labels),
        "classes": labels,
        "min_windows_threshold": min_windows,
        "n_train_rows_before_balance": int(len(tr)),
        "n_test_rows_before_balance": int(len(te)),
        "n_train_rows": int(len(tr_bal)),
        "n_test_rows": int(len(te_bal)),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "accuracy": round(acc, 6),
        "fit_predict_seconds": fit_seconds,
        "label_sanity": sanity,
        "confusion_matrix_csv": f"cm_{tag}.csv",
        "per_class_csv": f"per_class_{tag}.csv",
    }
    print(f"[{tag}] classes={len(labels)} train={len(tr_bal):,} test={len(te_bal):,} "
          f"macro_f1={macro_f1:.4f} acc={acc:.4f} flags={sanity['any_flag']}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-windows", type=int, default=100,
                    help="设备入选门槛：训练日与测试日都需 >= 该窗口数")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="传给主线 sample_balanced 的 max_rows（每类 max_rows//n_class 行）")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.features)
    cols = feature_columns(feats)
    days = sorted(feats["day"].unique())
    print(f"[in] {args.features} rows={len(feats):,} days={days} features={len(cols)}")
    if len(days) < 2:
        raise SystemExit("LORO 需要 >= 2 天数据")

    results = [
        run_pair(feats, a, b, cols, args.min_windows, args.max_rows,
                 args.random_state, args.n_jobs, args.out_dir)
        for a, b in permutations(days, 2)
    ]

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("classes", "label_sanity")}
        for r in results
    ])
    summary_path = args.out_dir / "loro_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    payload = {
        "command_line": sys.argv,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features_file": str(args.features),
        "n_feature_columns": len(cols),
        "feature_columns": cols,
        "rf_spec": "robust_iot_research.build_model('rf', ...) -> "
                   "n_estimators=500, class_weight='balanced' (协议 §7)",
        "balance": "robust_iot_research.sample_balanced (协议 §16.4)",
        "min_windows": args.min_windows,
        "max_rows": args.max_rows,
        "random_state": args.random_state,
        "results": results,
        "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in results])), 6),
        "macro_f1_min": round(float(np.min([r["macro_f1"] for r in results])), 6),
        "macro_f1_max": round(float(np.max([r["macro_f1"] for r in results])), 6),
        "any_label_flag": any(r["label_sanity"]["any_flag"] for r in results),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, cwd=REPO_ROOT).stdout.strip(),
    }
    json_path = args.out_dir / "loro_results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {summary_path}\n[out] {json_path}")
    print(f"[summary] macro-F1 mean={payload['macro_f1_mean']:.4f} "
          f"min={payload['macro_f1_min']:.4f} max={payload['macro_f1_max']:.4f} "
          f"any_label_flag={payload['any_label_flag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
