#!/usr/bin/env python3
"""历史 110 条主线结果的 `pred_proba` 重打分（协议 §20.3 / 执行计划 D7）。

背景
----
`results/robust_v2/` 的 110 条结果（11 任务 × 2 特征集 × 5 模型）产出于
2026-06-22，早于协议 §20.2 给 `evaluate_model` 加上 `pred_proba.csv` 落盘，
因此这批目录里**只有 `predictions.csv`（硬标签），没有概率**。
§7 的投票基线（soft voting）需要概率。

协议 §20.3 已判定**不必重训**：`model.joblib` 与 `feature_columns.json` 都在盘上，
测试集划分是确定性的（`fixed_split` 由轮次决定；`single_round` / `joint_validation`
是 `random_state=42` 的分层划分）。本脚本据此重建测试集、用既有模型重新打分。

**训练逻辑零重写**（§20.3 唯一实现纪律）：测试集重建**全部**走
`robust_iot_research.task_data` / `feature_columns` / `clean_x` / `fit_label_encoder`，
任务定义唯一来源是 `code/configs/research_experiments.json`。本文件不含任何
划分、特征或建模逻辑的副本。

硬门（不满足即跳过，绝不强算）
------------------------------
1. `model.joblib` 必须在当前 sklearn 下**无异常、无告警**地加载；
2. 重建测试集必须与既有 `predictions.csv` 逐行对齐
   （`source_file` / `window_id` / `window_start` / `true_label`）；
3. 模型对重建测试集的 **argmax 预测必须与 `predictions.csv` 的
   `predicted_label` 逐行一致**。

任何一条不满足 → 该 (task, feature_set, model) 如实记入跳过清单并跳过，
不写任何概率文件。

明确不做：OOF 重建
------------------
`oof_meta.csv`（stacking 训练侧的折外概率）**无法**从最终模型恢复——OOF 按定义
来自 K 个折内子模型，而落盘的 `model.joblib` 只有在全量训练集上重新拟合的
`final_estimator_` 与 `named_estimators_`，折内子模型从未持久化。用最终模型对训练集
打分得到的是**样本内**概率，与 OOF 语义不同，会系统性高估。因此本脚本
**只补 `pred_proba`，不产出任何 OOF 数值**。需要 OOF 的分析（E1 等）必须重训，
见 `results/e1_oof_arms*/` 与 G0 的全量落盘。

输出（只新增、绝不覆盖）
------------------------
- `results/robust_v2/raw_all/<task>/<feature_set>/<model>/pred_proba_rescored_20260829.csv`
  列布局与 `evaluate_model` 写的 `pred_proba.csv` 完全一致
  （`source_file, round, window_id, window_start, proba_<cls>…, true_label`）。
- `results/robust_v2/rescore_20260829/rescore_summary_20260829.csv`（逐条状态）
- `results/robust_v2/rescore_20260829/rescore_manifest_20260829.json`（§19.2 provenance）

用法
----
    # smoke：先跑 3 个任务看中间结果
    python code/scripts/utils/rescore_historical.py --tasks single_round_R2,loro_R2_R4_to_R3,joint_R2_R3_R4
    # 全量 110 条
    python code/scripts/utils/rescore_historical.py
    # 只体检不写文件
    python code/scripts/utils/rescore_historical.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

import robust_iot_research as R  # noqa: E402  （§20.3：划分/特征逻辑唯一来源）

RESULT_ROOT = REPO_ROOT / "results" / "robust_v2" / "raw_all"
CONFIG_PATH = REPO_ROOT / "code" / "configs" / "research_experiments.json"
FEATURES_CSV = RESULT_ROOT / "features_raw_all_w10.csv"
SUFFIX = "20260829"
OUT_NAME = f"pred_proba_rescored_{SUFFIX}.csv"
SUMMARY_DIR = REPO_ROOT / "results" / "robust_v2" / f"rescore_{SUFFIX}"

# robust_v2 的运行参数（`robust_iot_research.parse_args` 的默认值，未被命令行覆盖）。
# 由 D7 回归验证确认：用这组参数经 `task_data` 重建的测试集与落盘 predictions.csv
# 逐行一致（loro_R2_R4_to_R3 / single_round_R2 两个分支各验一次）。
RUN_ARGS = dict(test_size=0.3, random_state=42, max_rows=0,
                feature_mode="all", disable_feature_selection=True, n_jobs=1)

# `robust_iot_research.py` 当时是**以脚本方式**运行的，`SimpleStackingClassifier`
# 因此被 pickle 记成了 `__main__.SimpleStackingClassifier`。这里把同一个类对象挂回
# `__main__` 以便反序列化——**只是解 pickle 的名字绑定，不改任何类定义或行为**；
# 即便绑错了，下面的逐行一致硬门也会把它拦下来。
sys.modules["__main__"].SimpleStackingClassifier = R.SimpleStackingClassifier


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def _git_head() -> dict:
    out: dict = {"head": None, "dirty": None}
    try:
        out["head"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        out["dirty"] = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                           capture_output=True, text=True,
                                           check=True).stdout.strip())
    except Exception as exc:                      # noqa: BLE001
        out["error"] = str(exc)
    return out


def _versions() -> dict:
    import sklearn
    v = {"python": sys.version.split()[0], "platform": platform.platform(),
         "numpy": np.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    for name in ("scipy", "xgboost", "lightgbm"):
        try:
            v[name] = __import__(name).__version__
        except Exception:                         # noqa: BLE001
            v[name] = None
    return v


# --------------------------------------------------------------------------- #
# 单条重打分
# --------------------------------------------------------------------------- #
def load_model(path: Path) -> tuple[object | None, list[str], str | None]:
    """加载 joblib。返回 (model, warning_strings, error)。告警与异常都如实上报。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(path)
        except Exception as exc:                  # noqa: BLE001
            return None, [f"{w.category.__name__}: {w.message}" for w in caught], \
                   f"{type(exc).__name__}: {exc}"
    return model, [f"{w.category.__name__}: {w.message}" for w in caught], None


def rescore_one(model_dir: Path, task: dict, labels: list[str],
                features: pd.DataFrame, ns: argparse.Namespace,
                dry_run: bool) -> dict:
    rec: dict = {
        "task": task["name"], "task_type": task["type"],
        "feature_set": model_dir.parent.name, "model": model_dir.name,
        "status": "pending", "reason": "", "n_test": 0, "n_mismatch": -1,
        "match_rate": None, "n_features": 0, "load_warnings": "",
        "max_abs_rowsum_dev": None, "output": "",
    }

    pred_path = model_dir / "predictions.csv"
    fc_path = model_dir / "feature_columns.json"
    mdl_path = model_dir / "model.joblib"
    for p in (pred_path, fc_path, mdl_path):
        if not p.exists():
            rec.update(status="skipped", reason=f"missing {p.name}")
            return rec

    model, warns, err = load_model(mdl_path)
    rec["load_warnings"] = " | ".join(warns)
    if err is not None:
        rec.update(status="skipped", reason=f"joblib load failed: {err}")
        return rec
    if warns:
        # 协议要求：sklearn 版本不一致等告警一律视为不可信 → 跳过，不强算
        rec.update(status="skipped", reason="joblib load emitted warnings")
        return rec

    selected_columns = json.loads(fc_path.read_text(encoding="utf-8"))
    rec["n_features"] = len(selected_columns)

    # ---- 测试集重建：全部走 robust_iot_research，无任何本地副本 ----
    train_data, test_data, _y_train, y_test, _mtr, meta_test = R.task_data(features, task, ns)
    columns = R.feature_columns(train_data)
    missing = [c for c in selected_columns if c not in columns]
    if missing:
        rec.update(status="skipped", reason=f"feature_columns not in feature table: {missing[:3]}")
        return rec
    x_test = R.clean_x(test_data, columns)
    encoder = R.fit_label_encoder(labels)

    old = pd.read_csv(pred_path, encoding="utf-8-sig")
    rec["n_test"] = int(len(test_data))
    if len(old) != len(test_data):
        rec.update(status="skipped",
                   reason=f"row count differs: predictions.csv={len(old)} rebuilt={len(test_data)}")
        return rec

    # ---- 硬门 2：逐行对齐 ----
    aligned = (
        old["source_file"].to_numpy().tolist() == meta_test["source_file"].to_numpy().tolist()
        and old["window_id"].to_numpy().tolist() == meta_test["window_id"].to_numpy().tolist()
        and np.allclose(old["window_start"].to_numpy(),
                        meta_test["window_start"].to_numpy(), rtol=0, atol=1e-9)
        and old["true_label"].to_numpy().tolist() == y_test.to_numpy().tolist()
    )
    if not aligned:
        rec.update(status="skipped", reason="rebuilt test set not row-aligned with predictions.csv")
        return rec

    # ---- 打分（列位处理与 evaluate_model 完全一致）----
    try:
        proba = model.predict_proba(x_test[selected_columns])
    except Exception as exc:                      # noqa: BLE001
        rec.update(status="skipped", reason=f"predict_proba failed: {type(exc).__name__}: {exc}")
        return rec
    model_classes = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1]))).astype(int)
    proba_full = np.zeros((len(proba), len(encoder.classes_)), dtype=float)
    proba_full[:, model_classes] = proba

    # ---- 硬门 3：argmax 逐行一致 ----
    pred_labels = encoder.inverse_transform(proba_full.argmax(axis=1).astype(int))
    mismatch = int((pred_labels != old["predicted_label"].to_numpy()).sum())
    rec["n_mismatch"] = mismatch
    rec["match_rate"] = float(1.0 - mismatch / len(old)) if len(old) else None
    if mismatch:
        rec.update(status="skipped",
                   reason=f"argmax disagrees with predictions.csv on {mismatch}/{len(old)} rows")
        return rec

    # 行和 = 1 的容差取协议 §22.1 P1 的 1e-6。不能取更紧：XGBoost 的 predict_proba
    # 返回 float32，softmax 行和天然有 ~float32 eps (1.19e-7) 的偏差（实测
    # max|rowsum-1| = 1.19e-7 ~ 1.79e-7），而 rf / extra_trees / lightgbm / stacking
    # 是 float64（~1e-16）。1e-6 仍足以拦下真正的列位错配（那会给出 0.7 / 1.3 量级的行和）。
    dev = float(np.abs(proba_full.sum(axis=1) - 1.0).max())
    rec["max_abs_rowsum_dev"] = dev
    if dev > 1e-6:
        rec.update(status="skipped",
                   reason=f"probability rows do not sum to 1 (max dev {dev:.3e} > 1e-6)")
        return rec

    if dry_run:
        rec.update(status="ok_dry_run", reason="")
        return rec

    out_path = model_dir / OUT_NAME
    if out_path.exists():
        rec.update(status="skipped", reason=f"{OUT_NAME} already exists (refuse to overwrite)")
        return rec
    proba_df = meta_test[["round", "window_id", "window_start"]].copy()
    proba_df.insert(0, "source_file", meta_test["source_file"].to_numpy())
    for i, cls in enumerate(encoder.classes_):
        proba_df[f"proba_{cls}"] = proba_full[:, i]
    proba_df["true_label"] = y_test.to_numpy()
    proba_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    rec.update(status="ok", reason="", output=str(out_path.relative_to(REPO_ROOT)))
    return rec


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="历史 110 条 pred_proba 重打分（§20.3）")
    ap.add_argument("--tasks", default="", help="逗号分隔任务名子集（smoke 用）；缺省跑全部")
    ap.add_argument("--models", default="", help="逗号分隔模型名子集；缺省跑全部")
    ap.add_argument("--dry-run", action="store_true", help="只体检不写任何文件")
    cli = ap.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    labels = config["labels"]
    task_by_name = {t["name"]: t for t in config["evaluation_tasks"]}
    features = pd.read_csv(FEATURES_CSV)     # 与 build_feature_table 的读法一致
    ns = argparse.Namespace(**RUN_ARGS)

    want_tasks = {t.strip() for t in cli.tasks.split(",") if t.strip()}
    want_models = {m.strip() for m in cli.models.split(",") if m.strip()}

    model_dirs = sorted(
        d for d in RESULT_ROOT.glob("*/*/*")
        if d.is_dir() and d.parent.name in {"all_features", "selected_features"}
    )
    print(f"重打分范围：{RESULT_ROOT.relative_to(REPO_ROOT)}  共发现 {len(model_dirs)} 个模型目录")
    if cli.dry_run:
        print("  [dry-run] 只做一致性体检，不写任何文件")

    records: list[dict] = []
    for d in model_dirs:
        task_name = d.parent.parent.name
        if want_tasks and task_name not in want_tasks:
            continue
        if want_models and d.name not in want_models:
            continue
        if task_name not in task_by_name:
            records.append({"task": task_name, "task_type": None,
                            "feature_set": d.parent.name, "model": d.name,
                            "status": "skipped", "reason": "task not in config",
                            "n_test": 0, "n_mismatch": -1, "match_rate": None,
                            "n_features": 0, "load_warnings": "", "output": ""})
            continue
        rec = rescore_one(d, task_by_name[task_name], labels, features, ns, cli.dry_run)
        records.append(rec)
        flag = "OK " if rec["status"].startswith("ok") else "SKIP"
        detail = f"n={rec['n_test']} mismatch={rec['n_mismatch']}" if rec["n_test"] else rec["reason"]
        print(f"  [{flag}] {task_name}/{rec['feature_set']}/{rec['model']}: {detail}"
              + (f"  <- {rec['reason']}" if rec["status"] == "skipped" and rec["n_test"] else ""),
              flush=True)

    summary = pd.DataFrame(records)
    n_ok = int((summary["status"].str.startswith("ok")).sum())
    n_skip = int((summary["status"] == "skipped").sum())
    print(f"\n合计 {len(summary)} 条：一致并重打分 {n_ok}，跳过 {n_skip}")
    if n_skip:
        print("跳过清单：")
        for _, r in summary[summary["status"] == "skipped"].iterrows():
            print(f"  - {r['task']}/{r['feature_set']}/{r['model']}: {r['reason']}")

    if not cli.dry_run:
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(SUMMARY_DIR / f"rescore_summary_{SUFFIX}.csv",
                       index=False, encoding="utf-8-sig")
        manifest = {
            "script": "code/scripts/utils/rescore_historical.py",
            "protocol_sections": ["19.2", "20.3"],
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command_line": {"argv": list(sys.argv), "executable": sys.executable,
                             "cwd": str(Path.cwd())},
            "git": _git_head(),
            "versions": _versions(),
            "inputs": {"features_csv": str(FEATURES_CSV.relative_to(REPO_ROOT)),
                       "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
                       "result_root": str(RESULT_ROOT.relative_to(REPO_ROOT))},
            "run_args": RUN_ARGS,
            "hard_gates": [
                "joblib load without exception and without warnings",
                "rebuilt test set row-aligned with predictions.csv",
                "argmax(pred_proba) == predictions.csv predicted_label on every row",
                "probability rows sum to 1 (atol=1e-6, protocol 22.1 P1 tolerance; "
                "XGBoost predict_proba is float32 so its row sums deviate at ~1.2e-7)",
            ],
            "oof_not_reconstructed": (
                "OOF cannot be recovered from the persisted final model: the K fold-models "
                "that produce out-of-fold probabilities were never persisted, and scoring the "
                "training set with the final model yields in-sample probabilities with different "
                "semantics. This script writes pred_proba only."
            ),
            "counts": {"total": len(summary), "rescored": n_ok, "skipped": n_skip},
            "output_file_per_model": OUT_NAME,
        }
        (SUMMARY_DIR / f"rescore_manifest_{SUFFIX}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n汇总写入：{SUMMARY_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
