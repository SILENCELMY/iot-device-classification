"""E1 — OOF correction / Stacking robustness（协议 §12）

三臂设计。协议 §12 原定两臂（A 随机 K=5 / B 轮次分组）；本脚本增加 **A′ 诊断臂**
（随机折叠，折数与 B 对齐），用于把 A→B 的差异分解为两个独立来源：

    A  vs A′   = 纯折数效应（K=5 → K=n_rounds），划分方式都是随机
    A′ vs B    = 纯分组/泄漏效应（随机 → 按轮次分组），折数相同

不加 A′ 时 A→B 同时改变折数与分组方式，任何差异都无法归因。
A′ 仅用于归因分解，**不改变 §12 的预注册解读规则**（该规则只针对 B 臂）。
见 §23 Change Log 2026-08-28 条（澄清，非例外）。

**任务范围**（§12）：11 个主线任务（默认口径）+ G0 网格的 OOD 任务（`--grid`）。
G0 网格任务定义**唯一来源**是 `environment_grid_experiment.build_task_grid()`
（§20.3 / §11 唯一实现纪律），本文件不重复实现任务生成。

用法：
    python code/scripts/analysis/e1_oof_arms.py                     # 旗舰任务，单种子
    python code/scripts/analysis/e1_oof_arms.py --seeds 42,43,44,45,46
    python code/scripts/analysis/e1_oof_arms.py --tasks all --seeds 42,43,44,45,46
    # G0 网格（输出到 results/e1_oof_arms_g0/，不触碰主线 E1）：
    python code/scripts/analysis/e1_oof_arms.py --grid all                  # 150 OOD 任务
    python code/scripts/analysis/e1_oof_arms.py --grid s2plus               # |S|>=2 的 120 个
    python code/scripts/analysis/e1_oof_arms.py --grid s1                   # |S|=1 的 30 个
    python code/scripts/analysis/e1_oof_arms.py --grid all --shard 1/3      # 分片并行
    python code/scripts/analysis/e1_oof_arms.py --grid all --merge          # 合并分片
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

import robust_iot_research as R  # noqa: E402
import environment_grid_experiment as G  # noqa: E402  （§20.3：G0 任务定义唯一来源）

FEATURES_CSV = REPO_ROOT / "results" / "robust_v2" / "raw_all" / "features_raw_all_w10.csv"
OUT_DIR = REPO_ROOT / "results" / "e1_oof_arms"
OUT_DIR_GRID = REPO_ROOT / "results" / "e1_oof_arms_g0"
G0_METRICS_ROOT = REPO_ROOT / "results" / "g0_environment_grid" / "raw_all"
BASE_MODELS = ["rf", "xgboost", "lightgbm"]

# 协议 §12 任务范围：11 个既有主线任务（type 取自 code/configs/research_experiments.json）。
# `single_round_*` 只有 1 个轮次，无法按轮次分组 → B 臂按 §12 退化为 window_start 时间块。
#
# 注意：`single_round` 与 `joint_validation` 的训练/测试划分由 `task_data` 用
# `random_state=seed` 做分层随机切分，**换种子会同时改变划分**。因此这 4 个任务的跨种子
# std 混合了划分噪声与初始化噪声；同一种子内三臂共享划分，A/A′/B 的臂间对比仍然有效。
TASKS: list[dict] = [
    {"name": "loro_R2_R4_to_R3", "type": "fixed_split", "train_rounds": ["R2", "R4"], "test_rounds": ["R3"]},
    {"name": "loro_R2_R3_to_R4", "type": "fixed_split", "train_rounds": ["R2", "R3"], "test_rounds": ["R4"]},
    {"name": "loro_R3_R4_to_R2", "type": "fixed_split", "train_rounds": ["R3", "R4"], "test_rounds": ["R2"]},
    {"name": "position_R2_R3_R4_to_R5", "type": "fixed_split",
     "train_rounds": ["R2", "R3", "R4"], "test_rounds": ["R5"]},
    {"name": "jitter_R2_R3_R4_to_R6", "type": "fixed_split",
     "train_rounds": ["R2", "R3", "R4"], "test_rounds": ["R6"]},
    {"name": "jitter_R2_R3_R4_to_R7", "type": "fixed_split",
     "train_rounds": ["R2", "R3", "R4"], "test_rounds": ["R7"]},
    {"name": "jitter_R2_R3_R4_to_R6_R7", "type": "fixed_split",
     "train_rounds": ["R2", "R3", "R4"], "test_rounds": ["R6", "R7"]},
    {"name": "single_round_R2", "type": "single_round", "rounds": ["R2"]},
    {"name": "single_round_R3", "type": "single_round", "rounds": ["R3"]},
    {"name": "single_round_R4", "type": "single_round", "rounds": ["R4"]},
    {"name": "joint_R2_R3_R4", "type": "joint_validation", "rounds": ["R2", "R3", "R4"]},
]

FLAGSHIP = "loro_R2_R4_to_R3"

# B 臂的划分依据（§9.1 / §12）**只看训练集里的实际轮次数**：
#   n_train_rounds >= 2 → GroupKFold(group=round)；n_train_rounds == 1 → window_start 时间块。
# 这与 `SimpleStackingClassifier._splitter` 的运行时判据逐字一致。
# 历史上此处曾用固定任务名单（`SINGLE_ROUND = {"single_round_R2", ...}`）来标注，
# 那份名单不含 G0 的 `|S|=1` 任务（如 `g0_R3_to_R2`），会把单轮任务错标成 round_group。
# 名单已删除，勿再引入（EXECUTION_PLAN_20260829 D2 明确列为已知陷阱）。
def b_split_basis(n_train_rounds: int) -> str:
    return "round_group" if n_train_rounds >= 2 else "time_block"


# 臂定义：(oof_mode, cv)。B 的 cv=5 只是上限，实际折数动态——轮次分组时 = min(cv, 轮次数)，
# 单轮时间块时 = min(cv, n)。A′ 的 cv 写 None，表示**运行时动态取 B 的有效折数**：
# 只有折数对齐，A→A′ 才能解释为纯折数效应、A′→B 才能解释为纯分组效应。
# 注意：当 B 的有效折数恰为 5 时（单轮任务的时间块），A′ 退化为与 A 完全相同，
# 此时折数效应恒为 0、分组效应 = 全部差异，这是正确结果而非缺陷。
B_CV = 5
ARMS = {
    "A": ("random", 5),
    "A_prime": ("random", None),
    "B": ("grouped", B_CV),
}

GRID_SCOPES = ("s2plus", "s1", "all")


def grid_tasks(scope: str) -> list[dict]:
    """G0 网格的 OOD 任务子集。任务 dict 由 `environment_grid_experiment.build_task_grid()`
    生成（§20.3：不重复实现），此处只按 `n_sources` 过滤。

    IID 任务（`grid_kind` 以 `iid_` 开头）不属于 §12 的 E1 任务范围，一律排除。
    """
    ood = [t for t in G.build_task_grid() if t.get("grid_kind") == "ood"]
    if scope == "s2plus":
        return [t for t in ood if t["n_sources"] >= 2]
    if scope == "s1":
        return [t for t in ood if t["n_sources"] == 1]
    if scope == "all":
        return ood
    raise SystemExit(f"未知 --grid 取值：{scope!r}（可选 {GRID_SCOPES}）")


def build_stacking(seed: int, n_jobs: int, n_classes: int, oof_mode: str, cv: int):
    model = R.build_model("stacking", seed, n_jobs, n_classes)
    model.oof_mode = oof_mode
    model.cv = cv
    return model


def fold_records(model, x, y, train_round, window_start) -> list[dict]:
    """枚举 `_splitter` 的实际折叠分配（§19.2 第 2 条：保存划分记录）。

    轮级分组记轮名；时间块记该块 `window_start` 的实际边界（由分配结果导出，
    不重算 quantile —— 避免与 `_splitter` 内部公式产生第二份实现）。
    """
    recs: list[dict] = []
    tr = None if train_round is None else np.asarray(train_round)
    ws = None if window_start is None else np.asarray(window_start, dtype=float)
    grouped_by_round = (model.oof_mode == "grouped"
                        and tr is not None and len(np.unique(tr)) >= 2)
    for i, (train_idx, val_idx) in enumerate(model._splitter(x, y, train_round, window_start)):
        rec: dict = {"fold": i, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
        if model.oof_mode == "grouped":
            if grouped_by_round:
                rec["val_rounds"] = sorted({str(v) for v in tr[val_idx]})
                rec["train_rounds"] = sorted({str(v) for v in tr[train_idx]})
            elif ws is not None:
                block = ws[val_idx]
                rec["val_window_start_min"] = float(block.min())
                rec["val_window_start_max"] = float(block.max())
        recs.append(rec)
    return recs


def effective_folds(model, x, y, train_round, window_start) -> int:
    return sum(1 for _ in model._splitter(x, y, train_round, window_start))


def run_task(features: pd.DataFrame, task: dict, seed: int, args, labels: list[str]) -> tuple[list[dict], dict]:
    ns = argparse.Namespace(
        max_rows=10 ** 9, random_state=seed, test_size=0.3,
        disable_feature_selection=True, feature_mode="all", n_jobs=args.n_jobs,
    )
    train_data, test_data, y_train, y_test, meta_train, meta_test = R.task_data(features, task, ns)
    cols = R.feature_columns(train_data)
    x_train, x_test = R.clean_x(train_data, cols), R.clean_x(test_data, cols)
    enc = R.fit_label_encoder(labels)
    ye_train, ye_test = enc.transform(y_train), enc.transform(y_test)
    tr_round = meta_train["round"].to_numpy()
    tr_ws = meta_train["window_start"].to_numpy()
    n_train_rounds = int(len(np.unique(tr_round)))
    basis = b_split_basis(n_train_rounds)

    # 基模型与臂无关，每 (task, seed) 只算一次
    base_f1 = {}
    for m in BASE_MODELS:
        model = R.build_model(m, seed, args.n_jobs, len(labels))
        if model is None:
            continue
        model.fit(x_train[cols], ye_train)
        pred = enc.inverse_transform(model.predict(x_test[cols]).astype(int))
        metrics, _, _ = R.metric_summary(y_test.to_numpy(), pred, labels)
        base_f1[m] = metrics["macro_f1"]
    best_base_model = max(base_f1, key=base_f1.get)
    best_base_f1 = base_f1[best_base_model]

    rows = []
    prov_arms: dict[str, dict] = {}
    # 先测 B 臂的有效折数，A′ 与之对齐（见 ARMS 注释）
    b_probe = build_stacking(seed, args.n_jobs, len(labels), "grouped", B_CV)
    b_folds = effective_folds(b_probe, x_train[cols], ye_train, tr_round, tr_ws)

    grid_meta = {k: task[k] for k in ("n_sources", "target_env", "grid_kind") if k in task}
    for arm, (oof_mode, cv) in ARMS.items():
        cv_used = b_folds if cv is None else cv
        model = build_stacking(seed, args.n_jobs, len(labels), oof_mode, cv_used)
        folds = fold_records(model, x_train[cols], ye_train, tr_round, tr_ws)
        k = len(folds)
        prov_arms[arm] = {"oof_mode": oof_mode, "cv_requested": cv_used,
                          "folds_effective": k, "folds": folds}
        model.fit(x_train[cols], ye_train, train_round=tr_round, window_start=tr_ws)
        pred = enc.inverse_transform(model.predict(x_test[cols]).astype(int))
        metrics, _, _ = R.metric_summary(y_test.to_numpy(), pred, labels)
        rows.append({
            "task": task["name"], "seed": seed, "arm": arm,
            "oof_mode": oof_mode, "cv_requested": cv_used, "folds_effective": k,
            "b_folds_effective": b_folds,
            "aprime_aligned_with_b": bool(k == b_folds) if arm == "A_prime" else None,
            "n_train_rounds": n_train_rounds,
            "b_split_basis": basis,
            "stacking_f1": metrics["macro_f1"],
            "best_base_model": best_base_model, "best_base_f1": best_base_f1,
            "gain_absolute": metrics["macro_f1"] - best_base_f1,
            **{f"base_f1_{m}": v for m, v in base_f1.items()},
            **grid_meta,
        })
        print(f"    {arm:8s} oof={oof_mode:8s} K={k}"
              f"{' (=B)' if arm == 'A_prime' and k == b_folds else '':5s}  "
              f"stacking_f1={metrics['macro_f1']:.6f}  gain={rows[-1]['gain_absolute']:+.6f}",
              flush=True)

    prov = {
        "task": task["name"], "seed": seed, "task_type": task["type"],
        "train_rounds": task.get("train_rounds", task.get("rounds", [])),
        "test_rounds": task.get("test_rounds", task.get("rounds", [])),
        "train_samples": int(len(train_data)), "test_samples": int(len(test_data)),
        "n_train_rounds": n_train_rounds, "b_split_basis": basis,
        "b_folds_effective": b_folds, "feature_count": len(cols),
        **grid_meta,
        "arms": prov_arms,
    }
    return rows, prov


# --------------------------------------------------------------------------- #
# §19.2 持久化
# --------------------------------------------------------------------------- #
def _git_head(path: Path) -> dict:
    out = {"path": str(path.relative_to(REPO_ROOT)) if path != REPO_ROOT else ".",
           "head": None, "dirty": None}
    try:
        out["head"] = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
        out["dirty"] = bool(status.strip())
    except Exception as exc:                     # noqa: BLE001 —— 缺 git 不应中断实验
        out["error"] = str(exc)
    return out


def _versions() -> dict:
    import sklearn
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name in ("xgboost", "lightgbm", "scipy", "joblib"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                        # noqa: BLE001
            versions[name] = None
    return versions


def build_provenance(args, seeds, tasks, task_prov, out_dir, stem, scope) -> dict:
    return {
        "experiment": "E1 (OOF arms A / A_prime / B)",
        "protocol_sections": ["9.1", "12", "14", "19.2", "20.3"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_line": {"argv": list(sys.argv), "executable": sys.executable,
                         "cwd": str(Path.cwd())},
        "git": {"repo_root": _git_head(REPO_ROOT), "code": _git_head(REPO_ROOT / "code")},
        "versions": _versions(),
        "seeds": seeds,
        "scope": scope,
        "arms": {k: {"oof_mode": v[0], "cv": v[1]} for k, v in ARMS.items()},
        "inputs": {"features_csv": str(FEATURES_CSV.relative_to(REPO_ROOT)),
                   "feature_set": "all_features", "n_jobs": args.n_jobs},
        "reused_g0_results": False,     # 三臂全部重算，未复用 G0 落盘数值
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "output_stem": stem,
        "n_tasks": len(tasks),
        "tasks": [t["name"] for t in tasks],
        "fold_assignments": task_prov,
    }


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def write_outputs(df: pd.DataFrame, all_rows: list[dict], out_dir: Path, stem: str,
                  verbose: bool = True) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"e1_arms_raw{stem}.csv", index=False, encoding="utf-8-sig")

    # 归因分解：以 (task, seed) 为单位配对
    piv = df.pivot_table(index=["task", "seed"], columns="arm", values="gain_absolute")
    piv["fold_effect_A_to_Aprime"] = piv["A_prime"] - piv["A"]
    piv["group_effect_Aprime_to_B"] = piv["B"] - piv["A_prime"]
    piv["total_A_to_B"] = piv["B"] - piv["A"]
    piv = piv.reset_index()
    piv.to_csv(out_dir / f"e1_decomposition{stem}.csv", index=False, encoding="utf-8-sig")

    if verbose:
        print("\n" + "=" * 78)
        print("归因分解（gain 的变化量）")
        print("=" * 78)
        for _, r in piv.iterrows():
            tot = r["total_A_to_B"]
            share = (abs(r["fold_effect_A_to_Aprime"]) /
                     (abs(r["fold_effect_A_to_Aprime"]) + abs(r["group_effect_Aprime_to_B"]))
                     if (abs(r["fold_effect_A_to_Aprime"]) + abs(r["group_effect_Aprime_to_B"])) > 0 else float("nan"))
            print(f"  {r['task']} seed={int(r['seed'])}")
            print(f"    A  gain = {r['A']:+.6f}   (随机 K=5，历史口径)")
            print(f"    A′ gain = {r['A_prime']:+.6f}   (随机，折数对齐 B)")
            print(f"    B  gain = {r['B']:+.6f}   (分组：轮次或时间块)")
            print(f"    ├ 折数效应  A→A′ = {r['fold_effect_A_to_Aprime']:+.6f}")
            print(f"    ├ 分组效应 A′→B  = {r['group_effect_Aprime_to_B']:+.6f}")
            print(f"    └ 合计     A→B   = {tot:+.6f}   折数占比 {share:.1%}")

    with (out_dir / f"e1_arms{stem}.json").open("w", encoding="utf-8") as f:
        json.dump({"arms": {k: {"oof_mode": v[0], "cv": v[1]} for k, v in ARMS.items()},
                   "seeds": sorted({int(s) for s in df["seed"].unique()}),
                   "tasks": list(dict.fromkeys(df["task"].tolist())),
                   "rows": all_rows}, f, indent=2, ensure_ascii=False)
    print(f"\n输出：{out_dir.relative_to(REPO_ROOT)}/"
          f"{{e1_arms_raw{stem}.csv, e1_decomposition{stem}.csv, e1_arms{stem}.json, "
          f"provenance{stem}.json}}")
    return piv


# --------------------------------------------------------------------------- #
# 一致性校验（EXECUTION_PLAN D2 / §22.1 P1 回归容差 1e-6）
# --------------------------------------------------------------------------- #
def check_against_g0(df: pd.DataFrame, out_dir: Path, stem: str, tol: float = 1e-6) -> bool:
    """E1 基模型 F1 ≡ G0 基模型 F1；E1 B 臂 stacking F1 ≡ G0 stacking F1（seed 42）。"""
    items: list[dict] = []

    def g0_f1(task_name: str, model: str):
        path = G0_METRICS_ROOT / task_name / "all_features" / model / "metrics.json"
        if not path.exists():
            return None, str(path)
        with path.open(encoding="utf-8") as f:
            return float(json.load(f)["macro_f1"]), str(path)

    seed42 = df[df["seed"] == 42]
    for task_name, sub in seed42.groupby("task", sort=True):
        head = sub.iloc[0]
        for m in BASE_MODELS:
            col = f"base_f1_{m}"
            if col not in sub.columns or pd.isna(head[col]):
                continue
            ref, path = g0_f1(task_name, m)
            items.append({"task": task_name, "kind": f"base:{m}",
                          "e1": float(head[col]), "g0": ref, "g0_path": path,
                          "abs_diff": None if ref is None else abs(float(head[col]) - ref),
                          "pass": ref is not None and abs(float(head[col]) - ref) <= tol})
        b = sub[sub["arm"] == "B"]
        if not b.empty:
            e1_b = float(b.iloc[0]["stacking_f1"])
            ref, path = g0_f1(task_name, "stacking")
            items.append({"task": task_name, "kind": "arm_B:stacking",
                          "e1": e1_b, "g0": ref, "g0_path": path,
                          "abs_diff": None if ref is None else abs(e1_b - ref),
                          "pass": ref is not None and abs(e1_b - ref) <= tol})

    ok = bool(items) and all(i["pass"] for i in items)
    print("\n" + "=" * 78)
    print(f"G0 一致性校验（容差 {tol:g}）")
    print("=" * 78)
    for i in items:
        diff = "n/a" if i["abs_diff"] is None else f"{i['abs_diff']:.3e}"
        print(f"  [{'PASS' if i['pass'] else 'FAIL'}] {i['task']:24s} {i['kind']:18s} "
              f"E1={i['e1']:.12f}  G0={'None' if i['g0'] is None else format(i['g0'], '.12f')}  |Δ|={diff}")
    print(f"  → {sum(1 for i in items if i['pass'])}/{len(items)} 项通过，"
          f"整体 {'PASS' if ok else 'FAIL'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"g0_consistency{stem}.json").open("w", encoding="utf-8") as f:
        json.dump({"tolerance": tol, "all_pass": ok, "n_items": len(items),
                   "g0_metrics_root": str(G0_METRICS_ROOT.relative_to(REPO_ROOT)),
                   "items": items}, f, indent=2, ensure_ascii=False)
    print(f"  明细写入 {(out_dir / f'g0_consistency{stem}.json').relative_to(REPO_ROOT)}")
    return ok


# --------------------------------------------------------------------------- #
# 分片合并
# --------------------------------------------------------------------------- #
def merge_shards(out_root: Path, scope_stem: str, pool: list[dict], args) -> int:
    shard_dir = out_root / "shards"
    csvs = sorted(shard_dir.glob(f"e1_arms_raw{scope_stem}_shard*of*.csv"))
    if not csvs:
        raise SystemExit(f"没有可合并的分片：{shard_dir}/e1_arms_raw{scope_stem}_shard*of*.csv")
    totals = {int(p.stem.rsplit("of", 1)[1]) for p in csvs}
    if len(totals) != 1:
        raise SystemExit(f"分片总数不一致：{sorted(totals)}；请先清理 {shard_dir}")
    n_total = totals.pop()
    if len(csvs) != n_total:
        raise SystemExit(f"分片不全：找到 {len(csvs)}/{n_total} 个，缺片不得合并")

    frames, provs = [], []
    for p in csvs:
        frames.append(pd.read_csv(p, encoding="utf-8-sig"))
        pp = shard_dir / f"provenance{p.stem[len('e1_arms_raw'):]}.json"
        if not pp.exists():
            raise SystemExit(f"分片缺 provenance：{pp}")
        with pp.open(encoding="utf-8") as f:
            provs.append(json.load(f))

    # §19.2：分片必须来自同一份代码与同一套依赖，否则合并出来的表不可复现
    for field in ("git", "versions", "seeds", "arms", "inputs"):
        distinct = {json.dumps(pr[field], sort_keys=True, ensure_ascii=False) for pr in provs}
        if len(distinct) != 1:
            raise SystemExit(f"分片间 provenance 字段 {field!r} 不一致，拒绝合并：\n  "
                             + "\n  ".join(sorted(distinct)))

    df = pd.concat(frames, ignore_index=True)
    want = {t["name"] for t in pool}
    got = set(df["task"].unique())
    if got != want:
        raise SystemExit(f"合并后任务集不等于 --grid {args.grid} 全集："
                         f"缺 {sorted(want - got)[:5]}...（{len(want - got)} 个），"
                         f"多 {sorted(got - want)[:5]}...（{len(got - want)} 个）")
    order = {t["name"]: i for i, t in enumerate(pool)}
    df = df.sort_values(by=["task", "seed"], key=lambda s: s.map(order) if s.name == "task" else s,
                        kind="stable").reset_index(drop=True)
    # 经 CSV 往返后单元格是 numpy 标量 / NaN，json.dump 会失败；to_json 负责转原生类型与 null
    all_rows = json.loads(df.to_json(orient="records"))
    write_outputs(df, all_rows, out_root, scope_stem, verbose=False)

    merged = dict(provs[0])
    merged.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_line": {"argv": list(sys.argv), "executable": sys.executable,
                         "cwd": str(Path.cwd())},
        "scope": f"grid:{args.grid} (merged from {n_total} shards)",
        "output_dir": str(out_root.relative_to(REPO_ROOT)),
        "output_stem": scope_stem,
        "n_tasks": len(want),
        "tasks": [t["name"] for t in pool],
        "shards": [{"file": str(p.relative_to(REPO_ROOT)),
                    "argv": pr["command_line"]["argv"],
                    "n_tasks": pr["n_tasks"], "generated_utc": pr["generated_utc"]}
                   for p, pr in zip(csvs, provs)],
        "fold_assignments": [rec for pr in provs for rec in pr["fold_assignments"]],
    })
    with (out_root / f"provenance{scope_stem}.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"合并 {n_total} 个分片 → {len(df)} 行 / {len(want)} 任务，"
          f"写入 {out_root.relative_to(REPO_ROOT)}/")
    if args.check_g0:
        return 0 if check_against_g0(df, out_root, scope_stem) else 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="",
                   help="'all' 跑当前任务池全部，或逗号分隔任务名；"
                        "默认：主线模式只跑旗舰任务，--grid 模式跑该 scope 全部")
    p.add_argument("--grid", default="", choices=("",) + GRID_SCOPES,
                   help="跑 G0 网格 OOD 任务（s2plus=|S|>=2 的 120 个 / s1=|S|=1 的 30 个 / "
                        "all=150 个），输出到 results/e1_oof_arms_g0/")
    p.add_argument("--shard", default="", help="i/N：把任务列表按 [i-1::N] 取子集做分片并行")
    p.add_argument("--merge", action="store_true", help="合并 shards/ 下的分片输出（需 --grid）")
    p.add_argument("--check-g0", action="store_true",
                   help="与 results/g0_environment_grid/ 落盘 metrics.json 做 1e-6 一致性校验")
    p.add_argument("--seeds", default="42", help="逗号分隔，协议 §14 主表用 42,43,44,45,46")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    # 任务池：主线 11 任务，或 G0 网格子集（任务定义来自 environment_grid_experiment）
    if args.grid:
        pool, out_root, scope_stem = grid_tasks(args.grid), OUT_DIR_GRID, f"_{args.grid}"
    else:
        pool, out_root, scope_stem = TASKS, OUT_DIR, ""

    if args.merge:
        if not args.grid:
            raise SystemExit("--merge 需要同时给出 --grid")
        return merge_shards(out_root, scope_stem, pool, args)

    if not args.tasks:
        tasks = pool if args.grid else [t for t in pool if t["name"] == FLAGSHIP]
    elif args.tasks == "all":
        tasks = pool
    else:
        want = set(args.tasks.split(","))
        tasks = [t for t in pool if t["name"] in want]
        missing = want - {t["name"] for t in tasks}
        if missing:
            raise SystemExit(f"未知任务：{sorted(missing)}")

    shard_i = shard_n = None
    if args.shard:
        try:
            shard_i, shard_n = (int(v) for v in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard 格式应为 i/N，收到 {args.shard!r}") from None
        if not 1 <= shard_i <= shard_n:
            raise SystemExit(f"--shard 越界：{args.shard}")
        tasks = tasks[shard_i - 1::shard_n]      # 轮转切片：|S| 相近的任务分散到各片，负载更均衡
        if not tasks:
            raise SystemExit(f"分片 {args.shard} 为空")

    features = pd.read_csv(FEATURES_CSV)
    labels = sorted(features["label"].unique())

    # 部分运行（子集任务、分片或非标准种子集）**不得覆盖**全量结果的规范输出路径。
    # 2026-08-29 曾因此把一次 165 行的全量结果覆盖成 3 行 smoke 输出（数据靠 git 找回）。
    STD_SEEDS = [42, 43, 44, 45, 46]
    covers_pool = {t["name"] for t in tasks} == {t["name"] for t in pool} and shard_n is None
    # §14：G0 网格定位是覆盖失败模式，单种子（42）即为全量；主线 E1 全量是 5 种子。
    is_full = covers_pool and (seeds == [42] if args.grid else seeds == STD_SEEDS)
    if shard_n is not None:
        stem = f"{scope_stem}_shard{shard_i}of{shard_n}"
        out_dir = out_root / "shards"
        scope = f"grid:{args.grid} shard {shard_i}/{shard_n}"
    elif is_full:
        stem, out_dir = scope_stem, out_root
        scope = f"grid:{args.grid} full" if args.grid else "mainline full"
    else:
        stem = f"{scope_stem}_partial_{len(tasks)}task_{len(seeds)}seed"
        out_dir = out_root / "scratch"
        scope = f"partial ({len(tasks)} tasks x {len(seeds)} seeds)"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not is_full:
        print(f"  [部分运行] 输出写入 {out_dir.relative_to(REPO_ROOT)}/，"
              f"不触碰全量结果文件", flush=True)
    print(f"  任务 {len(tasks)}/{len(pool)}，种子 {seeds}，n_jobs={args.n_jobs}，"
          f"scope={scope}", flush=True)

    all_rows: list[dict] = []
    task_prov: list[dict] = []
    n_runs = len(tasks) * len(seeds)
    done = 0
    t_start = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        for seed in seeds:
            print(f"  [{i}/{len(tasks)}] {task['name']}  seed={seed}", flush=True)
            t0 = time.perf_counter()
            rows, prov = run_task(features, task, seed, args, labels)
            dt = time.perf_counter() - t0
            done += 1
            used = time.perf_counter() - t_start
            prov["elapsed_sec"] = round(dt, 3)
            all_rows.extend(rows)
            task_prov.append(prov)
            print(f"    ⏱ {dt:6.1f}s   已用 {used/60:6.1f}min   "
                  f"ETA {used / done * (n_runs - done) / 60:6.1f}min", flush=True)

    print(f"\n完成 {n_runs} 次运行，总耗时 {(time.perf_counter() - t_start) / 60:.1f} 分钟", flush=True)
    df = pd.DataFrame(all_rows)
    write_outputs(df, all_rows, out_dir, stem, verbose=not args.grid or len(tasks) <= 12)

    with (out_dir / f"provenance{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(build_provenance(args, seeds, tasks, task_prov, out_dir, stem, scope),
                  f, indent=2, ensure_ascii=False)

    if args.check_g0:
        return 0 if check_against_g0(df, out_dir, stem) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
