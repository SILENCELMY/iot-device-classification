"""E1 — OOF correction / Stacking robustness（协议 §12）

三臂设计。协议 §12 原定两臂（A 随机 K=5 / B 轮次分组）；本脚本增加 **A′ 诊断臂**
（随机折叠，折数与 B 对齐），用于把 A→B 的差异分解为两个独立来源：

    A  vs A′   = 纯折数效应（K=5 → K=n_rounds），划分方式都是随机
    A′ vs B    = 纯分组/泄漏效应（随机 → 按轮次分组），折数相同

不加 A′ 时 A→B 同时改变折数与分组方式，任何差异都无法归因。
A′ 仅用于归因分解，**不改变 §12 的预注册解读规则**（该规则只针对 B 臂）。
见 §23 Change Log 2026-08-28 条（澄清，非例外）。

用法：
    python code/scripts/analysis/e1_oof_arms.py                     # 旗舰任务，单种子
    python code/scripts/analysis/e1_oof_arms.py --seeds 42,43,44,45,46
    python code/scripts/analysis/e1_oof_arms.py --tasks all --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

import robust_iot_research as R  # noqa: E402

FEATURES_CSV = REPO_ROOT / "results" / "robust_v2" / "raw_all" / "features_raw_all_w10.csv"
OUT_DIR = REPO_ROOT / "results" / "e1_oof_arms"
BASE_MODELS = ["rf", "xgboost", "lightgbm"]

# 协议 §12 任务范围：11 个既有主线任务（type 取自 code/configs/research_experiments.json）。
# `single_round_*` 只有 1 个轮次，无法按轮次分组 → B 臂按 §12 退化为 window_start 时间块，
# 结果单独标注（见 SINGLE_ROUND 集合）。
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

# B 臂无法按轮次分组、改用时间块的任务（§12 要求单独标注）
SINGLE_ROUND = {"single_round_R2", "single_round_R3", "single_round_R4"}

FLAGSHIP = "loro_R2_R4_to_R3"

# 臂定义：(oof_mode, cv)。B 的 cv=5 只是上限，实际折数动态——轮次分组时 = min(cv, 轮次数)，
# 单轮时间块时 = min(cv, n)。A′ 的 cv 写 None，表示**运行时动态取 B 的有效折数**：
# 只有折数对齐，A→A′ 才能解释为纯折数效应、A′→B 才能解释为纯分组效应。
# 注意：当 B 的有效折数恰为 5 时（single_round 的时间块），A′ 退化为与 A 完全相同，
# 此时折数效应恒为 0、分组效应 = 全部差异，这是正确结果而非缺陷。
B_CV = 5
ARMS = {
    "A": ("random", 5),
    "A_prime": ("random", None),
    "B": ("grouped", B_CV),
}


def build_stacking(seed: int, n_jobs: int, n_classes: int, oof_mode: str, cv: int):
    model = R.build_model("stacking", seed, n_jobs, n_classes)
    model.oof_mode = oof_mode
    model.cv = cv
    return model


def effective_folds(model, x, y, train_round, window_start) -> int:
    return sum(1 for _ in model._splitter(x, y, train_round, window_start))


def run_task(features: pd.DataFrame, task: dict, seed: int, args, labels: list[str]) -> list[dict]:
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
    # 先测 B 臂的有效折数，A′ 与之对齐（见 ARMS 注释）
    b_probe = build_stacking(seed, args.n_jobs, len(labels), "grouped", B_CV)
    b_folds = effective_folds(b_probe, x_train[cols], ye_train, tr_round, tr_ws)

    for arm, (oof_mode, cv) in ARMS.items():
        cv_used = b_folds if cv is None else cv
        model = build_stacking(seed, args.n_jobs, len(labels), oof_mode, cv_used)
        k = effective_folds(model, x_train[cols], ye_train, tr_round, tr_ws)
        model.fit(x_train[cols], ye_train, train_round=tr_round, window_start=tr_ws)
        pred = enc.inverse_transform(model.predict(x_test[cols]).astype(int))
        metrics, _, _ = R.metric_summary(y_test.to_numpy(), pred, labels)
        rows.append({
            "task": task["name"], "seed": seed, "arm": arm,
            "oof_mode": oof_mode, "cv_requested": cv_used, "folds_effective": k,
            "b_folds_effective": b_folds,
            "aprime_aligned_with_b": bool(k == b_folds) if arm == "A_prime" else None,
            "n_train_rounds": int(len(np.unique(tr_round))),
            "b_split_basis": ("time_block" if task["name"] in SINGLE_ROUND else "round_group"),
            "stacking_f1": metrics["macro_f1"],
            "best_base_model": best_base_model, "best_base_f1": best_base_f1,
            "gain_absolute": metrics["macro_f1"] - best_base_f1,
            **{f"base_f1_{m}": v for m, v in base_f1.items()},
        })
        print(f"    {arm:8s} oof={oof_mode:8s} K={k}"
              f"{' (=B)' if arm == 'A_prime' and k == b_folds else '':5s}  "
              f"stacking_f1={metrics['macro_f1']:.6f}  gain={rows[-1]['gain_absolute']:+.6f}",
              flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=FLAGSHIP,
                   help="'all' 跑全部，或逗号分隔任务名；默认只跑旗舰任务")
    p.add_argument("--seeds", default="42", help="逗号分隔，协议 §14 主表用 42,43,44,45,46")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.tasks == "all":
        tasks = TASKS
    else:
        want = set(args.tasks.split(","))
        tasks = [t for t in TASKS if t["name"] in want]
        missing = want - {t["name"] for t in tasks}
        if missing:
            raise SystemExit(f"未知任务：{sorted(missing)}")

    features = pd.read_csv(FEATURES_CSV)
    labels = sorted(features["label"].unique())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 部分运行（子集任务或非标准种子集）**不得覆盖**全量结果的规范输出路径。
    # 2026-08-29 曾因此把一次 165 行的全量结果覆盖成 3 行 smoke 输出（数据靠 git 找回）。
    STD_SEEDS = [42, 43, 44, 45, 46]
    is_full = ({t["name"] for t in tasks} == {t["name"] for t in TASKS}
               and seeds == STD_SEEDS)
    if is_full:
        stem = ""
        out_dir = OUT_DIR
    else:
        stem = f"_partial_{len(tasks)}task_{len(seeds)}seed"
        out_dir = OUT_DIR / "scratch"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [部分运行] 输出写入 {out_dir.relative_to(REPO_ROOT)}/，"
              f"不触碰全量结果文件", flush=True)

    all_rows = []
    for task in tasks:
        for seed in seeds:
            print(f"  {task['name']}  seed={seed}", flush=True)
            all_rows.extend(run_task(features, task, seed, args, labels))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / f"e1_arms_raw{stem}.csv", index=False, encoding="utf-8-sig")

    # 归因分解：以 (task, seed) 为单位配对
    piv = df.pivot_table(index=["task", "seed"], columns="arm", values="gain_absolute")
    piv["fold_effect_A_to_Aprime"] = piv["A_prime"] - piv["A"]
    piv["group_effect_Aprime_to_B"] = piv["B"] - piv["A_prime"]
    piv["total_A_to_B"] = piv["B"] - piv["A"]
    piv = piv.reset_index()
    piv.to_csv(out_dir / f"e1_decomposition{stem}.csv", index=False, encoding="utf-8-sig")

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
                   "seeds": seeds, "tasks": [t["name"] for t in tasks],
                   "rows": all_rows}, f, indent=2, ensure_ascii=False)
    print(f"\n输出：{out_dir.relative_to(REPO_ROOT)}/"
          f"{{e1_arms_raw{stem}.csv, e1_decomposition{stem}.csv, e1_arms{stem}.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
