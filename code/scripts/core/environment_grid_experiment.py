"""G0 环境组合网格（协议 §8.5 第 5 条、§20.1、§21）

对每个目标环境 `e ∈ {R2..R7}`，源环境取 `S ⊆ {R2..R7}\\{e}`，`1 ≤ |S| ≤ 3`：
每个 `e` 有 C(5,1)+C(5,2)+C(5,3) = 25 个组合，共 **150 个 OOD 任务**；
另加 **6 个同环境 IID 任务**，合计协议 §8.5.5 所称的 **156 个任务**。

`|S|=1` 的 30 个有序对构成**同质**的环境×环境拓扑矩阵（§8.5.5），
替代已废弃的六环境 pairwise 矩阵（§4.4）。

**任务数 156，运行数 162**：6 个 IID 任务按 §8.5.4 各跑**两种划分**并并列报告——
  · `random`     —— 随机分层划分，只能称为「session 内上界」；
  · `time_block` —— 按 `window_start` 在每个 source_file 内分块，无相邻窗口泄漏。
两者是同一个 IID 任务的两次测量，不是两个任务；协议的 156 计的是任务。

**种子固定为 1 个**（§14：G0 用途是覆盖失败模式，不用于显著性结论）。

不重写训练逻辑（§20.3）：直接复用 `robust_iot_research` 的 `build_feature_table`、
`task_data`、`evaluate_task`、`build_model`、`feature_columns`、`sample_balanced`。
任务 dict 程序化生成，**不改 config JSON**（`evaluate_task` 只吃
`{"name","type","train_rounds","test_rounds"}`）。

用法：
    python code/scripts/core/environment_grid_experiment.py --smoke   # 5 任务 smoke test
    python code/scripts/core/environment_grid_experiment.py           # 全量 156
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))

import robust_iot_research as R  # noqa: E402

ENVIRONMENTS = ["R2", "R3", "R4", "R5", "R6", "R7"]
MAX_SOURCES = 3
FILTER_MODE = "raw_all"
WINDOW_SECONDS = 10.0
SEED = 42                                    # §14：G0 单种子
MODELS = ["rf", "xgboost", "lightgbm", "stacking"]
CACHE_SRC = REPO_ROOT / "results" / "robust_v2" / "raw_all" / "features_raw_all_w10.csv"
OUT_ROOT = REPO_ROOT / "results" / "g0_environment_grid"

# smoke test 的 5 个任务（§22.2 第 8 项：通过后再全量）。
# 覆盖四种形态：|S|=1 / |S|=2 / |S|=3 / IID 两种划分。
SMOKE_TASKS = [
    "g0_R2_to_R3",
    "g0_R2_R4_to_R3",
    "g0_R2_R4_R5_to_R3",
    "g0_iid_R2_random",
    "g0_iid_R2_time_block",
]


def build_task_grid() -> list[dict]:
    """生成运行列表：150 OOD + 6 IID×2 划分 = **162 次运行**，对应协议的 **156 个任务**。

    名称编码源→目标，便于后续按 |S| 聚合。
    """
    tasks: list[dict] = []
    for target in ENVIRONMENTS:
        sources = [r for r in ENVIRONMENTS if r != target]
        for k in range(1, MAX_SOURCES + 1):
            for combo in combinations(sources, k):
                tasks.append({
                    "name": f"g0_{'_'.join(combo)}_to_{target}",
                    "type": "fixed_split",
                    "train_rounds": list(combo),
                    "test_rounds": [target],
                    "n_sources": k,
                    "target_env": target,
                    "grid_kind": "ood",
                })
    # 6 个 IID 任务，每个跑两种划分（§8.5.4 要求并列报告）→ 12 次运行、6 个任务
    for env in ENVIRONMENTS:
        for mode in ("random", "time_block"):
            tasks.append({
                "name": f"g0_iid_{env}_{mode}",
                "type": "single_round",
                "rounds": [env],
                "split_mode": mode,
                "n_sources": 0,
                "target_env": env,
                "grid_kind": f"iid_{mode}",
                "iid_task_id": f"g0_iid_{env}",   # 两次运行共享的任务标识
            })
    return tasks


def count_protocol_tasks(runs: list[dict]) -> int:
    """协议口径的任务数：OOD 每次运行 = 1 任务；IID 的两种划分合计 1 任务。"""
    ood = sum(1 for t in runs if t["grid_kind"] == "ood")
    iid = len({t["iid_task_id"] for t in runs if t["grid_kind"].startswith("iid")})
    return ood + iid


def prepare_output_root(out_root: Path) -> None:
    """把特征缓存软链到新 output root，命中 build_feature_table 的缓存快路径。

    §20.3：`build_feature_table` 检查缓存中轮次是否齐全，齐全则直接返回，
    **不触发 tshark 重抽**。dataset/ 被权限锁定，一旦走到抽取分支会直接失败，
    所以这里先硬性校验缓存含全部 6 个轮次，缺则立刻报错而不是让它去读 pcap。
    """
    cache_dir = out_root / FILTER_MODE
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"features_{FILTER_MODE}_w{WINDOW_SECONDS:g}.csv"

    if not CACHE_SRC.exists():
        raise SystemExit(f"特征缓存不存在：{CACHE_SRC}")
    cached_rounds = set(pd.read_csv(CACHE_SRC, usecols=["round"])["round"].unique())
    missing = set(ENVIRONMENTS) - cached_rounds
    if missing:
        raise SystemExit(
            f"缓存缺少轮次 {sorted(missing)}；dataset/ 被权限锁定，无法重抽特征。"
            f"中止而不是让 build_feature_table 去读 pcap。"
        )
    if not dst.exists():
        try:
            dst.symlink_to(CACHE_SRC)
        except OSError:
            import shutil
            shutil.copy2(CACHE_SRC, dst)
    print(f"  特征缓存就位：{dst.relative_to(REPO_ROOT)} -> {CACHE_SRC.name}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="G0 环境组合网格（协议 §8.5）")
    p.add_argument("--smoke", action="store_true",
                   help=f"只跑 {len(SMOKE_TASKS)} 个任务的 smoke test（§22.2 第 8 项）")
    p.add_argument("--tasks", default="", help="逗号分隔任务名，覆盖 --smoke")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--max-rows", type=int, default=10 ** 9)
    args_cli = p.parse_args()

    grid = build_task_grid()
    n_proto = count_protocol_tasks(grid)
    by_name = {t["name"]: t for t in grid}
    if args_cli.tasks:
        want = [s.strip() for s in args_cli.tasks.split(",")]
        unknown = [w for w in want if w not in by_name]
        if unknown:
            raise SystemExit(f"未知任务：{unknown}")
        tasks, scope = [by_name[w] for w in want], "partial"
    elif args_cli.smoke:
        tasks, scope = [by_name[n] for n in SMOKE_TASKS], "smoke"
    else:
        tasks, scope = grid, "full"

    # 部分运行不得覆盖全量结果（2026-08-29 在 e1_oof_arms.py 上踩过这个坑）
    out_root = OUT_ROOT if scope == "full" else OUT_ROOT / "scratch" / scope
    print(f"G0 环境组合网格：{len(tasks)}/{len(grid)} 次运行，scope={scope}")
    print(f"  协议口径任务数：{n_proto}（150 OOD + 6 IID；IID 各跑 2 种划分 → {len(grid)} 次运行）")
    print(f"  输出根目录：{out_root.relative_to(REPO_ROOT)}")
    print(f"  模型：{args_cli.models}   种子：{SEED}（§14 单种子）")
    prepare_output_root(out_root)

    models = [m.strip() for m in args_cli.models.split(",")]
    args = argparse.Namespace(
        config="", dataset_root=Path("dataset"), output_root=out_root,
        window_seconds=WINDOW_SECONDS, min_packets_per_window=2,
        test_size=args_cli.test_size, random_state=SEED, n_jobs=args_cli.n_jobs,
        max_rows=args_cli.max_rows, filter_modes=FILTER_MODE,
        feature_mode="all", disable_feature_selection=True, force_extract=False,
    )

    features = R.build_feature_table(
        config={}, dataset_root=Path("dataset"), output_dir=out_root / FILTER_MODE,
        required=set(ENVIRONMENTS), filter_mode=FILTER_MODE,
        window_seconds=WINDOW_SECONDS, min_packets_per_window=2, force_extract=False,
    )
    labels = sorted(features["label"].unique())
    print(f"  特征表：{features.shape}，类别 {labels}\n", flush=True)

    summaries: list[dict] = []
    rankings: list[pd.DataFrame] = []
    t_start = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        t0 = time.perf_counter()
        R.evaluate_task(features, task, FILTER_MODE, models, args, {},
                        out_root, labels, summaries, rankings)
        # 把网格元信息补进 summary（evaluate_task 不认识这些键）
        for s in summaries:
            if s["task"] == task["name"]:
                s.setdefault("n_sources", task["n_sources"])
                s.setdefault("target_env", task["target_env"])
                s.setdefault("grid_kind", task["grid_kind"])
                s.setdefault("split_mode", task.get("split_mode", "fixed"))
        dt = time.perf_counter() - t0
        done = time.perf_counter() - t_start
        eta = done / i * (len(tasks) - i)
        print(f"  [{i:3d}/{len(tasks)}] {task['name']:32s} {dt:6.1f}s  "
              f"已用 {done/60:5.1f}min  ETA {eta/60:5.1f}min", flush=True)

    df = pd.DataFrame(summaries)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    with (out_root / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    # |S|=1 的同质 6×6 环境拓扑矩阵（§8.5.5，替代已废弃的六环境矩阵）
    pair = df[(df.get("n_sources") == 1) & (df["model"] == "rf")]
    if not pair.empty:
        mat = pd.DataFrame(index=ENVIRONMENTS, columns=ENVIRONMENTS, dtype=float)
        for _, r in pair.iterrows():
            src = r["train_rounds"][0] if isinstance(r["train_rounds"], list) else eval(str(r["train_rounds"]))[0]
            mat.loc[src, r["target_env"]] = r["macro_f1"]
        mat.to_csv(out_root / "env_topology_matrix_rf.csv", encoding="utf-8-sig")
        print(f"\n  |S|=1 拓扑矩阵（rf, macro_f1）已写入 env_topology_matrix_rf.csv"
              f"，非空格数 {int(mat.notna().sum().sum())}/30")

    print(f"\n完成 {len(tasks)} 任务，{len(summaries)} 条结果，"
          f"总耗时 {(time.perf_counter()-t_start)/60:.1f} 分钟")
    print(f"输出：{out_root.relative_to(REPO_ROOT)}/"
          "{summary_metrics.csv, summary_metrics.json, env_topology_matrix_rf.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
