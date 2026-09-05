"""【探索性,非协议】自采数据按多个窗长重抽特征。只做抽取，不跑任何模型。

动机：10 s 窗是"跨窗特征"最直接的形式——同一个抽取器在更长跨度上算 interarrival/burst，
周期性与心跳节律这些 10 s 内不存在的结构会自动进特征，且完全可部署
（判决延迟从 10 s 变 N s，与平滑同性质）。

窗长只取 10 s 的整数倍：占空比实测基频 10.06 s（duty-cycle-is-10s-window-already-matched），
30 s = 3 个周期、60 s = 6 个周期，周期对齐保持不破，同时纳入跨周期变化。
取 15/25 这类不对齐的值会把周期结构切碎，那正是那条记忆反对的。

产物：results/robust_v2/raw_all/features_raw_all_w30.csv 等，不覆盖 w10。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path("/home/lmy/iot-device-classification")
sys.path.insert(0, str(REPO / "code/scripts/core"))
import robust_iot_research as R          # noqa: E402

CONFIG   = REPO / "code/configs/research_experiments.json"
DATASET  = REPO / "dataset"
OUTDIR   = REPO / "results/robust_v2/raw_all"
ROUNDS   = {"R2","R3","R4","R5","R6","R7"}
WINDOWS  = [30.0, 60.0]                  # 10 s 已有
FILTER   = "raw_all"

def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    have = set(cfg["rounds"]) & ROUNDS
    missing = ROUNDS - have
    assert not missing, f"配置里缺轮次 {missing}"
    print(f"轮次 {sorted(have)}   设备 {list(cfg['device_dirs'])}", flush=True)
    for w in WINDOWS:
        out = OUTDIR / f"features_{FILTER}_w{w:g}.csv"
        if out.exists():
            print(f"[跳过] {out.name} 已存在", flush=True)
            continue
        t0 = time.time()
        print(f"\n=== 窗长 {w:g} s ===", flush=True)
        df = R.build_feature_table(
            config=cfg, dataset_root=DATASET, output_dir=OUTDIR,
            required=ROUNDS, filter_mode=FILTER, window_seconds=w,
            min_packets_per_window=2, force_extract=True)
        cols = R.feature_columns(df)
        print(f"  → {out.name}  {len(df)} 行  {len(cols)} 特征列  {time.time()-t0:.0f}s", flush=True)
        print(df.groupby(["round","label"]).size().unstack(fill_value=0).to_string(), flush=True)

if __name__ == "__main__":
    main()
