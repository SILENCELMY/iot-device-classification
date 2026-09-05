"""【探索性,非协议】D-Link 摄像头数据集特征抽取。

数据：Mendeley 84cc8grtkt，D-Link 14 台设备 5 个月每日采集（sha256 已校验）。
只用 8 台摄像头（2×Cam DCS-936L + 6×DayCam DCS-930L），因为：
  插座只采了 15 天、HomeHub 只采了 8 天 → 12 类任务的共同天数只有 3 天 / 0 天
  8 台摄像头共同覆盖 51 天 → 可用
门窗传感器无独立目录（只在 hub 流量里），共用流场景用不了。
空口帧（Network_Frames）实测 100% 是 Probe Request，无业务流量，用不了。

布局差异：D-Link 是【每设备每天一个 pcap】，而 UNSW/CIC 是每天一个含所有设备。
故直接对每个 (设备,天) 文件调 EG.extract_features_for_file —— 该函数内部
day_origin = 本文件首包，正好符合"一文件一设备一天"的语义。

价值：6 台同型号 DayCam（其中 4/5/6 连号同批次）是我们手上最大的同型号组，
CIC 最大只有 4 台 Gosund Plug。这是最纯粹的"同硬件同固件"身份型测试。
"""
from __future__ import annotations
import sys, time, re
from pathlib import Path
import pandas as pd

REPO = "/home/lmy/iot-device-classification"
sys.path.insert(0, REPO + "/code/scripts/core")
import extract_features_generic as EG

BASE = Path(REPO + "/dataset/dlink/D-Link-IoT-Datasets/Network_Packets")
OUT  = Path(REPO + "/results/dlink_cams")
DEVS = ["D-LinkCam1", "D-LinkCam2"] + [f"D-LinkDayCam{i}" for i in range(1, 7)]
MACS = {
    "D-LinkCam1": "b2:c5:54:44:0f:a4", "D-LinkCam2": "b2:c5:54:44:0f:11",
    "D-LinkDayCam1": "b0:c5:54:46:48:5d", "D-LinkDayCam2": "b0:c5:54:3d:3e:93",
    "D-LinkDayCam3": "b0:c5:54:3d:3f:8f", "D-LinkDayCam4": "b0:c5:54:42:8f:a6",
    "D-LinkDayCam5": "b0:c5:54:42:8f:88", "D-LinkDayCam6": "b0:c5:54:42:8f:e5",
}
WIN = 10.0
MIN_PKT = 2
N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def days_of(dev):
    out = set()
    for f in (BASE / dev).glob("*.pcap"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.pcap$", f.name)
        if m:
            out.add(m.group(1))
    return out


def main():
    t0 = time.time()
    common = sorted(set.intersection(*[days_of(d) for d in DEVS]))
    print(f"8 台共同覆盖 {len(common)} 天：{common[0]} .. {common[-1]}", flush=True)
    step = max(1, len(common) // N_DAYS)
    days = common[::step][:N_DAYS]
    print(f"本次抽 {len(days)} 天（均匀取样）：{days}", flush=True)

    mac_map = {v.lower(): k for k, v in MACS.items()}
    frames = []
    for dev in DEVS:
        for day in days:
            hits = list((BASE / dev).glob(f"*{day}.pcap"))
            if not hits:
                continue
            df = EG.extract_features_for_file(hits[0], day, mac_map, WIN, MIN_PKT, verbose=False)
            if df.empty:
                print(f"  [空] {dev} {day}", flush=True)
                continue
            frames.append(df)
        n = sum(len(f) for f in frames)
        print(f"  {dev:16s} 累计 {n} 窗   {time.time()-t0:.0f}s", flush=True)

    if not frames:
        sys.exit("没有抽到任何窗")
    R = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"features_dlink_w10_{len(days)}d.csv"
    R.to_csv(p, index=False)
    print(f"\n写出 {p}   {len(R)} 行 × {len(R.columns)} 列", flush=True)
    print("\n=== 逐设备逐天窗口数 ===", flush=True)
    print(R.groupby(["device", "day"]).size().unstack(fill_value=0).to_string(), flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
