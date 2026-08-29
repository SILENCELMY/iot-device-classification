#!/usr/bin/env python3
"""通用链路层（Ethernet）特征提取器 —— 协议 §20.1 新建项。

用途
----
UNSW IoT traces（网关侧 Ethernet 抓包，https://iotanalytics.unsw.edu.au/iottraces.html）的
每设备 / 10 秒非重叠窗口特征提取。**只做通用特征族**：

    len_*  /  interarrival_*  /  burst_*  /  subwin_*  /  up_*、down_*、up_down_*

802.11 专属特征（`subtype_*_ratio` / `retry_ratio` / `rssi_*` / `bssid_known` /
`unique_sa_count` / `unique_da_count` / `data_ratio` / `mgmt_ratio` / `ctrl_ratio` /
`null_data_ratio` / `qos_data_ratio`）在 Ethernet 抓包上不存在，**一概不做**（协议 §16.4）。

与主线实现的关系
----------------
特征定义与命名尽最大可能对齐 ``code/scripts/core/robust_iot_research.py``
的 ``summarize_window()``（第 241-439 行）。逐特征对照见
``results/unsw_pilot/FEATURE_ALIGNMENT.md``。所有统计辅助函数
（``quantile`` / ``safe_mean`` / ``safe_std`` / ``coefficient_of_variation``）
均逐字复制主线实现，以保证同一输入得到同一数值。

**唯二的语义改写**（链路层不同，无法照搬，已在对齐表标注）：

1. 方向判定：主线用 802.11 的 ``TA/DA vs BSSID``；本脚本用 ``eth.src/eth.dst vs 设备 MAC``。
   语义一致（up = 设备发出，down = 设备收到），机制不同。
2. ``side_packet_ratio`` / ``other_packet_ratio``：主线中 side = 802.11 管理/控制帧。
   Ethernet 无管理/控制帧，且按设备 MAC 归流后每个包必为 up 或 down，
   故这两列在本脚本中**恒为 0**（保留列名以维持列对齐，建模时会被 RF 自然忽略）。

协议约束
--------
- §16.2：pilot 以 **pcap** 为准。本脚本只吃 pcap，不吃官方 CSV。
- §16.1：网关 MAC ``14:cc:20:51:33:ea``，不作为待分类设备。

用法
----
    python3 extract_features_generic.py \
        --pcap-dir  dataset/unsw/pcap \
        --mac-map   dataset/unsw/device_mac_map.csv \
        --output    results/unsw_pilot/features_unsw_w10.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

# 主线 robust_iot_research.py 的 TSHARK_FIELDS 前两项 + Ethernet 地址。
# 802.11 字段（wlan.*/radiotap.*）一概不取。
TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "eth.src",
    "eth.dst",
]

# 网关 MAC（协议 §16.1）。作为设备的对端，不作为待分类设备。
GATEWAY_MAC = "14:cc:20:51:33:ea"

# 输出中的非特征列（对应主线 META_COLUMNS，按 UNSW 的环境轴改名：round -> day）
META_COLUMNS = {
    "device",
    "day",
    "label",
    "source_file",
    "window_id",
    "window_start",
    "window_end",
    "window_start_epoch",
}

# 主线的子窗口个数（robust_iot_research.py L410: sub_count = 5）
SUB_COUNT = 5

# 主线的 burst 阈值（L323: interarrival <= 0.10）与长间隔阈值（L325: >= 1.0）
BURST_IA_THRESHOLD = 0.10
LONG_GAP_THRESHOLD = 1.0


# --------------------------------------------------------------------------
# 统计辅助 —— 逐字复制 robust_iot_research.py L223-238，保证数值一致
# --------------------------------------------------------------------------

def quantile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.quantile(q))


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else 0.0


def safe_std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def coefficient_of_variation(mean_value: float, std_value: float) -> float:
    return float(std_value / abs(mean_value)) if abs(mean_value) > 1e-12 else 0.0


# --------------------------------------------------------------------------
# MAC 映射
# --------------------------------------------------------------------------

def load_mac_map(path: Path, include_non_iot: bool = False) -> dict[str, str]:
    """读取 device_mac_map.csv，返回 {mac(lower) -> device_id}。

    默认只保留 ``is_iot == 1`` 的条目；网关始终排除（它是对端，不是待分类设备）。
    """
    table = pd.read_csv(path, dtype=str).fillna("")
    mapping: dict[str, str] = {}
    for _, row in table.iterrows():
        mac = row["mac"].strip().lower()
        if not mac or mac == GATEWAY_MAC:
            continue
        if not include_non_iot and row.get("is_iot", "0").strip() != "1":
            continue
        mapping[mac] = row["device_id"].strip()
    return mapping


# --------------------------------------------------------------------------
# tshark 读取
# --------------------------------------------------------------------------

def tshark_command(pcap_path: Path, display_filter: str | None) -> list[str]:
    command = ["tshark", "-r", str(pcap_path)]
    if display_filter:
        command.extend(["-Y", display_filter])
    command.extend(["-T", "fields"])
    for field in TSHARK_FIELDS:
        command.extend(["-e", field])
    # 与主线一致：无表头、tab 分隔、occurrence=a
    command.extend(["-E", "header=n", "-E", "separator=\t", "-E", "occurrence=a"])
    return command


def read_packets(
    pcap_path: Path,
    mac_set: set[str],
    chunk_rows: int = 4_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """流式跑 tshark，返回 [time_epoch, length, eth_src, eth_dst] 的 DataFrame。

    只保留 ``eth.src`` 或 ``eth.dst`` 命中已知设备 MAC 的包（在 Python 侧过滤，
    避免构造超长的 tshark 显示过滤器）。
    """
    command = tshark_command(pcap_path, None)
    if verbose:
        print(f"[tshark] {' '.join(command[:6])} ... ({len(TSHARK_FIELDS)} fields)", flush=True)

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1 << 20,
    )
    frames: list[pd.DataFrame] = []
    total_read = 0
    try:
        reader = pd.read_csv(
            proc.stdout,
            sep="\t",
            header=None,
            names=["time_epoch", "length", "eth_src", "eth_dst"],
            dtype={"time_epoch": "float64", "length": "float64",
                   "eth_src": "string", "eth_dst": "string"},
            na_values=[""],
            keep_default_na=False,
            engine="c",
            on_bad_lines="skip",
            chunksize=chunk_rows,
            quoting=csv.QUOTE_NONE,
        )
        for chunk in reader:
            total_read += len(chunk)
            # occurrence=a 时一个字段可能是 "aa:..,bb:.." 逗号串（VLAN/隧道），取第一个
            for col in ("eth_src", "eth_dst"):
                s = chunk[col].fillna("")
                # 只在确有逗号时才做 split，省时间
                if s.str.contains(",", regex=False).any():
                    s = s.str.split(",", n=1).str[0]
                chunk[col] = s.str.lower()
            chunk = chunk.dropna(subset=["time_epoch", "length"])
            chunk = chunk[chunk["length"] > 0]
            keep = chunk["eth_src"].isin(mac_set) | chunk["eth_dst"].isin(mac_set)
            chunk = chunk[keep]
            if not chunk.empty:
                frames.append(chunk.reset_index(drop=True))
            if verbose:
                print(f"  ... read {total_read:,} packets, kept {sum(len(f) for f in frames):,}",
                      flush=True)
    finally:
        stderr_tail = ""
        if proc.stderr is not None:
            stderr_tail = proc.stderr.read()[-2000:]
        proc.stdout.close() if proc.stdout else None
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"tshark exited {rc} on {pcap_path}\n{stderr_tail}")

    if not frames:
        return pd.DataFrame(columns=["time_epoch", "length", "eth_src", "eth_dst"])
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[tshark] total {total_read:,} packets read, {len(out):,} kept "
              f"(matched a known device MAC)", flush=True)
    return out


# --------------------------------------------------------------------------
# 归流：一个包按 eth.src / eth.dst 进入相应设备的流
# --------------------------------------------------------------------------

def assign_device_streams(packets: pd.DataFrame, mac_map: dict[str, str]) -> pd.DataFrame:
    """把包展开成 (device, direction) 视图。

    - ``eth.src`` 命中设备 -> 该设备的 up 包；
    - ``eth.dst`` 命中设备 -> 该设备的 down 包；
    - 两端都是已知 IoT MAC 的局域网内流量会产生 2 行（对 A 记 up，对 B 记 down）——
      这是正确的每设备流语义，不是重复计数。
    """
    up = packets.loc[packets["eth_src"].isin(mac_map)].copy()
    up["device"] = up["eth_src"].map(mac_map)
    up["is_up"] = True

    down = packets.loc[packets["eth_dst"].isin(mac_map)].copy()
    down["device"] = down["eth_dst"].map(mac_map)
    down["is_up"] = False

    streams = pd.concat([up, down], ignore_index=True)
    return streams[["time_epoch", "length", "device", "is_up"]]


# --------------------------------------------------------------------------
# 单窗口特征 —— 对齐 robust_iot_research.py summarize_window()
# --------------------------------------------------------------------------

def summarize_window(
    group: pd.DataFrame,
    device: str,
    day: str,
    source_file: Path,
    window_id: int,
    window_seconds: float,
    day_origin: float,
) -> dict[str, Any]:
    """与主线 summarize_window 同名同序，只保留通用特征族。

    ``group`` 需含列：``time_epoch``、``length``、``is_up``、``relative_time``。
    """
    # --- 主线 L251-253：先按时间排序再做差分 ---
    group = group.sort_values("time_epoch", kind="stable")
    lengths = group["length"].astype(float)
    times = group["time_epoch"].astype(float)
    interarrival = times.diff().dropna()

    len_mean = safe_mean(lengths)
    len_std = safe_std(lengths)
    ia_mean = safe_mean(interarrival)
    ia_std = safe_std(interarrival)

    row: dict[str, Any] = {
        # --- 元数据（主线 L267-276 的对应项） ---
        "device": device,                    # 主线 "label"
        "label": device,                     # 冗余保留，便于直接复用主线以 label 为类名的代码
        "day": day,                          # 主线 "round"（UNSW 的环境轴 = 天）
        "source_file": str(source_file),
        "window_id": window_id,
        "window_start": float(group["relative_time"].min()),
        "window_end": float(group["relative_time"].max()),
        "window_start_epoch": float(day_origin + window_id * window_seconds),
        "packet_count": int(len(group)),
        "byte_count": float(lengths.sum()),

        # --- len_*（主线 L277-289，13 项） ---
        "len_mean": len_mean,
        "len_std": len_std,
        "len_min": float(lengths.min()),
        "len_max": float(lengths.max()),
        "len_range": float(lengths.max() - lengths.min()),
        "len_cv": coefficient_of_variation(len_mean, len_std),
        "len_p10": quantile(lengths, 0.10),
        "len_p25": quantile(lengths, 0.25),
        "len_p50": quantile(lengths, 0.50),
        "len_p75": quantile(lengths, 0.75),
        "len_p90": quantile(lengths, 0.90),
        "len_p95": quantile(lengths, 0.95),
        "len_iqr": quantile(lengths, 0.75) - quantile(lengths, 0.25),

        # --- interarrival_*（主线 L290-301，12 项） ---
        "interarrival_mean": ia_mean,
        "interarrival_std": ia_std,
        "interarrival_min": float(interarrival.min()) if len(interarrival) else 0.0,
        "interarrival_max": float(interarrival.max()) if len(interarrival) else 0.0,
        "interarrival_cv": coefficient_of_variation(ia_mean, ia_std),
        "interarrival_p10": quantile(interarrival, 0.10),
        "interarrival_p25": quantile(interarrival, 0.25),
        "interarrival_p50": quantile(interarrival, 0.50),
        "interarrival_p75": quantile(interarrival, 0.75),
        "interarrival_p90": quantile(interarrival, 0.90),
        "interarrival_p95": quantile(interarrival, 0.95),
        "interarrival_iqr": quantile(interarrival, 0.75) - quantile(interarrival, 0.25),
    }

    # --- burst_*（主线 L323-348，8 项） ---
    burst_interarrival = interarrival <= BURST_IA_THRESHOLD
    row["burst_packet_ratio"] = float(burst_interarrival.mean()) if len(burst_interarrival) else 0.0
    row["long_gap_ratio"] = (
        float((interarrival >= LONG_GAP_THRESHOLD).mean()) if len(interarrival) else 0.0
    )

    burst_starts = (
        np.concatenate([[True], ~burst_interarrival.to_numpy()[:-1]])
        if len(burst_interarrival) else np.array([True])
    )
    burst_id = np.cumsum(burst_starts.astype(int))
    burst_series = pd.Series(burst_id, index=interarrival.index)
    burst_sizes = burst_series.value_counts(sort=False)
    burst_count = int(len(burst_sizes))
    if burst_count > 0:
        burst_size_arr = burst_sizes.to_numpy()
        burst_size_mean = float(burst_size_arr.mean())
        burst_size_std = float(burst_size_arr.std(ddof=0)) if len(burst_size_arr) > 1 else 0.0
        burst_size_max = int(burst_size_arr.max())
        burst_size_min = int(burst_size_arr.min())
        burst_packet_fraction = float(burst_interarrival.sum()) / max(len(interarrival), 1)
    else:
        burst_size_mean = burst_size_std = burst_packet_fraction = 0.0
        burst_size_max = burst_size_min = 0
    row["burst_count"] = burst_count
    row["burst_size_mean"] = burst_size_mean
    row["burst_size_std"] = burst_size_std
    row["burst_size_max"] = burst_size_max
    row["burst_size_min"] = burst_size_min
    row["burst_packet_fraction"] = burst_packet_fraction

    # --- 方向（主线 L350-384）---------------------------------------------
    # 主线用 802.11 的 TA/DA vs BSSID；此处用 eth.src/eth.dst vs 设备 MAC。
    # 语义一致：up = 设备发出，down = 设备收到。
    # Ethernet 无管理/控制帧，且归流后每个包必为 up 或 down，
    # 故 side_/other_packet_ratio 恒为 0（保留列名以维持与主线的列对齐）。
    direction_label = np.where(group["is_up"].to_numpy(), "up", "down")
    n = len(group)
    row["up_packet_ratio"] = float((direction_label == "up").sum()) / max(n, 1)
    row["down_packet_ratio"] = float((direction_label == "down").sum()) / max(n, 1)
    row["side_packet_ratio"] = 0.0     # 退化：Ethernet 无 mgmt/ctrl 帧
    row["other_packet_ratio"] = 0.0    # 退化：归流后无「既非 up 也非 down」的包
    row["up_down_ratio"] = (
        float((direction_label == "up").sum()) / max(float((direction_label == "down").sum()), 1.0)
    )

    # --- 分方向的 len / ia 统计（主线 L386-408，11 项） ---
    up_mask_full = pd.Series(direction_label == "up", index=group.index)
    down_mask_full = pd.Series(direction_label == "down", index=group.index)
    ia_index = interarrival.index
    up_ia_mask = up_mask_full.reindex(ia_index, fill_value=False)
    down_ia_mask = down_mask_full.reindex(ia_index, fill_value=False)
    up_len = lengths[up_mask_full]
    down_len = lengths[down_mask_full]
    up_ia = interarrival[up_ia_mask]
    down_ia = interarrival[down_ia_mask]
    row["up_len_mean"] = safe_mean(up_len)
    row["up_len_std"] = safe_std(up_len)
    row["up_len_p50"] = quantile(up_len, 0.50)
    row["down_len_mean"] = safe_mean(down_len)
    row["down_len_std"] = safe_std(down_len)
    row["down_len_p50"] = quantile(down_len, 0.50)
    row["up_ia_mean"] = safe_mean(up_ia)
    row["up_ia_std"] = safe_std(up_ia)
    row["down_ia_mean"] = safe_mean(down_ia)
    row["down_ia_std"] = safe_std(down_ia)
    row["len_up_down_diff"] = abs(safe_mean(up_len) - safe_mean(down_len))

    # --- subwin_*（主线 L410-438，10 项） ---
    bins = np.linspace(0.0, window_seconds, SUB_COUNT + 1)
    positions = np.clip(
        group["relative_time"].to_numpy() - window_id * window_seconds, 0, window_seconds
    )
    packet_counts, _ = np.histogram(positions, bins=bins)
    length_arr = lengths.to_numpy()
    byte_sums = []
    for idx in range(SUB_COUNT):
        if idx == SUB_COUNT - 1:
            mask = (positions >= bins[idx]) & (positions <= bins[idx + 1])
        else:
            mask = (positions >= bins[idx]) & (positions < bins[idx + 1])
        byte_sums.append(float(length_arr[mask].sum()))
    packet_counts_series = pd.Series(packet_counts.astype(float))
    byte_sums_series = pd.Series(byte_sums)
    sub_packet_mean = safe_mean(packet_counts_series)
    sub_packet_std = safe_std(packet_counts_series)
    row.update(
        {
            "subwin_packet_mean": sub_packet_mean,
            "subwin_packet_std": sub_packet_std,
            "subwin_packet_min": float(packet_counts_series.min()),
            "subwin_packet_max": float(packet_counts_series.max()),
            "subwin_packet_cv": coefficient_of_variation(sub_packet_mean, sub_packet_std),
            "subwin_byte_mean": safe_mean(byte_sums_series),
            "subwin_byte_std": safe_std(byte_sums_series),
            "subwin_byte_min": float(byte_sums_series.min()),
            "subwin_byte_max": float(byte_sums_series.max()),
            "active_subwin_count": int((packet_counts_series > 0).sum()),
        }
    )
    return row


# --------------------------------------------------------------------------
# 单个 pcap 文件 -> 特征表
# --------------------------------------------------------------------------

def extract_features_for_file(
    pcap_path: Path,
    day: str,
    mac_map: dict[str, str],
    window_seconds: float,
    min_packets_per_window: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """一个日 pcap -> 每设备 / 每 10 秒窗口一行。

    **窗口栅格是全天共享的**（origin = 该 pcap 第一个包的时间戳），
    不是每设备各自从 0 起算 —— 否则不同设备的「窗口 0」落在不同墙钟时刻，
    跨设备窗口不可比。全天饱和的设备应得到约 86400/10 = 8640 个窗口
    （与协议 §16.1 记录的 Dropcam 8640 窗口一致）。
    """
    mac_set = set(mac_map)
    packets = read_packets(pcap_path, mac_set, verbose=verbose)
    if packets.empty:
        return pd.DataFrame()

    streams = assign_device_streams(packets, mac_map)
    if streams.empty:
        return pd.DataFrame()

    # 全天共享 origin（主线 L459 是每文件 origin；此处每文件 = 每天，语义相同）
    day_origin = float(streams["time_epoch"].min())
    streams["relative_time"] = streams["time_epoch"] - day_origin
    streams["window_id"] = np.floor(streams["relative_time"] / window_seconds).astype(np.int64)

    rows = []
    for (device, window_id), group in streams.groupby(["device", "window_id"], sort=True):
        if len(group) < min_packets_per_window:
            continue
        rows.append(
            summarize_window(
                group=group,
                device=str(device),
                day=day,
                source_file=pcap_path,
                window_id=int(window_id),
                window_seconds=window_seconds,
                day_origin=day_origin,
            )
        )
    if verbose:
        print(f"[{day}] {len(rows):,} windows over {streams['device'].nunique()} devices",
              flush=True)
    return pd.DataFrame(rows)


def feature_columns(features: pd.DataFrame) -> list[str]:
    """与主线 robust_iot_research.py L523-530 同逻辑。"""
    columns = []
    for column in features.columns:
        if column in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            columns.append(column)
    return columns


# --------------------------------------------------------------------------
# 输入发现
# --------------------------------------------------------------------------

def discover_pcaps(pcap_dir: Path, days: Iterable[str] | None) -> list[tuple[str, Path]]:
    """在 pcap_dir 中找 {day}.pcap（若只有 {day}.tar.gz 则报错要求先解包）。"""
    found: list[tuple[str, Path]] = []
    for path in sorted(pcap_dir.glob("*.pcap")):
        found.append((path.stem, path))
    if not found:
        tarballs = sorted(pcap_dir.glob("*.tar.gz"))
        if tarballs:
            raise SystemExit(
                f"{pcap_dir} 下只有 tar.gz，未找到 .pcap。请先解包：\n"
                + "\n".join(f"  tar -xzf {t} -C {pcap_dir}" for t in tarballs)
            )
        raise SystemExit(f"{pcap_dir} 下没有 .pcap 文件")
    if days:
        wanted = set(days)
        found = [(d, p) for d, p in found if d in wanted]
        missing = wanted - {d for d, _ in found}
        if missing:
            raise SystemExit(f"缺少这些日期的 pcap: {sorted(missing)}")
    return found


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UNSW Ethernet 通用特征提取（协议 §20.1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pcap-dir", type=Path, required=True,
                        help="含 {day}.pcap 的目录")
    parser.add_argument("--mac-map", type=Path, required=True,
                        help="device_mac_map.csv 路径")
    parser.add_argument("--output", type=Path, required=True,
                        help="输出 CSV 路径")
    parser.add_argument("--days", type=str, default=None,
                        help="逗号分隔的日期子集，如 16-09-30,16-09-23；默认全部")
    # 默认值与主线 robust_iot_research.py L77-78 一致
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--min-packets-per-window", type=int, default=2)
    parser.add_argument("--include-non-iot", action="store_true",
                        help="同时提取非 IoT 设备（手机/笔记本）。默认只做 IoT。网关始终排除。")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    mac_map = load_mac_map(args.mac_map, include_non_iot=args.include_non_iot)
    if not mac_map:
        raise SystemExit(f"{args.mac_map} 没有可用的设备 MAC")
    print(f"[map] {len(mac_map)} device MACs "
          f"({'IoT + non-IoT' if args.include_non_iot else 'IoT only'}), "
          f"gateway {GATEWAY_MAC} excluded", flush=True)

    days = [d.strip() for d in args.days.split(",")] if args.days else None
    targets = discover_pcaps(args.pcap_dir, days)
    print(f"[in] {len(targets)} pcap file(s): {[d for d, _ in targets]}", flush=True)

    frames = []
    for day, path in targets:
        t0 = time.time()
        frame = extract_features_for_file(
            pcap_path=path,
            day=day,
            mac_map=mac_map,
            window_seconds=args.window_seconds,
            min_packets_per_window=args.min_packets_per_window,
            verbose=not args.quiet,
        )
        print(f"[{day}] done in {time.time() - t0:.1f}s", flush=True)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit("没有提取到任何窗口")

    merged = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    cols = feature_columns(merged)
    print(f"[out] {args.output}  rows={len(merged):,}  "
          f"numeric_features={len(cols)}  devices={merged['device'].nunique()}  "
          f"days={merged['day'].nunique()}", flush=True)

    # --- §19.2 运行元信息落盘 ---
    meta = {
        "script": str(Path(__file__).resolve()),
        "command_line": sys.argv,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "window_seconds": args.window_seconds,
        "min_packets_per_window": args.min_packets_per_window,
        "include_non_iot": args.include_non_iot,
        "gateway_mac_excluded": GATEWAY_MAC,
        "tshark_fields": TSHARK_FIELDS,
        "n_device_macs": len(mac_map),
        "n_rows": int(len(merged)),
        "n_numeric_features": len(cols),
        "numeric_features": cols,
        "days": sorted(merged["day"].unique().tolist()),
        "devices": sorted(merged["device"].unique().tolist()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tshark": subprocess.run(["tshark", "--version"], capture_output=True, text=True
                                 ).stdout.splitlines()[0] if True else "",
        "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                   cwd=Path(__file__).resolve().parents[3]).stdout.strip(),
    }
    meta_path = args.output.with_suffix(".run_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
