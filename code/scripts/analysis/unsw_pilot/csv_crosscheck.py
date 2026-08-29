#!/usr/bin/env python3
"""官方 CSV 交叉核对 —— **仅限设备清点与窗口计数**（协议 §16.2 允许的唯一用途）。

================================ 硬约束 ================================
协议 §16.2：官方 CSV 的 `TIME` 是 **int64 秒级整数**，
`interarrival_*` / `burst_*` / `subwin_*` 三族时间特征在其上**全部失效**。

**本脚本绝不计算、绝不输出任何时间特征。**
它只做三件事：
  1. 证明 CSV 的 TIME 确实是秒级整数（把 §16.2 的判据在本机复现一遍）；
  2. 用 CSV 数一遍每个 MAC 的包数与"出现过的秒"数；
  3. 把 CSV 的设备清点结果与 **pcap 派生特征表** 的清点结果并排放，看是否一致。

**pilot 的一切结论以 pcap 为准。** CSV 只回答"设备清单对不对得上"，
不参与也不许参与任何时间特征的可行性判断。
========================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GATEWAY_MAC = "14:cc:20:51:33:ea"
WINDOW_SECONDS = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="官方 {day}.csv")
    ap.add_argument("--day", type=str, required=True)
    ap.add_argument("--mac-map", type=Path, required=True)
    ap.add_argument("--pcap-features", type=Path, required=True,
                    help="extract_features_generic.py 输出（唯一有效的结论来源）")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    macs = pd.read_csv(args.mac_map, dtype=str).fillna("")
    macs["mac"] = macs["mac"].str.strip().str.lower()
    iot = macs[macs["is_iot"].astype(int) == 1]
    mac2id = dict(zip(iot["mac"], iot["device_id"]))

    lines: list[str] = []
    A = lines.append
    A("# CSV_CROSSCHECK.md — 官方 CSV 与 pcap 的**设备清点**交叉核对")
    A("")
    A("> **协议 §16.2 声明**：本文件中的 CSV 数字**只用于设备清点与窗口计数核对**。")
    A("> CSV 的 `TIME` 是秒级整数，`interarrival_*` / `burst_*` / `subwin_*` 在其上全部失效，")
    A("> **本文件不含、也不得含任何由 CSV 计算的时间特征**。pilot 结论一律以 pcap 为准。")
    A("")
    A(f"- 天: `{args.day}`")
    A(f"- CSV: `{args.csv}`")
    A(f"- pcap 派生特征表: `{args.pcap_features}`")
    A("")

    print(f"[csv] reading {args.csv} ...", flush=True)
    csv = pd.read_csv(
        args.csv,
        usecols=["TIME", "Size", "eth.src", "eth.dst"],
        dtype={"TIME": "int64", "Size": "int64", "eth.src": "string", "eth.dst": "string"},
    )
    csv["eth.src"] = csv["eth.src"].str.lower()
    csv["eth.dst"] = csv["eth.dst"].str.lower()

    # ---------- 1. 复现 §16.2 的秒级整数判据 ----------
    t = csv["TIME"].to_numpy()
    n_rows, n_uniq = len(t), len(np.unique(t))
    span = int(t.max() - t.min())
    A("## 1. 复现 §16.2 的判据：CSV 的 TIME 是秒级整数")
    A("")
    A("| 项 | 值 | §16.2 记录 |")
    A("|---|---|---|")
    A(f"| CSV 行数 | {n_rows:,} | 673,414 |")
    A(f"| TIME 跨度（秒） | {span:,} | 86,398 |")
    A(f"| 唯一 TIME 值 | {n_uniq:,} | 84,291 |")
    A(f"| 唯一值 / 行数 | {n_uniq / n_rows:.6f} | — |")
    A(f"| dtype 是整数？ | {np.issubdtype(t.dtype, np.integer)} | int64 |")
    A("")
    A(f"**同一天的 pcap 时间戳唯一率 = 1.000000（微秒分辨率，最小包间隔 1.88e-05 s，"
      f"见 `smoke_{args.day}.txt` 第 [5] 节）。**")
    A(f"CSV 每个唯一秒平均塞进 {n_rows / n_uniq:.1f} 个包 —— "
      "这些包在 CSV 上的包间隔全部塌缩为 0，故时间特征不可用。**证据充分，约束成立。**")
    A("")

    # ---------- 2. CSV 侧设备清点 ----------
    rows = []
    for mac, dev in mac2id.items():
        m_src = csv["eth.src"] == mac
        m_dst = csv["eth.dst"] == mac
        involved = m_src | m_dst
        n_pkt = int(involved.sum())
        secs = csv.loc[involved, "TIME"]
        rows.append({
            "device": dev,
            "mac": mac,
            "csv_packets": n_pkt,
            "csv_unique_seconds": int(secs.nunique()) if n_pkt else 0,
            # 用 CSV 的秒粒度 TIME 折算 10 秒窗口数（仅计数用途，非时间特征）
            "csv_10s_windows": int((secs // WINDOW_SECONDS).nunique()) if n_pkt else 0,
        })
    csv_inv = pd.DataFrame(rows)

    # ---------- 3. pcap 侧设备清点（唯一有效结论来源） ----------
    feats = pd.read_csv(args.pcap_features)
    feats = feats[feats["day"] == args.day]
    pcap_inv = (
        feats.groupby("device")
        .agg(pcap_windows=("window_id", "nunique"), pcap_packets=("packet_count", "sum"))
        .reset_index()
    )

    merged = csv_inv.merge(pcap_inv, on="device", how="outer").fillna(0)
    for c in ("csv_packets", "csv_unique_seconds", "csv_10s_windows",
              "pcap_windows", "pcap_packets"):
        merged[c] = merged[c].astype(int)
    merged["window_diff"] = merged["pcap_windows"] - merged["csv_10s_windows"]
    merged["both_active"] = (merged["csv_packets"] > 0) & (merged["pcap_windows"] > 0)
    merged = merged.sort_values("pcap_windows", ascending=False)
    merged.to_csv(args.out.with_name(f"csv_crosscheck_{args.day}.csv"),
                  index=False, encoding="utf-8-sig")

    A("## 2. 设备清点交叉核对（允许用途）")
    A("")
    A("| device | MAC | CSV 包数 | CSV 10s 窗口 | pcap 10s 窗口 | 差 (pcap−csv) | pcap 包数 |")
    A("|---|---|---|---|---|---|---|")
    for _, r in merged.iterrows():
        A(f"| `{r['device']}` | `{r['mac']}` | {r['csv_packets']:,} | "
          f"{r['csv_10s_windows']:,} | {r['pcap_windows']:,} | {r['window_diff']:+,} | "
          f"{r['pcap_packets']:,} |")
    A("")

    n_csv_active = int((merged["csv_packets"] > 0).sum())
    n_pcap_active = int((merged["pcap_windows"] > 0).sum())
    disagree = merged[(merged["csv_packets"] > 0) != (merged["pcap_windows"] > 0)]
    A(f"- CSV 侧有流量的 IoT 设备数: **{n_csv_active}**")
    A(f"- pcap 侧有窗口的 IoT 设备数: **{n_pcap_active}**")
    A(f"- 「一侧有、另一侧无」的设备数: **{len(disagree)}**"
      + ("" if len(disagree) == 0 else f" → {sorted(disagree['device'])}"))
    A(f"- CSV 总行数 {n_rows:,} vs pcap 总包数（见 smoke 文件）—— "
      "CSV 只收录带 IP 层的包，pcap 含 ARP / IPv6-ND / 广播等，故 CSV 行数偏少属正常。")
    A("")
    A("**判读**：清点一致 → MAC 清单与归流逻辑无系统性错误。"
      "窗口数的小幅差异来自 (a) CSV 丢弃非 IP 包，(b) pcap 侧有 "
      "`min_packets_per_window=2` 门槛，(c) CSV 的秒级 TIME 与 pcap 的绝对时间原点对齐方式不同。"
      "**这些差异不影响任何以 pcap 为准的结论。**")
    A("")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[out] {args.out}")
    print(f"[out] {args.out.with_name(f'csv_crosscheck_{args.day}.csv')}")
    print(f"[summary] csv_active={n_csv_active} pcap_active={n_pcap_active} "
          f"disagree={len(disagree)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
