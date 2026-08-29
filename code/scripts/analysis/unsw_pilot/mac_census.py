#!/usr/bin/env python3
"""跨天 MAC 普查（五问之 2 的补强证据）。

对每个日 pcap 直接跑 tshark 取 `eth.src` / `eth.dst`，统计：
  - 官方清单内每个 MAC 的逐天包数（src / dst 分开）；
  - **清单外的单播 MAC**（排除广播/组播）及其包数 —— 这是检出
    「设备换了 MAC」或「清单不完整」的关键信号；
  - 清单内但某天完全消失的 MAC。

只碰 pcap（协议 §16.2）。
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path

import pandas as pd


def is_multicast_or_broadcast(mac: str) -> bool:
    """IEEE 802: 首字节最低位 = 1 → 组播/广播。"""
    try:
        return bool(int(mac.split(":")[0], 16) & 0x01)
    except (ValueError, IndexError):
        return False


def census_day(pcap: Path) -> tuple[collections.Counter, collections.Counter, int]:
    cmd = ["tshark", "-r", str(pcap), "-T", "fields", "-e", "eth.src", "-e", "eth.dst",
           "-E", "header=n", "-E", "separator=\t"]
    src, dst = collections.Counter(), collections.Counter()
    n = 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1 << 20)
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        parts += [""] * (2 - len(parts))
        n += 1
        if parts[0].strip():
            src[parts[0].strip().lower().split(",")[0]] += 1
        if parts[1].strip():
            dst[parts[1].strip().lower().split(",")[0]] += 1
    proc.wait()
    return src, dst, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", type=Path, required=True)
    ap.add_argument("--days", type=str, required=True)
    ap.add_argument("--mac-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    macs = pd.read_csv(args.mac_map, dtype=str).fillna("")
    macs["mac"] = macs["mac"].str.strip().str.lower()
    known = dict(zip(macs["mac"], macs["device_id"]))
    known_iot = {m: d for m, d in known.items()
                 if macs.set_index("mac").loc[m, "is_iot"] == "1"}

    days = [d.strip() for d in args.days.split(",")]
    per_day: dict[str, dict] = {}
    for day in days:
        pcap = args.pcap_dir / f"{day}.pcap"
        print(f"[census] {pcap} ...", flush=True)
        src, dst, n = census_day(pcap)
        per_day[day] = {"src": src, "dst": dst, "total_packets": n}
        print(f"[census] {day}: {n:,} packets, {len(src)} unique src, "
              f"{len(dst)} unique dst", flush=True)

    # ---- 表 1: 清单内 MAC 的逐天计数 ----
    rows = []
    for mac, dev in known.items():
        row = {"device_id": dev, "mac": mac,
               "is_iot": int(macs.set_index("mac").loc[mac, "is_iot"])}
        for day in days:
            row[f"{day}_src"] = per_day[day]["src"].get(mac, 0)
            row[f"{day}_dst"] = per_day[day]["dst"].get(mac, 0)
            row[f"{day}_total"] = row[f"{day}_src"] + row[f"{day}_dst"]
        row["days_present"] = sum(1 for d in days if row[f"{d}_total"] > 0)
        rows.append(row)
    known_tbl = pd.DataFrame(rows).sort_values(
        ["is_iot", "days_present"], ascending=[False, False])
    p1 = args.out_dir / "mac_census_known.csv"
    known_tbl.to_csv(p1, index=False, encoding="utf-8-sig")

    # ---- 表 2: 清单外单播 MAC ----
    rows = []
    seen: set[str] = set()
    for day in days:
        seen |= set(per_day[day]["src"]) | set(per_day[day]["dst"])
    for mac in sorted(seen):
        if mac in known or is_multicast_or_broadcast(mac):
            continue
        row = {"mac": mac}
        for day in days:
            row[f"{day}_src"] = per_day[day]["src"].get(mac, 0)
            row[f"{day}_dst"] = per_day[day]["dst"].get(mac, 0)
            row[f"{day}_total"] = row[f"{day}_src"] + row[f"{day}_dst"]
        row["grand_total"] = sum(row[f"{d}_total"] for d in days)
        row["days_present"] = sum(1 for d in days if row[f"{d}_total"] > 0)
        rows.append(row)
    unknown_tbl = (pd.DataFrame(rows).sort_values("grand_total", ascending=False)
                   if rows else pd.DataFrame(columns=["mac", "grand_total"]))
    p2 = args.out_dir / "mac_census_unlisted.csv"
    unknown_tbl.to_csv(p2, index=False, encoding="utf-8-sig")

    summary = {
        "days": days,
        "total_packets_per_day": {d: per_day[d]["total_packets"] for d in days},
        "n_known_macs": len(known),
        "n_known_iot_macs": len(known_iot),
        "iot_macs_present_all_days": int(
            ((known_tbl["is_iot"] == 1) & (known_tbl["days_present"] == len(days))).sum()),
        "iot_macs_present_never": int(
            ((known_tbl["is_iot"] == 1) & (known_tbl["days_present"] == 0)).sum()),
        "n_unlisted_unicast_macs": int(len(unknown_tbl)),
        "unlisted_unicast_packets_total": int(unknown_tbl["grand_total"].sum()) if len(unknown_tbl) else 0,
        "top_unlisted": (unknown_tbl.head(15).to_dict("records") if len(unknown_tbl) else []),
    }
    p3 = args.out_dir / "mac_census_summary.json"
    p3.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {p1}\n[out] {p2}\n[out] {p3}")
    print(f"[summary] IoT MACs present on all {len(days)} days: "
          f"{summary['iot_macs_present_all_days']}/{summary['n_known_iot_macs']}; "
          f"unlisted unicast MACs: {summary['n_unlisted_unicast_macs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
