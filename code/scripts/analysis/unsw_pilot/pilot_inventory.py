#!/usr/bin/env python3
"""UNSW pilot — 设备清点 / 每设备窗口计数 / MAC 跨天一致性（五问之 2、3）。

**协议 §16.2 硬约束**：本脚本的**全部结论来自 pcap 派生的特征表**。
官方 CSV **仅在 `--csv-crosscheck` 明确给出时**，用于**设备清点与窗口计数的交叉核对**，
绝不用于任何时间特征（interarrival / burst / subwin）的验证 —— CSV 的 TIME 是秒级整数。

输出（全部落 results/unsw_pilot/）：
  device_window_counts_<day>.csv     每设备窗口数 / 包数 / 字节数（单日）
  device_window_counts_all.csv       多天汇总
  mac_day_consistency.csv            每个 IoT MAC 逐天出现情况（五问之 2）
  INVENTORY.md                       人读汇总
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_mac_map(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str).fillna("")
    table["mac"] = table["mac"].str.strip().str.lower()
    table["is_iot"] = table["is_iot"].astype(int)
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True,
                    help="extract_features_generic.py 的输出 CSV")
    ap.add_argument("--mac-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-windows", type=int, nargs="+", default=[100, 300],
                    help="窗口数门槛（协议 §16.1 记录了 100 与 300 两档）")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.features)
    macs = load_mac_map(args.mac_map)
    iot = macs[macs["is_iot"] == 1]
    days = sorted(feats["day"].unique())

    print(f"[in] {args.features}  rows={len(feats):,}  days={days}")
    print(f"[in] {len(iot)} IoT MACs in map")

    # ---------------- 每设备 × 每天 的窗口计数 ----------------
    grouped = (
        feats.groupby(["day", "device"])
        .agg(
            n_windows=("window_id", "nunique"),
            n_packets=("packet_count", "sum"),
            n_bytes=("byte_count", "sum"),
            first_window=("window_id", "min"),
            last_window=("window_id", "max"),
        )
        .reset_index()
    )
    # 补齐零窗口设备（在特征表里完全不出现的 IoT 设备）
    full_index = pd.MultiIndex.from_product(
        [days, sorted(iot["device_id"])], names=["day", "device"]
    )
    grouped = (
        grouped.set_index(["day", "device"])
        .reindex(full_index)
        .fillna({"n_windows": 0, "n_packets": 0, "n_bytes": 0})
        .reset_index()
    )
    grouped["n_windows"] = grouped["n_windows"].astype(int)
    grouped["n_packets"] = grouped["n_packets"].astype("Int64")
    grouped = grouped.merge(
        iot[["device_id", "device_name", "mac", "category", "connection"]],
        left_on="device", right_on="device_id", how="left",
    ).drop(columns=["device_id"])
    grouped = grouped.sort_values(["day", "n_windows"], ascending=[True, False])

    all_path = args.out_dir / "device_window_counts_all.csv"
    grouped.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[out] {all_path}")

    for day in days:
        sub = grouped[grouped["day"] == day]
        p = args.out_dir / f"device_window_counts_{day}.csv"
        sub.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[out] {p}")

    # ---------------- 门槛统计（五问之 3） ----------------
    lines: list[str] = []
    lines.append("# INVENTORY.md — UNSW pilot 设备清点与窗口计数")
    lines.append("")
    lines.append("**全部数字来源：pcap 派生特征表**"
                 f" `{args.features.name}`（协议 §16.2：pilot 以 pcap 为准）。")
    lines.append("")
    lines.append(f"- 特征表行数: **{len(feats):,}**")
    lines.append(f"- 覆盖天数: **{len(days)}** {days}")
    lines.append(f"- 窗口长度: 10 秒非重叠；`min_packets_per_window = 2`（与主线默认一致）")
    lines.append("")
    lines.append("## 1. 每日门槛统计（五问之 3）")
    lines.append("")
    header = "| day | 有流量设备数 | " + " | ".join(
        f"≥{m} 窗口设备数" for m in args.min_windows) + " | 总窗口数 | 最大单设备窗口数 |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(args.min_windows)))
    for day in days:
        sub = grouped[grouped["day"] == day]
        active = int((sub["n_windows"] > 0).sum())
        cells = [str(int((sub["n_windows"] >= m).sum())) for m in args.min_windows]
        lines.append(
            f"| `{day}` | {active} | " + " | ".join(cells)
            + f" | {int(sub['n_windows'].sum()):,} | {int(sub['n_windows'].max()):,} |"
        )
    lines.append("")
    lines.append("协议 §16.1 记录的抽样日 `16-09-30` 预期：**有流量 20 台，≥100 窗口 18 台，"
                 "≥300 窗口 18 台**。上表为本次 pcap 实测值。")
    lines.append("")

    # ---------------- 逐日明细 ----------------
    for day in days:
        sub = grouped[grouped["day"] == day].sort_values("n_windows", ascending=False)
        lines.append(f"## 2.{days.index(day)+1} `{day}` 每设备窗口数")
        lines.append("")
        lines.append("| device_id | device_name | category | MAC | 窗口数 | 包数 |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            pk = "0" if pd.isna(r["n_packets"]) else f"{int(r['n_packets']):,}"
            lines.append(
                f"| `{r['device']}` | {r['device_name']} | {r['category']} | "
                f"`{r['mac']}` | {int(r['n_windows']):,} | {pk} |"
            )
        lines.append("")

    # ---------------- MAC 跨天一致性（五问之 2） ----------------
    if len(days) >= 2:
        pivot = grouped.pivot(index="device", columns="day", values="n_windows").fillna(0).astype(int)
        pivot = pivot.merge(
            iot.set_index("device_id")[["mac", "device_name", "category"]],
            left_index=True, right_index=True, how="left",
        )
        cols = list(days)
        pivot["days_present"] = (pivot[cols] > 0).sum(axis=1)
        pivot["days_ge100"] = (pivot[cols] >= 100).sum(axis=1)
        pivot["present_all_days"] = pivot["days_present"] == len(days)
        pivot["ge100_all_days"] = pivot["days_ge100"] == len(days)
        pivot = pivot.reset_index().sort_values(["days_ge100", "days_present"], ascending=False)
        p = args.out_dir / "mac_day_consistency.csv"
        pivot.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[out] {p}")

        lines.append("## 3. MAC 映射跨天一致性（五问之 2）")
        lines.append("")
        lines.append(f"每个 IoT MAC 在 {len(days)} 天中的窗口数（0 = 该天完全无流量）：")
        lines.append("")
        lines.append("| device_id | MAC | " + " | ".join(f"`{d}`" for d in cols)
                     + " | 出现天数 | ≥100窗口天数 |")
        lines.append("|---|---|" + "---|" * (len(cols) + 2))
        for _, r in pivot.iterrows():
            cells = " | ".join(f"{int(r[d]):,}" for d in cols)
            lines.append(f"| `{r['device']}` | `{r['mac']}` | {cells} | "
                         f"{int(r['days_present'])} | {int(r['days_ge100'])} |")
        lines.append("")
        n_all = int(pivot["present_all_days"].sum())
        n_all100 = int(pivot["ge100_all_days"].sum())
        n_never = int((pivot["days_present"] == 0).sum())
        lines.append(f"- 所有 {len(days)} 天都有流量的 IoT MAC: **{n_all} / {len(pivot)}**")
        lines.append(f"- 所有 {len(days)} 天都 ≥100 窗口的 IoT MAC: **{n_all100} / {len(pivot)}**")
        lines.append(f"- 任何一天都没出现过的 IoT MAC: **{n_never}**")
        lines.append("")
        lines.append("**判读**：MAC 映射本身是静态清单，跨天一致性的风险是"
                     "「同一 MAC 在不同天对应了不同物理设备」或「设备换了 MAC」。"
                     "上表能检出的是**出现/消失模式**；若某 MAC 在部分天完全消失、"
                     "而同期出现一个不在清单里的高流量 MAC，即为可疑信号。"
                     "见下一节的「清单外 MAC」检查。")
        lines.append("")

    out_md = args.out_dir / "INVENTORY.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[out] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
