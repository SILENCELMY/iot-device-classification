#!/usr/bin/env python
"""CIC IoT 2022：从 Device List.xlsx 机械生成 device_mac_map.csv。

协议 docs/PROTOCOL_CIC_RANGE_ANCHOR_20260904.md §2（sha256 5b6b1428…）要求：
「由 dataset/cic2022/raw/Device List.xlsx 机械生成（MAC → 设备名），生成脚本与产物
md5 落盘；设备名不做人工归并。」本文件即该生成脚本。

【必须记录的机械规则（协议未逐条列出，此处定死并落盘）】
1. `Category` 列用 ffill 前向填充 —— 源文件是 Excel 合并单元格，续行读出为 NaN。
2. `MAC Address` 统一 strip + 小写 + 分隔符归一为 ':' —— 源文件为大写冒号格式，
   而抽取器 load_mac_map 以小写 MAC 为键（其 docstring：{mac(lower) -> device_id}）。
3. `device_id` = `Device Name` 去掉全部非字母数字字符，**保留原大小写**，不做人工归并、
   不做同生态合并（7 台 Gosund 等必须保持为 7 个独立 id）。
4. `is_iot` = 1 全部置 1；若某行 Category/Device Name 命中网关/路由器关键词则置 0
   并在 stdout 报出（抽取器默认只保留 is_iot==1，且始终排除网关）。
5. `connection` 留空 —— 源文件无该列，**不臆造**。

解释器：~/anaconda3/bin/python（base 环境已有 openpyxl 3.0.10；iotcls 环境不被触碰、
不安装任何软件包，与 dataset/cic2022/MANIFEST.md 的纪律一致）。
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

SRC = Path("dataset/cic2022/raw/Device List.xlsx")
OUT = Path("dataset/cic2022/device_mac_map.csv")
GW_PAT = re.compile(r"gateway|router|switch|access\s*point|\bap\b", re.I)


def main() -> int:
    x = pd.read_excel(SRC)
    need = ["Category", "Device Name", "MAC Address"]
    missing = [c for c in need if c not in x.columns]
    if missing:
        print(f"[FAIL] 缺列 {missing}；实得 {x.columns.tolist()}")
        return 1

    x["Category"] = x["Category"].ffill()                       # 规则 1
    mac = (x["MAC Address"].astype(str).str.strip().str.lower()
           .str.replace("-", ":", regex=False))                 # 规则 2
    dev_id = (x["Device Name"].astype(str)
              .map(lambda s: re.sub(r"[^0-9A-Za-z]", "", s)))    # 规则 3

    out = pd.DataFrame({
        "device_id": dev_id,
        "device_name": x["Device Name"].astype(str).str.strip(),
        "mac": mac,
        "connection": "",                                       # 规则 5
        "category": x["Category"].astype(str).str.strip(),
        "is_iot": 1,                                            # 规则 4
    })

    gw = out["device_name"].str.contains(GW_PAT) | out["category"].str.contains(GW_PAT)
    if gw.any():
        out.loc[gw, "is_iot"] = 0
        print("[gateway] 置 is_iot=0：")
        print(out.loc[gw, ["device_id", "device_name", "category"]].to_string(index=False))
    else:
        print("[gateway] 清单中未命中网关/路由器关键词（抽取器仍会排除对端 MAC）")

    # ---- 机械核对，任一不过即中止
    bad = []
    if len(out) != 40:
        bad.append(f"行数 {len(out)} != 40")
    if out["mac"].duplicated().any():
        bad.append(f"MAC 重复 {out.loc[out['mac'].duplicated(keep=False), 'mac'].tolist()}")
    if out["device_id"].duplicated().any():
        bad.append("device_id 重复 "
                   f"{out.loc[out['device_id'].duplicated(keep=False), 'device_id'].tolist()}")
    if out["mac"].eq("").any() or out["mac"].isna().any():
        bad.append("有空 MAC")
    if out["device_id"].eq("").any():
        bad.append("有空 device_id")
    if not out["mac"].str.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}").all():
        bad.append("MAC 格式不合规："
                   f"{out.loc[~out['mac'].str.fullmatch(r'([0-9a-f]{2}:){5}[0-9a-f]{2}'), 'mac'].tolist()}")
    if bad:
        print("[FAIL] " + " / ".join(bad))
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    md5 = hashlib.md5(OUT.read_bytes()).hexdigest()

    print(f"\n[ok] {OUT}  行数={len(out)}  is_iot=1 的条目={int(out['is_iot'].sum())}")
    print(f"[md5] {md5}")
    print(f"[src_md5] {hashlib.md5(SRC.read_bytes()).hexdigest()}")
    print("\n[category 分布]")
    print(out["category"].value_counts().to_string())
    print("\n[同生态插座核对：名称含 gosund/teckin/yutron]")
    eco = out[out["device_name"].str.contains("gosund|teckin|yutron", case=False)]
    print(eco[["device_id", "device_name", "category"]].to_string(index=False)
          if len(eco) else "（未命中，需人工复核 Device Name 拼写）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
