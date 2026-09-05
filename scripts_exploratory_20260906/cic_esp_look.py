"""【探索性,非协议】CIC 的 4 个 ESP 型号：原始流量里到底差在哪？

动机：混淆矩阵指着这几对（型号级 0.073 缺口全在它们身上），逐类对搜遍 144 种配置
在 inner 上也只到 0.71–0.80 —— 配置层没有解。而我们的 94 列特征**全是量与时序**
（len_* / interarrival_* / burst_* / up_*、down_*），**没有任何"跟谁通信"的信息**。
时钟偏斜路已断（ESP 的 lwIP 不协商 TCP 时间戳，实测 0 个包带 tsval）。
所以先看：不同厂商的云端点 / DNS 名字 / 心跳周期 / TCP 指纹，有没有区分度。

**不建特征、不跑模型**，只把原始差异摆出来，由证据决定下一步。
每个量都要问一句：**跨天稳吗**（两天都看），否则就是会话身份不是设备身份。
"""
from __future__ import annotations
import subprocess, io, time, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/lmy/iot-device-classification/dataset/cic2022")
EXTR = ROOT/"extracted"
DAYS = [("1102_Idle",   EXTR/"2-Idle/2021_11_02_Idle.pcap"),
        ("1108_Idle",   EXTR/"2-Idle/2021_11_08_Idle.pcap")]

def esp_macs():
    m = pd.read_csv(ROOT/"device_mac_map.csv")
    s = m[m.device_id.str.contains("Gosund|Teckin|Yutron|GlobeLamp", case=False, na=False)]
    return dict(zip(s.mac.str.lower(), s.device_id))

def tsh(pcap, disp, fields):
    cmd=["tshark","-r",str(pcap),"-Y",disp,"-T","fields"]
    for f in fields: cmd += ["-e",f]
    cmd += ["-E","separator=\t"]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not p.stdout.strip(): return pd.DataFrame(columns=fields)
    return pd.read_csv(io.StringIO(p.stdout),sep="\t",header=None,names=fields,dtype=str)

def main():
    t0=time.time(); macs=esp_macs()
    filt=" or ".join(f"eth.src=={m}" for m in macs)
    print(f"ESP 族 {len(macs)} 台：{sorted(set(macs.values()))}\n",flush=True)

    store={}
    for day,p in DAYS:
        if not p.exists(): print(f"[缺] {p}"); continue
        print(f"=== {day} ({p.stat().st_size/1e6:.0f} MB) ===",flush=True)

        # 1) DNS 查询名 —— 最直接的"跟谁通信"
        d=tsh(p, f"dns.flags.response==0 and ({filt})",
              ["eth.src","dns.qry.name"]).dropna()
        print(f"  DNS 查询 {len(d)} 条",flush=True)
        for mac,g in d.groupby("eth.src"):
            names=sorted(set(g["dns.qry.name"].dropna()))
            print(f"    {macs.get(mac.lower(),mac):26s} {len(names)} 个域名: {names[:4]}",flush=True)
        store[(day,"dns")]=d

        # 2) 目的 IP / 端口
        t=tsh(p, f"tcp.flags.syn==1 and tcp.flags.ack==0 and ({filt})",
              ["eth.src","ip.dst","tcp.dstport"]).dropna()
        print(f"  TCP SYN {len(t)} 条",flush=True)
        for mac,g in t.groupby("eth.src"):
            dst=g.groupby(["ip.dst","tcp.dstport"]).size().sort_values(ascending=False)
            top=[f"{a}:{b}({n})" for (a,b),n in dst.head(3).items()]
            print(f"    {macs.get(mac.lower(),mac):26s} {len(dst)} 个端点: {top}",flush=True)
        store[(day,"syn")]=t

        # 3) TCP/IP 指纹（首包 TTL、窗口、MSS）+ 心跳周期
        f=tsh(p, f"tcp and ({filt})",
              ["eth.src","frame.time_epoch","ip.ttl","tcp.window_size_value","tcp.len"]).dropna(subset=["eth.src"])
        for c in ["frame.time_epoch","ip.ttl","tcp.window_size_value","tcp.len"]:
            f[c]=pd.to_numeric(f[c].astype(str).str.split(",").str[0],errors="coerce")
        print(f"  TCP 包 {len(f)} 个",flush=True)
        for mac,g in f.groupby("eth.src"):
            g=g.sort_values("frame.time_epoch")
            ttl=g["ip.ttl"].mode()
            win=g["tcp.window_size_value"].mode()
            # 心跳：相邻 SYN/长静默后的首包间隔众数
            dt=np.diff(g["frame.time_epoch"].to_numpy())
            dt=dt[dt>1.0]
            per = float(np.median(dt)) if len(dt)>5 else float("nan")
            print(f"    {macs.get(mac.lower(),mac):26s} TTL={ttl.iloc[0] if len(ttl) else '-':>4}"
                  f"  win={win.iloc[0] if len(win) else '-':>6}"
                  f"  静默间隔中位={per:8.2f}s  n={len(g)}",flush=True)
        store[(day,"tcp")]=f
        print(flush=True)

    # 跨天一致性：DNS 域名集合
    print("=== 跨天一致性：DNS 域名集合是否稳定 ===",flush=True)
    if ("1102_Idle","dns") in store and ("1108_Idle","dns") in store:
        a,b=store[("1102_Idle","dns")],store[("1108_Idle","dns")]
        for mac in sorted(set(a["eth.src"])|set(b["eth.src"])):
            sa=set(a[a["eth.src"]==mac]["dns.qry.name"].dropna())
            sb=set(b[b["eth.src"]==mac]["dns.qry.name"].dropna())
            j = len(sa&sb)/max(len(sa|sb),1)
            print(f"  {macs.get(mac.lower(),mac):26s} 两天 Jaccard={j:.2f}  "
                  f"({len(sa)} / {len(sb)} 个域名，交 {len(sa&sb)})",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
