"""【探索性,非协议】CIC 全设备协议栈指纹表 —— 给同设备检测器做独立外部真值。

来由：ESP 抽样看到四个互不相干的观测量给出同一个二分（win 4380/2920、心跳 59.6/60.4 s、
做不做 DNS、连 1 个还是 3 个端点），且两天完全一致；而**已知修不动的对全部落在组内**，
没有一个跨组对进过困难名单。

本脚本把这套指纹扩到全部 36 台、两天，产出机器可读表，用途是**验证**而非建特征：

  检测器开火（域内≥0.90 且 跨捕获≤0.60，即"数据不支持这个标签区分"）
      ⇒ 该对的指纹距离应当 ≈ 0
  指纹明确不同（如 window 不同）
      ⇒ 检测器【绝不该】开火          ← 特异性硬检验，外部真值，从未进过分类器

指纹项（全部跨捕获稳定、与分类器所用的 61 列量/时序特征无交集）：
  ttl_mode, win_mode, heartbeat_median, does_dns, n_endpoints, dns_names, endpoints
两天各算一次，**只保留两天一致的项**（不一致的项不可用作身份）。
"""
from __future__ import annotations
import subprocess, io, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/lmy/iot-device-classification/dataset/cic2022")
EXTR = ROOT/"extracted"
DAYS = [("1102_Idle", EXTR/"2-Idle/2021_11_02_Idle.pcap"),
        ("1108_Idle", EXTR/"2-Idle/2021_11_08_Idle.pcap")]
OUT  = Path("/home/lmy/cic_probe/cic_fingerprint.csv")

def tsh(pcap, disp, fields):
    cmd=["tshark","-r",str(pcap),"-Y",disp,"-T","fields"]
    for f in fields: cmd+=["-e",f]
    cmd+=["-E","separator=\t"]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not p.stdout.strip(): return pd.DataFrame(columns=fields)
    return pd.read_csv(io.StringIO(p.stdout),sep="\t",header=None,names=fields,dtype=str)

def num(s): return pd.to_numeric(s.astype(str).str.split(",").str[0], errors="coerce")

def main():
    t0=time.time()
    m=pd.read_csv(ROOT/"device_mac_map.csv")
    m=m[m.is_iot==1] if "is_iot" in m else m
    mac2dev=dict(zip(m.mac.str.lower(), m.device_id))
    print(f"设备 {len(mac2dev)} 台",flush=True)

    rows=[]
    for day,p in DAYS:
        if not p.exists(): print(f"[缺] {p}",flush=True); continue
        print(f"\n=== {day} ({p.stat().st_size/1e6:.0f} MB) ===",flush=True)
        tcp=tsh(p,"tcp",["eth.src","frame.time_epoch","ip.ttl","tcp.window_size_value"])
        for c in ["frame.time_epoch","ip.ttl","tcp.window_size_value"]: tcp[c]=num(tcp[c])
        tcp=tcp.dropna(subset=["eth.src"])
        syn=tsh(p,"tcp.flags.syn==1 and tcp.flags.ack==0",["eth.src","ip.dst","tcp.dstport"]).dropna()
        dns=tsh(p,"dns.flags.response==0",["eth.src","dns.qry.name"]).dropna()
        print(f"  TCP {len(tcp)}  SYN {len(syn)}  DNS {len(dns)}",flush=True)

        for mac,dev in mac2dev.items():
            g=tcp[tcp["eth.src"].str.lower()==mac]
            if len(g)<50: continue
            g=g.sort_values("frame.time_epoch")
            dt=np.diff(g["frame.time_epoch"].to_numpy()); dt=dt[dt>1.0]
            s=syn[syn["eth.src"].str.lower()==mac]
            dq=dns[dns["eth.src"].str.lower()==mac]
            eps=sorted(set(s["ip.dst"]+":"+s["tcp.dstport"])) if len(s) else []
            nms=sorted(set(dq["dns.qry.name"])) if len(dq) else []
            rows.append({"day":day,"device":dev,"mac":mac,"n_tcp":len(g),
                "ttl_mode": float(g["ip.ttl"].mode().iloc[0]) if g["ip.ttl"].notna().any() else np.nan,
                "win_mode": float(g["tcp.window_size_value"].mode().iloc[0]) if g["tcp.window_size_value"].notna().any() else np.nan,
                "heartbeat": float(np.median(dt)) if len(dt)>5 else np.nan,
                "does_dns": int(len(dq)>0), "n_endpoints": len(eps),
                "endpoints": json.dumps(eps), "dns_names": json.dumps(nms)})
        print(f"  可用设备 {len([r for r in rows if r['day']==day])} 台",flush=True)

    R=pd.DataFrame(rows); R.to_csv(OUT,index=False)
    print(f"\n写出 {OUT}  {len(R)} 行",flush=True)

    print("\n=== 跨天一致性（只有两天一致的项才可用作身份）===",flush=True)
    both=R.groupby("device").filter(lambda g: g.day.nunique()==2)
    for col in ["ttl_mode","win_mode","heartbeat","does_dns","n_endpoints"]:
        ok=0; tot=0
        for dev,g in both.groupby("device"):
            v=g[col].values
            if pd.isna(v).any(): continue
            tot+=1
            same = abs(v[0]-v[1])<= (1.0 if col=="heartbeat" else 1e-9)
            ok+=int(same)
        print(f"  {col:14s} 两天一致 {ok}/{tot}",flush=True)

    print("\n=== 逐设备（1102 那天）===",flush=True)
    A=R[R.day=="1102_Idle"].sort_values(["win_mode","heartbeat"])
    print(A[["device","ttl_mode","win_mode","heartbeat","does_dns","n_endpoints","n_tcp"]]
          .to_string(index=False),flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
