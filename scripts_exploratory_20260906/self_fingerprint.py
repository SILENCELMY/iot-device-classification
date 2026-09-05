"""【探索性,非协议】自采三类的"指纹"：两个灯是不是同一款？

用户 2026-09-05：「有两个灯，如果他们的指纹差不多的话那我们现在应该考虑先把灯和传感器
区分开」。这是 CIC 那条判据在自采上的对应版本：
    指纹相同 → 本来就不该分成两类（要分得换模态）
    指纹不同 → 数据可证支持区分，分类器却做不到 ← 这才是要解决的

**难点**：三台 Zigbee/BLE 设备共用小米网关的 802.11 电台与 TCP 栈
（[[three-labels-share-one-transmitter]]），所以 TTL / tcp.window 这类栈指纹
**按构造完全相同**，取不到。能取到的对应物是【命令/响应突发的帧长结构】——
Zigbee 载荷大小不同会透到网关的 802.11 上行帧长上，且与加密无关。

**只做描述，不建特征、不跑模型。** 三个观测量，每个都问"跨轮次稳不稳"：
  A  帧长集合：各标签出现过哪些长度、两两 Jaccard
  B  帧长分布：两两总变差距离（TV）
  C  突发结构：静默(>1s)后首包长度的分布 —— 命令帧最可能落在这里

判据：若 TV(T1,XM) ≪ TV(T1,Sensor) 且 ≈ 轮次间自距离 → 两个灯同款，
      困难簇应重整为 {Lamp, Sensor} 两类；T1|XM 归入"不该分"。
"""
from __future__ import annotations
import subprocess, io, time, itertools
from pathlib import Path
import numpy as np, pandas as pd

DS = Path("/home/lmy/iot-device-classification/dataset")
GW = "54:ef:44:59:eb:4c"          # 小米网关（三台共用）
DEVS = {"Light_T1":"light_T1", "Light_XM":"light_xm", "Sensor":"sensor"}
ROUNDS = {"R2":"round2_normal","R3":"round3_normal","R4":"round4_normal"}
OUT = Path("/home/lmy/cic_probe/self_fingerprint.csv")

def frames(pcap):
    """取网关发出的帧的长度与时间。"""
    cmd=["tshark","-r",str(pcap),"-Y",f"wlan.ta=={GW}","-T","fields",
         "-e","frame.time_epoch","-e","frame.len","-E","separator=\t"]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not p.stdout.strip(): return pd.DataFrame(columns=["t","len"])
    d=pd.read_csv(io.StringIO(p.stdout),sep="\t",header=None,names=["t","len"])
    return d.apply(pd.to_numeric,errors="coerce").dropna()

def hist(v, lo=0, hi=400):
    h=np.bincount(np.clip(v.astype(int),lo,hi-1),minlength=hi).astype(float)
    return h/max(h.sum(),1)

def tv(a,b): return 0.5*np.abs(a-b).sum()

def main():
    t0=time.time(); data={}
    for lab,d in DEVS.items():
        for r,rd in ROUNDS.items():
            g=list((DS/d/rd).glob("*.pcapng"))+list((DS/d/rd).glob("*.pcap"))
            if not g: print(f"[缺] {d}/{rd}",flush=True); continue
            f=frames(g[0])
            if f.empty: print(f"[空] {lab} {r}",flush=True); continue
            f=f.sort_values("t")
            dt=np.diff(f.t.to_numpy())
            first=f.len.to_numpy()[1:][dt>1.0]          # 静默后首包 = 命令帧候选
            data[(lab,r)]={"all":f.len.to_numpy(),"first":first}
            print(f"  {lab:9s} {r}  帧 {len(f):6d}  静默后首包 {len(first):5d}  "
                  f"长度中位 {np.median(f.len):.0f}  唯一长度 {f.len.nunique()}",flush=True)

    keys=sorted(data)
    print(f"\n=== A 帧长集合 Jaccard（同轮次内，两两）===",flush=True)
    for r in ROUNDS:
        ks=[k for k in keys if k[1]==r]
        for (a,_),(b,_) in itertools.combinations(ks,2):
            sa=set(data[(a,r)]["all"]); sb=set(data[(b,r)]["all"])
            print(f"  {r}  {a:9s} vs {b:9s}  Jaccard={len(sa&sb)/max(len(sa|sb),1):.3f}"
                  f"  ({len(sa)}/{len(sb)} 种长度)",flush=True)

    print(f"\n=== B 帧长分布总变差 TV（0=完全相同, 1=完全不同）===",flush=True)
    rows=[]
    for r in ROUNDS:
        ks=[k[0] for k in keys if k[1]==r]
        if len(ks)<2: continue
        H={a:hist(data[(a,r)]["all"]) for a in ks}
        Hf={a:hist(data[(a,r)]["first"]) for a in ks if len(data[(a,r)]["first"])>20}
        for a,b in itertools.combinations(ks,2):
            t_all=tv(H[a],H[b]); t_f=tv(Hf[a],Hf[b]) if a in Hf and b in Hf else np.nan
            rows.append({"round":r,"a":a,"b":b,"tv_all":t_all,"tv_first":t_f,"kind":"跨标签"})
            print(f"  {r}  {a:9s} vs {b:9s}  TV(全部)={t_all:.3f}   TV(命令帧)={t_f:.3f}",flush=True)

    print(f"\n=== 参照：同一标签在不同轮次之间的自距离（噪声底）===",flush=True)
    for lab in DEVS:
        ks=[k for k in keys if k[0]==lab]
        for (a,ra),(b,rb) in itertools.combinations(ks,2):
            ha,hb=hist(data[(a,ra)]["all"]),hist(data[(b,rb)]["all"])
            fa=data[(a,ra)]["first"]; fb=data[(b,rb)]["first"]
            t_f=tv(hist(fa),hist(fb)) if len(fa)>20 and len(fb)>20 else np.nan
            rows.append({"round":f"{ra}-{rb}","a":lab,"b":lab,"tv_all":tv(ha,hb),
                         "tv_first":t_f,"kind":"同标签跨轮次"})
            print(f"  {lab:9s} {ra} vs {rb}  TV(全部)={tv(ha,hb):.3f}   TV(命令帧)={t_f:.3f}",flush=True)

    R=pd.DataFrame(rows); R.to_csv(OUT,index=False)
    print("\n=== 判读 ===",flush=True)
    X=R[R.kind=="跨标签"].groupby(["a","b"])[["tv_all","tv_first"]].mean().round(3)
    S=R[R.kind=="同标签跨轮次"].groupby("a")[["tv_all","tv_first"]].mean().round(3)
    print("跨标签平均 TV：\n"+X.to_string(),flush=True)
    print("\n同标签跨轮次平均 TV（噪声底）：\n"+S.to_string(),flush=True)
    print("\n  若 TV(Light_T1,Light_XM) ≈ 噪声底，而 TV(灯,Sensor) 明显更大",flush=True)
    print("  → 两个灯同款，困难簇应重整为 {Lamp, Sensor}；T1|XM 归'不该分'。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
