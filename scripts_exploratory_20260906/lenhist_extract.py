"""【探索性,非协议】帧长直方图特征族 —— 补上从未被表示过的那个观测量。

来由（用户 2026-09-05 提出的指纹视角）：
  指纹相同 → 本来就不该分成两类；指纹不同 → 数据可证支持区分，分类器却做不到 ← 要解决的
自采实测（self_fingerprint.py，原始 pcap，不经模型）：

    跨标签帧长分布 TV        同标签跨轮次噪声底
    T1 vs XM   0.426         T1      0.110
    T1 vs Sen  0.315         XM      0.120
    XM vs Sen  0.498         Sensor  0.239

三对全部在噪声底 3–4 倍以上 ⇒ **困难簇三对都属于"可证可分、只是难分"**，没有标签债。

而 TV 量的是长度直方图的**形状**，IoT 命令帧长度是少数离散值；
我们的 94 列基础池只有 `len_*` 的均值/标准差/分位数，89 列扩展全是分位数/突发/自相关/FFT
（`dir_*` 32 / `time_*` 19 / `seq_*` 18 / `timeup_*` 10 / `timedn_*` 10）——
**没有任何一列表示离散长度结构，矩把它平均掉了。**

本脚本按与主线完全相同的窗口规则（`window_id = floor((t − t_min)/10)`，全部包、len>0）
产出可与 94/89 列表按 (label, round, source_file, window_id) 直接连接的新列。

**目标长度只从源轮次 R2/R3/R4 导出**（按类间/类内方差比排序取 top-K），
R5/R6/R7 从不参与选择 —— 否则就是拿目标域挑特征。

产出：results/feature_expansion_20260905/features_lenhist_w10.csv
定位：候选特征族，交给逐类对验收闸门决定在哪些对上用
（method-is-candidate-acceptance-not-deletion），不是无条件加特征。
"""
from __future__ import annotations
import subprocess, io, time, sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

BASE  = Path("/home/lmy/iot-device-classification/results/robust_v2/raw_all/features_raw_all_w10.csv")
OUTD  = Path("/home/lmy/iot-device-classification/results/feature_expansion_20260905")
OUT   = OUTD/"features_lenhist_w10.csv"
WIN   = 10.0
SRC_ROUNDS = ["R2","R3","R4"]        # 只从这里选目标长度
TOPK  = 32
KEYS  = ["label","round","source_file","window_id"]

def packets(pcap):
    cmd=["tshark","-r",str(pcap),"-T","fields","-e","frame.time_epoch","-e","frame.len",
         "-E","separator=\t"]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not p.stdout.strip(): return pd.DataFrame(columns=["t","len"])
    d=pd.read_csv(io.StringIO(p.stdout),sep="\t",header=None,names=["t","len"])
    d["t"]=pd.to_numeric(d.t,errors="coerce"); d["len"]=pd.to_numeric(d["len"],errors="coerce")
    return d.dropna().query("len > 0")

def main():
    t0=time.time()
    base=pd.read_csv(BASE, usecols=KEYS)
    files=base[["label","round","source_file"]].drop_duplicates()
    print(f"基表 {len(base)} 窗   {len(files)} 个 pcap",flush=True)

    # ---- pass 1：逐窗长度计数 ----
    per={}                                  # (label,round,src,wid) -> Counter
    for _,row in files.iterrows():
        d=packets(row.source_file)
        if d.empty: print(f"  [空] {row.source_file}",flush=True); continue
        rel=d.t.to_numpy()-d.t.to_numpy().min()
        wid=np.floor(rel/WIN).astype(int)
        L=d["len"].to_numpy().astype(int)
        for w in np.unique(wid):
            per[(row.label,row["round"],row.source_file,int(w))]=Counter(L[wid==w])
        print(f"  {row.label:9s} {row['round']}  包 {len(d):6d}  窗 {len(np.unique(wid)):4d}",flush=True)

    # ---- 选目标长度：只用源轮次，类间/类内方差比 ----
    src=[k for k in per if k[1] in SRC_ROUNDS]
    alll=Counter()
    for k in src: alll.update(per[k].keys())
    cand=[L for L,c in alll.items() if c>=50]
    print(f"\n源轮次出现 ≥50 窗的长度 {len(cand)} 种",flush=True)
    frac={}
    for k in src:
        c=per[k]; tot=sum(c.values())
        frac[k]={L: c.get(L,0)/tot for L in cand}
    labs=sorted({k[0] for k in src})
    score={}
    for L in cand:
        by={lab: np.array([frac[k][L] for k in src if k[0]==lab]) for lab in labs}
        mus=np.array([v.mean() for v in by.values()])
        wit=np.mean([v.var() for v in by.values()])
        score[L]= mus.var()/(wit+1e-12)
    tgt=[L for L,_ in sorted(score.items(), key=lambda x:-x[1])[:TOPK]]
    print(f"目标长度 top{TOPK}（只由源轮次导出）：{sorted(tgt)}",flush=True)

    # ---- pass 2：materialize ----
    rows=[]
    for k,c in per.items():
        tot=sum(c.values())
        r=dict(zip(KEYS,k))
        for L in tgt:
            r[f"lenhist_cnt_{L}"]=c.get(L,0)
            r[f"lenhist_frac_{L}"]=c.get(L,0)/tot
        p=np.array([v/tot for v in c.values()])
        top=c.most_common(1)[0]
        r["lenhist_nuniq"]=len(c)
        r["lenhist_entropy"]=float(-(p*np.log(p+1e-12)).sum())
        r["lenhist_top1_len"]=int(top[0])
        r["lenhist_top1_frac"]=top[1]/tot
        r["lenhist_cover_topk"]=sum(c.get(L,0) for L in tgt)/tot
        rows.append(r)
    R=pd.DataFrame(rows)
    OUTD.mkdir(parents=True, exist_ok=True)
    m=base.merge(R,on=KEYS,how="left")
    miss=m.isna().any(axis=1).sum()
    print(f"\n连接：基表 {len(base)} 行 → 匹配 {len(m)-miss} 行，缺 {miss} 行",flush=True)
    m.to_csv(OUT,index=False)
    print(f"写出 {OUT}   {len(m)} 行 × {len(m.columns)} 列（含 4 键）",flush=True)

    # ---- 描述：新列在困难簇三类上的区分度（只看源轮次，避免看目标域）----
    G=["Light_T1","Light_XM","Sensor"]
    s=m[m["round"].isin(SRC_ROUNDS) & m.label.isin(G)]
    cols=[c for c in m.columns if c.startswith("lenhist_")]
    print(f"\n=== 源轮次上，困难簇三类的类间/类内方差比（top 12 列）===",flush=True)
    sc={}
    for c in cols:
        by=[s[s.label==g][c].to_numpy() for g in G]
        mus=np.array([v.mean() for v in by]); wit=np.mean([v.var() for v in by])
        sc[c]=mus.var()/(wit+1e-12)
    for c,v in sorted(sc.items(),key=lambda x:-x[1])[:12]:
        print(f"  {c:24s} F={v:8.3f}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
