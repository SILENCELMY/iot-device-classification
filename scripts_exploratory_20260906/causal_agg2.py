"""【探索性,非协议】因果特征聚合（续）：CIC 上会不会也饱和？UNSW 上还剩多少给第二段？

上一轮（UNSW，10 类，每 10 秒一判决）：
    证据 60s   块聚合 0.9916 / 因果平均概率 0.8206 / **因果特征聚合 0.9908**
    证据 300s  块聚合 0.9973 /        —        / **因果特征聚合 0.9999**
=> 块聚合的收益可以在不牺牲判决密度与延迟的前提下拿到，第一段应该换成特征层聚合。

本脚本回答两个跟进问题：

**Q1（CIC）** 因果特征聚合会不会也饱和在 0.82？
  块聚合在 CIC 上 300s→2400s 从 0.8209 到 0.8222 纹丝不动。
  因果特征聚合是**另一条独立路径**：若两者都饱和在同一处，
  那个饱和就不是某种聚合方式的特性，**是数据本身的** —— "身份型困难"的又一次独立验证。
  若它反而爬上去 → CIC 的结论要改。

**Q2（UNSW）** k 增大后残余错误还剩什么？第二段（逐类对定点修）还有没有作用面？
  用户判断"还有空间，因为第二段就是针对困难簇设计的"。
  这里给出直接证据：逐 k 报错误总数与逐对分布 —— Belkin 那对（占 57%）在不在。

口径：全部每 10 秒一次判决；因果 = 第 n 窗只用 [n−k+1, n] 的原始包。
适用边界：要求"一条流 = 一台设备"。UNSW/CIC 每台自有 MAC，合法；
自采网关三类共用 MAC，该层不可用（与平滑的非法用法同源）。
"""
from __future__ import annotations
import sys, time, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/core")
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import extract_features_generic as EG
import pilot_rf_loro as P, run_unsw_iid_reference as IID

WIN=10.0; MIN_PKT=2; KNN_K=5

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

EXTR=Path(REPO+"/dataset/cic2022/extracted")
JOBS=[
 dict(name="UNSW", macmap=Path(REPO+"/dataset/unsw/device_mac_map.csv"), typ=lambda s:s,
      aggs=[1,3,6,10,30],
      days=[("16-09-23",Path(REPO+"/dataset/unsw/pcap/16-09-23.pcap"),None),
            ("16-10-12",Path(REPO+"/dataset/unsw/pcap/16-10-12.pcap"),None)],
      featcsv=REPO+"/results/unsw_pilot/four_day/features_unsw_w10_4day.csv"),
 dict(name="CIC", macmap=Path(REPO+"/dataset/cic2022/device_mac_map.csv"), typ=TYPE,
      aggs=[1,6,30,60,120],
      days=[("2021_11_02_Idle",EXTR/"2-Idle/2021_11_02_Idle.pcap","/home/lmy/cic_probe/idle_1102.csv"),
            ("2021_11_08_Active",EXTR/"5-Active/2021_11_08_Active.pcap","/home/lmy/cic_probe/active_1108.csv")],
      featcsv=None),
]

def per_window(pcap, mac_map):
    pk=EG.read_packets(pcap,set(mac_map),verbose=False)
    st=EG.assign_device_streams(pk,mac_map)
    origin=float(st["time_epoch"].min())
    wid=np.floor((st["time_epoch"].to_numpy()-origin)/WIN).astype(np.int64)
    dev=st["device"].to_numpy(); ln=st["length"].to_numpy().astype(int)
    up=st["is_up"].to_numpy().astype(int)
    out={}; o=np.lexsort((wid,dev)); dev,wid,ln,up=dev[o],wid[o],ln[o],up[o]
    b=0
    for i in range(1,len(dev)+1):
        if i==len(dev) or dev[i]!=dev[b] or wid[i]!=wid[b]:
            if i-b>=MIN_PKT: out[(dev[b],int(wid[b]))]=Counter(zip(ln[b:i],up[b:i]))
            b=i
    return out

def run(job):
    t0=time.time(); mac_map=EG.load_mac_map(job["macmap"])
    # 设备门槛
    sets=[]
    for day,_p,csv in job["days"]:
        if csv: d=pd.read_csv(csv,low_memory=False,usecols=["device","label","day","window_id"])
        else:
            d=pd.read_csv(job["featcsv"],low_memory=False,
                          usecols=["device","label","day","window_id"]); d=d[d.day==day]
        sets.append(set(IID.day_gate(d,day)))
    devs=sorted(set.intersection(*sets))
    typ=job["typ"]; cls=sorted({typ(d) for d in devs})
    print(f"\n{'='*88}\n{job['name']}：设备 {len(devs)} → 类 {len(cls)}",flush=True)

    W={}
    for day,p,_c in job["days"]:
        W[day]=per_window(p,mac_map)
        print(f"  {day}: {len(W[day])} 窗  {time.time()-t0:.0f}s",flush=True)
    src=job["days"][0][0]; tgt=job["days"][1][0]

    vocab=Counter()
    for (dv,_w),c in W[src].items():
        if dv in devs: vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]; kidx={k:i for i,k in enumerate(keys)}
    print(f"  源天词表 {len(keys)} 项",flush=True)

    def causal(day,k):
        by=defaultdict(list)
        for (dv,w),c in W[day].items():
            if dv in devs: by[dv].append((w,c))
        X=[];y=[]
        for dv,items in by.items():
            items.sort(); cs=[c for _w,c in items]
            run_=Counter()
            for n in range(len(cs)):
                run_.update(cs[n])
                if n>=k: run_.subtract(cs[n-k]); run_+=Counter()   # 去掉滑出窗口的
                tot=sum(run_.values())
                if tot==0: continue
                v=np.zeros(len(keys),dtype=np.float32)
                for kk,cnt in run_.items():
                    if kk in kidx and cnt>0: v[kidx[kk]]=cnt
                X.append(np.sqrt(v/tot)); y.append(typ(dv))
        return np.asarray(X), np.asarray(y)

    rows=[]
    for k in job["aggs"]:
        Xs,ys=causal(src,k); Xt,yt=causal(tgt,k)
        kn=KNeighborsClassifier(n_neighbors=KNN_K,metric="euclidean",n_jobs=12)
        kn.fit(Xs,ys); p=kn.predict(Xt)
        f=f1_score(yt,p,average="macro",labels=cls)
        C=confusion_matrix(yt,p,labels=cls); off=C.copy(); np.fill_diagonal(off,0)
        tot=int(off.sum()); pr={}
        for i in range(len(cls)):
            for j in range(i+1,len(cls)):
                v=off[i,j]+off[j,i]
                if v: pr[(i,j)]=int(v)
        top=sorted(pr.items(),key=lambda x:-x[1])[:4]
        rows.append({"ds":job["name"],"k":k,"evid_s":k*10,"macro":f,
                     "n_err":tot,"n_dec":len(yt)})
        print(f"\n  k={k:3d}（{k*10:4d}s 证据，每 10s 一判决）  macro={f:.4f}  "
              f"错误 {tot} / {len(yt)}",flush=True)
        for (i,j),v in top:
            print(f"      {cls[i]:24s} {cls[j]:24s} {v:6d} ({v/max(tot,1):5.1%})",flush=True)
    return pd.DataFrame(rows)

def main():
    out=[run(j) for j in JOBS]
    R=pd.concat(out); R.to_csv("/home/lmy/cic_probe/causal_agg2.csv",index=False)
    print(f"\n{'='*88}\n=== 汇总：因果特征聚合曲线（每 10 秒一判决）===",flush=True)
    print(R.pivot_table(index=["k","evid_s"],columns="ds",values="macro").round(4).to_string(),flush=True)
    print("\n对照 块聚合（判决稀疏）：UNSW 60s 0.9916 / 300s 0.9973；"
          "CIC 300s 0.8209 / 2400s 0.8222",flush=True)
    print("\n判读 Q1：CIC 若同样饱和在 ~0.82 → 饱和是数据的性质而非聚合方式的，"
          "身份型困难再获独立验证。",flush=True)
    print("判读 Q2：UNSW 若 Belkin 那对随 k 增大而消失 → 第二段在 UNSW 上作用面归零，须如实报；"
          "若仍在 → 第二段仍有空间。",flush=True)

if __name__=="__main__":
    main()
