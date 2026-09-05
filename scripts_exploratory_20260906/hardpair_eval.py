"""【探索性,非协议】困难对专项评测：去掉可分的，只看不可分的。

用户 2026-09-05：「100 个人 99 个得癌症，全预测得癌症就有 99% 准确率，但根本没解决问题。
我们的方法第二段就是研究这个的，所以正确方向是去掉可分的，重点研究不可分的。」

**我此前的失误**：报"UNSW k=10 只剩 10 个错误"时用的是全局错误数，
没算那一对在【它自己的可判决区】上的准确率。9 个错误可能是 6000 个窗里错 9 个（已解决），
也可能是 20 个争议窗里错 9 个（没解决）—— 这两种在 macro=0.9995 里长得一模一样。

**本脚本的主指标**（不是全体 macro）：
  对每个困难对 (i,j)：
    可判决区  = 该窗的 top-1/top-2 恰为 {i,j}，且真类 ∈ {i,j}
    区上准确率 = top-1 判对的比例          ← 这才是"这对解没解决"
    并报 pair-only accuracy = 只在真类 ∈{i,j} 的窗上，二选一判对的比例
逐证据量 k 报，看聚合把它推到哪里。

困难对取法：k=1 时错误最多的前 N 对（不看目标域标签之外的信息，只是选评测对象）。
"""
from __future__ import annotations
import sys, time, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/core")
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import extract_features_generic as EG
import pilot_rf_loro as P, run_unsw_iid_reference as IID

WIN=10.0; MIN_PKT=2; KNN_K=5; TOPN=4
EXTR=Path(REPO+"/dataset/cic2022/extracted")

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

JOBS=[
 dict(name="UNSW", macmap=Path(REPO+"/dataset/unsw/device_mac_map.csv"), typ=lambda s:s,
      ks=[1,3,6,10,30],
      days=[("16-09-23",Path(REPO+"/dataset/unsw/pcap/16-09-23.pcap"),None),
            ("16-10-12",Path(REPO+"/dataset/unsw/pcap/16-10-12.pcap"),None)],
      featcsv=REPO+"/results/unsw_pilot/four_day/features_unsw_w10_4day.csv"),
 dict(name="CIC", macmap=Path(REPO+"/dataset/cic2022/device_mac_map.csv"), typ=TYPE,
      ks=[1,6,30,120],
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
    sets=[]
    for day,_p,csv in job["days"]:
        d=(pd.read_csv(csv,low_memory=False,usecols=["device","label","day","window_id"])
           if csv else pd.read_csv(job["featcsv"],low_memory=False,
                usecols=["device","label","day","window_id"]).query("day==@day"))
        sets.append(set(IID.day_gate(d,day)))
    devs=sorted(set.intersection(*sets)); typ=job["typ"]
    cls=sorted({typ(d) for d in devs}); ci={c:i for i,c in enumerate(cls)}
    print(f"\n{'='*90}\n{job['name']}：{len(devs)} 台 → {len(cls)} 类",flush=True)
    W={d:per_window(p,mac_map) for d,p,_c in job["days"]}
    src=job["days"][0][0]; tgt=job["days"][1][0]
    vocab=Counter()
    for (dv,_w),c in W[src].items():
        if dv in devs: vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]; kidx={k:i for i,k in enumerate(keys)}

    def causal(day,k):
        by=defaultdict(list)
        for (dv,w),c in W[day].items():
            if dv in devs: by[dv].append((w,c))
        X=[];y=[]
        for dv,items in by.items():
            items.sort(); cs=[c for _w,c in items]
            for n in range(len(cs)):
                acc=Counter()
                for c in cs[max(0,n-k+1):n+1]: acc.update(c)
                tot=sum(acc.values())
                if tot==0: continue
                v=np.zeros(len(keys),dtype=np.float32)
                for kk,cnt in acc.items():
                    if kk in kidx: v[kidx[kk]]=cnt
                X.append(np.sqrt(v/tot)); y.append(ci[typ(dv)])
        return np.asarray(X), np.asarray(y)

    hard=None; rows=[]
    for k in job["ks"]:
        Xs,ys=causal(src,k); Xt,yt=causal(tgt,k)
        kn=KNeighborsClassifier(n_neighbors=KNN_K,metric="euclidean",n_jobs=12)
        kn.fit(Xs,ys); Pm=kn.predict_proba(Xt)
        order=[list(kn.classes_).index(i) if i in kn.classes_ else -1 for i in range(len(cls))]
        Q=np.zeros((len(Xt),len(cls)))
        for i,o in enumerate(order):
            if o>=0: Q[:,i]=Pm[:,o]
        oo=np.argsort(-Q,axis=1); t1,t2=oo[:,0],oo[:,1]
        mac=f1_score(yt,t1,average="macro",labels=np.arange(len(cls)))
        if hard is None:
            err=defaultdict(int)
            for a,b in zip(yt,t1):
                if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
            hard=[p for p,_ in sorted(err.items(),key=lambda x:-x[1])[:TOPN]]
            print(f"  困难对（由 k=1 的错误量选出，仅用于选评测对象）：",flush=True)
            for i,j in hard: print(f"    {cls[i]} | {cls[j]}",flush=True)
        print(f"\n  --- k={k}（{k*10}s 证据）  全体 macro={mac:.4f} ---",flush=True)
        for i,j in hard:
            D=(((t1==i)&(t2==j))|((t1==j)&(t2==i)))&np.isin(yt,[i,j])
            pm=np.isin(yt,[i,j])
            reg=float((t1[D]==yt[D]).mean()) if D.sum()>0 else np.nan
            pair=float((t1[pm]==yt[pm]).mean()) if pm.sum()>0 else np.nan
            rows.append({"ds":job["name"],"k":k,"evid_s":k*10,
                         "pair":f"{cls[i]}|{cls[j]}","n_region":int(D.sum()),
                         "acc_region":reg,"n_pair":int(pm.sum()),"acc_pair":pair,
                         "macro_all":mac})
            print(f"    {cls[i][:20]:20s}|{cls[j][:20]:20s} 可判决区 n={int(D.sum()):6d} "
                  f"区上准确率={reg:.4f}   该对全部窗 n={int(pm.sum()):6d} 对内准确率={pair:.4f}",
                  flush=True)
    return pd.DataFrame(rows)

def main():
    R=pd.concat([run(j) for j in JOBS])
    R.to_csv("/home/lmy/cic_probe/hardpair_eval.csv",index=False)
    print(f"\n{'='*90}\n=== 主指标：困难对在可判决区上的准确率（随证据量）===",flush=True)
    for ds in R.ds.unique():
        S=R[R.ds==ds]
        print(f"\n{ds}",flush=True)
        print(S.pivot_table(index="pair",columns="evid_s",values="acc_region").round(4).to_string(),flush=True)
    print("\n对照：全体 macro",flush=True)
    print(R.pivot_table(index="ds",columns="evid_s",values="macro_all").round(4).to_string(),flush=True)
    print("\n判读：全体 macro 由简单类撑着；只有'区上准确率'能回答'这对解没解决'。",flush=True)

if __name__=="__main__":
    main()
