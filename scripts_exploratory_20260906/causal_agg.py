"""【探索性,非协议】因果特征聚合：能不能在不牺牲判决密度的前提下拿到块聚合的收益？

对照位置极干净 —— 同样的表示（包长频率分布）、同样的分类器（Hellinger kNN）、
同样 60 秒证据的三种用法，前两个已测：

    块聚合 agg=6         0.9916    每 60 秒一次判决，延迟 60 秒，需缓冲整块
    因果平均概率 k=6      0.8206    每 10 秒一次判决，延迟 10 秒
    因果特征聚合 k=6      ？        每 10 秒一次判决，延迟 10 秒    ← 本脚本

做法：在第 n 个窗，用 [n−k+1, n] 的**原始包计数**合并成直方图再分类。
只用过去，判决密度与延迟和 k=1 完全相同，但直方图由 k×10 秒的包估计而成。

**适用边界**：要求"一条流 = 一台设备"。UNSW / CIC 每台设备自有 MAC，全程合法；
自采的网关三类共用一个 MAC，在那条流上聚合会把三台的包混进同一窗，**不可用**
（与平滑的非法用法同源，见 smoothing 的 +0.031 / −0.257 对照）。

三条臂全部在【每 10 秒一次判决】的口径下报，与块聚合的粒度差异单独注明。
"""
from __future__ import annotations
import sys, time
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

WIN=10.0; MIN_PKT=2; KNN_K=5; AGGS=[1,3,6,10,30]

DATASETS={
 "UNSW": dict(pcap=Path(REPO+"/dataset/unsw/pcap"), pat="{d}.pcap",
   macmap=Path(REPO+"/dataset/unsw/device_mac_map.csv"),
   feat=REPO+"/results/unsw_pilot/four_day/features_unsw_w10_4day.csv",
   src="16-09-23", inner="16-09-30", outer="16-10-12", typ=lambda s:s),
}

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

def run(name, cfg):
    t0=time.time()
    mac_map=EG.load_mac_map(cfg["macmap"])
    fe=pd.read_csv(cfg["feat"],low_memory=False,usecols=["device","label","day","window_id"])
    days=[cfg["src"],cfg["inner"],cfg["outer"]]
    devs=sorted(set.intersection(*[set(IID.day_gate(fe[fe.day==d],d)) for d in days]))
    print(f"\n{'='*88}\n{name}：设备 {len(devs)} 台",flush=True)
    W={d:per_window(cfg["pcap"]/cfg["pat"].format(d=d),mac_map) for d in days}
    for d in days: print(f"  {d}: {len(W[d])} 窗  {time.time()-t0:.0f}s",flush=True)

    vocab=Counter()
    for (dv,_w),c in W[cfg["src"]].items():
        if dv in devs: vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]; kidx={k:i for i,k in enumerate(keys)}
    print(f"  源天词表 {len(keys)} 项",flush=True)

    def causal(day, k):
        """每窗一行；第 n 行的直方图由 [n-k+1, n] 的原始计数合并而成（只用过去）。"""
        by=defaultdict(list)
        for (dv,w),c in W[day].items():
            if dv in devs: by[dv].append((w,c))
        X=[]; y=[]
        for dv,items in by.items():
            items.sort()
            cs=[c for _w,c in items]
            for n in range(len(cs)):
                acc=Counter()
                for c in cs[max(0,n-k+1):n+1]: acc.update(c)
                tot=sum(acc.values())
                if tot==0: continue
                v=np.zeros(len(keys),dtype=np.float32)
                for kk,cnt in acc.items():
                    if kk in kidx: v[kidx[kk]]=cnt
                X.append(np.sqrt(v/tot)); y.append(dv)
        return np.asarray(X), np.asarray(y)

    cls=sorted(devs); rows=[]
    for k in AGGS:
        Xs,ys=causal(cfg["src"],k)
        for tgt in (cfg["inner"],cfg["outer"]):
            Xt,yt=causal(tgt,k)
            kn=KNeighborsClassifier(n_neighbors=KNN_K,metric="euclidean",n_jobs=12)
            kn.fit(Xs,ys); p=kn.predict(Xt)
            f=f1_score(yt,p,average="macro",labels=cls)
            rows.append({"ds":name,"k":k,"evid_s":k*10,"tgt":tgt,"macro":f,"n_dec":len(yt)})
            print(f"  因果特征聚合 k={k:2d}（{k*10:3d}s 证据，每 10s 一判决）  "
                  f"{tgt}  macro={f:.4f}  判决数 {len(yt)}",flush=True)
    return pd.DataFrame(rows)

def main():
    out=[]
    for name,cfg in DATASETS.items():
        out.append(run(name,cfg))
    R=pd.concat(out); R.to_csv("/home/lmy/cic_probe/causal_agg.csv",index=False)
    print(f"\n{'='*88}\n=== 同证据量的三种用法对拍（UNSW，10 类，outer=16-10-12）===",flush=True)
    print("  证据    块聚合(判决稀疏)   因果平均概率   因果特征聚合   判决密度",flush=True)
    ref_block={60:0.9916, 300:0.9973}
    ref_prob ={60:0.8206, 300:None}
    for k in AGGS:
        if k==1: continue
        e=k*10
        r=R[(R.k==k)&(R.tgt=="16-10-12")]
        if r.empty: continue
        ca=r.macro.iloc[0]
        b=ref_block.get(e); pr=ref_prob.get(e)
        print(f"  {e:4d}s   {('%.4f'%b) if b else '   —   '}            "
              f"{('%.4f'%pr) if pr else '   —   '}        {ca:.4f}        每 10 秒",flush=True)
    print("\n  基准 k=1（10s 证据）：",
          f"{R[(R.k==1)&(R.tgt=='16-10-12')].macro.iloc[0]:.4f}",flush=True)
    print("\n判读：因果特征聚合若接近块聚合 → 块聚合的收益可以在不牺牲判决密度下拿到，",flush=True)
    print("      是方法的实质改进；若接近因果平均概率 → 收益来自粗粒度本身，不可移植。",flush=True)

if __name__=="__main__":
    main()
