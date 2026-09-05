"""【探索性,非协议】ByteIoT 同类数对照 + 因果平滑是否方法无关。

上一轮预检用了 15/11 类而我们主线是 10 类，**不同类数的 macro 不能直接比**，
所以"同粒度下我们更好"这句话还不硬。本脚本补严谨对照：

  · 设备集合用主线同一个 `IID.day_gate`（10 类）
  · 同源天 16-09-23、同目标天 16-09-30 / 16-10-12
  · 判决粒度固定 10 秒（每窗一判决）

三条臂，回答两个问题：
  A  byteiot          包长频率分布 + Hellinger + kNN                同粒度同类数，它到底多少
  B  byteiot+平滑     在 A 的 predict_proba 上做【因果】平滑         我们的第一段是否方法无关
                      按设备分组、只用过去 k 窗，k 在 inner(0930) 上选
  C  ours             我们 base+lenhist + RF（已知 0.8463 / 0.8548）  参照

关键区别（上一轮我讲反了，用户纠正）：
  因果平滑 k=10   用 100 秒证据，**每 10 秒出一次判决**，延迟 10 秒，只用过去
  块聚合 agg=30   用 300 秒证据，每 300 秒出一次判决，延迟 300 秒，需缓冲整块
同样的证据量，判决密度差 30 倍。B 臂就是要把这一点量出来。
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

PCAP=Path(REPO+"/dataset/unsw/pcap")
MACMAP=Path(REPO+"/dataset/unsw/device_mac_map.csv")
FEAT=REPO+"/results/unsw_pilot/four_day/features_unsw_w10_4day.csv"
SRC="16-09-23"; INNER="16-09-30"; OUTER="16-10-12"
WIN=10.0; MIN_PKT=2; KS=[1,3,5]; SMOOTH=[1,3,6,10,20,30]

def per_window(day, mac_map):
    pk=EG.read_packets(PCAP/f"{day}.pcap",set(mac_map),verbose=False)
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

def cmeanM(Pm, grp, k):
    """按流分组的因果滑动平均：第 n 个窗用 [n-k+1, n]，只用过去。"""
    if k<=1: return Pm
    o=np.empty_like(Pm)
    for u in np.unique(grp):
        i=np.where(grp==u)[0]; V=Pm[i]
        C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def main():
    t0=time.time(); mac_map=EG.load_mac_map(MACMAP)
    # 设备集合：与主线同一个门槛
    fe=pd.read_csv(FEAT,low_memory=False,usecols=["device","label","day","window_id"])
    devs=sorted(set(IID.day_gate(fe[fe.day==SRC],SRC))
              & set(IID.day_gate(fe[fe.day==INNER],INNER))
              & set(IID.day_gate(fe[fe.day==OUTER],OUTER)))
    print(f"主线口径设备集合：{len(devs)} 台\n{devs}\n",flush=True)

    W={d:per_window(d,mac_map) for d in (SRC,INNER,OUTER)}
    for d in (SRC,INNER,OUTER): print(f"  {d}: {len(W[d])} 窗   {time.time()-t0:.0f}s",flush=True)

    vocab=Counter()
    for (dv,_w),c in W[SRC].items():
        if dv in devs: vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]; kidx={k:i for i,k in enumerate(keys)}
    print(f"\n源天词表 {len(keys)} 项",flush=True)

    def mat(day):
        items=sorted([(dv,w,c) for (dv,w),c in W[day].items() if dv in devs])
        X=np.zeros((len(items),len(keys)),dtype=np.float32)
        y=[]; g=[]
        for r,(dv,w,c) in enumerate(items):
            tot=sum(c.values())
            for k,n in c.items():
                if k in kidx: X[r,kidx[k]]=n
            X[r]=np.sqrt(X[r]/max(tot,1))
            y.append(dv); g.append(dv)
        return X, np.asarray(y), np.asarray(g)

    Xs,ys,_=mat(SRC)
    cls=sorted(devs)
    rows=[]
    for tgt in (INNER,OUTER):
        Xt,yt,gt=mat(tgt)
        for k in KS:
            kn=KNeighborsClassifier(n_neighbors=k,metric="euclidean",n_jobs=12)
            kn.fit(Xs,ys)
            Pm=kn.predict_proba(Xt)
            order=[list(kn.classes_).index(c) for c in cls]
            Pm=Pm[:,order]
            f0=f1_score(yt,[cls[i] for i in Pm.argmax(1)],average="macro",labels=cls)
            rows.append({"tgt":tgt,"k":k,"smooth":1,"macro":f0})
            print(f"  byteiot  {tgt}  k={k}  平滑=1   macro={f0:.4f}",flush=True)
            for s in SMOOTH[1:]:
                p=[cls[i] for i in cmeanM(Pm,gt,s).argmax(1)]
                f=f1_score(yt,p,average="macro",labels=cls)
                rows.append({"tgt":tgt,"k":k,"smooth":s,"macro":f})
                print(f"  byteiot  {tgt}  k={k}  平滑={s:<2d}  macro={f:.4f}",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/byteiot_control.csv",index=False)
    print("\n=== A：同 10 类、10 秒判决，ByteIoT 不平滑 ===",flush=True)
    A=R[R.smooth==1].groupby("tgt").macro.max().round(4)
    print(A.to_string(),flush=True)
    print("   对照 我们 base+lenhist：16-09-30 0.8463 / 16-10-12 0.8548",flush=True)

    print("\n=== B：加上我们的因果平滑（k 在 inner 0930 上选，判决仍每 10 秒一次）===",flush=True)
    inner=R[R.tgt==INNER]
    bs=int(inner.loc[inner.macro.idxmax(),"smooth"]); bk=int(inner.loc[inner.macro.idxmax(),"k"])
    print(f"   inner 选出：kNN k={bk}，平滑窗={bs}",flush=True)
    sel=R[(R.k==bk)&(R.smooth==bs)]
    print(sel[["tgt","macro"]].to_string(index=False),flush=True)
    o0=R[(R.tgt==OUTER)&(R.smooth==1)].macro.max()
    o1=R[(R.tgt==OUTER)&(R.k==bk)&(R.smooth==bs)].macro.iloc[0]
    print(f"\n   outer：不平滑 {o0:.4f} → 平滑 {o1:.4f}   Δ={o1-o0:+.4f}",flush=True)
    print("   判读：若平滑把它也大幅抬起来 → 我们的第一段是【方法无关】的，",flush=True)
    print("         且在 10 秒判决密度下就能拿到块聚合的收益，这是可写的强结论。",flush=True)
    print("\n=== 完整表 ===",flush=True)
    print(R.pivot_table(index=["tgt","k"],columns="smooth",values="macro").round(4).to_string(),flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
