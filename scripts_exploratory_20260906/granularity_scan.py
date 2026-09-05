"""【探索性,非协议】判决粒度扫描 —— 困难簇是真的，还是短窗造成的？

来由：ByteIoT 核心（包长频率分布 + Hellinger + kNN）在 UNSW 跨天上
  10s 窗   0.7127 / 0.8090   （比我们的 0.8463 / 0.8548 差）
  60s 聚合 0.9916 / 0.9921
  300s 聚合 0.9973 / 1.0000  ← 跨天完美
=> 「跨天识别难」在很大程度上是**判决粒度定在 10 秒**造成的。
审稿人必问："那为什么不聚合？"

**决定性问题**：CIC 的 ESP 困难簇（前三对占 68.5% 错误）能不能被聚合修好？
  能  → 困难簇大部分是短窗产物，我们的问题大幅缩水，框架要重做
  不能 → 困难簇与粒度无关，是真的，我们的问题成立

CIC 每台设备有自己的 MAC，**按设备聚合是合法的**（部署时看得见 MAC），
所以这里没有"聚合等于作弊"的问题 —— 这也正是它作为判据的价值。
（自采的网关三类不同：三台共用一个 MAC，聚合会把标签混掉，另做。）

两条臂，用来区分"粒度对谁有用"：
  bytiot  包长频率分布 + Hellinger + kNN      （ByteIoT 核心）
  ours    我们的 61 列基础特征逐窗取均值 + RF  （同粒度下的我们）
并在每个粒度报**逐对错误集中度**：ESP 那三对是否还在。
"""
from __future__ import annotations
import sys, time, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/core")
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import extract_features_generic as EG
import pilot_rf_loro as P, run_unsw_iid_reference as IID

EXTR=Path(REPO+"/dataset/cic2022/extracted")
MACMAP=Path(REPO+"/dataset/cic2022/device_mac_map.csv")
SRC=("2021_11_02_Idle", EXTR/"2-Idle/2021_11_02_Idle.pcap",
     "/home/lmy/cic_probe/idle_1102.csv")
TGT=("2021_11_08_Active", EXTR/"5-Active/2021_11_08_Active.pcap",
     "/home/lmy/cic_probe/active_1108.csv")
WIN=10.0; MIN_PKT=2; AGGS=[1,6,30,60,120,240]; KS=[1,3,5]; NJ=12; SEED=42

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

def per_window(pcap, mac_map):
    pk=EG.read_packets(pcap,set(mac_map),verbose=False)
    st=EG.assign_device_streams(pk,mac_map)
    origin=float(st["time_epoch"].min())
    wid=np.floor((st["time_epoch"].to_numpy()-origin)/WIN).astype(np.int64)
    dev=st["device"].to_numpy(); ln=st["length"].to_numpy().astype(int)
    up=st["is_up"].to_numpy().astype(int)
    out={}; order=np.lexsort((wid,dev)); dev,wid,ln,up=dev[order],wid[order],ln[order],up[order]
    b=0
    for i in range(1,len(dev)+1):
        if i==len(dev) or dev[i]!=dev[b] or wid[i]!=wid[b]:
            if i-b>=MIN_PKT: out[(dev[b],int(wid[b]))]=Counter(zip(ln[b:i],up[b:i]))
            b=i
    return out

def main():
    t0=time.time(); mac_map=EG.load_mac_map(MACMAP)
    W={}
    for tag,pcap,_csv in (SRC,TGT):
        W[tag]=per_window(pcap,mac_map)
        print(f"  {tag}: {len(W[tag])} 窗   {time.time()-t0:.0f}s",flush=True)

    # 设备门槛与主线一致
    dsrc=pd.read_csv(SRC[2],low_memory=False,usecols=["device","label","day","window_id"])
    dtgt=pd.read_csv(TGT[2],low_memory=False,usecols=["device","label","day","window_id"])
    devs=sorted(set(IID.day_gate(dsrc,SRC[0])) & set(IID.day_gate(dtgt,TGT[0])))
    print(f"设备 {len(devs)} → 类 {len({TYPE(d) for d in devs})}",flush=True)

    vocab=Counter()
    for (dv,_w),c in W[SRC[0]].items():
        if dv in devs: vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]; kidx={k:i for i,k in enumerate(keys)}
    print(f"源天词表 {len(keys)} 项\n",flush=True)

    # 我们的 61 列特征表（按 device/window_id 取，聚合时求均值）
    FS={}
    for tag,_p,csv in (SRC,TGT):
        d=pd.read_csv(csv,low_memory=False)
        d=d[d.device.isin(devs)]
        cols=P.feature_columns(d)
        M=np.nan_to_num(d[cols].to_numpy(dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
        pos={(dv,int(w)):i for i,(dv,w) in
             enumerate(zip(d["device"].to_numpy(), d["window_id"].to_numpy()))}
        FS[tag]=(M,pos,cols)

    def build(tag, agg):
        by=defaultdict(list)
        for (dv,w),c in W[tag].items():
            if dv in devs: by[dv].append((w,c))
        Xb=[]; Xo=[]; y=[]
        M,pos,cols=FS[tag]
        for dv,items in by.items():
            items.sort()
            for s in range(0,len(items)-agg+1,agg):
                grp=items[s:s+agg]
                acc=Counter()
                for _w,c in grp: acc.update(c)
                tot=sum(acc.values())
                if tot==0: continue
                v=np.zeros(len(keys),dtype=np.float32)
                for k,n in acc.items():
                    if k in kidx: v[kidx[k]]=n
                Xb.append(np.sqrt(v/tot))
                ridx=[pos[(dv,w)] for w,_ in grp if (dv,w) in pos]
                if not ridx: Xb.pop(); continue
                Xo.append(M[ridx].mean(axis=0))
                y.append(TYPE(dv))
        return np.asarray(Xb), np.asarray(Xo), np.asarray(y)

    rows=[]
    for agg in AGGS:
        Xbs,Xos,ys=build(SRC[0],agg); Xbt,Xot,yt=build(TGT[0],agg)
        cls=sorted(set(ys)&set(yt))
        ms=np.isin(ys,cls); mt=np.isin(yt,cls)
        Xbs,Xos,ys=Xbs[ms],Xos[ms],ys[ms]; Xbt,Xot,yt=Xbt[mt],Xot[mt],yt[mt]
        print(f"=== agg={agg} ({agg*10}s 单元)  {len(cls)} 类  "
              f"训练 {len(ys)} 测试 {len(yt)} ===",flush=True)
        best=None
        for k in KS:
            kn=KNeighborsClassifier(n_neighbors=k,metric="euclidean",n_jobs=NJ)
            kn.fit(Xbs,ys); p=kn.predict(Xbt)
            f=f1_score(yt,p,average="macro",labels=cls)
            rows.append({"agg":agg,"unit_s":agg*10,"arm":"byteiot","k":k,"macro":f})
            print(f"  byteiot k={k}  macro={f:.4f}",flush=True)
            if best is None or f>best[0]: best=(f,p)
        rf=RandomForestClassifier(n_estimators=300,random_state=SEED,
                                  class_weight="balanced",n_jobs=NJ)
        rf.fit(Xos,ys); po=rf.predict(Xot)
        fo=f1_score(yt,po,average="macro",labels=cls)
        rows.append({"agg":agg,"unit_s":agg*10,"arm":"ours61","k":0,"macro":fo})
        print(f"  ours61(RF)   macro={fo:.4f}",flush=True)

        for nm,pp in [("byteiot",best[1]),("ours61",po)]:
            C=confusion_matrix(yt,pp,labels=cls); off=C.copy(); np.fill_diagonal(off,0)
            tot=off.sum()
            if tot==0: print(f"    {nm}: 零错误",flush=True); continue
            pr={}
            for i in range(len(cls)):
                for j in range(i+1,len(cls)):
                    v=off[i,j]+off[j,i]
                    if v: pr[(i,j)]=v
            top=sorted(pr.items(),key=lambda x:-x[1])[:5]
            print(f"    {nm}: 错误 {tot}，前 5 对占 {sum(v for _,v in top)/tot:.1%}",flush=True)
            for (i,j),v in top:
                print(f"       {cls[i]:22s} {cls[j]:22s} {v:6d} ({v/tot:5.1%})",flush=True)
        print(flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/granularity_scan.csv",index=False)
    print("=== 汇总（每粒度取最好）===",flush=True)
    B=R.loc[R.groupby(["agg","arm"]).macro.idxmax()]
    print(B.pivot_table(index=["agg","unit_s"],columns="arm",values="macro").round(4).to_string(),flush=True)
    print("\n判读：ESP 三对在 300s 聚合后若消失 → 困难簇是短窗产物，框架要重做；",flush=True)
    print("      若仍占主要错误   → 困难簇与粒度无关，我们的问题成立。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
