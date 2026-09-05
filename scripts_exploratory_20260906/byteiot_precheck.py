"""【探索性,非协议】ByteIoT 预检：包长频率分布 + Hellinger + kNN，在我们的跨天划分上。

来由：ByteIoT（IEEE TNSM 2021）在 UNSW 报 99.08%，但那几乎确定是同期划分。
我们的 lenhist 近似（top-32 判别性长度 + 树模型）在 UNSW 跨天上 ≈0，
但那不是它的方法，不能替它下结论。本脚本按论文描述实现其核心，只改划分。

**这是预检不是正式复现**：正式复现必须"先在他们的划分上复现出 99.08%，再改划分"，
否则是稻草人。本脚本只回答一个量级问题：**跨天时它落在 0.85 还是 0.95？**

锚点：GeMID 系统对比 IoTDevID / Kitsune，其报出的最好跨站数是 DD **0.894**。
若本预检也落在 0.85–0.90，则"跨环境天花板约 0.89"这个判断自洽，威胁解除。

方法核心（按论文描述）：
  · 每个单元建【双向】包长频率分布（up / down 分开计数）
  · 距离 = Hellinger：对频率开平方后取欧氏距离 / sqrt(2)，故可直接用 kNN(euclidean)
  · 分类器 = kNN
单元粒度两种，因为 ByteIoT 按流/会话聚合而我们按 10s 窗，**粒度差异属于方法差异不是口径差异**：
  win    10s 窗（与我们的数直接可比）
  aggN   同设备连续 N 个窗聚合（逼近其会话粒度，N=6/30 即 1 分钟/5 分钟）
"""
from __future__ import annotations
import subprocess, io, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, "/home/lmy/iot-device-classification/code/scripts/core")
import extract_features_generic as EG

PCAP=Path("/home/lmy/iot-device-classification/dataset/unsw/pcap")
MACMAP=Path("/home/lmy/iot-device-classification/dataset/unsw/device_mac_map.csv")
SRC="16-09-23"; TGTS=["16-09-30","16-10-12"]
WIN=10.0; MIN_PKT=2; KS=[1,3,5]; AGGS=[1,6,30]
MIN_WINDOWS=50          # 与主线 day_gate 一致的设备门槛

def per_window(day, mac_map):
    p=PCAP/f"{day}.pcap"
    packets=EG.read_packets(p,set(mac_map),verbose=False)
    streams=EG.assign_device_streams(packets,mac_map)
    origin=float(streams["time_epoch"].min())
    wid=np.floor((streams["time_epoch"].to_numpy()-origin)/WIN).astype(np.int64)
    dev=streams["device"].to_numpy(); ln=streams["length"].to_numpy().astype(int)
    updn=streams["is_up"].to_numpy().astype(int) if "is_up" in streams else np.ones(len(ln),int)
    out={}
    order=np.lexsort((wid,dev))
    dev,wid,ln,updn=dev[order],wid[order],ln[order],updn[order]
    b=0
    for i in range(1,len(dev)+1):
        if i==len(dev) or dev[i]!=dev[b] or wid[i]!=wid[b]:
            if i-b>=MIN_PKT:
                out[(dev[b],int(wid[b]))]=Counter(zip(ln[b:i],updn[b:i]))
            b=i
    return out

def main():
    t0=time.time()
    mac_map=EG.load_mac_map(MACMAP)
    print(f"设备 {len(mac_map)} 台",flush=True)
    W={}
    for day in [SRC]+TGTS:
        W[day]=per_window(day,mac_map)
        print(f"  {day}: {len(W[day])} 窗   {time.time()-t0:.0f}s",flush=True)

    # 词表只由源天建立（不看目标天）
    vocab=Counter()
    for c in W[SRC].values(): vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]
    kidx={k:i for i,k in enumerate(keys)}
    print(f"\n源天词表（(长度,方向) 出现≥20 窗）: {len(keys)} 项",flush=True)

    def mat(day, agg):
        """返回 (X 频率矩阵已开平方, y 设备名)。agg=每 agg 个连续窗合并。"""
        by=defaultdict(list)
        for (dv,w),c in W[day].items(): by[dv].append((w,c))
        rows=[]; labs=[]
        for dv,items in by.items():
            items.sort()
            for s in range(0,len(items)-agg+1,agg):
                acc=Counter()
                for _w,c in items[s:s+agg]: acc.update(c)
                v=np.zeros(len(keys),dtype=np.float32); tot=0
                for k,n in acc.items():
                    tot+=n
                    if k in kidx: v[kidx[k]]=n
                if tot==0: continue
                rows.append(np.sqrt(v/tot))          # Hellinger：开平方后欧氏 = Hellinger
                labs.append(dv)
        return np.asarray(rows), np.asarray(labs)

    print("\n=== ByteIoT 核心（包长频率分布 + Hellinger + kNN），我们的跨天划分 ===",flush=True)
    res=[]
    for agg in AGGS:
        Xs,ys=mat(SRC,agg)
        keep=[d for d,n in Counter(ys).items() if n>=MIN_WINDOWS//agg or n>=5]
        m=np.isin(ys,keep); Xs,ys=Xs[m],ys[m]
        for tgt in TGTS:
            Xt,yt=mat(tgt,agg)
            devs=sorted(set(ys)&set(yt))
            ms=np.isin(ys,devs); mt=np.isin(yt,devs)
            A,ay=Xs[ms],ys[ms]; B,by_=Xt[mt],yt[mt]
            if len(A)==0 or len(B)==0: continue
            cls=sorted(devs)
            for k in KS:
                kn=KNeighborsClassifier(n_neighbors=k,metric="euclidean",n_jobs=12)
                kn.fit(A,ay); p=kn.predict(B)
                f=f1_score(by_,p,average="macro",labels=cls)
                res.append({"agg":agg,"unit_s":agg*10,"tgt":tgt,"k":k,"n_cls":len(cls),
                            "n_train":len(A),"n_test":len(B),"macro":f})
                print(f"  agg={agg:2d}({agg*10:3d}s单元)  {tgt}  k={k}  "
                      f"{len(cls)} 类  训练 {len(A):6d} 测试 {len(B):6d}  macro={f:.4f}",flush=True)
            # 错误集中度（取最好的 k）
            best=max([r for r in res if r["agg"]==agg and r["tgt"]==tgt],key=lambda r:r["macro"])
            kn=KNeighborsClassifier(n_neighbors=best["k"],metric="euclidean",n_jobs=12)
            kn.fit(A,ay); p=kn.predict(B)
            C=confusion_matrix(by_,p,labels=cls); off=C.copy(); np.fill_diagonal(off,0)
            tot=off.sum()
            if tot:
                pr={}
                for i in range(len(cls)):
                    for j in range(i+1,len(cls)):
                        v=off[i,j]+off[j,i]
                        if v: pr[(i,j)]=v
                top=sorted(pr.items(),key=lambda x:-x[1])[:5]
                print(f"     错误 {tot}，前 5 对占 {sum(v for _,v in top)/tot:.1%}",flush=True)
                for (i,j),v in top:
                    print(f"       {cls[i]:24s} {cls[j]:24s} {v:5d} ({v/tot:5.1%})",flush=True)

    R=pd.DataFrame(res); R.to_csv("/home/lmy/cic_probe/byteiot_precheck.csv",index=False)
    print("\n=== 汇总（每种粒度取最好的 k）===",flush=True)
    B=R.loc[R.groupby(["agg","tgt"]).macro.idxmax()]
    print(B[["agg","unit_s","tgt","k","n_cls","macro"]].to_string(index=False),flush=True)
    print("\n对照：我们的 base+lenhist 跨天 best_base  16-09-30 0.8463 / 16-10-12 0.8548",flush=True)
    print("      GeMID 报出的最好跨站数 DD = 0.894",flush=True)
    print("      ByteIoT 原文 UNSW 99.08%（同期划分）",flush=True)
    print("\n判读：落在 0.85–0.90 → 跨环境天花板约 0.89 自洽，威胁解除；",flush=True)
    print("      落在 0.95+     → 我们的空间被挤压，需重新定位。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
