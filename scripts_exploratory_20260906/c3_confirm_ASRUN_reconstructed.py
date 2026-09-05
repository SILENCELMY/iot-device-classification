"""【重建版本 —— 非当时保存的副本】

本文件是 C3 确证测试（登记 C3-CONFIRM-UNSW16）【实际运行时】的 c3_confirm.py 的
**重建**，不是当时保存的副本。

**为什么需要重建**：原脚本位于 `~/cic_probe/`，在仓库外、无版本控制。
结果落盘于 2026-09-06 00:02–00:24，脚本被就地修改到 01:54（审计修正 8/9/1）。
**产出确证结果的那一版因此不复存在。** 这是一次真实的可复现性失误，
由外部审阅（fable，2026-09-06）指出。

**重建方式**：三处改动均由明确的字符串补丁做出，逐字可逆。本文件由
`c3_confirm.py`（01:54 版）逆转以下三处得到：
    缺陷 8  RF 恢复 class_weight="balanced"（与 fit_w 的 sample_weight 相乘）
    缺陷 9  恢复 `except Exception: m.fit(X,y)` 静默退回
    缺陷 1  恢复按 dict 顺序边写边量的施加
并删除当时尚未接线的 `cmean1`。

**可信度**：provenance 清楚，但**未经运行验证与当时的输出逐位一致**。
引用 C3 结果时应同时引用 `results/c3_confirmatory_20260905/` 下的原始输出与日志。

**教训已入规**：见 `docs/SPEC_MECHANICAL_RULES.md` —— 承重脚本必须在仓库内、
必须在实验前提交、协议必须钉脚本哈希。
"""
"""C3 确证测试 —— UNSW 16 天保留集，一次性。

协议：`docs/PROTOCOL_C3_CONFIRMATORY_20260905.md`（预登记）
登记：`C3-CONFIRM-UNSW16`，commit 59372d1
保留集在本脚本之前 **0 次读取**。

两臂，各跑在自己有证据的那台仪器上（协议 §4）：

  臂 A（诊断，判 C3a）  因果计数聚合 + Hellinger + kNN(5)
                        算法逐行复用 `causal_agg2.py` 的 per_window / causal，
                        只把"一次装全部天"改成"逐天装、算完即释放"（内存 + 得到逐天曲线）
  臂 B（管线，判 C3b/C3c） RF(300) on 130 列 + 因果概率平滑 + 逐对闸门（区 R3∩R1，整区记账）

判据（事前，不得改）：
  C3a  曲线单调升至 ≥0.99，且 k=1 与 k=10 的差 ≥0.15
  C3b  第二段过闸 ≤1 对，端到端 |Δ| < 0.005          ← 预测【无效】
  C3c  Δ(第二段) ≥ −0.002

附带诊断（不参与判定，协议 §4 未冻结报告内容）：
  * k 窗缓冲的**实际时间跨度**中位数 —— `causal` 按"最近 k 个【已观测】窗"聚合，
    而少于 2 包的窗被丢掉，所以 "k=10 → 100s" 是下界，不是等式。必须量出来。
  * 逐天 macro（保留天与训练天间隔 1–17 天），次要终点。
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/core")
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import extract_features_generic as EG
import pilot_rf_loro as P, run_unsw_iid_reference as IID

OUT=Path(REPO+"/results/c3_confirmatory_20260905"); OUT.mkdir(exist_ok=True)
FULL=REPO+"/results/unsw_features_full/features_day_%s.csv"
LH_MAIN=REPO+"/results/feature_expansion_20260905/lenhist_unsw_w10.csv"
RES_F=REPO+"/results/unsw_reserve_16day/features_unsw_w10_reserve16.csv"
RES_L=REPO+"/results/unsw_reserve_16day/lenhist_unsw_w10_reserve16.csv"
PCAP=REPO+"/dataset/unsw/pcap/%s.pcap"
MACMAP=Path(REPO+"/dataset/unsw/device_mac_map.csv")

TRAIN="16-09-23"; TUNE="16-09-30"
RESERVE=["16-09-24","16-09-25","16-09-26","16-09-27","16-09-28","16-09-29",
         "16-10-01","16-10-02","16-10-03","16-10-04","16-10-05","16-10-06",
         "16-10-07","16-10-08","16-10-09","16-10-10"]
WIN=10.0; MIN_PKT=2; KNN_K=5; AGGS=[1,3,6,10,30]
NJ=12; SEEDS=[42,43]; KBASE=[1,3,5,10,20]; MARGIN=0.02; MIN_CELL=40; TOPN=8
REGIONS=["R2","R3nR1","R3"]   # 协议 §4 修正 2：区进入搜索空间，逐对选
BLEND_W=0.5   # 协议 §4 修正 3：融合而非硬替换。等权，【不调参】——
              # 单天调 w 会系统性选出 w=1.0（硬替换），恰是施加天最差的那个

# ============================ 共用：设备门槛 ============================
def gate_reserve():
    d=pd.read_csv(RES_F,low_memory=False,encoding="utf-8-sig",
              usecols=["device","label","day","window_id"])
    return {day:set(IID.day_gate(d[d.day==day],day)) for day in RESERVE}

def gate_day(day):
    d=pd.read_csv(FULL%day,low_memory=False,encoding="utf-8-sig",
              usecols=["device","label","day","window_id"])
    return set(IID.day_gate(d,day))

# ============================ 臂 A ============================
def per_window(pcap, mac_map):
    """逐行复用 causal_agg2.per_window，额外返回每窗的时间戳中位数用于量真实跨度。"""
    pk=EG.read_packets(Path(pcap),set(mac_map),verbose=False)
    st=EG.assign_device_streams(pk,mac_map)
    origin=float(st["time_epoch"].min())
    t=st["time_epoch"].to_numpy()
    wid=np.floor((t-origin)/WIN).astype(np.int64)
    dev=st["device"].to_numpy(); ln=st["length"].to_numpy().astype(int)
    up=st["is_up"].to_numpy().astype(int)
    out={}; o=np.lexsort((wid,dev)); dev,wid,ln,up,t=dev[o],wid[o],ln[o],up[o],t[o]
    b=0
    for i in range(1,len(dev)+1):
        if i==len(dev) or dev[i]!=dev[b] or wid[i]!=wid[b]:
            if i-b>=MIN_PKT:
                out[(dev[b],int(wid[b]))]=(Counter(zip(ln[b:i],up[b:i])), float(t[b]))
            b=i
    return out

def causal(W, devs, k, keys, kidx):
    """逐行复用 causal_agg2.causal 的聚合逻辑；额外返回缓冲的真实时间跨度。"""
    by=defaultdict(list)
    for (dv,w),(c,t0) in W.items():
        if dv in devs: by[dv].append((w,c,t0))
    X=[];y=[];span=[]
    for dv,items in by.items():
        items.sort(key=lambda z:z[0])
        cs=[c for _w,c,_t in items]; ts=[t for _w,_c,t in items]
        run_=Counter()
        for n in range(len(cs)):
            run_.update(cs[n])
            if n>=k: run_.subtract(cs[n-k]); run_+=Counter()
            tot=sum(run_.values())
            if tot==0: continue
            v=np.zeros(len(keys),dtype=np.float32)
            for kk,cnt in run_.items():
                if kk in kidx and cnt>0: v[kidx[kk]]=cnt
            X.append(np.sqrt(v/tot)); y.append(dv)
            lo=max(0,n-k+1); span.append(ts[n]-ts[lo]+WIN)
    return np.asarray(X), np.asarray(y), np.asarray(span)

def arm_a(devs, res_gates):
    t0=time.time()
    print(f"\n{'#'*100}\n### 臂 A —— 诊断仪器（因果计数聚合 + Hellinger + kNN{KNN_K}）\n{'#'*100}",flush=True)
    mac_map=EG.load_mac_map(MACMAP)
    Ws=per_window(PCAP%TRAIN, mac_map)
    print(f"训练天 {TRAIN}: {len(Ws)} 窗   {time.time()-t0:.0f}s",flush=True)

    vocab=Counter()
    for (dv,_w),(c,_t) in Ws.items():
        if dv in devs: vocab.update(c.keys())
    keys=[k for k,n in vocab.items() if n>=20]; kidx={k:i for i,k in enumerate(keys)}
    print(f"源天词表 {len(keys)} 项（协议要求：只从 {TRAIN} 导出）",flush=True)

    models={}
    for k in AGGS:
        Xs,ys,_=causal(Ws,devs,k,keys,kidx)
        kn=KNeighborsClassifier(n_neighbors=KNN_K,metric="euclidean",n_jobs=NJ)
        kn.fit(Xs,ys); models[k]=kn
        print(f"  k={k:3d} 训练集 {Xs.shape}   {time.time()-t0:.0f}s",flush=True)
    del Ws

    acc={k:{"y":[],"p":[]} for k in AGGS}; spans={k:[] for k in AGGS}; per_day=[]
    for day in RESERVE:
        Wd=per_window(PCAP%day, mac_map)
        g=res_gates[day] & devs
        row={"day":day,"n_gate":len(g)}
        for k in AGGS:
            Xt,yt,sp=causal(Wd,g,k,keys,kidx)
            if len(Xt)==0: row[f"k{k}"]=np.nan; continue
            p=models[k].predict(Xt)
            row[f"k{k}"]=f1_score(yt,p,average="macro",labels=sorted(devs))
            row[f"n{k}"]=len(yt)
            acc[k]["y"].append(yt); acc[k]["p"].append(p); spans[k].append(sp)
        per_day.append(row)
        print(f"  {day}: 门 {len(g)} 台  "
              +"  ".join(f"k{k}={row.get(f'k{k}',float('nan')):.4f}" for k in AGGS)
              +f"   {time.time()-t0:.0f}s",flush=True)
        del Wd

    print(f"\n=== 臂 A 主判据：合并 16 天 ===",flush=True)
    rows=[]
    for k in AGGS:
        y=np.concatenate(acc[k]["y"]); p=np.concatenate(acc[k]["p"])
        sp=np.concatenate(spans[k])
        f=f1_score(y,p,average="macro",labels=sorted(devs))
        err=int((y!=p).sum())
        C=confusion_matrix(y,p,labels=sorted(devs)); off=C.copy(); np.fill_diagonal(off,0)
        cl=sorted(devs); pr={}
        for i in range(len(cl)):
            for j in range(i+1,len(cl)):
                v=off[i,j]+off[j,i]
                if v: pr[(cl[i],cl[j])]=int(v)
        top=sorted(pr.items(),key=lambda x:-x[1])[:3]
        rows.append(dict(k=k,evid_nominal=k*10,macro=f,n_err=err,n_dec=len(y),
                         span_med=float(np.median(sp)),span_p90=float(np.percentile(sp,90))))
        print(f"  k={k:3d}（标称 {k*10:4d}s；实际跨度中位 {np.median(sp):6.1f}s  "
              f"p90 {np.percentile(sp,90):7.1f}s）  macro={f:.4f}  错 {err}/{len(y)}",flush=True)
        for (a,b),v in top:
            print(f"        {a:24s} {b:24s} {v:6d} ({v/max(err,1):5.1%})",flush=True)
    A=pd.DataFrame(rows); A.to_csv(OUT/"arm_a_pooled.csv",index=False)
    pd.DataFrame(per_day).to_csv(OUT/"arm_a_per_day.csv",index=False)

    sat=A.macro.max()
    def _m(k):
        r=A[A.k==k]
        return float(r.macro.iloc[0]) if len(r) else np.nan
    lift=_m(10)-_m(1)
    mono=bool((A.sort_values("k").macro.diff().dropna()>=-0.002).all())
    print(f"\n--- C3a 判定 ---",flush=True)
    print(f"  饱和值 max={sat:.4f}   判据 ≥0.99   → {'通过' if sat>=0.99 else '不通过'}",flush=True)
    print(f"  k=1→k=10 提升 {lift:+.4f}   判据 ≥0.15  → "
          f"{'通过' if lift>=0.15 else ('缺 k 行，无法判' if np.isnan(lift) else '不通过')}",flush=True)
    print(f"  单调（容差 0.002）{mono}",flush=True)
    return dict(sat=sat,lift=lift,mono=mono,table=A.to_dict("records"))

# ============================ 臂 B ============================
def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]
        C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def blend(Ps,q,i,j,w=BLEND_W):
    """专用二分类器的 q=P(j) 与基模型【限制到{i,j}】的概率等权平均。

    硬替换（w=1）把基模型的信息整个扔掉。已烧天实测：Belkin 对在两个施加天上，
    在位者 0.6294/0.6322 高于任何单个专用二分类器的天花板 0.6266/0.6275，
    而 w=0.5 给 0.6807/0.6771 —— 两边的错不重叠，平均才都保住。
    """
    sub=Ps[:,[i,j]]
    pbase=sub[:,1]/np.clip(sub.sum(1),1e-12,None)
    return (1.0-w)*pbase+w*q

def MK(n,seed):
    if n=="lr": return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    return RandomForestClassifier(n_estimators=200,random_state=seed,
                                  class_weight="balanced",n_jobs=NJ)

def fit_w(m,name,X,y):
    w=compute_sample_weight("balanced",y)
    try:
        if name=="lr": m.fit(X,y,logisticregression__sample_weight=w)
        else:          m.fit(X,y,sample_weight=w)
    except Exception: m.fit(X,y)
    return m

def arm_b(devs, seed):
    t0=time.time()
    print(f"\n{'#'*100}\n### 臂 B —— 完整管线（RF300 + 因果概率平滑 + 逐对闸门（区逐对选∈{REGIONS}，整区记账））  seed={seed}\n{'#'*100}",flush=True)
    LH=pd.read_csv(LH_MAIN); lc=[c for c in LH.columns if c.startswith("lenhist_")]
    def load(day):
        d=pd.read_csv(FULL%day,low_memory=False)
        d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
        return d.sort_values(["device","window_id"]).reset_index(drop=True)
    Dtr=load(TRAIN); Dtu=load(TUNE)
    base=[c for c in P.feature_columns(Dtr) if not c.startswith("lenhist_")]
    cols=base+lc
    print(f"特征池 {len(base)}+{len(lc)}={len(cols)} 列   {time.time()-t0:.0f}s",flush=True)

    RL=pd.read_csv(RES_L); RL=RL[RL.day.isin(RESERVE)]
    Dre=pd.read_csv(RES_F,low_memory=False,encoding="utf-8-sig")
    Dre.columns=[c.lstrip("﻿") for c in Dre.columns]
    Dre=Dre.merge(RL[["device","day","window_id"]+lc],on=["device","day","window_id"],how="left")
    Dre=Dre[Dre.device.isin(devs)].sort_values(["day","device","window_id"]).reset_index(drop=True)
    print(f"保留集 {len(Dre)} 窗  lenhist 缺 {int(Dre[lc[0]].isna().sum())}   {time.time()-t0:.0f}s",flush=True)

    le=LabelEncoder().fit(sorted(devs))
    def XY(d, with_day=False):
        d=d[d.device.isin(devs)]
        X=np.asarray(P.clean_x(d,cols),dtype=np.float32)
        y=le.transform(d.device)
        g=(d.day.astype(str)+"|"+d.device.astype(str)).to_numpy() if with_day \
          else d.device.to_numpy()
        return X,y,g
    Xs,ys,_=XY(Dtr); Xu,yu,gu=XY(Dtu); Xo,yo,go=XY(Dre,with_day=True)
    day_o=Dre.day.to_numpy()
    del Dtr,Dre
    print(f"训练 {len(ys)}  选参 {len(yu)}  保留 {len(yo)}   平滑分组=(day,device)"
          f"   {time.time()-t0:.0f}s",flush=True)

    rf=RandomForestClassifier(n_estimators=300,random_state=seed,
                              class_weight="balanced",n_jobs=NJ)
    rf.fit(Xs,ys)
    Pu=rf.predict_proba(Xu); Po=rf.predict_proba(Xo)
    L=np.arange(len(devs))
    kb=max(((k,f1_score(yu,cmeanM(Pu,gu,k).argmax(1),average="macro",labels=L))
            for k in KBASE),key=lambda x:x[1])[0]
    Pu_s=cmeanM(Pu,gu,kb); Po_s=cmeanM(Po,go,kb)
    f_base=f1_score(yo,Po.argmax(1),average="macro",labels=L)
    f_sm  =f1_score(yo,Po_s.argmax(1),average="macro",labels=L)
    print(f"平滑窗 kb={kb}（在 {TUNE} 上选）",flush=True)
    print(f"保留集 macro：base={f_base:.4f}   +第一段(平滑)={f_sm:.4f}",flush=True)

    ui=np.argsort(-Pu_s,axis=1); u1,u2,u3=ui[:,0],ui[:,1],ui[:,2]
    oi=np.argsort(-Po_s,axis=1); o1,o2,o3=oi[:,0],oi[:,1],oi[:,2]
    err=Counter()
    for a,b in zip(yu,u1):
        if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
    hard=[p for p,_ in err.most_common(TOPN)]
    print(f"\n困难对（{TUNE} 错误量前 {TOPN}）：",flush=True)
    for i,j in hard:
        print(f"  {le.classes_[i]}|{le.classes_[j]}  错 {err[(i,j)]}",flush=True)

    fams={}
    for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
    idx={c:n for n,c in enumerate(cols)}
    MASKS=[("none",np.arange(len(cols)))]
    for f,v in fams.items():
        keep=np.array([idx[c] for c in cols if c not in set(v)])
        if len(keep)>=5: MASKS.append((f,keep))
    MD=dict(MASKS); CANDS=["lr","rf"]
    def region(rn,t1,t2,t3,i,j):
        """闸门的【作用集合】。记账必须在同一个集合上做 —— 协议 §4 修正 1。"""
        both=((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j))
        if rn=="R3":    return both
        if rn=="R3nR1": return both&((t1==i)|(t1==j))
        if rn=="R2":    return ((t1==i)&(t2==j))|((t1==j)&(t2==i))
        raise ValueError(rn)

    cfgs={}
    print(f"\n=== 第二段：在 {TUNE} 上导出配置（{len(MASKS)} 掩码 × {CANDS}，区逐对选，整区记账）===",flush=True)
    for i,j in hard:
        ms=np.isin(ys,[i,j])
        if len(np.unique(ys[ms]))<2: continue
        y1=(ys[ms]==j).astype(int); cache={}
        for mn,keep in MASKS:                        # 概率只算一次，三个区共用
            for nm in CANDS:
                key=(mn,nm)
                if key not in cache:
                    try: cache[key]=fit_w(MK(nm,seed),nm,Xs[ms][:,keep],y1
                                          ).predict_proba(Xu[:,keep])[:,1]
                    except Exception: cache[key]=None
        best=(None,None,None,-1.0,-1.0)              # (区, 掩码, 模型, 配置, 在位者)
        n_ok=0                                       # 够大的区数，区分两种不过闸
        for rn in REGIONS:
            sel=region(rn,u1,u2,u3,i,j)              # 作用集合
            if sel.sum()<MIN_CELL: continue
            n_ok+=1
            inc=float((u1[sel]==yu[sel]).mean())     # 【整区记账】：与作用集合同一个
            n3=int((sel&~np.isin(yu,[i,j])).sum())
            loc=(None,None,-1.0)
            for (mn,nm),q in cache.items():
                if q is None: continue
                mix=blend(Pu_s,q,i,j)
                new=u1.copy(); new[sel]=np.where(mix[sel]>=0.5,j,i)
                a=float((new[sel]==yu[sel]).mean())
                if a>loc[2]: loc=(mn,nm,a)
            print(f"    {rn:6s} 区 n={int(sel.sum()):6d} (第三类 {n3}, "
                  f"{n3/max(int(sel.sum()),1):5.1%})  {inc:.4f}→{loc[2]:.4f} "
                  f"({loc[1]}/掩{loc[0]})  {'过闸' if loc[2]>inc+MARGIN else '不过闸'}",flush=True)
            if loc[2]-inc>best[3]-best[4]: best=(rn,loc[0],loc[1],loc[2],inc)
        rn,mn,nm,a,inc=best
        if rn is None:
            why="三个区全太小" if n_ok==0 else f"{n_ok} 个区够大，但没有配置能改善"
            print(f"  {le.classes_[i]}|{le.classes_[j]}  不过闸（{why}）",flush=True); continue
        g=a>inc+MARGIN
        print(f"  {le.classes_[i]:22s}|{le.classes_[j]:22s} → 选 {rn} "
              f"{inc:.4f}→{a:.4f} ({nm}/掩{mn})  {'**过闸**' if g else '不过闸'}",flush=True)
        if g: cfgs[(i,j)]=(rn,mn,nm,a)
    print(f"\n过闸 {len(cfgs)} 对   {time.time()-t0:.0f}s",flush=True)

    pred=o1.copy(); detail=[]
    for (i,j),(rn,mn,nm,_a) in cfgs.items():
        keep=MD[mn]; ms=np.isin(ys,[i,j])
        q=fit_w(MK(nm,seed),nm,Xs[ms][:,keep],(ys[ms]==j).astype(int)
                ).predict_proba(Xo[:,keep])[:,1]
        sel=region(rn,o1,o2,o3,i,j)
        if not sel.any():
            detail.append((le.classes_[i],le.classes_[j],0,np.nan,np.nan,0,rn)); continue
        n3=int((sel&~np.isin(yo,[i,j])).sum())
        inc_o=float((o1[sel]==yo[sel]).mean())          # 整区
        mix=blend(Po_s,q,i,j)
        pred[sel]=np.where(mix[sel]>=0.5,j,i)
        acc_o=float((pred[sel]==yo[sel]).mean())        # 整区，同一集合
        detail.append((le.classes_[i],le.classes_[j],int(sel.sum()),inc_o,acc_o,n3,rn))
    f_fix=f1_score(yo,pred,average="macro",labels=L)
    for a,b,n,x,y_,n3,rn in detail:
        if n==0:
            print(f"    {a:22s}|{b:22s} [{rn}] 区 n=0  【配置无处施加】",flush=True); continue
        print(f"    {a:22s}|{b:22s} [{rn}] 区 n={n:6d} (第三类 {n3}, {n3/max(n,1):5.1%})  "
              f"整区在位者={x:.4f} → 修后={y_:.4f} ({y_-x:+.4f})",flush=True)

    delta=f_fix-f_sm
    print(f"\n=== 臂 B 保留集（合并 16 天，seed={seed}）===",flush=True)
    print(f"  base            {f_base:.4f}",flush=True)
    print(f"  +第一段(平滑)     {f_sm:.4f}   Δ={f_sm-f_base:+.4f}",flush=True)
    print(f"  +第二段(定点修)    {f_fix:.4f}   Δ={delta:+.4f}",flush=True)

    pd_rows=[]
    for d_ in RESERVE:
        m=day_o==d_
        if m.sum()==0: continue
        pd_rows.append(dict(day=d_,n=int(m.sum()),
            base=f1_score(yo[m],Po[m].argmax(1),average="macro",labels=L),
            smooth=f1_score(yo[m],Po_s[m].argmax(1),average="macro",labels=L),
            fixed=f1_score(yo[m],pred[m],average="macro",labels=L)))
    PD=pd.DataFrame(pd_rows); PD.to_csv(OUT/f"arm_b_per_day_seed{seed}.csv",index=False)
    print(f"\n  逐天（次要终点，不参与判定）：",flush=True)
    for _,r in PD.iterrows():
        print(f"    {r.day}  n={int(r.n):6d}  base={r.base:.4f}  平滑={r.smooth:.4f}  "
              f"修后={r.fixed:.4f}",flush=True)
    return dict(seed=seed,kb=kb,base=f_base,smooth=f_sm,fixed=f_fix,
                delta=delta,n_gated=len(cfgs),detail=detail,
                cfgs={f'{le.classes_[i]}|{le.classes_[j]}':list(v)
                      for (i,j),v in cfgs.items()})

# ============================ 主 ============================
def main():
    T=time.time()
    print(f"C3 确证测试   协议 docs/PROTOCOL_C3_CONFIRMATORY_20260905.md   "
          f"登记 C3-CONFIRM-UNSW16 @ 59372d1",flush=True)
    res_gates=gate_reserve()
    sets=[gate_day(TRAIN),gate_day(TUNE)]+[res_gates[d] for d in RESERVE]
    devs=set.intersection(*sets)
    print(f"\n类集 = 训练∩选参∩保留16天 的 day_gate 交集 = {len(devs)} 类",flush=True)
    for d_ in sorted(devs): print(f"    {d_}",flush=True)

    a=arm_a(devs,res_gates)
    bs=[arm_b(devs,s) for s in SEEDS]

    print(f"\n{'='*100}\n=== C3 判定 ===\n{'='*100}",flush=True)
    c3a = a["sat"]>=0.99 and (a["lift"]>=0.15 if not np.isnan(a["lift"]) else False)
    print(f"C3a  饱和 {a['sat']:.4f}（≥0.99）  k1→k10 {a['lift']:+.4f}（≥0.15）"
          f"  → {'通过' if c3a else '不通过'}",flush=True)
    for b in bs:
        sgn=[(a,bb,y_-x) for a,bb,n,x,y_,_n3,_r in b["detail"] if n>0]
        neg=[t for t in sgn if t[2]<0]
        c3b = b["delta"]>0 and len(neg)==0
        c3c = b["delta"]>=-0.002
        print(f"C3b′ seed{b['seed']}  过闸 {b['n_gated']} 对  Δ={b['delta']:+.4f}（>0）  "
              f"区上符号翻转 {len(neg)}/{len(sgn)} 对（须 0）  "
              f"→ {'通过' if c3b else '不通过'}",flush=True)
        for a,bb,d in neg:
            print(f"       ✗ {a}|{bb} 区上 Δ={d:+.4f}（选参天为正，留出天转负）",flush=True)
        print(f"C3c  seed{b['seed']}  Δ={b['delta']:+.4f}（≥−0.002）"
              f"  → {'通过' if c3c else '不通过'}",flush=True)
    if len(bs)>1:
        print(f"\n种子间差：平滑 {abs(bs[0]['smooth']-bs[1]['smooth']):.4f}   "
              f"修后 {abs(bs[0]['fixed']-bs[1]['fixed']):.4f}",flush=True)
    json.dump(dict(arm_a=a,arm_b=bs,devs=sorted(devs)),
              open(OUT/"verdict.json","w"),ensure_ascii=False,indent=2,default=float)
    print(f"\n写出 {OUT}   总耗时 {time.time()-T:.0f}s",flush=True)

if __name__=="__main__":
    main()
