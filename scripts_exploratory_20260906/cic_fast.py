"""【探索性,非协议】CIC 强基线快速读数 —— 只回答"包长分布能不能吃动那 0.073"。

strong_baseline.py 的完整版在跑，但 11 万行 × 29 类 × 500 树 × 单线程要几小时。
本脚本同数据、同超参，只缩三处以拿方向性答案：
  · 1 个 seed（42），不是 3 个
  · 只跑 outer 那一天（1108_Active），不跑 1103
  · n_jobs 放开（RF/XGB 固定 random_state 下多线程结果一致，不改数值）

判据（对应用户的"根源不可区分 vs 可优化"）：
  base → base+lenhist 若吃掉大部分残余      → 0.073 是"我们没建那一族"，可优化
  若几乎不动                                → 该缺口在【已枚举的候选空间内】无解，
                                             且已有独立证据（同固件族/同端点/同心跳/
                                             无 TCP 时间戳）→ 候选"根源不可区分"
并报逐对错误集中度：残余是否仍集中在 ESP 那几对上。
"""
from __future__ import annotations
import os, sys, time, re
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import pilot_rf_loro as P, run_unsw_iid_reference as IID

EXP=REPO+"/results/feature_expansion_20260905"
NJ=12; SEED=42
SRC="2021_11_02_Idle"; TGT="2021_11_08_Active"
FILES={SRC:"/home/lmy/cic_probe/idle_1102.csv", TGT:"/home/lmy/cic_probe/active_1108.csv"}

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

def models(k):
    out=[("rf", RandomForestClassifier(n_estimators=500, random_state=SEED,
                                       class_weight="balanced", n_jobs=NJ))]
    try:
        from xgboost import XGBClassifier
        out.append(("xgboost", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", num_class=k,
            eval_metric="mlogloss", random_state=SEED, n_jobs=NJ)))
    except Exception: pass
    return out

def main():
    t0=time.time()
    LH=pd.read_csv(EXP+"/lenhist_cic_w10.csv")
    lcols=[c for c in LH.columns if c.startswith("lenhist_")]
    D={}
    for day,f in FILES.items():
        d=pd.read_csv(f,low_memory=False)
        d=d.merge(LH[LH.day==day][["device","window_id"]+lcols],
                  on=["device","window_id"], how="left")
        D[day]=d
        print(f"  {day}: {len(d)} 行",flush=True)
    base=[c for c in P.feature_columns(D[SRC]) if not c.startswith("lenhist_")]
    print(f"base {len(base)} 列   lenhist {len(lcols)} 列   n_jobs={NJ}",flush=True)

    devs=sorted(set(IID.day_gate(D[SRC],SRC)) & set(IID.day_gate(D[TGT],TGT)))
    classes=sorted({TYPE(x) for x in devs}); le=LabelEncoder().fit(classes)
    print(f"设备 {len(devs)} → 类 {len(classes)}   "
          f"合并 { {k:v for k,v in Counter(TYPE(x) for x in devs).items() if v>1} }",flush=True)

    def XY(day):
        d=D[day]; d=d[d.device.isin(devs)]
        return d, le.transform([TYPE(x) for x in d.device])

    res={}
    for pname,cols in [("base",base),("base+lenhist",base+lcols)]:
        ds,ys=XY(SRC); Xs=np.asarray(P.clean_x(ds,cols),dtype=float)
        dt,yt=XY(TGT); Xt=np.asarray(P.clean_x(dt,cols),dtype=float)
        best=(-1,None,None)
        for mn,m in models(len(classes)):
            m.fit(Xs,ys); p=m.predict(Xt)
            f=f1_score(yt,p,average="macro",labels=np.arange(len(classes)))
            print(f"  {pname:14s} {mn:9s} macro={f:.4f}   {time.time()-t0:.0f}s",flush=True)
            if f>best[0]: best=(f,mn,p)
        res[pname]=best
        f,mn,p=best
        C=confusion_matrix(yt,p,labels=np.arange(len(classes)))
        off=C.copy(); np.fill_diagonal(off,0); tot=off.sum()
        pairs={}
        for i in range(len(classes)):
            for j in range(i+1,len(classes)):
                v=off[i,j]+off[j,i]
                if v>0: pairs[(i,j)]=v
        top=sorted(pairs.items(),key=lambda x:-x[1])[:8]
        print(f"\n  === {pname} best={mn} macro={f:.4f}  错误 {tot} 个，"
              f"前 8 对占 {sum(v for _,v in top)/max(tot,1):.1%} ===",flush=True)
        for (i,j),v in top:
            print(f"     {classes[i]:26s} {classes[j]:26s} {v:6d}  ({v/tot:5.1%})",flush=True)
        print(flush=True)

    a,b=res["base"][0],res["base+lenhist"][0]
    print(f"=== 判读 ===",flush=True)
    print(f"  base = {a:.4f}   base+lenhist = {b:.4f}   Δ = {b-a:+.4f}",flush=True)
    print("  大幅上升 → 0.073 是「我们没建那一族」，属可优化",flush=True)
    print("  几乎不动 → 在已枚举候选空间内无解，配合固件族/端点/心跳/无TCP时间戳"
          "四项独立证据，可作为「根源不可区分」的候选",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
