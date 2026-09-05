"""【探索性,非协议】补闸门盲点：可判决区判据从 top-2 放宽。

**发现的盲点**（2026-09-05）：闸门只在"该对争 top-2"的窗上工作，而
`GlobeLamp|GosundPlug` 13771 个窗里只有 30 个（0.2%）进区 —— 模型不是犹豫，
是**确信地判错**，真类连 top-2 都进不去。于是这类对**根本不进入闸门视野**。
CIC 上方法零增益，至少一部分不是"没有配置能修"，而是"没看见"。

**三种区定义对拍**：
  R2（现行）  {top1,top2} = {i,j}     模型在两类间犹豫时才可见
  R3          {i,j} ⊆ top-3
  R1          top1 ∈ {i,j}            只要判成其中一个就考虑 ← 覆盖"确信地判错"

**R1 的安全性**：部署时只看 top-1，**无需标签**；且不引入新伤害 ——
那些窗基模型本来就判成 i 或 j，修复只在两者间翻转；真类是第三类的窗，修不修都错。
闸门逻辑不变：在位者与配置在**同一个区**上量，要求配置 > 在位者 + MARGIN。

**判据**：
  区变大且存在配置能打赢在位者 → 盲点是真的，补上之后 CIC 从"看不见"变成"看得见"
  区变大但没有配置能打赢       → 那几对确实修不动，但方法从"默默不动"变成
                                 "看过之后明确拒绝" —— 论文分量完全不同
"""
from __future__ import annotations
import sys, time, re
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import pilot_rf_loro as P, run_unsw_iid_reference as IID

EXP=REPO+"/results/feature_expansion_20260905"
SRC=("2021_11_02_Idle","/home/lmy/cic_probe/idle_1102.csv")
INNER=("2021_11_03_Active","/home/lmy/cic_probe/active_1103.csv")
NJ=12; SEED=42; MARGIN=0.02; MIN_CELL=40; TOPN=6

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

def MK(n, k=2):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,C=1.0))
    return RandomForestClassifier(n_estimators=200,random_state=SEED,
                                  class_weight="balanced",n_jobs=NJ)

def fit_w(m,name,X,y):
    w=compute_sample_weight("balanced",y)
    try:
        if name=="lr": m.fit(X,y,logisticregression__sample_weight=w)
        else:          m.fit(X,y,sample_weight=w)
    except Exception: m.fit(X,y)
    return m

def main():
    t0=time.time()
    LH=pd.read_csv(EXP+"/lenhist_cic_w10.csv")
    lc=[c for c in LH.columns if c.startswith("lenhist_")]
    D={}
    for day,csv in (SRC,INNER):
        d=pd.read_csv(csv,low_memory=False)
        d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
        D[day]=d
    base=[c for c in P.feature_columns(D[SRC[0]]) if not c.startswith("lenhist_")]
    cols=base+lc
    devs=sorted(set(IID.day_gate(D[SRC[0]],SRC[0])) & set(IID.day_gate(D[INNER[0]],INNER[0])))
    classes=sorted({TYPE(x) for x in devs}); le=LabelEncoder().fit(classes)
    print(f"设备 {len(devs)} → 类 {len(classes)}   base {len(base)} + lenhist {len(lc)}",flush=True)

    def XY(day):
        d=D[day]; d=d[d.device.isin(devs)]
        return (np.asarray(P.clean_x(d,cols),dtype=float),
                le.transform([TYPE(x) for x in d.device]))
    Xs,ys=XY(SRC[0]); Xt,yt=XY(INNER[0])
    print(f"训练 {len(ys)}  测试 {len(yt)}   {time.time()-t0:.0f}s",flush=True)

    rf=RandomForestClassifier(n_estimators=300,random_state=SEED,
                              class_weight="balanced",n_jobs=NJ)
    rf.fit(Xs,ys); Pm=rf.predict_proba(Xt)
    oo=np.argsort(-Pm,axis=1); t1,t2,t3=oo[:,0],oo[:,1],oo[:,2]
    print(f"基模型完成  {time.time()-t0:.0f}s",flush=True)

    err=Counter()
    for a,b in zip(yt,t1):
        if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
    hard=[p for p,_ in err.most_common(TOPN)]
    print(f"\n困难对（inner 错误量前 {TOPN}）：",flush=True)
    for i,j in hard: print(f"  {classes[i]} | {classes[j]}   错 {err[(i,j)]}",flush=True)

    fams={}
    for c in cols:
        fams.setdefault(c.split("_")[0],[]).append(c)
    MASKS=[("none",np.arange(len(cols)))]
    idx={c:n for n,c in enumerate(cols)}
    for f,v in fams.items():
        keep=np.array([idx[c] for c in cols if c not in set(v)])
        if len(keep)>=5: MASKS.append((f,keep))
    CANDS=["lr","rf"]
    print(f"\n掩码 {len(MASKS)}  模型 {CANDS}",flush=True)

    REG={"R2":lambda i,j:(((t1==i)&(t2==j))|((t1==j)&(t2==i))),
         "R3":lambda i,j:(np.isin(i,[t1,t2,t3])&np.isin(j,[t1,t2,t3])) if False else
             ((((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j)))),
         "R1":lambda i,j:((t1==i)|(t1==j))}

    rows=[]
    for i,j in hard:
        ms=np.isin(ys,[i,j])
        if len(np.unique(ys[ms]))<2: continue
        y1=(ys[ms]==j).astype(int)
        cache={}
        print(f"\n{'='*80}\n{classes[i]} | {classes[j]}",flush=True)
        for rn,fn in REG.items():
            sel=fn(i,j); Dm=sel&np.isin(yt,[i,j])
            n=int(Dm.sum())
            if n<MIN_CELL:
                print(f"  {rn}: 区太小 n={n}",flush=True)
                rows.append({"pair":f"{classes[i]}|{classes[j]}","region":rn,"n":n,
                             "inc":np.nan,"best":np.nan,"mask":"","model":"","gated":False})
                continue
            inc=float((t1[Dm]==yt[Dm]).mean())
            best=(None,None,-1.0)
            for mn,keep in MASKS:
                for nm in CANDS:
                    key=(mn,nm)
                    if key not in cache:
                        try:
                            cache[key]=fit_w(MK(nm),nm,Xs[ms][:,keep],y1
                                             ).predict_proba(Xt[:,keep])[:,1]
                        except Exception: cache[key]=None
                    q=cache[key]
                    if q is None: continue
                    acc=float((np.where(q[Dm]>=0.5,j,i)==yt[Dm]).mean())
                    if acc>best[2]: best=(mn,nm,acc)
            g=best[2]>inc+MARGIN
            rows.append({"pair":f"{classes[i]}|{classes[j]}","region":rn,"n":n,
                         "inc":inc,"best":best[2],"mask":best[0],"model":best[1],"gated":bool(g)})
            print(f"  {rn}: 区 n={n:6d}  在位者={inc:.4f}  最好配置={best[2]:.4f} "
                  f"({best[1]}/掩{best[0]})  {'**过闸**' if g else '不过闸'}",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/region_fix.csv",index=False)
    print(f"\n{'='*80}\n=== 汇总：区大小 ===",flush=True)
    print(R.pivot_table(index="pair",columns="region",values="n").to_string(),flush=True)
    print("\n=== 在位者准确率 ===",flush=True)
    print(R.pivot_table(index="pair",columns="region",values="inc").round(4).to_string(),flush=True)
    print("\n=== 最好配置 ===",flush=True)
    print(R.pivot_table(index="pair",columns="region",values="best").round(4).to_string(),flush=True)
    print("\n=== 过闸 ===",flush=True)
    print(R.pivot_table(index="pair",columns="region",values="gated",aggfunc="first").to_string(),flush=True)
    print(f"\n各区定义下的过闸对数：",flush=True)
    print(R.groupby("region").gated.sum().to_string(),flush=True)
    print("\n判读：区变大且有配置能打赢 → 盲点补上后 CIC 从'看不见'变'看得见'；",flush=True)
    print("      区变大但打不赢 → 那几对确实修不动，但方法从'默默不动'变'明确拒绝'。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
