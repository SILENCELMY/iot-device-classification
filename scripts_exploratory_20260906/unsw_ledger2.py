"""【探索性,非协议】UNSW 全类对账本 v2 —— 修 v1 的分类规则错误。

**v1 的错**：用 `在位者`（真类∈{i,j} 的窗上 top-1 准确率）判"解没解"。
该量含【判到第三类】的泄漏，于是被 {i,j} 里较差那个类的**边际召回**支配，
与这一对本身无关。结果 8 个"可分但未解"全部含 `BelkinWemoSwitch`，
而它们的错误量是 0–8 个窗、对内准确率 0.95–1.00 —— 全是幻影。

这是今天第四次犯同一类错（`pair_auc` 用 LR 量树的部署、`hardpair_eval` 用 8 类准确率
冒充对内准确率、`region_fix` 用 kNN 诊断 RF 的区）：**量的对象与问的问题不一致**。

**v2 的判据**：
  解没解   →  该对【自己的】错误量 err_share，以及【对内准确率】
  可不可分 →  【专用上界】，且与对内在【同样的平滑口径】下比

分类：
  已解决        err_share < 1%  且  对内 ≥ 0.95
  ★可分但未解    专用上界 ≥ 0.90  且（err_share ≥ 1% 或 对内 < 0.95）   ← 靶
  部分可分      0.60 ≤ 专用上界 < 0.90
  不可分        专用上界 < 0.60

另并报每个类的**边际召回**，把 v1 的幻影来源摆在明面上。
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
from collections import Counter
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import pilot_rf_loro as P, run_unsw_iid_reference as IID

FULL=REPO+"/results/unsw_features_full/features_day_%s.csv"
EXP =REPO+"/results/feature_expansion_20260905/lenhist_unsw_w10.csv"
DAYS={"src":"16-09-23","inner":"16-09-30","outer":"16-10-12"}
NJ=12; SEED=42; KBASE=[1,3,5,10,20]; OUT="/home/lmy/cic_probe/unsw_ledger2.csv"

def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]
        C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def cmean1(v,g,k):
    """一维（二分类正类概率）的因果滑动平均，与多类同口径。"""
    if k<=1: return v
    o=np.empty_like(v)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=v[i]
        C=np.concatenate([[0.0],np.cumsum(V)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def main():
    t0=time.time()
    LH=pd.read_csv(EXP); lc=[c for c in LH.columns if c.startswith("lenhist_")]
    D={}
    for tag,day in DAYS.items():
        d=pd.read_csv(FULL%day,low_memory=False)
        d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
        D[tag]=d.sort_values(["device","window_id"]).reset_index(drop=True)
    base=[c for c in P.feature_columns(D["src"]) if not c.startswith("lenhist_")]
    cols=base+lc
    devs=sorted(set.intersection(*[set(IID.day_gate(D[t],DAYS[t])) for t in DAYS]))
    le=LabelEncoder().fit(devs)

    def XYG(tag):
        d=D[tag]; d=d[d.device.isin(devs)]
        return (np.asarray(P.clean_x(d,cols),dtype=float),
                le.transform(d.device), d["device"].to_numpy())
    Xs,ys,_=XYG("src"); Xi,yi,gi=XYG("inner"); Xo,yo,go=XYG("outer")

    rf=RandomForestClassifier(n_estimators=300,random_state=SEED,
                              class_weight="balanced",n_jobs=NJ)
    rf.fit(Xs,ys)
    Pi=rf.predict_proba(Xi); Po=rf.predict_proba(Xo)
    L=np.arange(len(devs))
    kb=max(((k,f1_score(yi,cmeanM(Pi,gi,k).argmax(1),average="macro",labels=L))
            for k in KBASE),key=lambda x:x[1])[0]
    Po_s=cmeanM(Po,go,kb); top1=Po_s.argmax(1)
    print(f"共同设备 {len(devs)} 类  C(10,2)=45 对   {len(cols)} 列   平滑 kb={kb}",flush=True)
    print(f"outer macro: 无平滑={f1_score(yo,Po.argmax(1),average='macro',labels=L):.4f}  "
          f"平滑={f1_score(yo,top1,average='macro',labels=L):.4f}",flush=True)

    rc=recall_score(yo,top1,average=None,labels=L)
    print(f"\n=== 每类边际召回（v1 幻影的来源：含它的对，'在位者'必然低）===",flush=True)
    for n,d_ in enumerate(devs):
        flag="  ← 最低" if rc[n]==rc.min() else ""
        print(f"  {d_:24s} 召回 {rc[n]:.4f}   窗 {int((yo==n).sum()):6d}{flag}",flush=True)

    err=Counter()
    for a,b in zip(yo,top1):
        if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
    TOT=int((yo!=top1).sum())
    print(f"\nouter 总错 {TOT} / {len(yo)} = {TOT/len(yo):.4f}",flush=True)

    rows=[]
    for i,j in combinations(range(len(devs)),2):
        pm=np.isin(yo,[i,j]); n=int(pm.sum())
        e=err[(i,j)]; share=e/TOT if TOT else 0.0
        inc=float((top1[pm]==yo[pm]).mean())
        sub=Po_s[pm][:,[i,j]]
        internal=float((np.where(sub[:,1]>sub[:,0],j,i)==yo[pm]).mean())
        ms=np.isin(ys,[i,j])
        b=RandomForestClassifier(n_estimators=200,random_state=SEED,
                                 class_weight="balanced",n_jobs=NJ)
        b.fit(Xs[ms],(ys[ms]==j).astype(int))
        q_raw=b.predict_proba(Xo)[:,1]
        spec_raw=float((np.where(q_raw[pm]>=0.5,j,i)==yo[pm]).mean())
        q_s=cmean1(q_raw,go,kb)                      # 与多类同口径的平滑
        spec=float((np.where(q_s[pm]>=0.5,j,i)==yo[pm]).mean())
        if   share<0.01 and internal>=0.95:  cls="已解决"
        elif spec>=0.90:                     cls="★可分但未解"
        elif spec>=0.60:                     cls="部分可分"
        else:                                cls="不可分"
        rows.append(dict(a=devs[i],b=devs[j],n=n,err=e,share=share,inc=inc,
                         internal=internal,spec=spec,spec_raw=spec_raw,cls=cls))
    R=pd.DataFrame(rows); R.to_csv(OUT,index=False)

    print(f"\n{'='*120}\n=== 按错误量排序（全 45 对）===",flush=True)
    print(f"{'类对':52s} {'窗n':>6s} {'错':>5s} {'错占比':>7s} {'在位者':>7s} {'对内':>7s} "
          f"{'上界平滑':>8s} {'上界原始':>8s}  分类",flush=True)
    for _,r in R.sort_values("err",ascending=False).iterrows():
        print(f"{r.a+'|'+r.b:52s} {r.n:6d} {r.err:5d} {r.share:7.4f} {r.inc:7.4f} "
              f"{r.internal:7.4f} {r.spec:8.4f} {r.spec_raw:8.4f}  {r.cls}",flush=True)

    print(f"\n{'='*120}\n=== 分类统计 ===",flush=True)
    print(R.groupby("cls").agg(对数=("cls","size"),错误量=("err","sum"),
                               错占比=("share","sum")).to_string(),flush=True)

    tgt=R[R.cls=="★可分但未解"].sort_values("err",ascending=False)
    print(f"\n=== ★ 靶（上界≥0.90 而【自己的】错误量≥1% 或 对内<0.95）：{len(tgt)} 对 ===",flush=True)
    if len(tgt)==0:
        print("  【空】",flush=True)
        print("  停止判据在 UNSW（10s 窗、130 列池）上满足：",flush=True)
        print("  上界高的对全部已解；未解的那对上界也低 —— 没有漏掉的可分未解对。",flush=True)
    else:
        for _,r in tgt.iterrows():
            print(f"  {r.a}|{r.b}  窗 {r.n}  错 {r.err}({r.share:.1%})  "
                  f"对内 {r.internal:.4f} → 上界 {r.spec:.4f}",flush=True)

    print(f"\n=== 未解的对（对内 < 0.95，按错误量）===",flush=True)
    U=R[R.internal<0.95].sort_values("err",ascending=False)
    if len(U)==0: print("  【无】",flush=True)
    for _,r in U.iterrows():
        print(f"  {r.a}|{r.b}  错 {r.err}({r.share:.1%})  对内 {r.internal:.4f}  "
              f"上界 {r.spec:.4f}（原始 {r.spec_raw:.4f}）  → {r.cls}",flush=True)
    print(f"\n  未解对合计扛 {U.share.sum():.1%} 的全体错误；"
          f"已解 {int((R.internal>=0.95).sum())} 对合计 {R[R.internal>=0.95].share.sum():.1%}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
