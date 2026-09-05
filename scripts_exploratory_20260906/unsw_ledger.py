"""【探索性,非协议】UNSW 全类对账本 —— 停止判据的仪器。

**为什么必须做**：停止判据是「在三个数据集找到所有可区分的点，修复，并解释不可修复的点」。
我们只审计过错误量最大的 6–8 对。**不把 C(10,2)=45 对全部过一遍，就无法声称
"没有漏掉可分但未解的对"** —— 那正是判据里唯一允许开新实验的情形。

三个逐对度量必须分开量（今天已犯错两次：`pair_auc` 用 LR 量树的部署、
`hardpair_eval` 用 8 类准确率冒充对内准确率）：

  在位者      部署管线的 top-1 在 真类∈{i,j} 的窗上对不对   ← 含第三类泄漏
  对内准确率   概率【限制到 {i,j} 两列】后 argmax 对不对      ← 只问"表示分不分得开"
  专用上界     src 上只用 {i,j} 重训一个 RF，在 outer 上量    ← 当前表示空间内的可分性上界

分类（可分性用【专用上界】判，不用在位者判）：
  已解决        err_share < 1%
  可分但未解 ★  专用上界 ≥ 0.90 且 在位者 < 0.90         ← 靶
  部分可分      0.60 ≤ 专用上界 < 0.90
  不可分        专用上界 < 0.60（≈ 掷硬币）

部署模型一律 RF（与 outer 端到端同一个），不用 LR。
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
from collections import Counter
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import pilot_rf_loro as P, run_unsw_iid_reference as IID

FULL=REPO+"/results/unsw_features_full/features_day_%s.csv"
EXP =REPO+"/results/feature_expansion_20260905/lenhist_unsw_w10.csv"
DAYS={"src":"16-09-23","inner":"16-09-30","outer":"16-10-12"}
NJ=12; SEED=42; KBASE=[1,3,5,10,20]; OUT="/home/lmy/cic_probe/unsw_ledger.csv"

def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]
        C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
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
        print(f"{tag} {day}: {len(d)} 窗  lenhist 缺 {d[lc[0]].isna().sum()}",flush=True)
    base=[c for c in P.feature_columns(D["src"]) if not c.startswith("lenhist_")]
    cols=base+lc
    devs=sorted(set.intersection(*[set(IID.day_gate(D[t],DAYS[t])) for t in DAYS]))
    le=LabelEncoder().fit(devs)
    print(f"\n共同设备 {len(devs)} 类  →  C({len(devs)},2)={len(devs)*(len(devs)-1)//2} 对"
          f"   特征 {len(base)}+{len(lc)}={len(cols)} 列",flush=True)
    for d_ in devs: print(f"    {d_}",flush=True)

    def XYG(tag):
        d=D[tag]; d=d[d.device.isin(devs)]
        return (np.asarray(P.clean_x(d,cols),dtype=float),
                le.transform(d.device), d["device"].to_numpy())
    Xs,ys,_=XYG("src"); Xi,yi,gi=XYG("inner"); Xo,yo,go=XYG("outer")
    print(f"\n训练 {len(ys)}  inner {len(yi)}  outer {len(yo)}   {time.time()-t0:.0f}s",flush=True)

    rf=RandomForestClassifier(n_estimators=300,random_state=SEED,
                              class_weight="balanced",n_jobs=NJ)
    rf.fit(Xs,ys)
    Pi=rf.predict_proba(Xi); Po=rf.predict_proba(Xo)
    L=np.arange(len(devs))
    kb=max(((k,f1_score(yi,cmeanM(Pi,gi,k).argmax(1),average="macro",labels=L))
            for k in KBASE),key=lambda x:x[1])[0]
    Po_s=cmeanM(Po,go,kb)
    f_raw=f1_score(yo,Po.argmax(1),average="macro",labels=L)
    f_sm =f1_score(yo,Po_s.argmax(1),average="macro",labels=L)
    top1=Po_s.argmax(1)
    print(f"平滑窗 kb={kb}（inner 选）   outer macro: 无平滑={f_raw:.4f}  平滑={f_sm:.4f}"
          f"   {time.time()-t0:.0f}s",flush=True)

    err=Counter()
    for a,b in zip(yo,top1):
        if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
    TOT=int((yo!=top1).sum())
    print(f"outer 总错 {TOT} / {len(yo)} 窗 = {TOT/len(yo):.4f}\n",flush=True)

    rows=[]
    print(f"=== 逐对（{len(devs)*(len(devs)-1)//2} 对，专用 RF 各一个）===",flush=True)
    for i,j in combinations(range(len(devs)),2):
        pm=np.isin(yo,[i,j]); n=int(pm.sum())
        e=err[(i,j)]; share=e/TOT if TOT else 0.0
        if n<20:
            rows.append(dict(a=devs[i],b=devs[j],n=n,err=e,share=share,
                             inc=np.nan,internal=np.nan,spec=np.nan,cls="窗太少"))
            continue
        inc=float((top1[pm]==yo[pm]).mean())
        sub=Po_s[pm][:,[i,j]]
        internal=float((np.where(sub[:,1]>sub[:,0],j,i)==yo[pm]).mean())
        ms=np.isin(ys,[i,j])
        if len(np.unique(ys[ms]))<2:
            spec=np.nan
        else:
            b=RandomForestClassifier(n_estimators=200,random_state=SEED,
                                     class_weight="balanced",n_jobs=NJ)
            b.fit(Xs[ms],(ys[ms]==j).astype(int))
            q=b.predict_proba(Xo[pm])[:,1]
            spec=float((np.where(q>=0.5,j,i)==yo[pm]).mean())
        if   share<0.01 and inc>=0.90:            cls="已解决"
        elif spec>=0.90 and inc<0.90:             cls="★可分但未解"
        elif spec>=0.60:                          cls="部分可分"
        else:                                     cls="不可分"
        rows.append(dict(a=devs[i],b=devs[j],n=n,err=e,share=share,
                         inc=inc,internal=internal,spec=spec,cls=cls))
    R=pd.DataFrame(rows); R.to_csv(OUT,index=False)

    print(f"\n{'='*112}\n=== 按错误量排序（全 {len(R)} 对）===",flush=True)
    print(f"{'类对':52s} {'窗n':>7s} {'错':>6s} {'错占比':>7s} {'在位者':>7s} {'对内':>7s} {'专用上界':>8s}  分类",flush=True)
    for _,r in R.sort_values("err",ascending=False).iterrows():
        nm=f"{r.a}|{r.b}"
        f=lambda v: "   nan" if pd.isna(v) else f"{v:.4f}"
        print(f"{nm:52s} {r.n:7d} {r.err:6d} {r.share:7.4f} "
              f"{f(r.inc)} {f(r.internal)} {f(r.spec):>8s}  {r.cls}",flush=True)

    print(f"\n{'='*112}\n=== 分类统计 ===",flush=True)
    g=R.groupby("cls").agg(对数=("cls","size"),错误量=("err","sum"),错占比=("share","sum"))
    print(g.to_string(),flush=True)

    tgt=R[R.cls=="★可分但未解"].sort_values("err",ascending=False)
    print(f"\n=== ★ 靶（可分但未解，专用上界≥0.90 而在位者<0.90）：{len(tgt)} 对 ===",flush=True)
    if len(tgt)==0:
        print("  【空】—— 在当前 130 列表示空间内，没有漏掉的可分未解对。",flush=True)
        print("  停止判据在 UNSW 上满足：可分的都解了，未解的都不可分（或已在部分可分名单内）。",flush=True)
    else:
        for _,r in tgt.iterrows():
            print(f"  {r.a}|{r.b}   窗 {r.n}  错 {r.err}({r.share:.1%})  "
                  f"在位者 {r.inc:.4f} → 专用上界 {r.spec:.4f}  可回收 {r.spec-r.inc:+.4f}",flush=True)
        print(f"\n  合计可回收错误量 {int(tgt.err.sum())} = 全体错误的 {tgt.share.sum():.1%}",flush=True)

    ins=R[R.cls=="不可分"].sort_values("err",ascending=False)
    print(f"\n=== 不可分（专用上界<0.60）：{len(ins)} 对，扛 {ins.share.sum():.1%} 错误 ===",flush=True)
    for _,r in ins.head(15).iterrows():
        print(f"  {r.a}|{r.b}   错 {r.err}  在位者 {r.inc:.4f}  专用上界 {r.spec:.4f}",flush=True)

    print(f"\n=== 部分可分（0.60≤上界<0.90）：{(R.cls=='部分可分').sum()} 对 ===",flush=True)
    for _,r in R[R.cls=="部分可分"].sort_values("err",ascending=False).head(15).iterrows():
        print(f"  {r.a}|{r.b}   错 {r.err}  在位者 {r.inc:.4f}  专用上界 {r.spec:.4f}",flush=True)

    print(f"\n写出 {OUT}   总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
