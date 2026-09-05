"""【探索性,非协议】v4：把覆盖从 top-2 扩到 top-K，K 在 inner 上选。

动机（why_cic.py 实测）：CIC 的错误只有 25.6% 是"真类排第二"，UNSW 是 56–64%。
top-2 覆盖在 CIC 上结构性够不着——闸门决策正确、修复本身正确，但下游传不到。

判决规则（对 top-2 退化为原方案）：
  以基模型 top-1 为擂主，按基概率降序让 top-2..top-K 依次挑战；
  若该 (擂主, 挑战者) 对有过闸配置，就由它裁决，胜者成为新擂主；没有配置则不换。
K 与 k_base 都在 inner 上选，outer 完全 held-out。
"""
from __future__ import annotations
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"
CANDS=["lr","rf","xgboost"]; KS=[1,3,5,10]; KBASE=[1,3,5,10]
TOPKS=[2,3]; MARGIN=0.01; SEEDS=(42,43)

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    return TC.make_model(n,2)
def cmean(p,g,k):
    if k<=1: return p
    o=np.empty_like(p)
    for u in np.unique(g):
        i=np.where(g==u)[0]; v=p[i]; c=np.cumsum(np.insert(v,0,0.0))
        for n in range(len(v)):
            lo=max(0,n-k+1); o[i[n]]=(c[n+1]-c[lo])/(n+1-lo)
    return o
def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o
def load(x): return x if isinstance(x,pd.DataFrame) else pd.read_csv(x,low_memory=False)

def topk_pairs(order,K):
    s=set()
    for row in order[:,:K]:
        for a in range(K):
            for b in range(a+1,K): s.add(tuple(sorted((row[a],row[b]))))
    return s

def decide(order,K,probs,cfgscore):
    """擂主赛：top-1 起，按序让 top-2..K 挑战；有配置的对由配置裁决。"""
    pred=order[:,0].copy()
    for slot in range(1,K):
        chall=order[:,slot]
        for (i,j),sc in cfgscore.items():
            m=((pred==i)&(chall==j))|((pred==j)&(chall==i))
            if not m.any(): continue
            pred[m]=np.where(sc[m]>=0.5, j, i)
    return pred

def task(tag,A,dayA,B,dayB,C,dayC):
    dfA,dfB,dfC=load(A),load(B),load(C)
    cols=P.feature_columns(dfA)
    devs=sorted(set(IID.day_gate(dfA,dayA))&set(IID.day_gate(dfB,dayB))&set(IID.day_gate(dfC,dayC)))
    le=LabelEncoder().fit(devs)
    print(f"\n{'='*98}\n{tag}   inner {dayA}→{dayB}   outer {dayA}→{dayC}   {len(devs)} 类",flush=True)
    rows=[]
    for seed in SEEDS:
        TC.SEED=seed
        def prep(df,sort=False):
            d=df[df.label.isin(devs)]
            if sort: d=d.sort_values(["label","window_start_epoch"])
            d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=seed)
            if sort: d=d.sort_values(["label","window_start_epoch"])
            return np.asarray(P.clean_x(d,cols),dtype=float), le.transform(d.label), np.asarray(d.label)
        Xa,ya,_=prep(dfA); Xb,yb,gb=prep(dfB,True); Xt,yt,gt=prep(dfC,True)
        bm=TC.make_model("xgboost",len(devs)); bm.fit(Xa,ya)
        Pb=bm.predict_proba(Xb)
        kb=max(((k,f1_score(yb,cmeanM(Pb,gb,k).argmax(1),average="macro")) for k in KBASE),key=lambda x:x[1])[0]
        Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1)
        rk=np.array([int(np.where(ob[n]==yb[n])[0][0]) for n in range(len(yb))])
        e=rk>0
        cov={K:float(((rk>=1)&(rk<K)).sum()/max(e.sum(),1)) for K in TOPKS}
        print(f"  seed{seed}  k_base={kb}   inner 错误中真类排名可及率: "
              + "  ".join(f"top{K}:{cov[K]*100:.1f}%" for K in TOPKS),flush=True)
        # 逐 K 导出配置 + 在 inner 上评，选 K
        allpairs=topk_pairs(ob,max(TOPKS))
        cfg={}
        for (i,j) in sorted(allpairs):
            ma=np.isin(ya,[i,j]); mb=np.isin(yb,[i,j])
            if len(np.unique(ya[ma]))<2 or len(np.unique(yb[mb]))<2: continue
            y2=(yb[mb]==j).astype(int)
            s=Pbs[mb][:,j]/np.clip(Pbs[mb][:,i]+Pbs[mb][:,j],1e-12,None)
            try: inc=roc_auc_score(y2,s)
            except Exception: continue
            y1=(ya[ma]==j).astype(int); best=(None,0,-1.0)
            for nm in CANDS:
                try:
                    m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb[mb])[:,1]
                except Exception: continue
                for k in KS:
                    v=roc_auc_score(y2,cmean(q,gb[mb],k))
                    if v>best[2]: best=(nm,k,v)
            if best[0] and best[2]>inc+MARGIN: cfg[(i,j)]=best
        # 在 inner 上给每个 K 打分（配置已拟合于 A，评在 B）
        sc_in={}
        for (i,j),(nm,k,_) in cfg.items():
            ms=np.isin(ya,[i,j])
            pm=MK(nm); pm.fit(Xa[ms],(ya[ms]==j).astype(int))
            sc_in[(i,j)]=cmean(pm.predict_proba(Xb)[:,1],gb,k)
        scores={K: f1_score(yb, decide(ob,K,Pbs,sc_in), average="macro") for K in TOPKS}
        Kbest=max(scores,key=scores.get)
        print(f"           过闸 {len(cfg)}/{len(allpairs)} 对   inner macro 按 K: "
              + "  ".join(f"top{K}:{scores[K]:.4f}" for K in TOPKS) + f"   → 选 K={Kbest}",flush=True)
        # outer
        Pt=bm.predict_proba(Xt); Pts=cmeanM(Pt,gt,kb); ot=np.argsort(-Pts,axis=1)
        sc_out={}
        for (i,j),(nm,k,_) in cfg.items():
            ms=np.isin(ya,[i,j])
            pm=MK(nm); pm.fit(Xa[ms],(ya[ms]==j).astype(int))
            sc_out[(i,j)]=cmean(pm.predict_proba(Xt)[:,1],gt,k)
        f_base=f1_score(yt,Pt.argmax(1),average="macro")
        f_sm=f1_score(yt,ot[:,0],average="macro")
        res={K: f1_score(yt, decide(ot,K,Pts,sc_out), average="macro") for K in TOPKS}
        print(f"           base={f_base:.4f}  平滑={f_sm:.4f}  " +
              "  ".join(f"top{K}={res[K]:.4f}({res[K]-f_sm:+.4f})" for K in TOPKS) +
              f"   **选中 K={Kbest} → {res[Kbest]:.4f} ({res[Kbest]-f_sm:+.4f})**",flush=True)
        rows.append({"seed":seed,"k_base":kb,"K":Kbest,"base":f_base,"smooth":f_sm,
                     **{f"top{K}":res[K] for K in TOPKS}, "chosen":res[Kbest],"n_cfg":len(cfg)})
    R=pd.DataFrame(rows)
    print(f"  → 均值 base={R.base.mean():.4f} 平滑={R.smooth.mean():.4f} "
          + " ".join(f"top{K}={R['top'+str(K)].mean():.4f}" for K in TOPKS)
          + f"  选中={R.chosen.mean():.4f}  Δ(选中−平滑)={(R.chosen-R.smooth).mean():+.4f}",flush=True)
    return R

if __name__=="__main__":
    with threadpool_limits(1):
        t0=time.time(); out=[]
        out.append(task("UNSW", UNSW%"16-09-23","16-09-23", UNSW%"16-09-30","16-09-30",
                        UNSW%"16-10-12","16-10-12"))
        out.append(task("CIC", "/home/lmy/cic_probe/idle_1102.csv","2021_11_02_Idle",
                        "/home/lmy/cic_probe/active_1103.csv","2021_11_03_Active",
                        "/home/lmy/cic_probe/active_1108.csv","2021_11_08_Active"))
        pd.concat(out).to_csv("/home/lmy/cic_probe/cfg_v4.csv",index=False)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
