"""【探索性,非协议】逐类对配置 =(模型, 观测时长) 联合搜索，在 inner 跨环境上导出。
闸门在"打算使用的时长"上评（此前在 k=1 上评，把 Belkin 这类需要长时程的对整个挡在门外）。

时长聚合的合法性：按【流分组】做因果滑动平均。部署时"这些窗来自同一个流"是已知的
（MAC/五元组），未知的只是这个流的身份标签。故用分组、不用标签值。
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
CANDS=["lr","rf","xgboost"]; KS=[1,3,5,10]; GATE=0.95; SEEDS=(42,43)

def MK(nm):
    if nm=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,C=1.0))
    return TC.make_model(nm,2)

def causal_mean(p, grp, k):
    """按流分组的因果滑动平均：只用当前及之前 k-1 个同流窗。"""
    if k<=1: return p
    out=np.empty_like(p)
    for g in np.unique(grp):
        idx=np.where(grp==g)[0]
        v=p[idx]; c=np.cumsum(np.insert(v,0,0.0))
        for n in range(len(v)):
            lo=max(0,n-k+1); out[idx[n]]=(c[n+1]-c[lo])/(n+1-lo)
    return out

def load(x): return x if isinstance(x,pd.DataFrame) else pd.read_csv(x,low_memory=False)

def prep(df, devs, cols, le, seed, sort=False):
    TC.SEED=seed
    d=df[df.label.isin(devs)]
    if sort: d=d.sort_values(["label","window_start_epoch"])
    d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=seed)
    if sort: d=d.sort_values(["label","window_start_epoch"])
    return (np.asarray(P.clean_x(d,cols),dtype=float), le.transform(d.label),
            np.asarray(d.label))

def derive(dfA,dayA,dfB,dayB,devs,cols,le,seed):
    """inner A→B：逐类对联合搜 (模型, 时长)。返回 {(i,j):(model,k,auc)}。"""
    Xa,ya,_ = prep(dfA,devs,cols,le,seed)
    Xb,yb,gb= prep(dfB,devs,cols,le,seed,sort=True)
    bm=TC.make_model("xgboost",len(devs)); bm.fit(Xa,ya)
    oo=np.argsort(-bm.predict_proba(Xb),axis=1)
    cand=sorted(set(map(tuple,np.sort(np.c_[oo[:,0],oo[:,1]],axis=1))))
    cfg={}
    for (i,j) in cand:
        ma=np.isin(ya,[i,j]); mb=np.isin(yb,[i,j])
        if len(np.unique(ya[ma]))<2 or len(np.unique(yb[mb]))<2: continue
        ya01=(ya[ma]==j).astype(int); yb01=(yb[mb]==j).astype(int)
        best=(None,0,-1.0)
        for nm in CANDS:
            try:
                m=MK(nm); m.fit(Xa[ma],ya01); q=m.predict_proba(Xb[mb])[:,1]
            except Exception: continue
            for k in KS:
                try: v=roc_auc_score(yb01, causal_mean(q, gb[mb], k))
                except Exception: continue
                if v>best[2]: best=(nm,k,v)
        if best[0]: cfg[(i,j)]=best
    return cfg, len(cand)

def evaluate(dfS,dfT,devs,cols,le,seed,cfg,tag):
    Xs,ys,_ = prep(dfS,devs,cols,le,seed)
    Xt,yt,gt= prep(dfT,devs,cols,le,seed,sort=True)
    base=TC.make_model("xgboost",len(devs)); base.fit(Xs,ys)
    o=np.argsort(-base.predict_proba(Xt),axis=1); top1,top2=o[:,0],o[:,1]
    f0=f1_score(yt,top1,average="macro")
    pred=top1.copy(); nov=0; used={}; ks={}
    for (i,j),(nm,k,auc) in cfg.items():
        if auc<GATE: continue
        ms=np.isin(ys,[i,j])
        if len(np.unique(ys[ms]))<2: continue
        mm=((top1==i)&(top2==j))|((top1==j)&(top2==i))
        if not mm.any(): continue
        pm=MK(nm); pm.fit(Xs[ms],(ys[ms]==j).astype(int))
        q=pm.predict_proba(Xt)[:,1]
        q=causal_mean(q, gt, k)               # 全序列聚合后再取被覆盖的位置
        pred[mm]=np.where(q[mm]>=0.5, j, i); nov+=int(mm.sum())
        used[nm]=used.get(nm,0)+1; ks[k]=ks.get(k,0)+1
    f1=f1_score(yt,pred,average="macro")
    print(f"  seed{seed} {tag}: base={f0:.4f}  joint={f1:.4f}  Δ={f1-f0:+.4f}   "
          f"过闸 {len(used) and sum(used.values())} 对 / {nov} 窗  模型{used} 时长{ks}",flush=True)
    return {"seed":seed,"tag":tag,"base":f0,"joint":f1,"delta":f1-f0,
            "n_pairs":sum(used.values()) if used else 0,"n_win":nov,
            "models":json.dumps(used),"ks":json.dumps(ks)}

def task(tag,A,dayA,B,dayB,C,dayC):
    dfA,dfB,dfC=load(A),load(B),load(C)
    cols=P.feature_columns(dfA)
    devs=sorted(set(IID.day_gate(dfA,dayA))&set(IID.day_gate(dfB,dayB))&set(IID.day_gate(dfC,dayC)))
    le=LabelEncoder().fit(devs)
    print(f"\n{'='*94}\n{tag}   inner {dayA}→{dayB}   outer {dayA}→{dayC}   "
          f"{len(devs)} 类  {len(cols)} 列",flush=True)
    rows=[]
    for seed in SEEDS:
        cfg,ncand=derive(dfA,dayA,dfB,dayB,devs,cols,le,seed)
        ok={k:v for k,v in cfg.items() if v[2]>=GATE}
        print(f"    inner 候选 {ncand} 对，过闸 {len(ok)} 对",flush=True)
        for (i,j),(nm,k,a) in sorted(ok.items(), key=lambda x:-x[1][2])[:6]:
            print(f"      {le.classes_[i][:20]}|{le.classes_[j][:20]:20s} → {nm} k={k} auc={a:.4f}",flush=True)
        rows.append(evaluate(dfA,dfC,devs,cols,le,seed,cfg,"outer"))
    R=pd.DataFrame(rows)
    print(f"  → 均值 Δmacro = {R.delta.mean():+.4f}",flush=True)
    return R

if __name__=="__main__":
    with threadpool_limits(1):
        t0=time.time(); out=[]
        out.append(task("UNSW", UNSW%"16-09-23","16-09-23", UNSW%"16-09-30","16-09-30",
                        UNSW%"16-10-12","16-10-12"))
        out.append(task("CIC（inner 1102，outer 1108）",
                        "/home/lmy/cic_probe/idle_1102.csv","2021_11_02_Idle",
                        "/home/lmy/cic_probe/active_1102.csv","2021_11_02_Active",
                        "/home/lmy/cic_probe/active_1108.csv","2021_11_08_Active"))
        pd.concat(out).to_csv("/home/lmy/cic_probe/cfg_joint.csv",index=False)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
