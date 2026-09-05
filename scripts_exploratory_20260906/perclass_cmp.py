"""【探索性,非协议】平滑基线在困难类上到底做到没做到：逐类 F1 四臂对比。
臂：base(k=1) / base+全局平滑 k=3（宏观最优）/ base+全局平滑 k=10 / cfg_joint(逐类对配置)"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"
CANDS=["lr","rf","xgboost"]; KS=[1,3,5,10]; GATE=0.95; SEED=42
def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    return TC.make_model(n,2)
def cmean(p,grp,k):
    if k<=1: return p
    out=np.empty_like(p)
    for g in np.unique(grp):
        i=np.where(grp==g)[0]; v=p[i]; c=np.cumsum(np.insert(v,0,0.0))
        for n in range(len(v)):
            lo=max(0,n-k+1); out[i[n]]=(c[n+1]-c[lo])/(n+1-lo)
    return out
def cmeanM(M,grp,k):
    if k<=1: return M
    out=np.empty_like(M)
    for g in np.unique(grp):
        i=np.where(grp==g)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); out[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return out
with threadpool_limits(1):
    t0=time.time(); TC.SEED=SEED
    A=pd.read_csv(UNSW%"16-09-23",low_memory=False); B=pd.read_csv(UNSW%"16-09-30",low_memory=False)
    C=pd.read_csv(UNSW%"16-10-12",low_memory=False)
    cols=P.feature_columns(A)
    devs=sorted(set(IID.day_gate(A,"16-09-23"))&set(IID.day_gate(B,"16-09-30"))&set(IID.day_gate(C,"16-10-12")))
    le=LabelEncoder().fit(devs)
    def prep(df,sort=False):
        d=df[df.label.isin(devs)]
        if sort: d=d.sort_values(["label","window_start_epoch"])
        d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=SEED)
        if sort: d=d.sort_values(["label","window_start_epoch"])
        return np.asarray(P.clean_x(d,cols),dtype=float), le.transform(d.label), np.asarray(d.label)
    Xa,ya,_=prep(A); Xb,yb,gb=prep(B,True); Xt,yt,gt=prep(C,True)
    # ---- inner 导出逐类对 (模型,时长)
    bm=TC.make_model("xgboost",len(devs)); bm.fit(Xa,ya)
    oo=np.argsort(-bm.predict_proba(Xb),axis=1)
    cand=sorted(set(map(tuple,np.sort(np.c_[oo[:,0],oo[:,1]],axis=1))))
    cfg={}
    for (i,j) in cand:
        ma=np.isin(ya,[i,j]); mb=np.isin(yb,[i,j])
        if len(np.unique(ya[ma]))<2 or len(np.unique(yb[mb]))<2: continue
        y1=(ya[ma]==j).astype(int); y2=(yb[mb]==j).astype(int)
        best=(None,0,-1)
        for nm in CANDS:
            try:
                m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb[mb])[:,1]
            except Exception: continue
            for k in KS:
                v=roc_auc_score(y2,cmean(q,gb[mb],k))
                if v>best[2]: best=(nm,k,v)
        if best[0]: cfg[(i,j)]=best
    # ---- 四臂
    base=TC.make_model("xgboost",len(devs)); base.fit(Xa,ya)
    Pm=base.predict_proba(Xt); o=np.argsort(-Pm,axis=1); top1,top2=o[:,0],o[:,1]
    arms={"base(k=1)":top1,
          "base+平滑k=3":cmeanM(Pm,gt,3).argmax(1),
          "base+平滑k=10":cmeanM(Pm,gt,10).argmax(1)}
    pred=top1.copy()
    for (i,j),(nm,k,a) in cfg.items():
        if a<GATE: continue
        ms=np.isin(ya,[i,j])
        if len(np.unique(ya[ms]))<2: continue
        mm=((top1==i)&(top2==j))|((top1==j)&(top2==i))
        if not mm.any(): continue
        pm=MK(nm); pm.fit(Xa[ms],(ya[ms]==j).astype(int))
        q=cmean(pm.predict_proba(Xt)[:,1],gt,k)
        pred[mm]=np.where(q[mm]>=0.5,j,i)
    arms["cfg_joint"]=pred
    print(f"UNSW outer 16-09-23 → 16-10-12   seed{SEED}   {len(devs)} 类\n",flush=True)
    hdr=f"{'类':24s}" + "".join(f"{k:>16s}" for k in arms)
    print(hdr,flush=True); print("-"*len(hdr),flush=True)
    F={k:f1_score(yt,v,average=None,labels=np.arange(len(devs))) for k,v in arms.items()}
    order=np.argsort(F["base(k=1)"])
    for c in order:
        print(f"{le.classes_[c][:24]:24s}" + "".join(f"{F[k][c]:16.4f}" for k in arms),flush=True)
    print("-"*len(hdr),flush=True)
    mac=[f1_score(yt,v,average="macro") for v in arms.values()]
    print("macro-F1".ljust(24) + "".join(format(x,"16.4f") for x in mac),flush=True)
    acc=[(v==yt).mean() for v in arms.values()]
    print("accuracy".ljust(24) + "".join(format(x,"16.4f") for x in acc),flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
