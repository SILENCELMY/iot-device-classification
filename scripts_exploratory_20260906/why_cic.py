"""【探索性,非协议】CIC 上能修的那一半为什么没被修：把错误质量、闸门决策、可覆盖性对齐看。
输出：
  A 平滑后基模型在 outer 上的逐类对错误质量（top 20）
  B v3 闸门放行的对（model, k, 配置 inner AUC, 在位者 inner AUC）
  C 对每个高错误对：过闸没有？没过是因为在位者已经够好，还是配置也修不动？
  D 这些错误里真类排第 2 的比例（覆盖是否够得着）
"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
CANDS=["lr","rf","xgboost"]; KS=[1,3,5,10]; MARGIN=0.01; SEED=42
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
with threadpool_limits(1):
    t0=time.time(); TC.SEED=SEED
    A=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    B=pd.read_csv("/home/lmy/cic_probe/active_1103.csv",low_memory=False)
    Cd=pd.read_csv("/home/lmy/cic_probe/active_1108.csv",low_memory=False)
    cols=P.feature_columns(A)
    devs=sorted(set(IID.day_gate(A,"2021_11_02_Idle"))&set(IID.day_gate(B,"2021_11_03_Active"))
                &set(IID.day_gate(Cd,"2021_11_08_Active")))
    le=LabelEncoder().fit(devs)
    def prep(df,sort=False):
        d=df[df.label.isin(devs)]
        if sort: d=d.sort_values(["label","window_start_epoch"])
        d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=SEED)
        if sort: d=d.sort_values(["label","window_start_epoch"])
        return np.asarray(P.clean_x(d,cols),dtype=float), le.transform(d.label), np.asarray(d.label)
    Xa,ya,_=prep(A); Xb,yb,gb=prep(B,True); Xt,yt,gt=prep(Cd,True)
    bm=TC.make_model("xgboost",len(devs)); bm.fit(Xa,ya)
    Pb=bm.predict_proba(Xb)
    from sklearn.metrics import f1_score
    kb=max(((k,f1_score(yb,cmeanM(Pb,gb,k).argmax(1),average="macro")) for k in KS),key=lambda x:x[1])[0]
    Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1)
    cand=set(map(tuple,np.sort(np.c_[ob[:,0],ob[:,1]],axis=1)))
    print(f"{len(devs)} 类  k_base={kb}  inner 候选 {len(cand)} 对",flush=True)
    # ---- outer 平滑后基模型的错误质量
    Pt=bm.predict_proba(Xt); Pts=cmeanM(Pt,gt,kb)
    o=np.argsort(-Pts,axis=1); t1,t2=o[:,0],o[:,1]
    rank=np.array([int(np.where(o[n]==yt[n])[0][0]) for n in range(len(yt))])
    Cm=confusion_matrix(yt,t1,labels=np.arange(len(devs)))
    err={}
    for i in range(len(devs)):
        for j in range(len(devs)):
            if i!=j and Cm[i,j]: 
                k2=tuple(sorted((i,j))); err[k2]=err.get(k2,0)+Cm[i,j]
    E=pd.Series(err).sort_values(ascending=False); tot=E.sum()
    print(f"平滑后 outer 错误 {tot}（{tot/len(yt)*100:.1f}%）  真类排第2 的占错误 {(rank==1).sum()/tot*100:.1f}%\n",flush=True)
    # ---- 逐对：在位者 vs 配置（都在 inner 上）
    def judge(i,j):
        ma=np.isin(ya,[i,j]); mb=np.isin(yb,[i,j])
        if len(np.unique(ya[ma]))<2 or len(np.unique(yb[mb]))<2: return None
        y2=(yb[mb]==j).astype(int)
        s=Pbs[mb][:,j]/np.clip(Pbs[mb][:,i]+Pbs[mb][:,j],1e-12,None)
        try: inc=roc_auc_score(y2,s)
        except Exception: return None
        y1=(ya[ma]==j).astype(int); best=(None,0,-1.0)
        for nm in CANDS:
            try:
                m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb[mb])[:,1]
            except Exception: continue
            for k in KS:
                v=roc_auc_score(y2,cmean(q,gb[mb],k))
                if v>best[2]: best=(nm,k,v)
        return inc,best
    print(f"{'类对':46s}{'错误':>6s}{'占比':>7s}{'top2内':>7s}{'在位者':>8s}{'配置':>8s}{'模型':>9s}{'过闸':>5s}",flush=True)
    print("-"*98,flush=True)
    for (i,j),n in E.head(20).items():
        r=judge(i,j)
        mm=((t1==i)&(t2==j))|((t1==j)&(t2==i))
        cover=int(((rank==1)&np.isin(yt,[i,j])&mm).sum())
        nm_=f"{le.classes_[i][:20]}|{le.classes_[j][:20]}"
        if r is None:
            print(f"{nm_:46s}{n:6d}{n/tot*100:6.1f}%{cover:7d}      —       —        —    —",flush=True); continue
        inc,(bm_,bk,ba)=r
        ok="✓" if (bm_ and ba>inc+MARGIN) else ""
        incand="" if (i,j) in cand else "(不在top2候选)"
        print(f"{nm_:46s}{n:6d}{n/tot*100:6.1f}%{cover:7d}{inc:8.4f}{ba:8.4f}{(bm_+' k'+str(bk)):>9s}{ok:>5s} {incand}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
