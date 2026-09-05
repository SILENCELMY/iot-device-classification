"""【探索性,非协议】补两个缺口：
 ③ 逐日漂移对（Gosund）在 (模型, 时长) 下到底能不能修 —— 此前只用减法/极性试过
 ②' v3 的逐类 F1 实测（此前只有 v2 的）
 附：CIC 上 v3 过闸的 10-12 对里，有没有 Gosund 这类逐日翻转的对
"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
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
KS=[1,3,5,10,20]; CANDS=["lr","rf","xgboost"]; SEED=42
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
    # ---------- ③ Gosund 逐日漂移对 ----------
    print("="*94, flush=True)
    print("③ 逐日漂移对在 (模型, 时长) 下能不能修   源 1102Idle → 目标 1108Active（跨日+跨状态）", flush=True)
    src=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    for tgtf,tag in (("active_1103.csv","1103Active（inner 用的那天）"),
                     ("active_1108.csv","1108Active（outer）")):
        tgt=pd.read_csv(f"/home/lmy/cic_probe/{tgtf}",low_memory=False)
        cols=P.feature_columns(src)
        devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) &
                    set(IID.day_gate(tgt,"2021_11_0"+("3" if "1103" in tgtf else "8")+"_Active")))
        s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=SEED)
        t=tgt[tgt.label.isin(devs)].sort_values(["label","window_start_epoch"])
        t=P.sample_balanced(t,max_rows=IID.MAX_ROWS,random_state=SEED)
        t=t.sort_values(["label","window_start_epoch"])
        print(f"\n  目标={tag}",flush=True)
        for a,b in [("GosundESP10ACD8Plug","GosundESP147FF9Plug"),
                    ("GosundESP032979Plug","GosundESP1ACEE1Socket")]:
            if a not in devs or b not in devs: continue
            ss=s[s.label.isin([a,b])]; tt=t[t.label.isin([a,b])]
            if ss.label.nunique()<2 or tt.label.nunique()<2: continue
            Xs=np.asarray(P.clean_x(ss,cols),dtype=float); ys=(ss.label.to_numpy()==b).astype(int)
            Xt=np.asarray(P.clean_x(tt,cols),dtype=float); yt=(tt.label.to_numpy()==b).astype(int)
            gt=np.asarray(tt.label)
            print(f"    {a[:20]}|{b[:20]}",flush=True)
            for nm in CANDS:
                m=MK(nm); m.fit(Xs,ys); q=m.predict_proba(Xt)[:,1]
                row=" ".join(f"k={k}:{roc_auc_score(yt,cmean(q,gt,k)):.4f}" for k in KS)
                print(f"      {nm:9s} {row}",flush=True)
    # ---------- ②' v3 逐类 F1 ----------
    print("\n"+"="*94, flush=True)
    print("②' v3 的逐类 F1（UNSW outer 16-09-23 → 16-10-12，seed42）", flush=True)
    A=pd.read_csv(UNSW%"16-09-23",low_memory=False); B=pd.read_csv(UNSW%"16-09-30",low_memory=False)
    Cd=pd.read_csv(UNSW%"16-10-12",low_memory=False)
    cols=P.feature_columns(A)
    devs=sorted(set(IID.day_gate(A,"16-09-23"))&set(IID.day_gate(B,"16-09-30"))&set(IID.day_gate(Cd,"16-10-12")))
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
    kb=max(((k,f1_score(yb,cmeanM(Pb,gb,k).argmax(1),average="macro")) for k in [1,3,5,10]),key=lambda x:x[1])[0]
    Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1)
    cand=sorted(set(map(tuple,np.sort(np.c_[ob[:,0],ob[:,1]],axis=1))))
    cfg={}
    for (i,j) in cand:
        ma=np.isin(ya,[i,j]); mb=np.isin(yb,[i,j])
        if len(np.unique(ya[ma]))<2 or len(np.unique(yb[mb]))<2: continue
        y2=(yb[mb]==j).astype(int)
        sinc=Pbs[mb][:,j]/np.clip(Pbs[mb][:,i]+Pbs[mb][:,j],1e-12,None)
        try: inc=roc_auc_score(y2,sinc)
        except Exception: continue
        y1=(ya[ma]==j).astype(int); best=(None,0,-1.0)
        for nm in CANDS:
            m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb[mb])[:,1]
            for k in [1,3,5,10]:
                v=roc_auc_score(y2,cmean(q,gb[mb],k))
                if v>best[2]: best=(nm,k,v)
        if best[0] and best[2]>inc+0.01: cfg[(i,j)]=(best[0],best[1],best[2],inc)
    Pt=bm.predict_proba(Xt); Pts=cmeanM(Pt,gt,kb)
    o=np.argsort(-Pts,axis=1); t1,t2=o[:,0],o[:,1]; pred=t1.copy()
    print(f"  k_base={kb}   过闸 {len(cfg)} 对：",flush=True)
    for (i,j),(nm,k,a,inc) in cfg.items():
        print(f"    {le.classes_[i]}|{le.classes_[j]}  → {nm} k={k}  inner配置 {a:.4f} vs 在位者 {inc:.4f}",flush=True)
        ms=np.isin(ya,[i,j]); mm=((t1==i)&(t2==j))|((t1==j)&(t2==i))
        if not mm.any(): continue
        pm=MK(nm); pm.fit(Xa[ms],(ya[ms]==j).astype(int))
        q=cmean(pm.predict_proba(Xt)[:,1],gt,k)
        pred[mm]=np.where(q[mm]>=0.5,j,i)
    F0=f1_score(yt,Pt.argmax(1),average=None,labels=np.arange(len(devs)))
    F1s=f1_score(yt,t1,average=None,labels=np.arange(len(devs)))
    F3=f1_score(yt,pred,average=None,labels=np.arange(len(devs)))
    print(f"\n  {'类':22s}{'base':>10s}{'+平滑':>10s}{'+v3':>10s}",flush=True)
    for c in np.argsort(F0):
        print(f"  {le.classes_[c][:22]:22s}{F0[c]:10.4f}{F1s[c]:10.4f}{F3[c]:10.4f}",flush=True)
    print(f"  {'macro':22s}{F0.mean():10.4f}{F1s.mean():10.4f}{F3.mean():10.4f}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
