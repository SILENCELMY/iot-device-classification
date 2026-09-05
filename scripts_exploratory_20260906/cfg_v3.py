"""【探索性,非协议】v3：平滑当在位基线，逐类对配置只在 inner 上打得赢在位者时才动手。

对 v2(cfg_joint) 的唯一改动：闸门从"绝对 AUC >= 0.95"改成"相对在位者有增益"。
  在位者 = 基模型多分类概率按同流因果平滑（k_base 在 inner 上选）
  逐类对配置 = (模型, 时长)，在 inner 上搜
  只有当 配置的 inner 逐类对 AUC  >  在位者的 inner 逐类对 AUC + MARGIN 时才覆盖
全部选择只用 inner（有标签的环境对），outer 完全 held-out。
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
CANDS=["lr","rf","xgboost"]; KS=[1,3,5,10]; KBASE=[1,3,5,10]; MARGIN=0.01; SEEDS=(42,43,44,45,46)

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

def load(x): return x if isinstance(x,pd.DataFrame) else pd.read_csv(x,low_memory=False)

def task(tag,A,dayA,B,dayB,C,dayC):
    dfA,dfB,dfC=load(A),load(B),load(C)
    cols=P.feature_columns(dfA)
    devs=sorted(set(IID.day_gate(dfA,dayA))&set(IID.day_gate(dfB,dayB))&set(IID.day_gate(dfC,dayC)))
    le=LabelEncoder().fit(devs)
    print(f"\n{'='*96}\n{tag}   inner {dayA}→{dayB}   outer {dayA}→{dayC}   {len(devs)} 类  {len(cols)} 列",flush=True)
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
        # ① 在 inner 上选全局平滑 k_base
        sc=[(k, f1_score(yb, cmeanM(Pb,gb,k).argmax(1), average="macro")) for k in KBASE]
        kb=max(sc,key=lambda x:x[1])[0]
        print(f"  seed{seed}  inner 选出的全局平滑 k_base={kb}  "
              f"(inner macro {dict((k,round(v,4)) for k,v in sc)})",flush=True)
        Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1)
        cand=sorted(set(map(tuple,np.sort(np.c_[ob[:,0],ob[:,1]],axis=1))))
        # ② 逐类对：在位者 vs 配置，都在 inner 上算
        cfg={}
        for (i,j) in cand:
            ma=np.isin(ya,[i,j]); mb=np.isin(yb,[i,j])
            if len(np.unique(ya[ma]))<2 or len(np.unique(yb[mb]))<2: continue
            y2=(yb[mb]==j).astype(int)
            # 在位者在这一对上的 AUC：用平滑后基模型的两类相对分数
            s_inc=Pbs[mb][:,j]/np.clip(Pbs[mb][:,i]+Pbs[mb][:,j],1e-12,None)
            try: inc=roc_auc_score(y2,s_inc)
            except Exception: continue
            y1=(ya[ma]==j).astype(int); best=(None,0,-1.0)
            for nm in CANDS:
                try:
                    m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb[mb])[:,1]
                except Exception: continue
                for k in KS:
                    v=roc_auc_score(y2,cmean(q,gb[mb],k))
                    if v>best[2]: best=(nm,k,v)
            if best[0] and best[2] > inc + MARGIN:
                cfg[(i,j)]=(best[0],best[1],best[2],inc)
        # ③ outer 评估
        Pt=bm.predict_proba(Xt); Pts=cmeanM(Pt,gt,kb)
        top1=Pts.argmax(1); o=np.argsort(-Pts,axis=1); t1,t2=o[:,0],o[:,1]
        f_base1=f1_score(yt,Pt.argmax(1),average="macro")
        f_smooth=f1_score(yt,top1,average="macro")
        pred=top1.copy(); nov=0; used={}; ks={}
        for (i,j),(nm,k,a,inc) in cfg.items():
            ms=np.isin(ya,[i,j])
            if len(np.unique(ya[ms]))<2: continue
            mm=((t1==i)&(t2==j))|((t1==j)&(t2==i))
            if not mm.any(): continue
            pm=MK(nm); pm.fit(Xa[ms],(ya[ms]==j).astype(int))
            q=cmean(pm.predict_proba(Xt)[:,1],gt,k)
            pred[mm]=np.where(q[mm]>=0.5,j,i); nov+=int(mm.sum())
            used[nm]=used.get(nm,0)+1; ks[k]=ks.get(k,0)+1
        f_v3=f1_score(yt,pred,average="macro")
        print(f"           base={f_base1:.4f}  +平滑k{kb}={f_smooth:.4f}  +逐类对配置={f_v3:.4f}   "
              f"Δ(v3−平滑)={f_v3-f_smooth:+.4f}  Δ(v3−base)={f_v3-f_base1:+.4f}",flush=True)
        print(f"           候选 {len(cand)} 对，**过闸 {len(cfg)} 对**（v2 是几乎全过）/ {nov} 窗  "
              f"模型{used} 时长{ks}",flush=True)
        rows.append({"seed":seed,"k_base":kb,"base":f_base1,"smooth":f_smooth,"v3":f_v3,
                     "n_gate":len(cfg),"n_cand":len(cand)})
    R=pd.DataFrame(rows)
    print(f"  → 均值  base={R.base.mean():.4f}  平滑={R.smooth.mean():.4f}  v3={R.v3.mean():.4f}   "
          f"Δ(v3−平滑)={(R.v3-R.smooth).mean():+.4f}",flush=True)
    return R

if __name__=="__main__":
    with threadpool_limits(1):
        t0=time.time(); out=[]
        out.append(task("UNSW", UNSW%"16-09-23","16-09-23", UNSW%"16-09-30","16-09-30",
                        UNSW%"16-10-12","16-10-12"))
        out.append(task("CIC 同型化（inner 1102Idle→1103Active，outer 1102Idle→1108Active）",
                        "/home/lmy/cic_probe/idle_1102.csv","2021_11_02_Idle",
                        "/home/lmy/cic_probe/active_1103.csv","2021_11_03_Active",
                        "/home/lmy/cic_probe/active_1108.csv","2021_11_08_Active"))
        pd.concat(out).to_csv("/home/lmy/cic_probe/cfg_v3.csv",index=False)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
