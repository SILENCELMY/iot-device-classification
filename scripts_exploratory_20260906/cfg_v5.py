"""【探索性,非协议】v5：闸门判据改在【可判决区】上、并改用【准确率】。

v3/v4 的缺陷（thr_incumbent.py 实测暴露）：
  闸门在"两类的全部窗口"上量逐类对 AUC，得到 1.0000；
  但判决只发生在"这两类争 top-2"的窗口上，那里 AUC 是 0.03。
  总体错了，而且 AUC 对切点免疫，两个毛病叠在一起。

v5 只改闸门的量，不改任何别的：
  可判决区 D_ij = inner 上 (i,j) 恰为平滑后基模型 top-2 的窗口
  在位者得分 = 平滑后基模型在 D_ij 上按 argmax 判这两类的准确率
  配置得分   = 逐类对 (模型,时长) 在 D_ij 上按 0.5 切点判的准确率
  过闸条件   = 配置得分 > 在位者得分 + MARGIN，且 |D_ij| >= MIN_CELL
"""
from __future__ import annotations
import os, sys, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
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
MARGIN=0.02; MIN_CELL=40; SEEDS=(42,43)

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

def task(tag,A,dayA,B,dayB,C,dayC):
    dfA,dfB,dfC=load(A),load(B),load(C)
    cols=P.feature_columns(dfA)
    devs=sorted(set(IID.day_gate(dfA,dayA))&set(IID.day_gate(dfB,dayB))&set(IID.day_gate(dfC,dayC)))
    le=LabelEncoder().fit(devs)
    print(f"\n{'='*96}\n{tag}   inner {dayA}→{dayB}   outer {dayA}→{dayC}   {len(devs)} 类",flush=True)
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
        Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1); b1,b2=ob[:,0],ob[:,1]
        cand=sorted(set(map(tuple,np.sort(np.c_[b1,b2],axis=1))))
        cfg={}; audit=[]
        for (i,j) in cand:
            # 可判决区：该对恰为 top-2，且真类是这两类之一（后者只用于打分，不用于选窗）
            D=(((b1==i)&(b2==j))|((b1==j)&(b2==i)))
            Dy=D&np.isin(yb,[i,j])
            if Dy.sum()<MIN_CELL: continue
            inc_acc=float((b1[Dy]==yb[Dy]).mean())      # 在位者在可判决区的准确率
            ma=np.isin(ya,[i,j])
            if len(np.unique(ya[ma]))<2: continue
            y1=(ya[ma]==j).astype(int); best=(None,0,-1.0)
            for nm in CANDS:
                try:
                    m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb)[:,1]
                except Exception: continue
                for k in KS:
                    qq=cmean(q,gb,k)
                    acc=float((np.where(qq[Dy]>=0.5,j,i)==yb[Dy]).mean())
                    if acc>best[2]: best=(nm,k,acc)
            if best[0] is None: continue
            audit.append((i,j,int(Dy.sum()),inc_acc,best[2],best[0],best[1]))
            if best[2]>inc_acc+MARGIN: cfg[(i,j)]=best
        # outer
        Pt=bm.predict_proba(Xt); Pts=cmeanM(Pt,gt,kb); ot=np.argsort(-Pts,axis=1)
        t1,t2=ot[:,0],ot[:,1]
        f_base=f1_score(yt,Pt.argmax(1),average="macro"); f_sm=f1_score(yt,t1,average="macro")
        pred=t1.copy(); nov=0; used={}
        for (i,j),(nm,k,a) in cfg.items():
            ms=np.isin(ya,[i,j])
            pm=MK(nm); pm.fit(Xa[ms],(ya[ms]==j).astype(int))
            q=cmean(pm.predict_proba(Xt)[:,1],gt,k)
            mm=((t1==i)&(t2==j))|((t1==j)&(t2==i))
            if not mm.any(): continue
            pred[mm]=np.where(q[mm]>=0.5,j,i); nov+=int(mm.sum()); used[nm]=used.get(nm,0)+1
        f_v5=f1_score(yt,pred,average="macro")
        print(f"  seed{seed} k_base={kb}  base={f_base:.4f} 平滑={f_sm:.4f} v5={f_v5:.4f}  "
              f"Δ(v5−平滑)={f_v5-f_sm:+.4f}   过闸 {len(cfg)}/{len(audit)} 对 / {nov} 窗  模型{used}",flush=True)
        Ad=pd.DataFrame(audit,columns=["i","j","n_D","inc_acc","cfg_acc","model","k"])
        Ad["gain"]=Ad.cfg_acc-Ad.inc_acc
        top=Ad.sort_values("gain",ascending=False).head(6)
        for r in top.itertuples():
            print(f"      {le.classes_[r.i][:20]}|{le.classes_[r.j][:20]:20s} "
                  f"可判决区 {r.n_D:5d}  在位者 {r.inc_acc:.4f} → 配置 {r.cfg_acc:.4f} "
                  f"({r.gain:+.4f}) {r.model} k{r.k}",flush=True)
        rows.append({"seed":seed,"base":f_base,"smooth":f_sm,"v5":f_v5,"n_gate":len(cfg)})
    R=pd.DataFrame(rows)
    print(f"  → 均值 base={R.base.mean():.4f} 平滑={R.smooth.mean():.4f} v5={R.v5.mean():.4f}  "
          f"Δ(v5−平滑)={(R.v5-R.smooth).mean():+.4f}",flush=True)
    return R

if __name__=="__main__":
    with threadpool_limits(1):
        t0=time.time(); out=[]
        out.append(task("UNSW", UNSW%"16-09-23","16-09-23", UNSW%"16-09-30","16-09-30",
                        UNSW%"16-10-12","16-10-12"))
        out.append(task("CIC", "/home/lmy/cic_probe/idle_1102.csv","2021_11_02_Idle",
                        "/home/lmy/cic_probe/active_1103.csv","2021_11_03_Active",
                        "/home/lmy/cic_probe/active_1108.csv","2021_11_08_Active"))
        pd.concat(out).to_csv("/home/lmy/cic_probe/cfg_v5.csv",index=False)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
