"""【探索性,非协议】v5 在【型号级】CIC 上重跑。

与 cfg_v5.py 唯一的差别：标签空间从物理实例改为设备型号（同型号多实例合并）。
理由见 README「更正」节：实例级标签是数据集产物，不是任务定义。
时间平滑的分组仍按【物理流】（原始 label = MAC），不按合并后的型号——
部署时知道「这些窗来自同一个 MAC」，不知道它是哪个型号。
"""
from __future__ import annotations
import os, sys, time, re
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
CANDS=["lr","rf","xgboost"]; KS=[1,3,5,10]; KBASE=[1,3,5,10]
MARGIN=0.02; MIN_CELL=40; SEEDS=(42,43)

def TYPE(s):
    if re.match(r"GosundESP.*(Plug|Socket)$", s): return "GosundPlugSocket"
    if re.match(r"TeckinPlug\d$", s):             return "TeckinPlug"
    if re.match(r"YutronPlug\d$", s):             return "YutronPlug"
    if re.match(r"AmazonAlexaEchoDot\d$", s):     return "AmazonAlexaEchoDot"
    return s

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

def task(tag, fA, dayA, fB, dayB, fC, dayC):
    dfA=pd.read_csv(fA,low_memory=False); dfB=pd.read_csv(fB,low_memory=False)
    dfC=pd.read_csv(fC,low_memory=False)
    cols=P.feature_columns(dfA)
    devs=sorted(set(IID.day_gate(dfA,dayA))&set(IID.day_gate(dfB,dayB))&set(IID.day_gate(dfC,dayC)))
    classes=sorted({TYPE(d) for d in devs}); le=LabelEncoder().fit(classes)
    from collections import Counter
    merged={k:v for k,v in Counter(TYPE(d) for d in devs).items() if v>1}
    print(f"\n{'='*96}\n{tag}   inner {dayA}→{dayB}   outer {dayA}→{dayC}",flush=True)
    print(f"  实例 {len(devs)} 台 → 型号 {len(classes)} 类   合并：{merged}",flush=True)
    rows=[]
    for seed in SEEDS:
        TC.SEED=seed
        def prep(df,sort=False):
            d=df[df.label.isin(devs)]
            if sort: d=d.sort_values(["label","window_start_epoch"])
            d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=seed)
            if sort: d=d.sort_values(["label","window_start_epoch"])
            X=np.asarray(P.clean_x(d,cols),dtype=float)
            y=le.transform([TYPE(x) for x in d.label])
            g=np.asarray(d.label)          # 分组按物理流，不按型号
            return X,y,g
        Xa,ya,_=prep(dfA); Xb,yb,gb=prep(dfB,True); Xt,yt,gt=prep(dfC,True)
        bm=TC.make_model("xgboost",len(classes)); bm.fit(Xa,ya)
        Pb=bm.predict_proba(Xb)
        kb=max(((k,f1_score(yb,cmeanM(Pb,gb,k).argmax(1),average="macro")) for k in KBASE),key=lambda x:x[1])[0]
        Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1); b1,b2=ob[:,0],ob[:,1]
        cand=sorted(set(map(tuple,np.sort(np.c_[b1,b2],axis=1))))
        cfg={}; audit=[]
        for (i,j) in cand:
            D=(((b1==i)&(b2==j))|((b1==j)&(b2==i)))&np.isin(yb,[i,j])
            if D.sum()<MIN_CELL: continue
            inc=float((b1[D]==yb[D]).mean())
            ma=np.isin(ya,[i,j])
            if len(np.unique(ya[ma]))<2: continue
            y1=(ya[ma]==j).astype(int); best=(None,0,-1.0)
            for nm in CANDS:
                try:
                    m=MK(nm); m.fit(Xa[ma],y1); q=m.predict_proba(Xb)[:,1]
                except Exception: continue
                for k in KS:
                    qq=cmean(q,gb,k)
                    acc=float((np.where(qq[D]>=0.5,j,i)==yb[D]).mean())
                    if acc>best[2]: best=(nm,k,acc)
            if best[0] is None: continue
            audit.append((i,j,int(D.sum()),inc,best[2],best[0],best[1]))
            if best[2]>inc+MARGIN: cfg[(i,j)]=best
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
        Ad=pd.DataFrame(audit,columns=["i","j","n_D","inc","cfg","model","k"]); Ad["gain"]=Ad.cfg-Ad.inc
        for r in Ad.sort_values("gain",ascending=False).head(6).itertuples():
            print(f"      {le.classes_[r.i][:22]}|{le.classes_[r.j][:22]:22s} 可判决区 {r.n_D:5d}  "
                  f"在位者 {r.inc:.4f} → 配置 {r.cfg:.4f} ({r.gain:+.4f}) {r.model} k{r.k}",flush=True)
        F=f1_score(yt,pred,average=None,labels=np.arange(len(classes)))
        Fs=f1_score(yt,t1,average=None,labels=np.arange(len(classes)))
        worst=np.argsort(Fs)[:5]
        print("      最差 5 类  " + "  ".join(
            f"{le.classes_[c][:18]} {Fs[c]:.3f}→{F[c]:.3f}" for c in worst),flush=True)
        rows.append({"seed":seed,"base":f_base,"smooth":f_sm,"v5":f_v5,"n_gate":len(cfg)})
    R=pd.DataFrame(rows)
    print(f"  → 均值 base={R.base.mean():.4f} 平滑={R.smooth.mean():.4f} v5={R.v5.mean():.4f}  "
          f"Δ(v5−平滑)={(R.v5-R.smooth).mean():+.4f}",flush=True)
    return R

if __name__=="__main__":
    with threadpool_limits(1):
        t0=time.time()
        R=task("CIC 型号级", "/home/lmy/cic_probe/idle_1102.csv","2021_11_02_Idle",
               "/home/lmy/cic_probe/active_1103.csv","2021_11_03_Active",
               "/home/lmy/cic_probe/active_1108.csv","2021_11_08_Active")
        R.to_csv("/home/lmy/cic_probe/cfg_v5_type.csv",index=False)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
