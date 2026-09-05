"""【探索性,非协议】关键对照：+0.0893 里有多少只是时间平滑。
  base_k     基模型多分类概率按同流因果滑动平均后 argmax（不做任何逐类对配置）
与 cfg_joint 用同样的分组、同样的 k、同样的 seed 与数据切分。"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"
KS=[1,3,5,10,20]

def causal_mean_mat(M, grp, k):
    if k<=1: return M
    out=np.empty_like(M)
    for g in np.unique(grp):
        idx=np.where(grp==g)[0]
        V=M[idx]; C=np.vstack([np.zeros(V.shape[1]), np.cumsum(V,axis=0)])
        for n in range(len(idx)):
            lo=max(0,n-k+1); out[idx[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return out

with threadpool_limits(1):
    t0=time.time()
    dfS=pd.read_csv(UNSW%"16-09-23",low_memory=False)
    dfB=pd.read_csv(UNSW%"16-09-30",low_memory=False)
    dfT=pd.read_csv(UNSW%"16-10-12",low_memory=False)
    cols=P.feature_columns(dfS)
    devs=sorted(set(IID.day_gate(dfS,"16-09-23"))&set(IID.day_gate(dfB,"16-09-30"))
                &set(IID.day_gate(dfT,"16-10-12")))
    le=LabelEncoder().fit(devs)
    print(f"UNSW outer 16-09-23 → 16-10-12   {len(devs)} 类  {len(cols)} 列",flush=True)
    print(f"（对照 cfg_joint：base 0.8414/0.8344，joint 0.9371/0.9172，Δ=+0.0893）\n",flush=True)
    for seed in (42,43):
        TC.SEED=seed
        s=P.sample_balanced(dfS[dfS.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        t=dfT[dfT.label.isin(devs)].sort_values(["label","window_start_epoch"])
        t=P.sample_balanced(t,max_rows=IID.MAX_ROWS,random_state=seed)
        t=t.sort_values(["label","window_start_epoch"])
        Xs=np.asarray(P.clean_x(s,cols),dtype=float); ys=le.transform(s.label)
        Xt=np.asarray(P.clean_x(t,cols),dtype=float); yt=le.transform(t.label)
        gt=np.asarray(t.label)
        m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys)
        Pm=m.predict_proba(Xt)
        row=[]
        for k in KS:
            pred=causal_mean_mat(Pm,gt,k).argmax(1)
            row.append((k, f1_score(yt,pred,average="macro"), (pred==yt).mean()))
        b=row[0][1]
        print(f"  seed{seed}  " + "   ".join(
            f"k={k}:{f:.4f}({f-b:+.4f})" for k,f,a in row),flush=True)
    print(f"\n判读：若 base_k=10 就已接近 0.93，则 +0.0893 主要来自时间平滑，"
          f"逐类对配置贡献甚微；若 base_k=10 只到 0.86–0.87，则配置是主因。",flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s",flush=True)
