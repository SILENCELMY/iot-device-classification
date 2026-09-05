"""【探索性,非协议】用【每对各自最好的模型】重测阈值：排序好的时候，切点还灵不灵。"""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0,REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0,REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0,REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"
def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    return TC.make_model(n,2)
def em(p,it=100):
    ps=0.5; pi=0.5
    for _ in range(it):
        w=(p*pi/ps)/(p*pi/ps+(1-p)*(1-pi)/(1-ps)); n=float(np.clip(w.mean(),1e-4,1-1e-4))
        if abs(n-pi)<1e-8: pi=n; break
        pi=n
    return pi
def one(tag,dfS,sd,dfT,td,pairs):
    cols=P.feature_columns(dfS); TC.SEED=42
    devs=sorted(set(IID.day_gate(dfS,sd))&set(IID.day_gate(dfT,td)))
    s=P.sample_balanced(dfS[dfS.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(dfT[dfT.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    print(f"\n{tag}\n  {chr(31)}",flush=True)
    print(f"  {\"类对 / 模型\":52s} {\"AUC\":>7s} {\"A源阈值\":>8s} {\"B中位\":>7s} {\"C EM\":>7s} {\"D上界\":>7s}",flush=True)
    for a,b in pairs:
        if a not in devs or b not in devs: continue
        ss=s[s.label.isin([a,b])]; tt=t[t.label.isin([a,b])]
        if ss.label.nunique()<2 or tt.label.nunique()<2: continue
        Xs=np.asarray(P.clean_x(ss,cols),dtype=float); ys=(ss.label.to_numpy()==b).astype(int)
        Xt=np.asarray(P.clean_x(tt,cols),dtype=float); yt=(tt.label.to_numpy()==b).astype(int)
        for nm in ("lr","rf","xgboost"):
            m=MK(nm); m.fit(Xs,ys); p=m.predict_proba(Xt)[:,1]
            auc=roc_auc_score(yt,p)
            A=((p>=0.5)==yt).mean(); B=((p>=np.median(p))==yt).mean()
            C=((p>=np.quantile(p,1-em(p)))==yt).mean()
            D=max(((p>=c)==yt).mean() for c in np.unique(np.quantile(p,np.linspace(0,1,201))))
            print(f"  {(a[:20]+chr(124)+b[:20]+\"  \"+nm):52s} {auc:7.4f} {A:8.4f} {B:7.4f} {C:7.4f} {D:7.4f}",flush=True)
with threadpool_limits(1):
    one("CIC 1102 Idle→Active",
        pd.read_csv("idle_1102.csv",low_memory=False),"2021_11_02_Idle",
        pd.read_csv("active_1102.csv",low_memory=False),"2021_11_02_Active",
        [("GlobeLampESPB1680C","GosundESP039AAFSocket")])
    one("UNSW 16-09-23 → 16-10-12",
        pd.read_csv(UNSW%"16-09-23",low_memory=False),"16-09-23",
        pd.read_csv(UNSW%"16-10-12",low_memory=False),"16-10-12",
        [("BelkinWemoMotion","BelkinWemoSwitch"),("NetatmoWelcome","TribySpeaker")])
