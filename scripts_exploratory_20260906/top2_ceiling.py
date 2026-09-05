"""【探索性,非协议】任何 top-2 覆盖方案的真实天花板：错误窗口里，真类排第几。"""
import os, sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"

def one(tag, dfS, sday, dfT, tday, focus=None):
    cols=P.feature_columns(dfS)
    devs=sorted(set(IID.day_gate(dfS,sday)) & set(IID.day_gate(dfT,tday)))
    le=LabelEncoder().fit(devs); TC.SEED=42
    s=P.sample_balanced(dfS[dfS.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(dfT[dfT.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    Xs=np.asarray(P.clean_x(s,cols),dtype=float); ys=le.transform(s.label)
    Xt=np.asarray(P.clean_x(t,cols),dtype=float); yt=le.transform(t.label)
    m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys)
    o=np.argsort(-m.predict_proba(Xt),axis=1)
    rank=np.array([int(np.where(o[n]==yt[n])[0][0]) for n in range(len(yt))])
    err=rank>0; ne=int(err.sum())
    print(f"\n{tag}   {len(devs)} 类  n={len(yt)}  错误 {ne}（{ne/len(yt)*100:.1f}%）",flush=True)
    for r,lab in ((1,"真类排第 2（top-2 覆盖可救）"),(2,"排第 3"),(3,"排第 4")):
        c=int((rank==r).sum())
        print(f"    {lab:28s} {c:5d}  占错误 {c/ne*100:5.1f}%  占全体 {c/len(yt)*100:5.2f}%",flush=True)
    c=int((rank>=4).sum())
    print(f"    {'排第 5 及以后（无救）':28s} {c:5d}  占错误 {c/ne*100:5.1f}%",flush=True)
    print(f"  ==> top-2 覆盖的**绝对天花板** = +{(rank==1).sum()/len(yt)*100:.2f} 个百分点准确率",flush=True)
    if focus:
        for a,b in focus:
            if a not in devs or b not in devs: continue
            ia,ib=le.transform([a])[0],le.transform([b])[0]
            mm=np.isin(yt,[ia,ib])
            e2=err&mm; n2=int(e2.sum())
            if n2==0: continue
            t2=int(((rank==1)&mm).sum())
            # 其中 top-2 恰好是这一对的
            pairtop=((o[:,0]==ia)&(o[:,1]==ib))|((o[:,0]==ib)&(o[:,1]==ia))
            print(f"    [{a[:18]}|{b[:18]}] 该两类错误 {n2}，真类排第2 的 {t2}"
                  f"（{t2/n2*100:.1f}%），其中 top-2 恰为该对 {int((e2&pairtop).sum())}",flush=True)

with threadpool_limits(1):
    BK=[("BelkinWemoMotion","BelkinWemoSwitch")]
    for sd,td in [("16-09-23","16-09-30"),("16-09-30","16-10-12"),("16-09-23","16-10-12")]:
        one(f"UNSW {sd} → {td}", pd.read_csv(UNSW%sd,low_memory=False), sd,
            pd.read_csv(UNSW%td,low_memory=False), td, BK)
    one("CIC 1102 Idle → Active",
        pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False),"2021_11_02_Idle",
        pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False),"2021_11_02_Active",
        [("GlobeLampESPB1680C","GosundESP039AAFSocket"),("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2")])
