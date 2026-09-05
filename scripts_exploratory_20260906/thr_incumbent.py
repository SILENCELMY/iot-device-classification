"""【探索性,非协议】阈值重测（此前用崩掉的 xgb 测过，无效）：
用【在位者自己的相对分数】看 CIC 高错误对上切点值多少。
  A 现状（相对分数 >= 0.5，即多分类 argmax 在这两类间的判决）
  B 目标域中位切点
  C EM 估先验后取分位（无标签，可部署）
  D oracle 切点（上界）
只在"真类排第2 且 top-2 恰为该对"的窗口上算——那是覆盖唯一够得着的集合。
"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
SEED=42
def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o
def em(p,it=200):
    ps=0.5; pi=0.5
    for _ in range(it):
        w=(p*pi/ps)/(p*pi/ps+(1-p)*(1-pi)/(1-ps)); n=float(np.clip(np.mean(w),1e-4,1-1e-4))
        if abs(n-pi)<1e-9: pi=n; break
        pi=n
    return pi
with threadpool_limits(1):
    t0=time.time(); TC.SEED=SEED
    A=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    Cd=pd.read_csv("/home/lmy/cic_probe/active_1108.csv",low_memory=False)
    cols=P.feature_columns(A)
    devs=sorted(set(IID.day_gate(A,"2021_11_02_Idle"))&set(IID.day_gate(Cd,"2021_11_08_Active")))
    le=LabelEncoder().fit(devs)
    def prep(df,sort=False):
        d=df[df.label.isin(devs)]
        if sort: d=d.sort_values(["label","window_start_epoch"])
        d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=SEED)
        if sort: d=d.sort_values(["label","window_start_epoch"])
        return np.asarray(P.clean_x(d,cols),dtype=float), le.transform(d.label), np.asarray(d.label)
    Xa,ya,_=prep(A); Xt,yt,gt=prep(Cd,True)
    bm=TC.make_model("xgboost",len(devs)); bm.fit(Xa,ya)
    Pt=cmeanM(bm.predict_proba(Xt),gt,10)
    o=np.argsort(-Pt,axis=1); t1,t2=o[:,0],o[:,1]
    print(f"{len(devs)} 类  平滑 k=10  outer macro={f1_score(yt,t1,average='macro'):.4f}",flush=True)
    print(f"\n{'类对':44s}{'可及窗':>7s}{'AUC':>8s}{'A现状':>8s}{'B中位':>8s}{'C EM':>8s}{'D上界':>8s}",flush=True)
    print("-"*92,flush=True)
    focus=[("GosundESP147FF9Plug","GosundESP1ACEE1Socket"),
           ("GosundESP032979Plug","GosundESP10ACD8Plug"),
           ("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2"),
           ("YutronPlug1","YutronPlug2"),
           ("TeckinPlug2","YutronPlug1"),
           ("GosundESP039AAFSocket","GosundESP0C3994Plug")]
    tot={"A":0.0,"B":0.0,"C":0.0,"D":0.0,"n":0}
    for a,b in focus:
        if a not in devs or b not in devs: continue
        i,j=le.transform([a])[0],le.transform([b])[0]
        mm=(((t1==i)&(t2==j))|((t1==j)&(t2==i)))&np.isin(yt,[i,j])
        n=int(mm.sum())
        if n<20: 
            print(f"{a[:20]+'|'+b[:20]:44s}{n:7d}   窗口太少",flush=True); continue
        s=Pt[mm][:,j]/np.clip(Pt[mm][:,i]+Pt[mm][:,j],1e-12,None)
        y=(yt[mm]==j).astype(int)
        if len(np.unique(y))<2: continue
        auc=roc_auc_score(y,s)
        Aa=((s>=0.5)==y).mean(); Bb=((s>=np.median(s))==y).mean()
        Cc=((s>=np.quantile(s,1-em(s)))==y).mean()
        Dd=max(((s>=c)==y).mean() for c in np.unique(np.quantile(s,np.linspace(0,1,201))))
        for k_,v_ in (("A",Aa),("B",Bb),("C",Cc),("D",Dd)): tot[k_]+=v_*n
        tot["n"]+=n
        print(f"{a[:20]+'|'+b[:20]:44s}{n:7d}{auc:8.4f}{Aa:8.4f}{Bb:8.4f}{Cc:8.4f}{Dd:8.4f}",flush=True)
    if tot["n"]:
        print("-"*92,flush=True)
        print(f"{'加权平均':44s}{tot['n']:7d}{'':8s}" +
              "".join(f"{tot[k]/tot['n']:8.4f}" for k in "ABCD"),flush=True)
    print(f"\n判读：若 D ≫ A 而 AUC 已很高 → 切点是真瓶颈，闸门应改用准确率而非 AUC。",flush=True)
    print(f"      若 C ≈ A → 无标签修法仍不够，需要更好的先验/切点估计。",flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s",flush=True)
