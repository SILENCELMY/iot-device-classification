"""【探索性,非协议】病因判定：困难/高AUC类对的错误，是"分不开"还是"切错地方"。
同一个源域二分类器、同一份排序，只换阈值：
  A 源域阈值(0.5)         —— 现状
  B 目标域中位数阈值       —— 假定目标域两类等量（本评估协议下成立，部署时需先估先验）
  C 目标域先验用 EM 估计   —— 完全无标签，可部署
  D 最优阈值(oracle)      —— 上界参考，不可部署
只读 CIC 1102 + UNSW。"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"

def em_prior(p, iters=100):
    """Saerens-Latinne-Decaestecker：从无标签目标分数估计正类先验。"""
    pi_s = 0.5; pi = 0.5
    for _ in range(iters):
        w = (p*pi/pi_s) / (p*pi/pi_s + (1-p)*(1-pi)/(1-pi_s))
        new = float(np.clip(w.mean(), 1e-4, 1-1e-4))
        if abs(new-pi) < 1e-8: pi=new; break
        pi = new
    return pi

def one(tag, src, tgt, sday, tday, pairs):
    cols=P.feature_columns(src)
    devs=sorted(set(IID.day_gate(src,sday)) & set(IID.day_gate(tgt,tday)))
    s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    print(f"\n{'='*100}\n{tag}",flush=True)
    print(f"  {'类对':46s} {'AUC':>7s} {'A源阈值':>8s} {'B中位':>8s} {'C EM':>8s} {'D上界':>8s} {'C-A':>7s}",flush=True)
    tot={"A":0,"B":0,"C":0,"D":0,"n":0}
    for a,b in pairs:
        if a not in devs or b not in devs: continue
        ss=s[s.label.isin([a,b])]; tt=tgt[tgt.label.isin([a,b])]
        if ss.label.nunique()<2 or tt.label.nunique()<2: continue
        Xs=np.asarray(P.clean_x(ss,cols),dtype=float); ys=(ss.label.to_numpy()==b).astype(int)
        Xt=np.asarray(P.clean_x(tt,cols),dtype=float); yt=(tt.label.to_numpy()==b).astype(int)
        m=TC.make_model("xgboost",2); m.fit(Xs,ys); p=m.predict_proba(Xt)[:,1]
        auc=roc_auc_score(yt,p)
        accA=((p>=0.5).astype(int)==yt).mean()
        accB=((p>=np.median(p)).astype(int)==yt).mean()
        pi=em_prior(p); thrC=np.quantile(p, 1-pi)
        accC=((p>=thrC).astype(int)==yt).mean()
        cand=np.unique(np.quantile(p,np.linspace(0,1,201)))
        accD=max((( p>=c).astype(int)==yt).mean() for c in cand)
        n=len(yt); tot["n"]+=n
        for k,v in (("A",accA),("B",accB),("C",accC),("D",accD)): tot[k]+=v*n
        print(f"  {a[:22]+'|'+b[:22]:46s} {auc:7.4f} {accA:8.4f} {accB:8.4f} {accC:8.4f} {accD:8.4f} "
              f"{accC-accA:+7.4f}",flush=True)
    if tot["n"]:
        print(f"  {'加权平均':46s} {'':7s} {tot['A']/tot['n']:8.4f} {tot['B']/tot['n']:8.4f} "
              f"{tot['C']/tot['n']:8.4f} {tot['D']/tot['n']:8.4f} {(tot['C']-tot['A'])/tot['n']:+7.4f}",flush=True)

with threadpool_limits(1):
    t0=time.time(); TC.SEED=42
    for sd,td in [("16-09-23","16-09-30"),("16-09-30","16-10-12"),("16-09-23","16-10-12")]:
        e=pd.read_csv(f"/home/lmy/cic_probe/unsw_err_{sd}_to_{td}.csv",index_col=0)
        import ast
        idx=[]
        for i in e.index:
            i=str(i)
            try:
                v=ast.literal_eval(i)
                v=tuple(v) if isinstance(v,(tuple,list)) else tuple(i.split("|"))
            except Exception:
                v=tuple(i.split("|"))
            idx.append(v)
        pr=[k for k in idx[:10] if len(k)==2]
        print(f"    解析出 {len(pr)} 个类对", flush=True)
        one(f"UNSW {sd} → {td}（错误最多的 10 对）",
            pd.read_csv(UNSW%sd,low_memory=False), pd.read_csv(UNSW%td,low_memory=False), sd, td, pr)
    car=pd.read_csv("/home/lmy/cic_probe/carrier.csv")
    cic=[("GlobeLampESPB1680C","GosundESP039AAFSocket"),
         ("GosundESP032979Plug","GosundESP10ACD8Plug"),
         ("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2"),
         ("TeckinPlug1","YutronPlug2"),("YutronPlug1","YutronPlug2"),
         ("GosundESP10ACD8Plug","GosundESP147FF9Plug")]
    one("CIC 1102 Idle → Active（错误最多的 6 对）",
        pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False),
        pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False),
        "2021_11_02_Idle","2021_11_02_Active", cic)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
