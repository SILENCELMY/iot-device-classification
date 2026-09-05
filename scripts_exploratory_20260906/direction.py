"""【探索性,非协议】源域方向：Idle→Active 是不是选反了。
Idle 下插座族几乎不可识别（F1 0.16–0.56），Active 下近乎完美（0.997）。
在"看不见身份"的一侧训练，是否是 CIC 低分的主因。"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
SEED=42
PLUGS=["GosundESP10ACD8Plug","GosundESP1ACEE1Socket","GosundESP039AAFSocket",
       "GlobeLampESPB1680C","YutronPlug2","GosundESP0C3994Plug","YutronPlug1",
       "TeckinPlug1","TeckinPlug2","GosundESP032979Plug","GosundESP147FF9Plug"]
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
    F={}
    for tag,f,day in (("I02","idle_1102","2021_11_02_Idle"),("A02","active_1102","2021_11_02_Active"),
                      ("I03","idle_1103","2021_11_03_Idle"),("A03","active_1103","2021_11_03_Active"),
                      ("I08","idle_1108","2021_11_08_Idle"),("A08","active_1108","2021_11_08_Active")):
        F[tag]=(pd.read_csv(f"/home/lmy/cic_probe/{f}.csv",low_memory=False),day)
    cols=P.feature_columns(F["I02"][0])
    devs=sorted(set.intersection(*[set(IID.day_gate(d,dy)) for d,dy in F.values()]))
    le=LabelEncoder().fit(devs); pl=[c for c in PLUGS if c in devs]
    pli=le.transform(pl)
    print(f"{len(devs)} 类（六个文件的交集）  插座族 {len(pl)} 台  {len(cols)} 列\n",flush=True)
    def run(s,t,tag):
        ds,_=F[s]; dt,_=F[t]
        a=P.sample_balanced(ds[ds.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=SEED)
        b=dt[dt.label.isin(devs)].sort_values(["label","window_start_epoch"])
        b=P.sample_balanced(b,max_rows=IID.MAX_ROWS,random_state=SEED).sort_values(["label","window_start_epoch"])
        Xs=np.asarray(P.clean_x(a,cols),dtype=float); ys=le.transform(a.label)
        Xt=np.asarray(P.clean_x(b,cols),dtype=float); yt=le.transform(b.label); gt=np.asarray(b.label)
        m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys)
        Pm=m.predict_proba(Xt)
        p1=Pm.argmax(1); p2=cmeanM(Pm,gt,10).argmax(1)
        fa=f1_score(yt,p1,average=None,labels=np.arange(len(devs)))
        fb=f1_score(yt,p2,average=None,labels=np.arange(len(devs)))
        print(f"  {tag:26s} macro={fa.mean():.4f}  +平滑={fb.mean():.4f}   "
              f"插座族 macro={fa[pli].mean():.4f} → {fb[pli].mean():.4f}   "
              f"其余={fa[[i for i in range(len(devs)) if i not in pli]].mean():.4f}",flush=True)
    print("=== 现有方向：在 Idle 上训练 ===",flush=True)
    run("I02","A02","I02 → A02  同日")
    run("I02","A08","I02 → A08  跨日")
    run("I02","I08","I02 → I08  同状态跨日")
    print("\n=== 反过来：在 Active 上训练 ===",flush=True)
    run("A02","I02","A02 → I02  同日")
    run("A02","A08","A02 → A08  同状态跨日")
    run("A02","I08","A02 → I08  跨日跨状态")
    print("\n=== 两侧都用（源 = Idle+Active 各半）===",flush=True)
    both=pd.concat([F["I02"][0],F["A02"][0]],ignore_index=True)
    F["B02"]=(both,None)
    a=P.sample_balanced(both[both.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=SEED)
    for tgt,tag in (("A08","B02 → A08"),("I08","B02 → I08")):
        dt,_=F[tgt]
        b=dt[dt.label.isin(devs)].sort_values(["label","window_start_epoch"])
        b=P.sample_balanced(b,max_rows=IID.MAX_ROWS,random_state=SEED).sort_values(["label","window_start_epoch"])
        Xs=np.asarray(P.clean_x(a,cols),dtype=float); ys=le.transform(a.label)
        Xt=np.asarray(P.clean_x(b,cols),dtype=float); yt=le.transform(b.label); gt=np.asarray(b.label)
        m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys); Pm=m.predict_proba(Xt)
        fa=f1_score(yt,Pm.argmax(1),average=None,labels=np.arange(len(devs)))
        fb=f1_score(yt,cmeanM(Pm,gt,10).argmax(1),average=None,labels=np.arange(len(devs)))
        print(f"  {tag:26s} macro={fa.mean():.4f}  +平滑={fb.mean():.4f}   "
              f"插座族 macro={fa[pli].mean():.4f} → {fb[pli].mean():.4f}   "
              f"其余={fa[[i for i in range(len(devs)) if i not in pli]].mean():.4f}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
