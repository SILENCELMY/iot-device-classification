"""【探索性,非协议】纯覆盖（不带掩码）：base top-2 命中即用逐类对二分类器重判。只读 1102。
选对口径只用源域信息（源域内时间块 AUC），不看目标标签。"""
import os, sys, time, itertools
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
with threadpool_limits(1):
    t0=time.time()
    src=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    tgt=pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
    cols=P.feature_columns(src)
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    le=LabelEncoder().fit(devs)
    d0=src[src.label.isin(devs)].sort_values("window_start_epoch")
    blk=TC.time_blocks(np.asarray(d0["window_start_epoch"]))
    res=[]
    for seed in (42,43,44):
        TC.SEED=seed
        s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        t=P.sample_balanced(tgt[tgt.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
        ys=le.transform(s.label); yt=le.transform(t.label)
        base=TC.make_model("xgboost",len(devs)); base.fit(Xs,ys)
        pp=base.predict_proba(Xt); order=np.argsort(-pp,axis=1)
        top1,top2=order[:,0],order[:,1]
        f_base=f1_score(yt,top1,average="macro"); a_base=(top1==yt).mean()
        # 源域内时间块 AUC（只用源域，合法的选对依据）
        Xall=np.asarray(P.clean_x(d0,cols),dtype=float); yall=le.transform(d0.label)
        tr,te=blk<4,blk==4
        need=set(map(tuple,np.sort(np.c_[top1,top2],axis=1)))
        src_auc={}; pairmods={}
        for (i,j) in need:
            mi=np.isin(yall,[i,j]); Xi,yi=Xall[mi&tr],yall[mi&tr]; Xj,yj=Xall[mi&te],yall[mi&te]
            if len(np.unique(yi))<2 or len(np.unique(yj))<2: continue
            mm=TC.make_model("xgboost",2); mm.fit(Xi,(yi==j).astype(int))
            try: src_auc[(i,j)]=roc_auc_score((yj==j).astype(int), mm.predict_proba(Xj)[:,1])
            except Exception: continue
            ms=np.isin(ys,[i,j])
            if len(np.unique(ys[ms]))<2: continue
            pm=TC.make_model("xgboost",2); pm.fit(Xs[ms],(ys[ms]==j).astype(int))
            pairmods[(i,j)]=pm
        for thr,nm in ((0.0,"全部覆盖"),(0.95,"源域AUC>=0.95才覆盖"),(0.99,"源域AUC>=0.99才覆盖")):
            pred=top1.copy(); n_ov=0
            for (i,j),pm in pairmods.items():
                if src_auc.get((i,j),0.0) < thr: continue
                m=((top1==i)&(top2==j))|((top1==j)&(top2==i))
                if not m.any(): continue
                _p=np.asarray(pm.predict(Xt[m]))
                if _p.ndim>1: _p=_p.argmax(axis=1)
                pred[m]=np.where(_p.ravel()==1, j, i); n_ov+=int(m.sum())
            res.append({"seed":seed,"arm":nm,"macro_f1":f1_score(yt,pred,average="macro"),
                        "acc":(pred==yt).mean(),"n_override":n_ov,"n_pairs":len(pairmods)})
            print(f"  seed{seed} {nm:22s} macroF1={res[-1]['macro_f1']:.4f} acc={res[-1]['acc']:.4f} "
                  f"覆盖窗口 {n_ov}/{len(yt)}",flush=True)
        res.append({"seed":seed,"arm":"base(无覆盖)","macro_f1":f_base,"acc":a_base,
                    "n_override":0,"n_pairs":len(pairmods)})
        print(f"  seed{seed} {'base(无覆盖)':22s} macroF1={f_base:.4f} acc={a_base:.4f}"
              f"   候选类对 {len(pairmods)}   {time.time()-t0:.0f}s",flush=True)
    R=pd.DataFrame(res); R.to_csv("/home/lmy/cic_probe/pure_override.csv",index=False)
    print("\n=== 汇总（3 seed 均值）===",flush=True)
    g=R.groupby("arm")[["macro_f1","acc"]].mean()
    b=g.loc["base(无覆盖)"]
    g["Δmacro"]=(g.macro_f1-b.macro_f1); g["Δacc"]=(g.acc-b.acc)
    print(g.round(4).to_string(),flush=True)
