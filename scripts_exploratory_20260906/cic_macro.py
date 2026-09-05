"""【探索性,非协议】CIC 1102：删 down/burst 族到底动不动 macro-F1。烧毁隔离：只读 1102。"""
import os, sys
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
IDLE="/home/lmy/cic_probe/idle_1102.csv"; ACTIVE="/home/lmy/cic_probe/active_1102.csv"
assert "1102" in IDLE and "1102" in ACTIVE, "烧毁隔离违规"
with threadpool_limits(1):
    src=pd.read_csv(IDLE,low_memory=False); tgt=pd.read_csv(ACTIVE,low_memory=False)
    cols=P.feature_columns(src); lab="label" if "label" in src.columns else "device"
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    fams=TC.derive_families(cols)
    le=LabelEncoder().fit(devs)
    print(f"{len(devs)} 类  {len(cols)} 列  down={len(fams['down'])} burst={len(fams['burst'])}", flush=True)
    ARMS=[("full",set()),("drop_down",set(fams["down"])),("drop_burst",set(fams["burst"])),
          ("drop_down_burst",set(fams["down"])|set(fams["burst"]))]
    rows=[]
    for seed in (42,43,44):
        TC.SEED=seed
        s=P.sample_balanced(src[src[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        t=P.sample_balanced(tgt[tgt[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        ys=le.transform(np.asarray(s[lab])); yt=le.transform(np.asarray(t[lab]))
        for arm,drop in ARMS:
            use=[c for c in cols if c not in drop]
            Xs=np.asarray(P.clean_x(s,use),dtype=float); Xt=np.asarray(P.clean_x(t,use),dtype=float)
            for mn in ("rf","xgboost","lightgbm"):
                m=TC.make_model(mn,len(devs)); m.fit(Xs,ys); pr=m.predict(Xt)
                rows.append({"seed":seed,"arm":arm,"model":mn,"n_cols":len(use),
                             "macro_f1":f1_score(yt,pr,average="macro"),"acc":accuracy_score(yt,pr)})
                print(f"  seed{seed} {arm:16s} {mn:9s} cols={len(use):3d} "
                      f"macroF1={rows[-1]['macro_f1']:.4f} acc={rows[-1]['acc']:.4f}", flush=True)
    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/probe_macro.csv",index=False)
    print("\n=== 按 arm 汇总 ===", flush=True)
    print(R.groupby("arm")[["macro_f1","acc"]].agg(["mean","std"]).round(4).to_string(), flush=True)
    bb=R.loc[R.groupby(["seed","arm"]).macro_f1.idxmax()]
    piv=bb.pivot(index="seed",columns="arm",values="macro_f1")
    print("\n=== best_base（每 seed 取三模型最大 macro）===", flush=True)
    print(piv.round(4).to_string(), flush=True)
    for a in ("drop_down","drop_burst","drop_down_burst"):
        dd=(piv[a]-piv["full"])
        print(f"  Δ({a} - full) 逐 seed {dd.round(4).tolist()}  均值 {dd.mean():+.4f}", flush=True)
    print("\n=== AMCREST 单类 F1（full vs drop_down，best_base 口径）===", flush=True)
