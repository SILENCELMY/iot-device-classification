"""【探索性,非协议】CIC 的 70% 是迁移丢的还是根本到不了：同域 36 类 vs 跨域 36 类。"""
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
    idle=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    act =pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
    a08 =pd.read_csv("/home/lmy/cic_probe/active_1108.csv",low_memory=False)
    cols=P.feature_columns(idle)
    devs=sorted(set(IID.day_gate(idle,"2021_11_02_Idle"))&set(IID.day_gate(act,"2021_11_02_Active"))
                &set(IID.day_gate(a08,"2021_11_08_Active")))
    le=LabelEncoder().fit(devs)
    print(f"{len(devs)} 类  {len(cols)} 列",flush=True)

    def indomain(df, day, tag):
        """同域：时间块划分，前 4 块训 / 第 5 块测（与项目约定一致）。"""
        d=df[df.label.isin(devs)].sort_values("window_start_epoch")
        blk=TC.time_blocks(np.asarray(d["window_start_epoch"]))
        X=np.asarray(P.clean_x(d,cols),dtype=float); y=le.transform(d.label)
        g=np.asarray(d.label)
        tr,te=blk<4,blk==4
        m=TC.make_model("xgboost",len(devs)); m.fit(X[tr],y[tr])
        Pm=m.predict_proba(X[te])
        p1=Pm.argmax(1)
        # 平滑版（同流因果，k=10）
        dte=d[te].sort_values(["label","window_start_epoch"])
        Xs2=np.asarray(P.clean_x(dte,cols),dtype=float); y2=le.transform(dte.label); g2=np.asarray(dte.label)
        Pm2=cmeanM(m.predict_proba(Xs2),g2,10)
        f=f1_score(y[te],p1,average="macro"); fs=f1_score(y2,Pm2.argmax(1),average="macro")
        print(f"  {tag:34s} macro={f:.4f}  acc={accuracy_score(y[te],p1):.4f}   "
              f"+平滑k10 macro={fs:.4f}",flush=True)
        return f1_score(y[te],p1,average=None,labels=np.arange(len(devs)))

    print("\n=== 同域（时间块留出，第 5 块）===",flush=True)
    F_idle=indomain(idle,"2021_11_02_Idle","Idle 1102 内部")
    F_act =indomain(act ,"2021_11_02_Active","Active 1102 内部")

    print("\n=== 跨域 ===",flush=True)
    def cross(src,sday,tgt,tday,tag):
        s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=SEED)
        t=tgt[tgt.label.isin(devs)].sort_values(["label","window_start_epoch"])
        t=P.sample_balanced(t,max_rows=IID.MAX_ROWS,random_state=SEED).sort_values(["label","window_start_epoch"])
        Xs=np.asarray(P.clean_x(s,cols),dtype=float); ys=le.transform(s.label)
        Xt=np.asarray(P.clean_x(t,cols),dtype=float); yt=le.transform(t.label); gt=np.asarray(t.label)
        m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys)
        Pm=m.predict_proba(Xt)
        f=f1_score(yt,Pm.argmax(1),average="macro")
        fs=f1_score(yt,cmeanM(Pm,gt,10).argmax(1),average="macro")
        print(f"  {tag:34s} macro={f:.4f}  acc={accuracy_score(yt,Pm.argmax(1)):.4f}   "
              f"+平滑k10 macro={fs:.4f}",flush=True)
        return f1_score(yt,Pm.argmax(1),average=None,labels=np.arange(len(devs)))
    F_x1=cross(idle,"Idle",act,"Active","1102Idle → 1102Active")
    F_x2=cross(idle,"Idle",a08,"Active","1102Idle → 1108Active")

    print("\n=== 逐类 F1 分布 ===",flush=True)
    D=pd.DataFrame({"类":le.classes_,"同域Idle":F_idle,"同域Active":F_act,
                    "跨域同日":F_x1,"跨域跨日":F_x2}).sort_values("同域Idle")
    print(D.round(3).to_string(index=False),flush=True)
    for c in ("同域Idle","同域Active","跨域同日","跨域跨日"):
        v=D[c].to_numpy()
        print(f"  {c}: 中位 {np.median(v):.3f}  <0.5 的类 {int((v<0.5).sum())}/{len(v)}  "
              f"<0.8 的类 {int((v<0.8).sum())}/{len(v)}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
