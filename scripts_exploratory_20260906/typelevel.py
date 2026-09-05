"""【探索性,非协议】按【设备型号】而非【物理实例】评估 CIC。
理由：11 台 ESP 插座同芯片同固件，实例级身份需物理层指纹，流量统计原理上做不到；
且现实用途（分段/策略/清点/异常基线）全是型号级。实例级标签是数据集产物，不是任务定义。
两种合并粒度都报，并与实例级并列。"""
import os, sys, time, re
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
# 型号映射：只合并"同型号的多个物理实例"，不跨型号、不跨功能
def to_type_fine(s):          # 细粒度：Plug 与 Socket 视为不同型号
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "TeckinPlug"
    if re.match(r"YutronPlug\d$", s):      return "YutronPlug"
    if re.match(r"AmazonAlexaEchoDot\d$",s):return "AmazonAlexaEchoDot"
    return s
def to_type_coarse(s):        # 粗粒度：Gosund 全系合一
    t=to_type_fine(s)
    return "GosundPlugSocket" if t in ("GosundPlug","GosundSocket") else t
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
    a08 =pd.read_csv("/home/lmy/cic_probe/active_1108.csv",low_memory=False)
    a02 =pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
    cols=P.feature_columns(idle)
    devs=sorted(set(IID.day_gate(idle,"2021_11_02_Idle"))&set(IID.day_gate(a02,"2021_11_02_Active"))
                &set(IID.day_gate(a08,"2021_11_08_Active")))
    for name,fn in (("实例级（现状）",lambda s:s),("型号级 细",to_type_fine),("型号级 粗",to_type_coarse)):
        lab=[fn(d) for d in devs]
        classes=sorted(set(lab))
        le=LabelEncoder().fit(classes)
        print(f"\n=== {name}：{len(classes)} 类（实例 {len(devs)} 台）===",flush=True)
        if name!="实例级（现状）":
            from collections import Counter
            merged={k:v for k,v in Counter(lab).items() if v>1}
            print(f"    合并：{merged}",flush=True)
        def run(src,sday,tgt,tday,tag):
            s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=SEED)
            t=tgt[tgt.label.isin(devs)].sort_values(["label","window_start_epoch"])
            t=P.sample_balanced(t,max_rows=IID.MAX_ROWS,random_state=SEED).sort_values(["label","window_start_epoch"])
            Xs=np.asarray(P.clean_x(s,cols),dtype=float); ys=le.transform([fn(x) for x in s.label])
            Xt=np.asarray(P.clean_x(t,cols),dtype=float); yt=le.transform([fn(x) for x in t.label])
            gt=np.asarray(t.label)      # 分组仍按物理流（MAC），不用合并后的标签
            m=TC.make_model("xgboost",len(classes)); m.fit(Xs,ys)
            Pm=m.predict_proba(Xt)
            f=f1_score(yt,Pm.argmax(1),average="macro")
            fs=f1_score(yt,cmeanM(Pm,gt,10).argmax(1),average="macro")
            print(f"    {tag:26s} macro={f:.4f}  acc={accuracy_score(yt,Pm.argmax(1)):.4f}   "
                  f"+平滑k10 macro={fs:.4f}",flush=True)
            return f1_score(yt,cmeanM(Pm,gt,10).argmax(1),average=None,labels=np.arange(len(classes)))
        run(idle,"I",a02,"A","1102Idle → 1102Active")
        F=run(idle,"I",a08,"A","1102Idle → 1108Active")
        low=[(classes[i],F[i]) for i in np.argsort(F)[:6]]
        print(f"    最差 6 类（跨日+平滑）：" + "  ".join(f"{c[:22]} {v:.3f}" for c,v in low),flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
