"""【探索性,非协议】CIC 1102：强反转对是"跨域翻转"还是"同域也反转"。只读 1102。"""
import os, sys, itertools
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
IDLE="/home/lmy/cic_probe/idle_1102.csv"; ACTIVE="/home/lmy/cic_probe/active_1102.csv"
assert "1102" in IDLE and "1102" in ACTIVE, "烧毁隔离违规"
FOCUS=[("GosundESP032979Plug","GosundESP1ACEE1Socket"),
       ("GosundESP10ACD8Plug","GosundESP147FF9Plug"),
       ("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2"),
       ("TeckinPlug1","TeckinPlug2"),          # 对照：同型号但基线好
       ("YutronPlug1","YutronPlug2"),          # 对照
       ("AMCRESTWiFiCamera","iRobotRoomba")]   # 对照：override 命中对
with threadpool_limits(1):
    TC.SEED=42
    src=pd.read_csv(IDLE,low_memory=False); tgt=pd.read_csv(ACTIVE,low_memory=False)
    cols=P.feature_columns(src); lab="label" if "label" in src.columns else "device"
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    s=P.sample_balanced(src[src[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(tgt[tgt[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
    ys=np.asarray(s[lab]); yt=np.asarray(t[lab])
    full=np.arange(len(cols))
    # 同域：Idle 内部二分（半训半测），以及 Active 内部二分
    def in_domain(X,y,a,b,seed=42):
        m=np.isin(y,[a,b]);  Xa,ya=X[m],y[m]
        if len(np.unique(ya))<2: return None
        i1,i2=train_test_split(np.arange(len(ya)),test_size=0.5,random_state=seed,stratify=ya)
        return TC.pair_auc(Xa[i1],ya[i1],Xa[i2],ya[i2],a,b,full)
    print(f"{len(devs)} 类  {len(cols)} 列\n", flush=True)
    print(f"{'类对':52s} {'Idle内':>8s} {'Active内':>9s} {'Idle→Active':>12s}", flush=True)
    out=[]
    for a,b in FOCUS:
        if a not in devs or b not in devs:
            print(f"{a+'|'+b:52s}  —— 有一侧未过门槛", flush=True); continue
        ii=in_domain(Xs,ys,a,b); aa=in_domain(Xt,yt,a,b)
        cc=TC.pair_auc(Xs,ys,Xt,yt,a,b,full)
        out.append({"pair":f"{a}|{b}","idle_in":ii,"active_in":aa,"cross":cc})
        f=lambda v: "  None " if v is None else f"{v:8.4f}"
        print(f"{a+'|'+b:52s} {f(ii)} {f(aa)}   {f(cc)}", flush=True)
    pd.DataFrame(out).to_csv("/home/lmy/cic_probe/probe_invert.csv",index=False)
    print("\n判读：同域高(>0.9) + 跨域低(<0.2) = 纯跨域翻转，方法管辖内", flush=True)
    print("      同域也低              = 不可识别或标签映射问题，方法管辖外", flush=True)
