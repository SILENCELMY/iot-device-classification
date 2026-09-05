"""【探索性,非协议】CIC 1102：同域 AUC 在【时间块划分】下还剩多少。只读 1102。
随机划分 vs 时间块划分（前4块训/第5块测，沿用 run_histint 约定）对照。"""
import os, sys
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
FOCUS=[("GosundESP10ACD8Plug","GosundESP147FF9Plug"),
       ("GosundESP032979Plug","GosundESP1ACEE1Socket"),
       ("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2"),
       ("YutronPlug1","YutronPlug2"),
       ("TeckinPlug1","TeckinPlug2"),
       ("AMCRESTWiFiCamera","iRobotRoomba")]
with threadpool_limits(1):
    TC.SEED=42
    src=pd.read_csv(IDLE,low_memory=False); tgt=pd.read_csv(ACTIVE,low_memory=False)
    cols=P.feature_columns(src); lab="label" if "label" in src.columns else "device"
    assert "window_start" not in cols, "window_start 不该在特征列里"
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    full=np.arange(len(cols))

    def two_splits(df, a, b):
        """返回 (随机划分AUC, 时间块划分AUC)。不做 sample_balanced，用全量窗口保时间完整。"""
        d = df[df[lab].isin([a,b])].sort_values("window_start").reset_index(drop=True)
        if d[lab].nunique() < 2: return None, None
        X = np.asarray(P.clean_x(d, cols), dtype=float); y = np.asarray(d[lab])
        i1,i2 = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42, stratify=y)
        rnd = TC.pair_auc(X[i1],y[i1],X[i2],y[i2],a,b,full)
        blk = TC.time_blocks(np.asarray(d["window_start_epoch"]))
        tr, te = blk<4, blk==4
        tmp = None
        if len(np.unique(y[tr]))==2 and len(np.unique(y[te]))==2:
            tmp = TC.pair_auc(X[tr],y[tr],X[te],y[te],a,b,full)
        return rnd, tmp

    print(f"{len(devs)} 类  {len(cols)} 列", flush=True)
    hdr=f"{'类对':46s} {'Idle随机':>9s} {'Idle时块':>9s} {'Act随机':>9s} {'Act时块':>9s} {'跨域':>8s}"
    print(hdr, flush=True); print("-"*len(hdr), flush=True)
    s=P.sample_balanced(src[src[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(tgt[tgt[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
    ys=np.asarray(s[lab]); yt=np.asarray(t[lab])
    out=[]
    for a,b in FOCUS:
        if a not in devs or b not in devs: continue
        ir,it = two_splits(src,a,b); ar,at = two_splits(tgt,a,b)
        cr = TC.pair_auc(Xs,ys,Xt,yt,a,b,full)
        out.append({"pair":f"{a}|{b}","idle_rand":ir,"idle_tblk":it,
                    "act_rand":ar,"act_tblk":at,"cross":cr})
        f=lambda v:"     None" if v is None else f"{v:9.4f}"
        print(f"{a+'|'+b:46s} {f(ir)} {f(it)} {f(ar)} {f(at)} {f(cr)[1:]}", flush=True)
    pd.DataFrame(out).to_csv("/home/lmy/cic_probe/probe_temporal.csv",index=False)
    print("\n时块 ≈0.5 而随机 ≈1.0  →  同域高分是捕获内时间轨迹泄漏，不是设备身份", flush=True)
