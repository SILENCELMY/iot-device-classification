"""【探索性,非协议】困难对到底能不能救：
  路1 非线性模型 —— LR（现有诊断量） vs RF/xgboost 的跨域逐类对 AUC
  路2 观测时长   —— 连续 k 个窗的概率平均后 AUC 随 k 的变化
UNSW 三个跨日任务 + CIC 1102。"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"
KS=[1,2,3,5,10,20,40]

def prob_auc(mk, Xs, ys01, Xt, yt01):
    m=mk(); m.fit(Xs, ys01)
    p=m.predict_proba(Xt)[:,1]
    return roc_auc_score(yt01, p), p

def agg_auc(p, y01, order, k):
    """按目标域时间顺序，每类内部连续 k 窗取概率均值。"""
    out_p, out_y = [], []
    for cls in (0,1):
        idx=order[y01[order]==cls]
        for st in range(0, len(idx)-k+1, k):
            ch=idx[st:st+k]
            out_p.append(p[ch].mean()); out_y.append(cls)
    if len(set(out_y))<2 or len(out_y)<10: return None, len(out_y)
    return roc_auc_score(out_y, out_p), len(out_y)

def run(tag, src, tgt, sday, tday, pairs):
    cols=P.feature_columns(src)
    devs=sorted(set(IID.day_gate(src,sday)) & set(IID.day_gate(tgt,tday)))
    s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    print(f"\n{'='*92}\n{tag}   {len(devs)} 类  {len(cols)} 列",flush=True)
    for a,b in pairs:
        if a not in devs or b not in devs:
            print(f"  {a}|{b}: 有一侧未过门槛",flush=True); continue
        ss=s[s.label.isin([a,b])]
        tt=tgt[tgt.label.isin([a,b])].sort_values("window_start_epoch")
        if ss.label.nunique()<2 or tt.label.nunique()<2: continue
        Xs=np.asarray(P.clean_x(ss,cols),dtype=float); ys01=(ss.label.to_numpy()==b).astype(int)
        Xt=np.asarray(P.clean_x(tt,cols),dtype=float); yt01=(tt.label.to_numpy()==b).astype(int)
        order=np.arange(len(yt01))          # 已按时间排序
        print(f"\n  {a} | {b}   源 {len(ys01)} 窗 / 目标 {len(yt01)} 窗",flush=True)
        mods={"LR(现诊断量)": lambda: make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=2000, C=1.0)),
              "RandomForest": lambda: TC.make_model("rf",2),
              "xgboost":      lambda: TC.make_model("xgboost",2)}
        for nm,mk in mods.items():
            try: a1,p = prob_auc(mk, Xs, ys01, Xt, yt01)
            except Exception as e: print(f"    {nm:14s} 失败 {e}",flush=True); continue
            row=[]
            for k in KS:
                v,n=agg_auc(p,yt01,order,k)
                row.append(f"k={k}:{'  --  ' if v is None else f'{v:.4f}'}")
            print(f"    {nm:14s} " + "  ".join(row),flush=True)

with threadpool_limits(1):
    t0=time.time(); TC.SEED=42
    BELKIN=[("BelkinWemoMotion","BelkinWemoSwitch"),("NetatmoWelcome","TribySpeaker")]
    for sd,td in [("16-09-23","16-09-30"),("16-09-30","16-10-12"),("16-09-23","16-10-12")]:
        run(f"UNSW {sd} → {td}", pd.read_csv(UNSW%sd,low_memory=False),
            pd.read_csv(UNSW%td,low_memory=False), sd, td, BELKIN)
    CICP=[("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2"),("YutronPlug1","YutronPlug2"),
          ("TeckinPlug1","YutronPlug2"),("GlobeLampESPB1680C","GosundESP039AAFSocket")]
    run("CIC 1102 Idle → Active", pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False),
        pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False),
        "2021_11_02_Idle","2021_11_02_Active", CICP)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
