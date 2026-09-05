"""【探索性,非协议】Gosund 反转对：哪根轴在分它们，Idle 与 Active 的方向是否相反。只读 1102。"""
import os, sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_two_channel as TC
src=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
tgt=pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
cols=P.feature_columns(src); lab="label"
for a,b in [("GosundESP10ACD8Plug","GosundESP147FF9Plug"),
            ("GosundESP032979Plug","GosundESP1ACEE1Socket")]:
    print(f"\n{'='*88}\n{a}  vs  {b}", flush=True)
    rec=[]
    for name,df in (("Idle",src),("Active",tgt)):
        d=df[df[lab].isin([a,b])]
        X=P.clean_x(d,cols); y=np.asarray(d[lab])
        for c in cols:
            va,vb=X.loc[y==a,c].to_numpy(),X.loc[y==b,c].to_numpy()
            if len(va)<20 or len(vb)<20: continue
            u=mannwhitneyu(va,vb,alternative="two-sided").statistic/(len(va)*len(vb))  # =P(a>b)
            rec.append({"dom":name,"col":c,"P_a_gt_b":u,"mean_a":va.mean(),"mean_b":vb.mean()})
    R=pd.DataFrame(rec).pivot(index="col",columns="dom",values="P_a_gt_b")
    R["|Idle-0.5|"]=(R.Idle-0.5).abs()
    R["方向反转"]=np.sign(R.Idle-0.5)!=np.sign(R.Active-0.5)
    top=R.sort_values("|Idle-0.5|",ascending=False).head(10)
    print(f"  Idle 里最能分的 10 列（P(a>b)，0.5=不可分，>0.5 表示 {a[:14]} 更大）", flush=True)
    print(top[["Idle","Active","方向反转"]].round(4).to_string(), flush=True)
    strong=R[R["|Idle-0.5|"]>=0.25]
    print(f"  Idle 强判别列（|P-0.5|>=0.25）{len(strong)} 条，其中方向在 Active 反转的 "
          f"{int(strong['方向反转'].sum())} 条 ({strong['方向反转'].mean()*100:.0f}%)", flush=True)
    m=pd.DataFrame(rec)
    for c in ("packet_count","byte_count"):
        if c in cols:
            q=m[m.col==c].set_index("dom")
            print(f"  {c:14s} Idle: {a[:14]}={q.loc['Idle','mean_a']:.1f} {b[:14]}={q.loc['Idle','mean_b']:.1f}"
                  f"   Active: {a[:14]}={q.loc['Active','mean_a']:.1f} {b[:14]}={q.loc['Active','mean_b']:.1f}", flush=True)
