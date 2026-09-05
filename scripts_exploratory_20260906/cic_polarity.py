"""【探索性,非协议】极性可迁移性：跨域 AUC 的"反转"在 1102/1103/1108 三个配对日间是否同向。
预注册预测（写在跑之前）：Gosund10ACD8|147FF9 在 1103、1108 上的跨域 AUC 应落在 0.10–0.25。"""
import os, sys, itertools
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
DAYS={"1102":"2021_11_02","1103":"2021_11_03","1108":"2021_11_08"}
FOCUS=[("GosundESP10ACD8Plug","GosundESP147FF9Plug"),
       ("GosundESP032979Plug","GosundESP1ACEE1Socket"),
       ("AmazonAlexaEchoDot1","AmazonAlexaEchoDot2"),
       ("TeckinPlug1","TeckinPlug2"),
       ("AMCRESTWiFiCamera","iRobotRoomba")]
with threadpool_limits(1):
    TC.SEED=42
    per={}
    for tag,d in DAYS.items():
        src=pd.read_csv(f"/home/lmy/cic_probe/idle_{tag}.csv",low_memory=False)
        tgt=pd.read_csv(f"/home/lmy/cic_probe/active_{tag}.csv",low_memory=False)
        cols=P.feature_columns(src)
        devs=sorted(set(IID.day_gate(src,d+"_Idle")) & set(IID.day_gate(tgt,d+"_Active")))
        s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
        t=P.sample_balanced(tgt[tgt.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
        Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
        ys=np.asarray(s.label); yt=np.asarray(t.label); full=np.arange(len(cols))
        r={}
        for a,b in itertools.combinations(devs,2):
            v=TC.pair_auc(Xs,ys,Xt,yt,a,b,full)
            if v is not None: r[f"{a}|{b}"]=v
        per[tag]=r
        print(f"{tag}: {len(devs)} 台设备  {len(r)} 对可测  反转对(<0.5) {sum(v<0.5 for v in r.values())}",flush=True)
    A=pd.DataFrame(per).dropna()
    A.to_csv("/home/lmy/cic_probe/polarity_cross_auc.csv")
    print(f"\n三天都可测的类对 {len(A)} 对", flush=True)
    print("\n=== 预注册的焦点对 ===", flush=True)
    for a,b in FOCUS:
        k=f"{a}|{b}"
        if k in A.index: print(f"  {k:48s} " + "  ".join(f"{t}={A.loc[k,t]:.4f}" for t in DAYS), flush=True)
        else: print(f"  {k:48s} 某天不可测", flush=True)
    print("\n=== 极性一致性（全体类对）===", flush=True)
    tags=list(DAYS)
    for i in range(len(tags)):
        for j in range(i+1,len(tags)):
            x,y=A[tags[i]],A[tags[j]]
            agree=((x<0.5)==(y<0.5)).mean()
            print(f"  {tags[i]} vs {tags[j]}: Pearson r({tags[i]}-0.5, {tags[j]}-0.5) = "
                  f"{np.corrcoef(x-0.5,y-0.5)[0,1]:+.4f}   符号一致率 {agree*100:.1f}%", flush=True)
    inv0=A[A[tags[0]]<0.4]
    print(f"\n=== 1102 上明确反转(<0.4)的 {len(inv0)} 对，在另两天 ===", flush=True)
    if len(inv0): print(inv0.round(4).to_string(), flush=True)
    print("\n判读：符号一致率≈100% 且 r 高 → 极性可导出、可迁移，② 成立", flush=True)
    print("      符号一致率≈50%            → 逐日偶然，② 死", flush=True)
