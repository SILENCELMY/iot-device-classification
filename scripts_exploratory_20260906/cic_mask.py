"""【探索性,非协议】可行性：存不存在一个列子集，能把 Gosund 反转对救回来。只读 1102。
注意：这里的子集是在同一天上挑的，属于"存在性"测试，不是可迁移性测试。"""
import os, sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
with threadpool_limits(1):
    TC.SEED=42
    src=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    tgt=pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
    cols=P.feature_columns(src); lab="label"
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    s=P.sample_balanced(src[src[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(tgt[tgt[lab].isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
    ys=np.asarray(s[lab]); yt=np.asarray(t[lab])
    fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
    def auc(a,b,drop):
        k=np.array([idx[c] for c in cols if c not in drop])
        return None if len(k)==0 else TC.pair_auc(Xs,ys,Xt,yt,a,b,k)
    IA=set(c for c in cols if "ia_" in c or "interarrival" in c)
    for a,b in [("GosundESP10ACD8Plug","GosundESP147FF9Plug"),
                ("GosundESP032979Plug","GosundESP1ACEE1Socket")]:
        print(f"\n{'='*76}\n{a} vs {b}", flush=True)
        SETS=[("全 61 列（基线）",set()),
              ("删 interarrival 族",set(fams["interarrival"])),
              ("删 up 族",set(fams["up"])),
              ("删 interarrival ∪ up",set(fams["interarrival"])|set(fams["up"])),
              (f"删所有时序列（{len(IA)} 条：*ia_* / interarrival*）",IA)]
        for nm,dr in SETS:
            v=auc(a,b,dr)
            print(f"  {nm:44s} 剩 {len(cols)-len(dr&set(cols)):3d} 列  AUC={v:.4f}", flush=True)
        # 贪心逐列删除（存在性上界）
        cur=set(); best=auc(a,b,cur); hist=[]
        for step in range(12):
            cand=[(auc(a,b,cur|{c}),c) for c in cols if c not in cur]
            cand=[(v,c) for v,c in cand if v is not None]
            v,c=max(cand)
            if v<=best+1e-6: break
            cur.add(c); best=v; hist.append((c,v))
        print(f"  贪心逐列删除 {len(cur)} 列后 AUC={best:.4f}   （存在性上界，同日挑选）", flush=True)
        for c,v in hist: print(f"      -{c:24s} → {v:.4f}", flush=True)
