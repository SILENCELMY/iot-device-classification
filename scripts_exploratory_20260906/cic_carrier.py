"""【探索性,非协议】困难簇的真正问题：判别力的【载体族】是不是决定了跨域能不能活。
和以往所有诊断相反：这里测"只用某族能分到多少"（载体），不是"删掉某族会怎样"（毒药）。
只读已烧的 1102。"""
import os, sys, itertools, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
TIMING={"interarrival","subwin"}   # 事前声明：时序通道（up/down 含 ia 列，单独看）
with threadpool_limits(1):
    t0=time.time(); TC.SEED=42
    src=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    tgt=pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
    cols=P.feature_columns(src)
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
    famidx={f:np.array([idx[c] for c in v]) for f,v in fams.items() if len(v)>0}
    full=np.arange(len(cols))
    # 跨域：源 Idle 拟合、目标 Active 评
    s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(tgt[tgt.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
    ys=np.asarray(s.label); yt=np.asarray(t.label)
    # 同域：Idle 内时间块划分（前4块训/第5块测）
    d0=src[src.label.isin(devs)].sort_values("window_start_epoch")
    blk=TC.time_blocks(np.asarray(d0["window_start_epoch"]))
    X0=np.asarray(P.clean_x(d0,cols),dtype=float); y0=np.asarray(d0.label)
    tr,te=blk<4, blk==4
    Xi,yi,Xj,yj = X0[tr],y0[tr],X0[te],y0[te]
    rows=[]; pairs=list(itertools.combinations(devs,2))
    for n,(a,b) in enumerate(pairs):
        inf = TC.pair_auc(Xi,yi,Xj,yj,a,b,full)
        cro = TC.pair_auc(Xs,ys,Xt,yt,a,b,full)
        if inf is None or cro is None: continue
        r={"pair":f"{a}|{b}","in_full":inf,"cross_full":cro}
        for f,ix in famidx.items():
            v=TC.pair_auc(Xi,yi,Xj,yj,a,b,ix)          # 同域单族 = 载体强度
            r["in_"+f]= np.nan if v is None else v
        rows.append(r)
        if (n+1)%150==0: print(f"  {n+1}/{len(pairs)}  {time.time()-t0:.0f}s",flush=True)
    R=pd.DataFrame(rows)
    fcols=[c for c in R.columns if c.startswith("in_") and c!="in_full"]
    # 载体 = 同域单族 |AUC-0.5| 最大的族
    dev_=(R[fcols]-0.5).abs()
    R["carrier"]=dev_.idxmax(axis=1).str.replace("in_","",regex=False)
    R["carrier_str"]=dev_.max(axis=1)
    R["inverted"]=R.cross_full<0.5
    R["hard"]=R.cross_full<0.7
    R.to_csv("/home/lmy/cic_probe/carrier.csv",index=False)
    print(f"\n可测类对 {len(R)}  同域中位 {R.in_full.median():.4f}  跨域中位 {R.cross_full.median():.4f}",flush=True)
    print("\n=== 按载体族分组：跨域还剩多少 ===",flush=True)
    g=R.groupby("carrier").agg(n=("pair","size"), 同域=("in_full","median"),
                               跨域=("cross_full","median"), 反转率=("inverted","mean"),
                               困难率=("hard","mean")).sort_values("跨域")
    print(g.round(4).to_string(),flush=True)
    print("\n=== 困难对(跨域<0.7) vs 正常对 的载体分布 ===",flush=True)
    ct=pd.crosstab(R.carrier,R.hard,normalize="columns")*100
    ct.columns=["正常对%","困难对%"]
    print(ct.round(1).to_string(),flush=True)
    print(f"\n时序族({'/'.join(sorted(TIMING))})当载体的类对：",flush=True)
    m=R.carrier.isin(TIMING)
    print(f"  是时序载体 {m.sum():3d} 对  跨域中位 {R[m].cross_full.median():.4f}  困难率 {R[m].hard.mean()*100:.1f}%",flush=True)
    print(f"  非时序载体 {(~m).sum():3d} 对  跨域中位 {R[~m].cross_full.median():.4f}  困难率 {R[~m].hard.mean()*100:.1f}%",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
