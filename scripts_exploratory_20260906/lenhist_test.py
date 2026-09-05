"""【探索性,非协议】帧长直方图特征族有没有增量？—— 最直接的一测，不带方法机制。

只问：在困难簇（网关三类）上，183 列 vs 183+69 列，跨轮次 macro 差多少。
单列 F 比很弱（最高 0.32），但弱特征可以合起来有用；也可能完全没用。
两种都要能看出来，所以同时报：
  · 网关三类（方法真正作用处）与 5 类
  · 三个 outer 单元分开报（pos_R5 是位移最大的）
  · 三个模型 × 3 seed，并按 best_base 取
另报 inner（R2R3→R4）上的同一对比 —— 若 inner 涨而 outer 不涨，就是过拟合源域。
"""
from __future__ import annotations
import os, sys, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import run_two_channel as TC

NEW89 = REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
LENH  = REPO+"/results/feature_expansion_20260905/features_lenhist_w10.csv"
KEYS  = ["label","round","source_file","window_id"]
GATEWAY=["Light_T1","Light_XM","Sensor"]
UNITS=[("inner_R4",["R2","R3"],"R4"),
       ("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
MODELS=["rf","xgboost","lightgbm"]; SEEDS=[42,43,44]

def main():
    t0=time.time()
    with threadpool_limits(1):
        d=TC.Data(); new=pd.read_csv(NEW89); lh=pd.read_csv(LENH)
        df=d.df.merge(new,on=KEYS,how="inner").merge(lh,on=KEYS,how="inner")
        assert len(df)==len(d.df), f"行数变了 {len(df)} vs {len(d.df)}"
        base=[c for c in TC.feature_columns(df) if not c.startswith("lenhist_")]
        lcols=[c for c in df.columns if c.startswith("lenhist_")]
        POOLS={"183":base, "183+lenhist":base+lcols}
        L5=sorted(df.label.unique()); i5={c:n for n,c in enumerate(L5)}
        GIDX=[i5[c] for c in GATEWAY]
        print(f"{len(df)} 行   183 池 {len(base)} 列   lenhist {len(lcols)} 列",flush=True)

        rows=[]
        for uname,src,tgt in UNITS:
            s=df[df["round"].isin(src)].sort_values(["label","window_start"],kind="mergesort")
            t=df[df["round"]==tgt].sort_values(["label","window_start"],kind="mergesort")
            ys=np.array([i5[x] for x in s.label]); yt=np.array([i5[x] for x in t.label])
            gm=np.isin(yt,GIDX)
            for pname,cols in POOLS.items():
                Xs=np.asarray(TC.clean_x(s,cols),dtype=float)
                Xt=np.asarray(TC.clean_x(t,cols),dtype=float)
                # 网关三类：只用网关子集训练与评测
                ms=np.isin(ys,GIDX)
                loc={c:k for k,c in enumerate(GIDX)}
                yg_s=np.array([loc[v] for v in ys[ms]]); yg_t=np.array([loc[v] for v in yt[gm]])
                for seed in SEEDS:
                    TC.SEED=seed
                    for mn in MODELS:
                        m5=TC.make_model(mn,len(L5))
                        if m5 is None: continue
                        m5.fit(Xs,ys)
                        f5=f1_score(yt,m5.predict(Xt),average="macro",labels=np.arange(len(L5)))
                        m3=TC.make_model(mn,len(GIDX)); m3.fit(Xs[ms],yg_s)
                        f3=f1_score(yg_t,m3.predict(Xt[gm]),average="macro",
                                    labels=np.arange(len(GIDX)))
                        rows.append({"unit":uname,"pool":pname,"seed":seed,"model":mn,
                                     "macro5":f5,"gw3":f3})
            print(f"  {uname} 完成 {time.time()-t0:.0f}s",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/lenhist_test.csv",index=False)
    bb5=R.loc[R.groupby(["unit","pool","seed"]).macro5.idxmax()]
    bb3=R.loc[R.groupby(["unit","pool","seed"]).gw3.idxmax()]
    print("\n=== 网关三类 macro（best_base）===",flush=True)
    P3=bb3.pivot_table(index="unit",columns="pool",values="gw3").round(4)
    P3["Δ"]=(P3["183+lenhist"]-P3["183"]).round(4)
    print(P3.to_string(),flush=True)
    print("\n=== 5 类 macro（best_base）===",flush=True)
    P5=bb5.pivot_table(index="unit",columns="pool",values="macro5").round(4)
    P5["Δ"]=(P5["183+lenhist"]-P5["183"]).round(4)
    print(P5.to_string(),flush=True)
    print("\n=== 逐模型（三个 outer 单元均值，网关三类）===",flush=True)
    O=R[R.unit!="inner_R4"]
    print(O.pivot_table(index="model",columns="pool",values="gw3").round(4).to_string(),flush=True)
    o3=P3.drop(index="inner_R4")["Δ"].mean(); i3=P3.loc["inner_R4","Δ"]
    print(f"\ninner Δ={i3:+.4f}   三 outer 单元均值 Δ={o3:+.4f}",flush=True)
    print("判读：inner 涨而 outer 不涨 → 过拟合源域；两边都涨 → 真增量；都不涨 → 这族没用。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
