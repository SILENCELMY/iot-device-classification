"""【探索性,非协议】窗长对比：10 s 窗是不是已经够了？

正确的问法不是"w30 比 w10 好吗"，而是【w30 比 w10 + k=3 平滑好吗】——
两者用的证据量相同（30 秒），差别只在怎么用：
    平滑  把三个 10 s 窗的【预测】平均   —— 只降方差，不产生新特征
    长窗  在 30 s 跨度上【重算特征】     —— interarrival/burst/周期性跨越多个 10 s 周期
只有长窗赢了，才说明跨窗结构带了 10 s 内不存在的信息。

且 w10+k=3 在部署上还占两个便宜：判决延迟相同（都要等 30 s 证据），但每 10 s 出一次
结果（长窗 30 s 才出一次），决策数多 3 倍。所以长窗必须【明显】赢才值得换。

窗长只取 10 s 整数倍：占空比实测基频 10.06 s，30 s=3 周期、60 s=6 周期，周期对齐不破。

口径统一：三个窗长都用 94 列基础池（89 列扩展特征只在 w10 上抽过，用了就不可比）。
因此本脚本的绝对值不能与 183 列的 flat_183=0.8072 直接比，只能横向比三个窗长。
平滑按【发射机】分组（部署时可见的 MAC），与两段式一致。
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

RAW = REPO+"/results/robust_v2/raw_all/features_raw_all_w{w}.csv"
WINS=[10,30,60]
GATEWAY=["Light_T1","Light_XM","Sensor"]
def TX(l): return "Gateway" if l in GATEWAY else l
OUTER=[("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
MODELS=["rf","xgboost","lightgbm"]
SEEDS=[42,43,44]
# 平滑窗 k：w10 上 k=3/6 分别等价于 30 s / 60 s 证据
KMAP={10:[1,3,6],30:[1,2],60:[1]}

def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def main():
    t0=time.time(); rows=[]
    with threadpool_limits(1):
        for w in WINS:
            path=RAW.format(w=w)
            if not os.path.exists(path):
                print(f"[缺] {path}",flush=True); continue
            df=pd.read_csv(path)
            cols=TC.feature_columns(df)
            L5=sorted(df["label"].unique()); i5={c:n for n,c in enumerate(L5)}
            n_by=df.groupby("round").size().to_dict()
            print(f"\n=== 窗长 {w}s   {len(df)} 行  {len(cols)} 特征列  "
                  f"轮次行数 {[(r,n_by.get(r,0)) for r in ['R2','R3','R4','R5','R6','R7']]}",flush=True)

            def block(rounds, mix=False):
                s=df[df["round"].isin(rounds)].copy()
                s["tx"]=[TX(x) for x in s["label"]]
                s=s.sort_values(["tx","window_start"],kind="mergesort") if mix \
                  else s.sort_values(["label","window_start"],kind="mergesort")
                return (np.asarray(TC.clean_x(s,cols),dtype=float),
                        np.array([i5[x] for x in s["label"]]),
                        np.asarray(s["tx"]))

            for uname,src,tgt in OUTER:
                Xs,ys,_ = block(src); Xt,yt,gt = block([tgt], mix=True)
                gmask=np.isin(yt,[i5[c] for c in GATEWAY])
                for seed in SEEDS:
                    TC.SEED=seed
                    for mn in MODELS:
                        m=TC.make_model(mn,len(L5))
                        if m is None: continue
                        m.fit(Xs,ys); P=m.predict_proba(Xt)
                        for k in KMAP[w]:
                            p=cmeanM(P,gt,k).argmax(1)
                            rows.append({"w":w,"k":k,"evid_s":w*k,"unit":uname,"seed":seed,
                                "model":mn,
                                "macro":f1_score(yt,p,average="macro",labels=np.arange(len(L5))),
                                "gw":f1_score(yt[gmask],p[gmask],average="macro",
                                              labels=[i5[c] for c in GATEWAY]),
                                "n_dec":len(yt)})
                print(f"  {uname} 完成  {time.time()-t0:.0f}s",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/win_compare.csv",index=False)
    bb=R.loc[R.groupby(["w","k","unit","seed"]).macro.idxmax()]      # best_base 口径

    print("\n=== 单模型 xgboost：5 类窗口级 macro-F1（三单元 × 3 seed 均值）===",flush=True)
    X=R[R.model=="xgboost"]
    print(X.pivot_table(index=["w","k","evid_s"],values=["macro","gw","n_dec"]).round(4).to_string(),flush=True)

    print("\n=== best_base（逐 单元,seed 取三模型最大）===",flush=True)
    print(bb.pivot_table(index=["w","k","evid_s"],values=["macro","gw"]).round(4).to_string(),flush=True)

    print("\n=== 同证据量对拍（这才是判据）===",flush=True)
    def get(w,k,col="macro"):
        s=bb[(bb.w==w)&(bb.k==k)][col]
        return s.mean() if len(s) else float("nan")
    for ev,(a,b) in [(30,((10,3),(30,1))), (60,((10,6),(60,1)))]:
        m1,m2=get(*a),get(*b); g1,g2=get(*a,"gw"),get(*b,"gw")
        n1=len(bb[(bb.w==a[0])&(bb.k==a[1])]); n2=len(bb[(bb.w==b[0])&(bb.k==b[1])])
        print(f"  {ev}s 证据:  w{a[0]}+k={a[1]} macro={m1:.4f} 网关={g1:.4f}   "
              f"vs  w{b[0]}原生 macro={m2:.4f} 网关={g2:.4f}   "
              f"长窗−平滑 = {m2-m1:+.4f} (网关 {g2-g1:+.4f})",flush=True)
    print(f"\n  基准 w10 k=1: macro={get(10,1):.4f} 网关={get(10,1,'gw'):.4f}",flush=True)
    print("\n  判读：长窗−平滑 明显为正 → 跨窗特征有真信息，值得换（代价：决策数降 3~6 倍）",flush=True)
    print("        ≈0 或为负        → 10 s 窗已够，平滑是更便宜且决策更密的用法",flush=True)

    print("\n=== 逐单元（看是否只在 pos_R5 上不同）===",flush=True)
    print(bb.pivot_table(index=["w","k"],columns="unit",values="macro").round(4).to_string(),flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
