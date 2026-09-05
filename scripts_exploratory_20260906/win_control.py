"""【探索性,非协议】窗长对比的样本量控制。

win_compare 的结论是 w10（macro 0.7640）打赢 w30（0.7138）和 w60（0.7230）。
但有个混淆没排掉：窗越长，行数越少 ——
    w10  训练 R2R3R4 ≈ 5700 行
    w30  ≈ 1900 行   （1/3）
    w60  ≈ 950 行    （1/6）
长窗是输在【特征更差】还是输在【训练样本更少】？

做法：把 w10 的训练集下采样到与 w30 / w60 相同的行数（按类分层，多个抽样重复），
在同一个目标轮上评测。
    若 w10@1900行 仍明显赢 w30@1900行 → 10 s 特征本身更好，长窗确实无用
    若 w10@1900行 掉到 w30 水平       → 差距只是样本量，长窗的特征不比 10 s 差，
                                        真正的代价是"同样时长的采集换来更少的窗"

顺带补一个 win_compare 缺的干净对照：非网关类（Camera/Socket）上平滑是合法的
（一个发射机 = 一台设备），单独报它们的 F1，看平滑在合法处是否照常有益。
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
GATEWAY=["Light_T1","Light_XM","Sensor"]
def TX(l): return "Gateway" if l in GATEWAY else l
OUTER=[("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
MODELS=["rf","xgboost","lightgbm"]
SEEDS=[42,43,44]
SUBS=[0,1,2]                      # 下采样重复次数（0/1/2 三个抽样）

def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def strat_sub(y, n_target, rs):
    """按类分层下采样到总行数 ≈ n_target。"""
    rng=np.random.RandomState(rs); keep=[]
    frac=min(1.0, n_target/len(y))
    for c in np.unique(y):
        idx=np.where(y==c)[0]
        k=max(2, int(round(len(idx)*frac)))
        keep.append(rng.choice(idx, size=min(k,len(idx)), replace=False))
    return np.sort(np.concatenate(keep))

def load(w):
    df=pd.read_csv(RAW.format(w=w))
    cols=TC.feature_columns(df)
    L5=sorted(df["label"].unique()); i5={c:n for n,c in enumerate(L5)}
    def block(rounds, mix=False):
        s=df[df["round"].isin(rounds)].copy()
        s["tx"]=[TX(x) for x in s["label"]]
        s=s.sort_values(["tx","window_start"],kind="mergesort") if mix \
          else s.sort_values(["label","window_start"],kind="mergesort")
        return (np.asarray(TC.clean_x(s,cols),dtype=float),
                np.array([i5[x] for x in s["label"]]),
                np.asarray(s["tx"]))
    return block, L5, i5, len(cols)

def main():
    t0=time.time(); rows=[]
    with threadpool_limits(1):
        B={}; N={}
        for w in (10,30,60):
            B[w]=load(w)
            Xs,ys,_=B[w][0](["R2","R3","R4"]); N[w]=len(ys)
            print(f"w{w}: 训练 {len(ys)} 行  {B[w][3]} 特征列",flush=True)

        for uname,src,tgt in OUTER:
            for w in (10,30,60):
                block,L5,i5,_=B[w]
                Xs,ys,_=block(src); Xt,yt,gt=block([tgt],mix=True)
                gid=[i5[c] for c in GATEWAY]; gm=np.isin(yt,gid)
                ngm=~gm
                # 全量 + 下采样到 w30 / w60 的行数
                plans=[("full", np.arange(len(ys)), 0)]
                if w==10:
                    for tw in (30,60):
                        for r in SUBS:
                            plans.append((f"sub{tw}", strat_sub(ys,N[tw],1000*tw+r), r))
                for pname, sel, rep in plans:
                    for seed in SEEDS:
                        TC.SEED=seed
                        for mn in MODELS:
                            m=TC.make_model(mn,len(L5))
                            if m is None: continue
                            m.fit(Xs[sel],ys[sel]); P=m.predict_proba(Xt)
                            for k in ([1,3] if w==10 else [1]):
                                p=cmeanM(P,gt,k).argmax(1)
                                rows.append({"w":w,"k":k,"plan":pname,"rep":rep,
                                    "n_train":len(sel),"unit":uname,"seed":seed,"model":mn,
                                    "macro":f1_score(yt,p,average="macro",labels=np.arange(len(L5))),
                                    "gw":f1_score(yt[gm],p[gm],average="macro",labels=gid),
                                    "nongw":f1_score(yt[ngm],p[ngm],average="macro",
                                                     labels=[i5[c] for c in L5 if c not in GATEWAY])})
            print(f"  {uname} 完成 {time.time()-t0:.0f}s",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/win_control.csv",index=False)
    bb=R.loc[R.groupby(["w","k","plan","rep","unit","seed"]).macro.idxmax()]

    print("\n=== 样本量控制（best_base，三单元 × 3 seed；w10 下采样再 × 3 抽样）===",flush=True)
    A=bb[bb.k==1].groupby(["w","plan"]).agg(n_train=("n_train","mean"),
        macro=("macro","mean"), gw=("gw","mean"), nongw=("nongw","mean")).round(4)
    print(A.to_string(),flush=True)

    g=lambda w,pl,c="macro": bb[(bb.k==1)&(bb.w==w)&(bb.plan==pl)][c].mean()
    print("\n=== 判据：同训练行数下 w10 vs 长窗 ===",flush=True)
    for tw in (30,60):
        a,b_=g(10,f"sub{tw}"), g(tw,"full")
        ag,bg=g(10,f"sub{tw}","gw"), g(tw,"full","gw")
        print(f"  ≈{int(bb[(bb.w==tw)&(bb.plan=='full')].n_train.mean())} 行:  "
              f"w10下采样 macro={a:.4f} 网关={ag:.4f}   vs  w{tw}原生 macro={b_:.4f} 网关={bg:.4f}   "
              f"w10−w{tw} = {a-b_:+.4f} (网关 {ag-bg:+.4f})",flush=True)
    print(f"\n  参照 w10 全量: macro={g(10,'full'):.4f} 网关={g(10,'full','gw'):.4f}",flush=True)
    print("  判读：w10−长窗 仍明显为正 → 10 s 特征本身更好，长窗无用",flush=True)
    print("        ≈0                → 差距只是样本量，长窗特征不差，代价在窗数变少",flush=True)

    print("\n=== 平滑在【合法处】（非网关类，一发射机=一设备）===",flush=True)
    S=bb[(bb.w==10)&(bb.plan=="full")].groupby("k")[["macro","gw","nongw"]].mean().round(4)
    print(S.to_string(),flush=True)
    print("  nongw 列是 Camera/Socket 的 macro —— 平滑在这里合法。",flush=True)
    print("  gw 列下跌属预期：扁平 5 类模型 + 按发射机平滑 = 把网关内三台的概率平均掉，",flush=True)
    print("  正是两段式设计要避免的，不能当作平滑的反证。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
