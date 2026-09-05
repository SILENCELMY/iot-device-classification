"""【探索性,非协议】共识臂：不选 inner，让 9 种 inner 投票。

来由：inner_sweep 实测 —— 9 种 inner 切法的 Δ 从 +0.0041 到 +0.0306，
【跨度 0.0265 超过方法自身的平均增益】，而 `|S|` 与 `min在位者` 都预测不了它。
（更正记录：cfg_ablation 得出的"功劳全在 |S|=1"是抽样噪声 —— 它的两个臂恰好
抽中 9 个里的第 7 名与第 9 名。）

既然没有规则能挑对 inner，就【不挑】：9 个 inner 各自提名 (类对, 掩码, 模型)，
只接受得票数 ≥ m 的。这样：
  1. 消掉"选 inner"这个自由参数，不需要猜；
  2. 正面打 winner's curse —— 只在某一个 inner 上碰巧最优的配置拿不到多数票；
  3. 可核对：报 m 的整条曲线，不是挑一个 m。

对照基准（inner_sweep 实测，seed42 三单元均值 Δ）：
    最好 +0.0306（R4>R3 与 R2R4>R3 并列）  9 个均值 +0.0213  最差 +0.0041（R2R3>R4）
判据：共识若落在均值以上、且随 m 平稳 → 是真改进，可写进方法；
      若只在某个 m 上好 → 又是一次挑参数，不算。

配置来源：inner_sweep_picks.csv（9 inner × 各 seed 的全部过闸记录），不重搜。
"""
from __future__ import annotations
import os, sys, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import run_two_channel as TC

NEW89 = REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
PICKS = "/home/lmy/cic_probe/inner_sweep_picks.csv"
KEYS  = ["label","round","source_file","window_id"]
GATEWAY=["Light_T1","Light_XM","Sensor"]
def TX(l): return "Gateway" if l in GATEWAY else l
KB_INNER=(["R2","R3"],"R4")
OUTER=[("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
KBASE=[1,3,5,10,20]
MS=[1,2,3,4,5,6,7,8,9]          # 票数门槛，报整条曲线

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,C=1.0))
    return TC.make_model(n,2)

def fit_w(m,name,X,y):
    w=compute_sample_weight("balanced",y)
    try:
        if name=="lr": m.fit(X,y,logisticregression__sample_weight=w)
        else:          m.fit(X,y,sample_weight=w)
    except Exception: m.fit(X,y)
    return m

def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def main():
    t0=time.time()
    P=pd.read_csv(PICKS)
    print(f"picks {len(P)} 行   inner {P.inner.nunique()} 种   seed {sorted(P.seed.unique())}",flush=True)
    with threadpool_limits(1):
        d=TC.Data(); new=pd.read_csv(NEW89)
        df=d.df.merge(new,on=KEYS,how="inner"); assert len(df)==len(d.df)
        cols=TC.feature_columns(df); L5=list(d.enc.classes_)
        TX3=sorted(set(TX(c) for c in L5)); G3=[c for c in L5 if TX(c)=="Gateway"]
        i5={c:n for n,c in enumerate(L5)}; GIDX=[i5[c] for c in G3]
        fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none",np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f,keep))
        MD={m:k for m,k in MASKS}

        def block(rounds,mix=False):
            s=df[df["round"].isin(rounds)].copy(); s["tx"]=[TX(x) for x in s["label"]]
            s=s.sort_values(["tx","window_start"],kind="mergesort") if mix \
              else s.sort_values(["label","window_start"],kind="mergesort")
            return (np.asarray(TC.clean_x(s,cols),dtype=float),
                    np.array([i5[x] for x in s["label"]]), np.asarray(s["tx"]))

        def gw_proba(Xtr,ytr,Xte):
            loc={c:k for k,c in enumerate(GIDX)}
            m=TC.make_model("xgboost",len(GIDX)); m.fit(Xtr,np.array([loc[v] for v in ytr]))
            Pr=m.predict_proba(Xte); out=np.zeros((len(Xte),len(L5)))
            for k,c in enumerate(GIDX): out[:,c]=Pr[:,k]
            return out

        n_inner=P.inner.nunique()
        rows=[]
        for seed in sorted(P.seed.unique()):
            TC.SEED=int(seed)
            # --- 投票：逐类对统计得票与众数配置 ---
            Q=P[P.seed==seed]
            votes={}
            for pr,g in Q.groupby("pair"):
                cfgc=Counter(zip(g["mask"],g["model"]))
                (mn,nm),nmode = cfgc.most_common(1)[0]
                votes[pr]=(len(g), mn, nm, nmode, dict(cfgc))
            print(f"\n=== seed{seed}  逐类对得票（共 {n_inner} 个 inner）===",flush=True)
            for pr,(v,mn,nm,nmode,cc) in sorted(votes.items(), key=lambda x:-x[1][0]):
                print(f"  {pr:22s} 得票 {v}/{n_inner}   众数配置 {nm}/掩{mn} ({nmode} 票)"
                      f"   全部: {dict(cc)}",flush=True)

            Xa,ya5,_=block(KB_INNER[0]); Xb,yb5,gb=block([KB_INNER[1]],mix=True)
            ya3=np.array([TX3.index(TX(L5[c])) for c in ya5])
            yb3=np.array([TX3.index(TX(L5[c])) for c in yb5])
            m1=TC.make_model("xgboost",len(TX3)); m1.fit(Xa,ya3); P1=m1.predict_proba(Xb)
            kb=max(((k,f1_score(yb3,cmeanM(P1,gb,k).argmax(1),average="macro")) for k in KBASE),
                   key=lambda x:x[1])[0]

            CFGS={}
            for m in MS:
                c={}
                for pr,(v,mn,nm,_n,_cc) in votes.items():
                    if v>=m:
                        a,b_=pr.split("|"); c[(i5[a],i5[b_])]=(mn,nm)
                CFGS[f"m>={m}"]=c

            for uname,src,tgt in OUTER:
                Xs,ys5,_=block(src); Xt,yt5,gt=block([tgt],mix=True)
                ys3=np.array([TX3.index(TX(L5[c])) for c in ys5])
                mf=TC.make_model("xgboost",len(L5)); mf.fit(Xs,ys5)
                f_flat=f1_score(yt5,mf.predict_proba(Xt).argmax(1),average="macro",
                                labels=np.arange(len(L5)))
                st=TC.make_model("xgboost",len(TX3)); st.fit(Xs,ys3)
                p1=cmeanM(st.predict_proba(Xt),gt,kb).argmax(1)
                gsel=p1==TX3.index("Gateway")
                base=np.empty(len(yt5),dtype=int)
                for n,t in enumerate(TX3):
                    if t!="Gateway": base[p1==n]=i5[t]
                gm_s=np.isin(ys5,GIDX)
                if gsel.any():
                    og=np.argsort(-gw_proba(Xs[gm_s],ys5[gm_s],Xt[gsel]),axis=1)
                    g1,g2=og[:,0],og[:,1]; base[gsel]=g1
                f_h0=f1_score(yt5,base,average="macro",labels=np.arange(len(L5)))
                gmask=np.isin(yt5,GIDX)
                gwf=lambda p: f1_score(yt5[gmask],p[gmask],average="macro",labels=GIDX)
                fc={}; line=[]
                for aname,cfg in CFGS.items():
                    pred=base.copy()
                    if gsel.any() and cfg:
                        pg=g1.copy()
                        for (i,j),(mn,nm) in cfg.items():
                            key=(i,j,mn,nm)
                            if key not in fc:
                                keep=MD[mn]; ms=np.isin(ys5,[i,j])
                                fc[key]=None if len(np.unique(ys5[ms]))<2 else \
                                    fit_w(MK(nm),nm,Xs[ms][:,keep],(ys5[ms]==j).astype(int)
                                          ).predict_proba(Xt[gsel][:,keep])[:,1]
                            q=fc[key]
                            if q is None: continue
                            sel=((g1==i)&(g2==j))|((g1==j)&(g2==i))
                            if sel.any(): pg[sel]=np.where(q[sel]>=0.5,j,i)
                        pred[gsel]=pg
                    f_=f1_score(yt5,pred,average="macro",labels=np.arange(len(L5)))
                    rows.append({"unit":uname,"seed":seed,"arm":aname,"m":int(aname.split(">=")[1]),
                                 "n_gate":len(cfg),"flat":f_flat,"hier":f_h0,"cfg":f_,
                                 "delta":f_-f_h0,"gw_hier":gwf(base),"gw_cfg":gwf(pred)})
                    line.append(f"{aname}={f_:.4f}")
                print(f"  {uname} seed{seed} 分层={f_h0:.4f}  "+"  ".join(line),flush=True)

        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/cfg_consensus.csv",index=False)
        print("\n=== 共识曲线（三单元 × 全 seed 均值）===",flush=True)
        A=R.groupby("m").agg(n_gate=("n_gate","mean"),cfg=("cfg","mean"),
                             delta=("delta","mean"),gw=("gw_cfg","mean")).round(4)
        print(A.to_string(),flush=True)
        print(f"\n分层（不修）= {R.hier.mean():.4f}    扁平 = {R.flat.mean():.4f}",flush=True)
        print("\n对照 inner_sweep（seed42 三单元 Δ）：最好 +0.0306  9 个均值 +0.0213  最差 +0.0041",flush=True)
        print("判读：共识落在 9 个均值以上且随 m 平稳 → 真改进；只在某个 m 上好 → 又是挑参数。",flush=True)
        print("\n=== 逐单元 ===",flush=True)
        print(R.pivot_table(index="m",columns="unit",values="delta").round(4).to_string(),flush=True)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
