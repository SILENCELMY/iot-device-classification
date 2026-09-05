"""【探索性,非协议】强基线之上：逐类对闸门有没有区分力？—— 承重检验。

背景（2026-09-05 实测）：帧长直方图族在自采上 +0.1008（183 → 183+lenhist = 0.8074 → 0.9082），
在 UNSW 上 ≈0（+0.002/+0.004，主错误对不降反升）。**同一族一个池子决定性、一个池子惰性**
⇒ 全局挑一组特征不成立，逐对才是对的粒度（见 literature-position-2026-09）。

于是"验收程序"这个贡献从有趣变成**承重**：它必须表现出区分力，否则只是"什么都加"。

**预注册（结果前写死）：**
  A 在网关各对上【留】lenhist，且仍能找到该改的对   → 有区分力，贡献成立
  B 无差别接受一切                                  → 只是"什么都加"，不是验收
  C 在实测有效的对上【掩掉】lenhist                  → 闸门判据有问题
  D 强基线之上增益 ≈ 0                              → 诚实结论：补强基线后无作用面，须报

机制上不需要改代码：族导出规则按列名首个下划线段分组，所以 `lenhist_*` 自动成为可掩族，
`dir_*`/`time_*`/`seq_*`（我们自造的 89 列扩展）同样各自成族 ——
**因此本实验同时在问：程序会不会区别对待"物理属性族"与"环境耦合的统计量族"？**

臂：
  flat_strong   平铺 5 类，252 列（183+lenhist）
  h2_strong     两段式（发射机+平滑 → 网关内平铺）
  h2_cfg        h2 + 逐类对定点修（在 |S|=1 inner 上导出，九种切法均为正，见 inner_sweep）
并报：每一对被选中的配置里，lenhist 是被留还是被掩。
"""
from __future__ import annotations
import os, sys, time, hashlib
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

NEW89=REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
LENH =REPO+"/results/feature_expansion_20260905/features_lenhist_w10.csv"
KEYS=["label","round","source_file","window_id"]
GATEWAY=["Light_T1","Light_XM","Sensor"]
def TX(l): return "Gateway" if l in GATEWAY else l
KB_INNER=(["R2","R3"],"R4")
INNER_A=(["R2"],"R3")
OUTER=[("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
CANDS=["lr","rf","xgboost","lightgbm"]; KBASE=[1,3,5,10,20]
MARGIN=0.02; MIN_CELL=40; SEEDS=[42,43,44]

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
    with threadpool_limits(1):
        d=TC.Data(); new=pd.read_csv(NEW89); lh=pd.read_csv(LENH)
        df=d.df.merge(new,on=KEYS,how="inner").merge(lh,on=KEYS,how="inner")
        assert len(df)==len(d.df), f"行数变了 {len(df)} vs {len(d.df)}"
        cols=TC.feature_columns(df); L5=list(d.enc.classes_)
        i5={c:n for n,c in enumerate(L5)}
        TX3=sorted(set(TX(c) for c in L5)); GIDX=[i5[c] for c in GATEWAY]
        fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none",np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f,keep))
        MD={m:k for m,k in MASKS}
        nlh=sum(1 for c in cols if c.startswith("lenhist_"))
        print(f"池 {len(cols)} 列（其中 lenhist {nlh} 列）  族 {len(fams)} 个  掩码候选 {len(MASKS)}",flush=True)
        print(f"族名: {sorted(fams)}",flush=True)
        assert "lenhist" in fams, "lenhist 没有成族，族导出规则与预期不符"

        def block(rounds,mix=False):
            s=df[df["round"].isin(rounds)].copy(); s["tx"]=[TX(x) for x in s["label"]]
            s=s.sort_values(["tx","window_start"],kind="mergesort") if mix \
              else s.sort_values(["label","window_start"],kind="mergesort")
            return (np.asarray(TC.clean_x(s,cols),dtype=float),
                    np.array([i5[x] for x in s["label"]]), np.asarray(s["tx"]))

        def gw_proba(Xtr,ytr,Xte):
            loc={c:k for k,c in enumerate(GIDX)}
            m=TC.make_model("xgboost",len(GIDX)); m.fit(Xtr,np.array([loc[v] for v in ytr]))
            P=m.predict_proba(Xte); out=np.zeros((len(Xte),len(L5)))
            for k,c in enumerate(GIDX): out[:,c]=P[:,k]
            return out

        rows=[]; picks=[]
        for seed in SEEDS:
            TC.SEED=seed
            Xa,ya,_=block(KB_INNER[0]); Xb,yb,gb=block([KB_INNER[1]],mix=True)
            ya3=np.array([TX3.index(TX(L5[c])) for c in ya])
            yb3=np.array([TX3.index(TX(L5[c])) for c in yb])
            m1=TC.make_model("xgboost",len(TX3)); m1.fit(Xa,ya3); P1=m1.predict_proba(Xb)
            kb=max(((k,f1_score(yb3,cmeanM(P1,gb,k).argmax(1),average="macro")) for k in KBASE),
                   key=lambda x:x[1])[0]

            XA,yA,_=block(INNER_A[0]); XT,yT,_=block([INNER_A[1]])
            ma=np.isin(yA,GIDX); mt=np.isin(yT,GIDX)
            oo=np.argsort(-gw_proba(XA[ma],yA[ma],XT[mt]),axis=1); a1,a2=oo[:,0],oo[:,1]
            cfg={}
            print(f"\n=== seed{seed}  kb={kb} ===",flush=True)
            for (i,j) in sorted(set(map(tuple,np.sort(np.c_[a1,a2],axis=1)))):
                if i==j: continue
                D=(((a1==i)&(a2==j))|((a1==j)&(a2==i)))&np.isin(yT[mt],[i,j])
                if D.sum()<MIN_CELL: continue
                inc=float((a1[D]==yT[mt][D]).mean())
                ms=np.isin(yA[ma],[i,j])
                if len(np.unique(yA[ma][ms]))<2: continue
                y1=(yA[ma][ms]==j).astype(int)
                scores={}
                for mn,keep in MASKS:
                    for nm in CANDS:
                        try:
                            q=fit_w(MK(nm),nm,XA[ma][ms][:,keep],y1).predict_proba(XT[mt][:,keep])[:,1]
                        except Exception: continue
                        scores[(mn,nm)]=float((np.where(q[D]>=0.5,j,i)==yT[mt][D]).mean())
                if not scores: continue
                (mn,nm),acc=max(scores.items(),key=lambda x:x[1])
                # 关键观测：掩掉 lenhist 时最好能到多少 vs 保留时最好能到多少
                best_keep=max((v for (k_,_n),v in scores.items() if k_!="lenhist"), default=np.nan)
                best_masklh=max((v for (k_,_n),v in scores.items() if k_=="lenhist"), default=np.nan)
                pen=best_keep-best_masklh
                picks.append({"seed":seed,"pair":f"{L5[i]}|{L5[j]}","n":int(D.sum()),
                    "inc":inc,"acc":acc,"mask":mn,"model":nm,
                    "best_with_lenhist":best_keep,"best_masking_lenhist":best_masklh,
                    "lenhist_value":pen,"gated":bool(acc>inc+MARGIN)})
                flag="**过闸**" if acc>inc+MARGIN else "不过闸"
                print(f"  {L5[i]}|{L5[j]:9s} n={int(D.sum()):4d} 在位者 {inc:.4f} → 最好 {acc:.4f}"
                      f"  {nm}/掩{mn}  {flag}   留lenhist最好 {best_keep:.4f} / 掩掉最好 "
                      f"{best_masklh:.4f}  (lenhist 值 {pen:+.4f})",flush=True)
                if acc>inc+MARGIN: cfg[(i,j)]=(mn,nm,acc)

            for uname,src,tgt in OUTER:
                Xs,ys,_=block(src); Xt,yt,gt=block([tgt],mix=True)
                ys3=np.array([TX3.index(TX(L5[c])) for c in ys])
                mf=TC.make_model("xgboost",len(L5)); mf.fit(Xs,ys)
                f_flat=f1_score(yt,mf.predict(Xt),average="macro",labels=np.arange(len(L5)))
                st=TC.make_model("xgboost",len(TX3)); st.fit(Xs,ys3)
                p1=cmeanM(st.predict_proba(Xt),gt,kb).argmax(1)
                gsel=p1==TX3.index("Gateway")
                base=np.empty(len(yt),dtype=int)
                for n,t in enumerate(TX3):
                    if t!="Gateway": base[p1==n]=i5[t]
                gm_s=np.isin(ys,GIDX)
                pred=base.copy()
                if gsel.any():
                    og=np.argsort(-gw_proba(Xs[gm_s],ys[gm_s],Xt[gsel]),axis=1)
                    g1,g2=og[:,0],og[:,1]; base[gsel]=g1
                    pg=g1.copy()
                    for (i,j),(mn,nm,_a) in cfg.items():
                        keep=MD[mn]; ms=np.isin(ys,[i,j])
                        if len(np.unique(ys[ms]))<2: continue
                        sel=((g1==i)&(g2==j))|((g1==j)&(g2==i))
                        if not sel.any(): continue
                        q=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys[ms]==j).astype(int)
                                ).predict_proba(Xt[gsel][:,keep])[:,1]
                        pg[sel]=np.where(q[sel]>=0.5,j,i)
                    pred=base.copy(); pred[gsel]=pg
                F=lambda p: f1_score(yt,p,average="macro",labels=np.arange(len(L5)))
                gmask=np.isin(yt,GIDX)
                GW=lambda p: f1_score(yt[gmask],p[gmask],average="macro",labels=GIDX)
                rows.append({"unit":uname,"seed":seed,"kb":kb,"n_gate":len(cfg),
                  "flat_strong":f_flat,"h2_strong":F(base),"h2_cfg":F(pred),
                  "gw_h2":GW(base),"gw_cfg":GW(pred)})
                print(f"  {uname} seed{seed}  平铺={f_flat:.4f}  两段={F(base):.4f}  "
                      f"+定点修={F(pred):.4f}   网关 {GW(base):.4f}→{GW(pred):.4f}",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/cfg_strong.csv",index=False)
    P=pd.DataFrame(picks); P.to_csv("/home/lmy/cic_probe/cfg_strong_picks.csv",index=False)
    print("\n=== 强基线之上（3 seed 均值）===",flush=True)
    print(R.groupby("unit")[["flat_strong","h2_strong","h2_cfg","gw_h2","gw_cfg"]].mean().round(4).to_string(),flush=True)
    print(f"\n总均值  平铺={R.flat_strong.mean():.4f}  两段={R.h2_strong.mean():.4f}  "
          f"+定点修={R.h2_cfg.mean():.4f}",flush=True)
    print(f"  两段−平铺 = {R.h2_strong.mean()-R.flat_strong.mean():+.4f}",flush=True)
    print(f"  定点修增益 = {R.h2_cfg.mean()-R.h2_strong.mean():+.4f}   过闸 {R.n_gate.mean():.1f} 对",flush=True)
    print(f"  网关三类：{R.gw_h2.mean():.4f} → {R.gw_cfg.mean():.4f} "
          f"({R.gw_cfg.mean()-R.gw_h2.mean():+.4f})",flush=True)
    print("\n=== 预注册判读：闸门有没有区分力 ===",flush=True)
    print(P.groupby("pair")[["inc","acc","best_with_lenhist","best_masking_lenhist",
                             "lenhist_value"]].mean().round(4).to_string(),flush=True)
    print("\n被选中的掩码分布：",flush=True)
    print(P[P.gated].groupby(["pair","mask","model"]).size().to_string(),flush=True)
    nlh_mask=int((P[P.gated]["mask"]=="lenhist").sum())
    print(f"\n  过闸配置里选择【掩掉 lenhist】的：{nlh_mask}/{int(P.gated.sum())}",flush=True)
    print(f"  lenhist 的逐对价值（留−掩）：均值 {P.lenhist_value.mean():+.4f}  "
          f"最小 {P.lenhist_value.min():+.4f}  最大 {P.lenhist_value.max():+.4f}",flush=True)
    print("  A 留且仍有该改的对 → 有区分力；B 无差别接受；C 在有效对上掩掉 → 判据有问题；",flush=True)
    print("  D 增益≈0 → 补强基线后无作用面，须如实报。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
