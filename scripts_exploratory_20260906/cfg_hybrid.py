"""【探索性,非协议】混合臂：`|S|=1` 检测，`|S|=2` 验收。

来由：cfg_ablation 定案「功劳全在 inner 的 |S|=1，第二环境验证零贡献」。
机制猜想（见 README「为什么 |S|=1 更灵敏」）：rssi 在同位置轮次间也在漂；训 1 轮 → 零
rssi 多样性 → 换任何一轮都崩 → 闸门看见"这对靠不稳定特征分开"；训 2–3 轮 → 范围内插值
→ 在位者正常 → 闸门看不见；outer 训 3 轮但 R5 在范围外 → 照样崩。
即 |S|=1 是【敏感性测试】，不是同分布代理。

由此与 rssi-ablation-is-the-origin 的"|S|≥2 是硬要求"调和：两者是两件事 ——
    检测「哪些对靠不稳定特征」  用 |S|=1  最敏感，宁可多报
    决策「要不要全局删 rssi」    用 |S|≥2  |S|=1 会一律说删，过度删除
现方法两件事都用 |S|=1（cfg_AB 的 B 也是 |S|=1，所以等于没验证）。本脚本补上混合设计。

臂：
  A        检测+选配置+验收 全在 A=R2→R3（|S|=1）        —— 现最优，= cfg_A
  S        全在 S=R2+R3→R4（|S|=2）                      —— = cfg_S
  hyb@tol  在 A 上检测并选配置，再要求在 S 上【不添乱】：acc_S ≥ inc_S − tol
           tol ∈ {0.00, 0.05, 0.10, 1.00}   （tol=1.00 等价于不验收 = A）

关键观测量（无论臂的胜负如何都要看）：救命配置在 S 上的 (inc_S, acc_S)。
  acc_S ≈ inc_S  → "不添乱"验收白送，可以常开
  acc_S ≪ inc_S  → 修复是【拿源域范围内准确率换范围外准确率】，这是方法性质的发现，
                   且说明"不添乱"这条验收方向本身是错的。
"""
from __future__ import annotations
import os, sys, time, json, hashlib
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
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
KEYS  = ["label","round","source_file","window_id"]
MD5_A = "703984b6ad2fde2f45e0cce1c6df31be"
MD5_B = "e0586ea2b55c6913a614a1f94772e701"
GATEWAY=["Light_T1","Light_XM","Sensor"]
def TX(l): return "Gateway" if l in GATEWAY else l

INNER_A = (["R2"], "R3")          # |S|=1  检测/选配置
INNER_S = (["R2","R3"], "R4")     # |S|=2  验收
KB_INNER= (["R2","R3"], "R4")
OUTER=[("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
CANDS=["lr","rf","xgboost","lightgbm"]
KBASE=[1,3,5,10,20]
TOLS=[0.00,0.05,0.10,1.00]
MARGIN=0.02; MIN_CELL=40; SEEDS=[42,43]

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return TC.make_model(n,2)

def fit_w(m,name,X,y):
    w=compute_sample_weight("balanced", y)
    try:
        if name=="lr": m.fit(X,y, logisticregression__sample_weight=w)
        else:          m.fit(X,y, sample_weight=w)
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

def load183():
    a=hashlib.md5(TC.CACHE.read_bytes()).hexdigest()
    b=hashlib.md5(open(NEW89,"rb").read()).hexdigest()
    assert a==MD5_A and b==MD5_B, "特征缓存 md5 不符"
    d=TC.Data(); new=pd.read_csv(NEW89)
    df=d.df.merge(new,on=KEYS,how="inner")
    assert len(df)==len(d.df)
    return df,d

def main():
    t0=time.time()
    with threadpool_limits(1):
        df,d=load183(); cols=TC.feature_columns(df)
        L5=list(d.enc.classes_); TX3=sorted(set(TX(c) for c in L5))
        G3=[c for c in L5 if TX(c)=="Gateway"]; i5={c:n for n,c in enumerate(L5)}
        GIDX=[i5[c] for c in G3]
        fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none",np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f,keep))
        MD={m:k for m,k in MASKS}
        print(f"183 列  网关内 {G3}  掩码 {len(MASKS)}  模型 {CANDS}  tol {TOLS}",flush=True)

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

        def gw_block(rounds):
            X,y,_=block(rounds); m=np.isin(y,GIDX); return X[m],y[m]

        def region(src,tgt):
            Xs_,ys_=gw_block(src); Xt_,yt_=gw_block([tgt])
            oo=np.argsort(-gw_proba(Xs_,ys_,Xt_),axis=1)
            return Xs_,ys_,Xt_,yt_,oo[:,0],oo[:,1]

        def score(i,j,mn,nm,Xs_,ys_,Xt_,yt_,p1_,p2_):
            D=(((p1_==i)&(p2_==j))|((p1_==j)&(p2_==i)))&np.isin(yt_,[i,j])
            if D.sum()<MIN_CELL: return None,None,0
            inc=float((p1_[D]==yt_[D]).mean()); ms=np.isin(ys_,[i,j])
            if len(np.unique(ys_[ms]))<2: return None,None,0
            keep=MD[mn]
            try:
                q=fit_w(MK(nm),nm,Xs_[ms][:,keep],(ys_[ms]==j).astype(int)).predict_proba(Xt_[:,keep])[:,1]
            except Exception: return None,None,0
            return inc, float((np.where(q[D]>=0.5,j,i)==yt_[D]).mean()), int(D.sum())

        rows=[]
        for seed in SEEDS:
            TC.SEED=seed
            Xa,ya5,_=block(KB_INNER[0]); Xb,yb5,gb=block([KB_INNER[1]],mix=True)
            ya3=np.array([TX3.index(TX(L5[c])) for c in ya5])
            yb3=np.array([TX3.index(TX(L5[c])) for c in yb5])
            m1=TC.make_model("xgboost",len(TX3)); m1.fit(Xa,ya3); P1=m1.predict_proba(Xb)
            kb=max(((k,f1_score(yb3,cmeanM(P1,gb,k).argmax(1),average="macro")) for k in KBASE),
                   key=lambda x:x[1])[0]
            print(f"\n=== seed{seed}  第一段 kb={kb} ===",flush=True)

            XA_s,yA_s,XA_t,yA_t,a1,a2 = region(*INNER_A)
            XS_s,yS_s,XS_t,yS_t,s1,s2 = region(*INNER_S)

            # --- 臂 A：检测+选配置+验收 全在 |S|=1 ---
            cfgA={}; onS={}
            for (i,j) in sorted(set(map(tuple,np.sort(np.c_[a1,a2],axis=1)))):
                if i==j: continue
                best=(None,None,-1.,None)
                for mn,_ in MASKS:
                    for nm in CANDS:
                        inc,acc,_n=score(i,j,mn,nm,XA_s,yA_s,XA_t,yA_t,a1,a2)
                        if acc is None: continue
                        if acc>best[2]: best=(mn,nm,acc,inc)
                if best[0] is None or best[3] is None: continue
                if best[2]>best[3]+MARGIN:
                    cfgA[(i,j)]=(best[0],best[1],best[2])
                    incS,accS,nS = score(i,j,best[0],best[1],XS_s,yS_s,XS_t,yS_t,s1,s2)
                    onS[(i,j)]=(incS,accS,nS)
                    tag = "S 上不可评（可判决区太小）" if accS is None else \
                          f"S 上 在位者 {incS:.4f} → 配置 {accS:.4f} ({accS-incS:+.4f}, n={nS})"
                    print(f"  [A检测] {L5[i]}|{L5[j]:9s} A: {best[3]:.4f}→{best[2]:.4f} "
                          f"{best[1]}/掩{best[0]}   {tag}",flush=True)

            # --- 臂 S：全在 |S|=2 ---
            cfgS={}
            for (i,j) in sorted(set(map(tuple,np.sort(np.c_[s1,s2],axis=1)))):
                if i==j: continue
                best=(None,None,-1.,None)
                for mn,_ in MASKS:
                    for nm in CANDS:
                        inc,acc,_n=score(i,j,mn,nm,XS_s,yS_s,XS_t,yS_t,s1,s2)
                        if acc is None: continue
                        if acc>best[2]: best=(mn,nm,acc,inc)
                if best[0] is None or best[3] is None: continue
                if best[2]>best[3]+MARGIN: cfgS[(i,j)]=(best[0],best[1],best[2])

            # --- 混合臂：A 上检测选配置，S 上"不添乱"验收 ---
            CFGS={"A":cfgA,"S":cfgS}
            for tol in TOLS:
                c={}
                for k_,v in cfgA.items():
                    incS,accS,_n = onS.get(k_,(None,None,0))
                    if accS is None:  c[k_]=v            # S 上不可评 → 放行（保守：不因缺证据拒绝）
                    elif accS >= incS-tol: c[k_]=v
                CFGS[f"hyb{tol:.2f}"]=c
            print("  过闸对数  " + "  ".join(f"{k_}={len(v)}" for k_,v in CFGS.items()),flush=True)

            # --- outer ---
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
                fc={}
                out={}
                for aname,cfg in CFGS.items():
                    pred=base.copy()
                    if gsel.any() and cfg:
                        pg=g1.copy()
                        for (i,j),(mn,nm,_a) in cfg.items():
                            key=(i,j,mn,nm)
                            if key not in fc:
                                keep=MD[mn]; ms=np.isin(ys5,[i,j])
                                fc[key]=None if len(np.unique(ys5[ms]))<2 else \
                                    fit_w(MK(nm),nm,Xs[ms][:,keep],
                                          (ys5[ms]==j).astype(int)).predict_proba(Xt[gsel][:,keep])[:,1]
                            q=fc[key]
                            if q is None: continue
                            sel=((g1==i)&(g2==j))|((g1==j)&(g2==i))
                            if sel.any(): pg[sel]=np.where(q[sel]>=0.5,j,i)
                        pred[gsel]=pg
                    out[aname]=(f1_score(yt5,pred,average="macro",labels=np.arange(len(L5))),gwf(pred))
                    rows.append({"unit":uname,"seed":seed,"arm":aname,"flat":f_flat,"hier":f_h0,
                                 "cfg":out[aname][0],"delta":out[aname][0]-f_h0,
                                 "gw_hier":gwf(base),"gw_cfg":out[aname][1],"n_gate":len(cfg)})
                print(f"  {uname} seed{seed} flat={f_flat:.4f} 分层={f_h0:.4f}  " +
                      "  ".join(f"{a}={v[0]:.4f}" for a,v in out.items()),flush=True)

        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/cfg_hybrid.csv",index=False)
        print("\n=== 各臂（三单元 × 2 seed 均值）===",flush=True)
        A=R.groupby("arm").agg(n_gate=("n_gate","mean"),cfg=("cfg","mean"),
                               delta=("delta","mean"),gw=("gw_cfg","mean")).round(4)
        print(A.sort_values("delta",ascending=False).to_string(),flush=True)
        print(f"\n分层（不修）= {R.hier.mean():.4f}   扁平 = {R.flat.mean():.4f}",flush=True)
        print("\n=== 逐单元 Δ ===",flush=True)
        print(R.pivot_table(index="arm",columns="unit",values="delta").round(4).to_string(),flush=True)
        print("\n判读：hyb 各 tol 若与 A 持平 → 「不添乱」验收白送，可常开；",flush=True)
        print("      若 tol 小时掉到 S 水平 → 修复必然牺牲源域范围内准确率，该验收方向错误。",flush=True)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
