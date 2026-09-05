"""【探索性,非协议】三层判决：按物理拓扑分层，而不是按标签平铺。

用户 2026-09-05 给出的接入结构：
    Socket / Camera   → WiFi   → 路由器     自有 802.11 电台
    Light_XM          → 蓝牙   → 智能网关   网关代发
    Light_T1 / Sensor → Zigbee → 智能网关   网关代发

于是困难簇是「1 个 BLE + 2 个 Zigbee」，难度分层而非一团。逐对审计印证（_ckpt_sweep）：

    类对                协议关系          出现/9 inner  在位者均值  众数配置
    Light_XM|Sensor    BLE vs Zigbee         4         0.7198    lightgbm/掩rssi
    Light_T1|Light_XM  Zigbee vs BLE         8         0.5916    xgboost /掩rssi
    Light_T1|Sensor    Zigbee vs Zigbee      9         0.4639    lr      /掩rssi  ← 唯一退到 LR

最容易那对只在 4/9 个 inner 里进过可判决区（多数时候不争 top-2），最难那对 9/9 全进。
=> BLE 对 Zigbee 基本已解决，残余全在同协议那一对。

本脚本测：把网关内的 3 分类拆成【BLE vs Zigbee】+【Zigbee 内二分】两级，是否优于平铺 3 类。
依据 hierarchical-decision-works（09-04）：两层赢平铺 +0.0528，部分原因是"少了混淆目标，
簇内更易分"—— 同一逻辑应在这里再起一次作用。

**分组的可导出性**：BLE/Zigbee 这个划分不是外部知识走后门 —— `self_fingerprint.py` 实测
帧长分布 TV 已独立还原它（跨协议 0.426/0.498 > 同协议 0.315，噪声底 0.110–0.239），
顺序完全一致且测于知道协议之前。本脚本用已知分组实现，可导出性由那份实测背书。

臂：
  flat5      平铺 5 类
  h2         两层（现方法）：发射机 3 类 + 平滑 → 网关内平铺 3 类
  h2_cfg     h2 + 逐类对定点修
  h3         三层：发射机 + 平滑 → BLE/Zigbee 二分 → Zigbee 内二分
  h3_cfg     h3 + 只在 Zigbee 内那一对上定点修
"""
from __future__ import annotations
import os, sys, time, hashlib
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

NEW89=REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
KEYS=["label","round","source_file","window_id"]
BLE=["Light_XM"]; ZIG=["Light_T1","Sensor"]; GATEWAY=BLE+ZIG
def TX(l): return "Gateway" if l in GATEWAY else l
def L2(l): return "BLE" if l in BLE else "Zigbee"

INNER=(["R2","R3"],"R4")
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
        d=TC.Data(); new=pd.read_csv(NEW89)
        df=d.df.merge(new,on=KEYS,how="inner"); assert len(df)==len(d.df)
        cols=TC.feature_columns(df); L5=list(d.enc.classes_)
        i5={c:n for n,c in enumerate(L5)}
        TX3=sorted(set(TX(c) for c in L5)); GIDX=[i5[c] for c in GATEWAY]
        ZIDX=[i5[c] for c in ZIG]; BIDX=[i5[c] for c in BLE]
        fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none",np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f,keep))
        MD={m:k for m,k in MASKS}
        print(f"5 类 {L5}   发射机 {TX3}   BLE {BLE}   Zigbee {ZIG}",flush=True)

        def block(rounds,mix=False):
            s=df[df["round"].isin(rounds)].copy(); s["tx"]=[TX(x) for x in s["label"]]
            s=s.sort_values(["tx","window_start"],kind="mergesort") if mix \
              else s.sort_values(["label","window_start"],kind="mergesort")
            return (np.asarray(TC.clean_x(s,cols),dtype=float),
                    np.array([i5[x] for x in s["label"]]), np.asarray(s["tx"]))

        def sub(Xtr,ytr,Xte,idxs,model="xgboost"):
            """在 idxs 这几类的子集上训分类器，返回散射回 5 列的概率。"""
            loc={c:k for k,c in enumerate(idxs)}
            m=TC.make_model(model,len(idxs)); m.fit(Xtr,np.array([loc[v] for v in ytr]))
            P=m.predict_proba(Xte); out=np.zeros((len(Xte),len(L5)))
            for k,c in enumerate(idxs): out[:,c]=P[:,k]
            return out

        rows=[]
        for seed in SEEDS:
            TC.SEED=seed
            # 第一段平滑窗（在 inner 上选）
            Xa,ya,_=block(INNER[0]); Xb,yb,gb=block([INNER[1]],mix=True)
            ya3=np.array([TX3.index(TX(L5[c])) for c in ya])
            yb3=np.array([TX3.index(TX(L5[c])) for c in yb])
            m1=TC.make_model("xgboost",len(TX3)); m1.fit(Xa,ya3); P1=m1.predict_proba(Xb)
            kb=max(((k,f1_score(yb3,cmeanM(P1,gb,k).argmax(1),average="macro")) for k in KBASE),
                   key=lambda x:x[1])[0]

            # 逐类对配置：在 |S|=1 inner 上导出（九种切法均为正，这里取其中一种，
            # 分布见 inner_sweep；本实验比的是层级结构不是 inner 选法）
            XA,yA,_=block(INNER_A[0]); XT,yT,_=block([INNER_A[1]])
            ma=np.isin(yA,GIDX); mt=np.isin(yT,GIDX)
            PA=sub(XA[ma],yA[ma],XT[mt],GIDX)
            oo=np.argsort(-PA,axis=1); a1,a2=oo[:,0],oo[:,1]
            cfg={}
            for (i,j) in sorted(set(map(tuple,np.sort(np.c_[a1,a2],axis=1)))):
                if i==j: continue
                D=(((a1==i)&(a2==j))|((a1==j)&(a2==i)))&np.isin(yT[mt],[i,j])
                if D.sum()<MIN_CELL: continue
                inc=float((a1[D]==yT[mt][D]).mean())
                ms=np.isin(yA[ma],[i,j])
                if len(np.unique(yA[ma][ms]))<2: continue
                y1=(yA[ma][ms]==j).astype(int); best=(None,None,-1.)
                for mn,keep in MASKS:
                    for nm in CANDS:
                        try:
                            q=fit_w(MK(nm),nm,XA[ma][ms][:,keep],y1).predict_proba(XT[mt][:,keep])[:,1]
                        except Exception: continue
                        acc=float((np.where(q[D]>=0.5,j,i)==yT[mt][D]).mean())
                        if acc>best[2]: best=(mn,nm,acc)
                if best[1] and best[2]>inc+MARGIN:
                    cfg[(i,j)]=best
                    print(f"  seed{seed} 过闸 {L5[i]}|{L5[j]:9s} {inc:.4f}→{best[2]:.4f} "
                          f"{best[1]}/掩{best[0]}",flush=True)
            print(f"  seed{seed} kb={kb}  过闸 {len(cfg)} 对",flush=True)

            for uname,src,tgt in OUTER:
                Xs,ys,_=block(src); Xt,yt,gt=block([tgt],mix=True)
                ys3=np.array([TX3.index(TX(L5[c])) for c in ys])
                mf=TC.make_model("xgboost",len(L5)); mf.fit(Xs,ys)
                f_flat=f1_score(yt,mf.predict(Xt),average="macro",labels=np.arange(len(L5)))
                st=TC.make_model("xgboost",len(TX3)); st.fit(Xs,ys3)
                p1=cmeanM(st.predict_proba(Xt),gt,kb).argmax(1)
                gsel=p1==TX3.index("Gateway")
                stub=np.empty(len(yt),dtype=int)
                for n,t in enumerate(TX3):
                    if t!="Gateway": stub[p1==n]=i5[t]
                gm_s=np.isin(ys,GIDX)
                out={}
                if gsel.any():
                    # --- h2：网关内平铺 3 类 ---
                    Pg=sub(Xs[gm_s],ys[gm_s],Xt[gsel],GIDX)
                    og=np.argsort(-Pg,axis=1); g1,g2=og[:,0],og[:,1]
                    h2=stub.copy(); h2[gsel]=g1
                    # --- h3：BLE/Zigbee 二分 → Zigbee 内二分 ---
                    y2=np.array([0 if L2(L5[c])=="BLE" else 1 for c in ys[gm_s]])
                    mm=TC.make_model("xgboost",2); mm.fit(Xs[gm_s],y2)
                    isz=mm.predict(Xt[gsel])==1
                    zs=np.isin(ys,ZIDX)
                    g3=np.full(gsel.sum(), BIDX[0], dtype=int)
                    if isz.any():
                        Pz=sub(Xs[zs],ys[zs],Xt[gsel][isz],ZIDX)
                        g3[isz]=np.argsort(-Pz,axis=1)[:,0]
                    h3=stub.copy(); h3[gsel]=g3
                    # --- 定点修 ---
                    def apply_cfg(pred_g, top1, top2):
                        pg=pred_g.copy()
                        for (i,j),(mn,nm,_a) in cfg.items():
                            keep=MD[mn]; ms=np.isin(ys,[i,j])
                            if len(np.unique(ys[ms]))<2: continue
                            sel=((top1==i)&(top2==j))|((top1==j)&(top2==i))
                            if not sel.any(): continue
                            q=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys[ms]==j).astype(int)
                                    ).predict_proba(Xt[gsel][:,keep])[:,1]
                            pg[sel]=np.where(q[sel]>=0.5,j,i)
                        return pg
                    h2c=stub.copy(); h2c[gsel]=apply_cfg(g1,g1,g2)
                    # h3 的 top-2：Zigbee 内是二分，另一侧固定为 BLE
                    t1=g3.copy(); t2=np.where(isz, BIDX[0], ZIDX[0])
                    for k_ in range(len(t2)):
                        if isz[k_]:
                            t2[k_]= ZIDX[1] if g3[k_]==ZIDX[0] else ZIDX[0]
                    h3c=stub.copy(); h3c[gsel]=apply_cfg(g3,t1,t2)
                else:
                    h2=h3=h2c=h3c=stub.copy()
                F=lambda p: f1_score(yt,p,average="macro",labels=np.arange(len(L5)))
                gmask=np.isin(yt,GIDX)
                GW=lambda p: f1_score(yt[gmask],p[gmask],average="macro",labels=GIDX)
                rec={"unit":uname,"seed":seed,"kb":kb,"n_gate":len(cfg),
                     "flat5":f_flat,"h2":F(h2),"h2_cfg":F(h2c),"h3":F(h3),"h3_cfg":F(h3c),
                     "gw_h2":GW(h2),"gw_h2_cfg":GW(h2c),"gw_h3":GW(h3),"gw_h3_cfg":GW(h3c)}
                rows.append(rec)
                print(f"    {uname} seed{seed}  flat={f_flat:.4f}  h2={rec['h2']:.4f} "
                      f"h2_cfg={rec['h2_cfg']:.4f}  h3={rec['h3']:.4f} h3_cfg={rec['h3_cfg']:.4f}"
                      f"   网关 h2={rec['gw_h2']:.4f} h3={rec['gw_h3']:.4f}",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/cfg_hier3.csv",index=False)
    print("\n=== 5 类窗口级 macro-F1（3 seed 均值）===",flush=True)
    print(R.groupby("unit")[["flat5","h2","h2_cfg","h3","h3_cfg"]].mean().round(4).to_string(),flush=True)
    print("\n=== 网关三类 macro ===",flush=True)
    print(R.groupby("unit")[["gw_h2","gw_h2_cfg","gw_h3","gw_h3_cfg"]].mean().round(4).to_string(),flush=True)
    print("\n=== 总均值 ===",flush=True)
    for c in ["flat5","h2","h2_cfg","h3","h3_cfg"]:
        print(f"  {c:8s} {R[c].mean():.4f}",flush=True)
    print(f"\n  h3 − h2         = {R.h3.mean()-R.h2.mean():+.4f}",flush=True)
    print(f"  h3_cfg − h2_cfg = {R.h3_cfg.mean()-R.h2_cfg.mean():+.4f}",flush=True)
    print(f"  网关 h3 − h2    = {R.gw_h3.mean()-R.gw_h2.mean():+.4f}",flush=True)
    print("\n判读：h3 明显 > h2 → 按物理拓扑分层有增益，且分组可由帧长指纹导出；",flush=True)
    print("      h3 ≈ h2 → 三层不值得，两层已够；h3 < h2 → 二分级的错误传播吃掉了收益。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
