"""【探索性,非协议】自采数据上的新方法（P0-1），两段式。

设计依据（用户提出）：平滑的分组依据是"同一个可观测流"，而自采恰好有三个流——
Camera 自己的发射机、小米网关（内含 Light_T1/Light_XM/Sensor 三台）、Socket 自己的发射机
（见 three-labels-share-one-transmitter）。**平滑在发射机这一层合法**：部署时看得见 MAC。
流内部有几台设备是下一层的问题，那一层不能平滑（三台在同一个 MAC 里交织）。

  第一段  3 类（Camera / Gateway / Socket）   平滑合法，按发射机分组
  第二段  网关流内部 3 类（T1 / XM / Sensor）  不平滑，逐类对配置修复
  最终仍报【5 类窗口级 macro-F1】，标签空间不变。

保守实现：自采是"一台设备抓一小时"分开采的，但部署时三台交织。
故第一段把网关三类合成一个组、按 window_start 排序，人为造出交织流，
让平滑面对真实部署条件，不占数据集结构的便宜。

inner / outer（与 UNSW 的 inner 23→30 / outer 23→12 同构）：
  inner : R2+R3 → R4              只用源轮次，绝不触碰 R5/R6/R7
  outer : R2+R3+R4 → R5 / R6 / R7
特征池 183 列（与 override_183 同一个池，flat_183=0.8072 / override_183=0.8210 可直接对照）。
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

GATEWAY = ["Light_T1","Light_XM","Sensor"]      # 共用小米网关 54:ef:44:59:eb:4c
def TX(lab):                                     # 发射机（= 部署时可见的 MAC）
    return "Gateway" if lab in GATEWAY else lab

INNER_SRC, INNER_TGT = ["R2","R3"], "R4"
# 双 inner：两个互不重叠的源轮次对（协议 §2.4 内层历史 LORO，只用源轮次）
INNER_A = (["R2"], "R3")
INNER_B = (["R3"], "R2")
OUTER = [("pos_R5", ["R2","R3","R4"], "R5"),
         ("jit_R6", ["R2","R3","R4"], "R6"),
         ("jit_R7", ["R2","R3","R4"], "R7")]
CANDS=["lr","rf","xgboost","lightgbm"]
KBASE=[1,3,5,10,20]          # 第一段的平滑窗（在 inner 上选）
MARGIN=0.02; MIN_CELL=40
SEEDS=[42,43,44,45,46]
CKPT="/home/lmy/cic_probe/_ckpt_self"

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return TC.make_model(n,2)

def fit_w(m,name,X,y):
    w=compute_sample_weight("balanced", y)
    try:
        if name=="lr": m.fit(X,y, logisticregression__sample_weight=w)
        else:          m.fit(X,y, sample_weight=w)
    except Exception:
        m.fit(X,y)
    return m

def cmeanM(M,g,k):
    """按流分组的因果滑动平均。g 是【发射机】不是标签。"""
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def _fp():
    return hashlib.sha256(repr((sorted(CANDS),KBASE,MARGIN,MIN_CELL,"hier")).encode()).hexdigest()[:8]

def ck_load(seed):
    d={}
    try:
        with open(f"{CKPT}/hier_s{seed}_{_fp()}.jsonl") as f:
            for ln in f:
                r=json.loads(ln); d[(r["i"],r["j"])]=r["v"]
    except FileNotFoundError: pass
    return d

def ck_append(seed,i,j,v):
    os.makedirs(CKPT, exist_ok=True)
    with open(f"{CKPT}/hier_s{seed}_{_fp()}.jsonl","a") as f:
        f.write(json.dumps({"i":int(i),"j":int(j),"v":v})+"\n"); f.flush(); os.fsync(f.fileno())

def load183():
    a=hashlib.md5(TC.CACHE.read_bytes()).hexdigest()
    b=hashlib.md5(open(NEW89,"rb").read()).hexdigest()
    assert a==MD5_A, f"94 列缓存 md5 不符 {a}"
    assert b==MD5_B, f"89 列新特征 md5 不符 {b}"
    d=TC.Data(); new=pd.read_csv(NEW89)
    df=d.df.merge(new, on=KEYS, how="inner")
    assert len(df)==len(d.df), f"连接后行数变了 {len(df)} vs {len(d.df)}"
    return df, d

def gw_proba(Xtr, ytr5, Xte, n5, gidx):
    """在网关子集上训 3 类，再把概率散射回 5 列（其余列为 0），
    这样下游全程在 5 类索引空间里，不用来回映射。"""
    loc={c:k for k,c in enumerate(gidx)}
    yloc=np.array([loc[v] for v in ytr5])
    m=TC.make_model("xgboost",len(gidx)); m.fit(Xtr,yloc)
    P=m.predict_proba(Xte)
    out=np.zeros((len(Xte), n5), dtype=float)
    for k,c in enumerate(gidx): out[:,c]=P[:,k]
    return out

def main():
    t0=time.time()
    with threadpool_limits(1):
        df, d = load183()
        cols = TC.feature_columns(df)
        L5   = list(d.enc.classes_)                      # 5 类顺序
        TX5  = [TX(c) for c in L5]
        TX3  = sorted(set(TX5))                          # ['Camera','Gateway','Socket']
        G3   = [c for c in L5 if TX(c)=="Gateway"]       # 网关三类
        i5   = {c:n for n,c in enumerate(L5)}
        fams = TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none", np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f, keep))
        MD={m:k for m,k in MASKS}
        print(f"183 列  5 类 {L5}",flush=True)
        print(f"发射机 3 类 {TX3}   网关内 {G3}",flush=True)
        print(f"掩码候选 {len(MASKS)}  模型 {CANDS}  第一段平滑窗候选 {KBASE}",flush=True)

        def block(rounds, mix_gateway=False):
            """mix_gateway: 把网关三类按 window_start 交织排序，模拟部署时的混合流。"""
            s=df[df["round"].isin(rounds)].copy()
            s["tx"]=[TX(x) for x in s["label"]]
            s=s.sort_values(["tx","window_start"], kind="mergesort") if mix_gateway \
              else s.sort_values(["label","window_start"], kind="mergesort")
            return (np.asarray(TC.clean_x(s,cols),dtype=float),
                    np.array([i5[x] for x in s["label"]]),
                    np.asarray(s["tx"]))

        Xa,ya5,_    = block(INNER_SRC)
        Xb,yb5,gb   = block([INNER_TGT], mix_gateway=True)
        ya3=np.array([TX3.index(TX(L5[c])) for c in ya5])
        yb3=np.array([TX3.index(TX(L5[c])) for c in yb5])

        rows=[]
        gidx=[i5[c] for c in G3]
        for seed in SEEDS:
            TC.SEED=seed
            # ---------- 第一段：3 类发射机 + 平滑（在 inner 上选 k）----------
            m1=TC.make_model("xgboost",len(TX3)); m1.fit(Xa,ya3)
            P1=m1.predict_proba(Xb)
            kb=max(((k, f1_score(yb3, cmeanM(P1,gb,k).argmax(1), average="macro"))
                    for k in KBASE), key=lambda x:x[1])[0]
            f1_in=f1_score(yb3, cmeanM(P1,gb,kb).argmax(1), average="macro")
            print(f"\n  seed{seed}  第一段 inner：k={kb}  3类 macro={f1_in:.4f}"
                  f"（不平滑 {f1_score(yb3,P1.argmax(1),average='macro'):.4f}）",flush=True)

            # ---------- 第二段：网关流内部逐类对配置（inner 上导出）----------
            gm_a = np.isin(ya5, [i5[c] for c in G3])
            gm_b = np.isin(yb5, [i5[c] for c in G3])
            Xa_g, ya_g = Xa[gm_a], ya5[gm_a]
            Xb_g, yb_g = Xb[gm_b], yb5[gm_b]
            P2=gw_proba(Xa_g, ya_g, Xb_g, len(L5), gidx)
            o2=np.argsort(-P2,axis=1); c1,c2=o2[:,0],o2[:,1]
            cand=sorted(set(map(tuple,np.sort(np.c_[c1,c2],axis=1))))
            done=ck_load(seed); cfg={}; audit=[]
            for (i,j) in cand:
                if i==j: continue
                if (i,j) in done:
                    nD,inc,acc,mn,nm = done[(i,j)]
                    audit.append((i,j,nD,inc,acc,mn,nm))
                    if acc>inc+MARGIN: cfg[(i,j)]=(mn,nm,acc)
                    continue
                D=(((c1==i)&(c2==j))|((c1==j)&(c2==i)))&np.isin(yb_g,[i,j])
                if D.sum()<MIN_CELL: continue
                inc=float((c1[D]==yb_g[D]).mean())
                ma=np.isin(ya_g,[i,j])
                if len(np.unique(ya_g[ma]))<2: continue
                y1=(ya_g[ma]==j).astype(int); best=(None,None,-1.0)
                for mn,keep in MASKS:
                    for nm in CANDS:
                        try:
                            mm_=fit_w(MK(nm),nm,Xa_g[ma][:,keep],y1)
                            q=mm_.predict_proba(Xb_g[:,keep])[:,1]
                        except Exception: continue
                        acc=float((np.where(q[D]>=0.5,j,i)==yb_g[D]).mean())
                        if acc>best[2]: best=(mn,nm,acc)
                if best[1] is None: continue
                audit.append((i,j,int(D.sum()),inc,best[2],best[0],best[1]))
                ck_append(seed,i,j,[int(D.sum()),inc,best[2],best[0],best[1]])
                if best[2]>inc+MARGIN: cfg[(i,j)]=best
            print(f"      第二段 inner：候选 {len(cand)} 对，过闸 {len(cfg)} 对",flush=True)

            # ---------- 第二段（双 inner 版）：在 A 上选，要求在 B 上也成立 ----------
            def gw_block(rounds):
                Xr,yr,_ = block(rounds)
                m = np.isin(yr,[i5[c] for c in G3])
                return Xr[m], yr[m]
            def region(Xs_,ys_,Xt_,yt_):
                oo=np.argsort(-gw_proba(Xs_,ys_,Xt_,len(L5),gidx),axis=1)
                return oo[:,0],oo[:,1]
            XA_s,yA_s = gw_block(INNER_A[0]); XA_t,yA_t = gw_block([INNER_A[1]])
            XB_s,yB_s = gw_block(INNER_B[0]); XB_t,yB_t = gw_block([INNER_B[1]])
            a1,a2 = region(XA_s,yA_s,XA_t,yA_t)
            b1,b2 = region(XB_s,yB_s,XB_t,yB_t)
            def score_cfg(i,j,mn,nm,Xs_,ys_,Xt_,yt_,p1_,p2_):
                D=(((p1_==i)&(p2_==j))|((p1_==j)&(p2_==i)))&np.isin(yt_,[i,j])
                if D.sum()<MIN_CELL: return None,None
                inc=float((p1_[D]==yt_[D]).mean())
                ms=np.isin(ys_,[i,j])
                if len(np.unique(ys_[ms]))<2: return None,None
                keep=MD[mn]
                try:
                    pm=fit_w(MK(nm),nm,Xs_[ms][:,keep],(ys_[ms]==j).astype(int))
                    q=pm.predict_proba(Xt_[:,keep])[:,1]
                except Exception: return None,None
                return inc, float((np.where(q[D]>=0.5,j,i)==yt_[D]).mean())
            candA=sorted(set(map(tuple,np.sort(np.c_[a1,a2],axis=1))))
            cfg2={}; n_A_pass=0
            for (i,j) in candA:
                if i==j: continue
                bestA=(None,None,-1.0,None)
                for mn,_k in MASKS:
                    for nm in CANDS:
                        inc,acc = score_cfg(i,j,mn,nm,XA_s,yA_s,XA_t,yA_t,a1,a2)
                        if acc is None: continue
                        if acc>bestA[2]: bestA=(mn,nm,acc,inc)
                if bestA[0] is None or bestA[3] is None: continue
                if bestA[2] <= bestA[3]+MARGIN: continue
                n_A_pass += 1
                incB,accB = score_cfg(i,j,bestA[0],bestA[1],XB_s,yB_s,XB_t,yB_t,b1,b2)
                if accB is None or incB is None: continue
                if accB > incB+MARGIN:
                    cfg2[(i,j)]=(bestA[0],bestA[1],bestA[2])
                    print(f"        [双inner] {L5[i]}|{L5[j]:9s} A: {bestA[3]:.4f}→{bestA[2]:.4f}"
                          f"  B: {incB:.4f}→{accB:.4f}  {bestA[1]}/掩{bestA[0]}  **通过**",flush=True)
                else:
                    print(f"        [双inner] {L5[i]}|{L5[j]:9s} A: {bestA[3]:.4f}→{bestA[2]:.4f}"
                          f"  B: {incB:.4f}→{accB:.4f}  {bestA[1]}/掩{bestA[0]}  B 上不成立，拒绝",flush=True)
            print(f"      第二段双inner：A 上过闸 {n_A_pass} 对 → B 上确认 {len(cfg2)} 对",flush=True)
            for (i,j),(mn,nm,a) in sorted(cfg.items(), key=lambda x:-x[1][2]):
                r=[x for x in audit if x[0]==i and x[1]==j][0]
                print(f"        {L5[i]}|{L5[j]:9s} 可判决区 {r[2]:4d}  在位者 {r[3]:.4f} → "
                      f"配置 {a:.4f} ({a-r[3]:+.4f})  {nm}/掩{mn}",flush=True)

            # ---------- outer ----------
            for uname, src, tgt in OUTER:
                Xs,ys5,_   = block(src)
                Xt,yt5,gt  = block([tgt], mix_gateway=True)
                ys3=np.array([TX3.index(TX(L5[c])) for c in ys5])
                yt3=np.array([TX3.index(TX(L5[c])) for c in yt5])
                # 朴素 5 类基线（对照 flat_183）
                mflat=TC.make_model("xgboost",len(L5)); mflat.fit(Xs,ys5)
                pflat=mflat.predict_proba(Xt).argmax(1)
                f_flat=f1_score(yt5,pflat,average="macro",labels=np.arange(len(L5)))
                # 第一段
                s1=TC.make_model("xgboost",len(TX3)); s1.fit(Xs,ys3)
                p1=cmeanM(s1.predict_proba(Xt),gt,kb).argmax(1)
                # 第二段：只在被判为 Gateway 的窗口上跑
                gsel = p1==TX3.index("Gateway")
                gm_s = np.isin(ys5,[i5[c] for c in G3])
                pred_h=np.empty(len(yt5),dtype=int)
                for n,t in enumerate(TX3):
                    if t=="Gateway": continue
                    pred_h[p1==n]=i5[t]
                if gsel.any():
                    Pg=gw_proba(Xs[gm_s], ys5[gm_s], Xt[gsel], len(L5), gidx)
                    og=np.argsort(-Pg,axis=1); g1,g2=og[:,0],og[:,1]
                    pred_g=g1.copy()
                    for (i,j),(mn,nm,a) in cfg.items():
                        keep=MD[mn]; ms=np.isin(ys5,[i,j])
                        if len(np.unique(ys5[ms]))<2: continue
                        sel=((g1==i)&(g2==j))|((g1==j)&(g2==i))
                        if not sel.any(): continue
                        pm=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys5[ms]==j).astype(int))
                        q=pm.predict_proba(Xt[gsel][:,keep])[:,1]
                        pred_g[sel]=np.where(q[sel]>=0.5,j,i)
                    pred_g2=g1.copy()
                    for (i,j),(mn,nm,a) in cfg2.items():
                        keep=MD[mn]; ms=np.isin(ys5,[i,j])
                        if len(np.unique(ys5[ms]))<2: continue
                        sel=((g1==i)&(g2==j))|((g1==j)&(g2==i))
                        if not sel.any(): continue
                        pm=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys5[ms]==j).astype(int))
                        q=pm.predict_proba(Xt[gsel][:,keep])[:,1]
                        pred_g2[sel]=np.where(q[sel]>=0.5,j,i)
                    pred_h0=pred_h.copy(); pred_h0[gsel]=g1      # 分层但不修
                    pred_h2=pred_h.copy(); pred_h2[gsel]=pred_g2 # 分层 + 双inner修
                    pred_h[gsel]=pred_g                           # 分层 + 单inner修
                else:
                    pred_h0=pred_h.copy(); pred_h2=pred_h.copy()
                f_h0=f1_score(yt5,pred_h0,average="macro",labels=np.arange(len(L5)))
                f_h =f1_score(yt5,pred_h ,average="macro",labels=np.arange(len(L5)))
                f_h2=f1_score(yt5,pred_h2,average="macro",labels=np.arange(len(L5)))
                # 并报：网关三类单独的 macro（方法真正作用的地方）
                # 依据 experiment_protocol_final.md:257 —— Socket 在 110 条结果中 96 条 F1=1.000，
                # 5 类 macro 里有 1/5 是免费的，增益被稀释 1/5
                gidx=[i5[c] for c in G3]
                gmask=np.isin(yt5,gidx)
                gw=lambda p: f1_score(yt5[gmask],p[gmask],average="macro",labels=gidx)
                g_flat,g_h0,g_h,g_h2 = gw(pflat),gw(pred_h0),gw(pred_h),gw(pred_h2)
                print(f"        网关三类 macro  flat={g_flat:.4f}  分层={g_h0:.4f}  "
                      f"+单inner={g_h:.4f}  +双inner={g_h2:.4f}",flush=True)
                F=lambda p: f1_score(yt5,p,average=None,labels=np.arange(len(L5)))
                print(f"    {uname} seed{seed}  flat={f_flat:.4f}  分层={f_h0:.4f}"
                      f"  +单inner修={f_h:.4f}  +双inner修={f_h2:.4f}   "
                      f"Δ(单)={f_h-f_h0:+.4f}  Δ(双)={f_h2-f_h0:+.4f}",flush=True)
                Ff,Fh=F(pflat),F(pred_h)
                print("        逐类  " + "  ".join(
                    f"{L5[c][:9]} {Ff[c]:.3f}→{Fh[c]:.3f}" for c in range(len(L5))),flush=True)
                rows.append({"unit":uname,"seed":seed,"flat":f_flat,"hier":f_h0,
                             "hier_cfg":f_h,"hier_cfg2":f_h2,"k_base":kb,
                             "gw_flat":g_flat,"gw_hier":g_h0,"gw_cfg":g_h,"gw_cfg2":g_h2,
                             "n_gate":len(cfg),"n_gate2":len(cfg2)})
        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/cfg_self_hier.csv",index=False)
        print("\n=== 汇总（5 类窗口级 macro-F1）===",flush=True)
        print(R.groupby("unit")[["flat","hier","hier_cfg","hier_cfg2"]].mean().round(4).to_string(),flush=True)
        print(f"\n三单元总均值  flat={R.flat.mean():.4f}  分层={R.hier.mean():.4f}  "
              f"+单inner修={R.hier_cfg.mean():.4f}  +双inner修={R.hier_cfg2.mean():.4f}",flush=True)
        print(f"过闸对数  单inner {R.n_gate.mean():.1f} 对   双inner {R.n_gate2.mean():.1f} 对",flush=True)
        print("\n=== 网关三类 macro（方法真正作用处，不含免费的 Socket/Camera）===",flush=True)
        print(R.groupby("unit")[["gw_flat","gw_hier","gw_cfg","gw_cfg2"]].mean().round(4).to_string(),flush=True)
        print(f"三单元均值  flat={R.gw_flat.mean():.4f}  分层={R.gw_hier.mean():.4f}  "
              f"+单inner={R.gw_cfg.mean():.4f}  +双inner={R.gw_cfg2.mean():.4f}",flush=True)
        print(f"参照：override_183 的 flat_183=0.8072、override_183=0.8210",flush=True)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
