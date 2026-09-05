"""【探索性,非协议】inner 切法扫描：`|S|=1` 的优势是规则、是原则、还是运气？

背景：cfg_ablation 已定案——"双 inner 验证"零贡献，全部功劳在 inner 的 `|S|`：
    |S|=2 (R2+R3→R4)  Δ=+0.0050   过闸 2 对
    |S|=1 (R2→R3)     Δ=+0.0188   过闸 1 对
一个我没有理论的参数，占了方法总增益的 73%。必须搞清楚。

轮次几何（目录名实证）：R2/R3/R4 = normal（同位置同操作），R5 = positionB，R6/R7 = jitter。
=> 没有任何 inner 任务里含位置漂移，"inner 复现 outer 的 regime"这个说法不成立。
   剩下的机制猜想是【放大】：R2/R3/R4 之间只差时间与噪声；训两轮 → 模型见到两份 rssi
   实现、自动降权 → 在位者在 R4 上正常 → 闸门看不见病；训一轮 → 模型死抓这一轮的 rssi
   → 在位者在 R3 上崩 → 闸门看见病。outer 训三轮降权更多，但 R5 的位移远大于 normal
   轮次间漂移，照样崩。即：`|S|=1` 是用小位移把同一个病放大到可见的【压力测试】。

预注册（结果出来前写死，不得事后改）：
  H1 原则  —— 6 个 |S|=1 分裂成能用/不能用两组，且"inner 暴露的最低在位者" min_inc
             能预测 outer Δ（Spearman ρ 显著为负）。=> 得到设计时可算的选择规则。
  H2 规则  —— 6 个 |S|=1 全部打赢 3 个 |S|=2。=> `|S|=1` 是硬规则（仍缺解释）。
  H3 运气  —— 只有 R2→R3 能用，min_inc 与 outer Δ 无关。
             => +0.0188 是 9 选 1 抽中的，方法真实增益退回 ≈ +0.005。

判据：min_inc 与 outer Δ 的 Spearman ρ。|ρ|≥0.7 且方向为负 → H1；
      否则看 |S|=1 组是否整体高于 |S|=2 组（组间不重叠 → H2）；都不满足 → H3。
注意：min_inc 只用 inner 数据算，选 inner 时【不看 outer】，所以 H1 成立的话规则是可部署的。

固定项：第一段平滑窗 kb 仍从原 inner (R2+R3→R4) 选，全臂共用 —— 只让第二段 config 随
inner 变，隔离出纯 |S| 效应。outer 的基础预测按 (unit,seed) 缓存，9 个 cfg 复用。
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
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import run_two_channel as TC

NEW89 = REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
KEYS  = ["label","round","source_file","window_id"]
MD5_A = "703984b6ad2fde2f45e0cce1c6df31be"
MD5_B = "e0586ea2b55c6913a614a1f94772e701"

GATEWAY = ["Light_T1","Light_XM","Sensor"]
def TX(lab): return "Gateway" if lab in GATEWAY else lab

# 9 种 inner 切法（只用源轮次 R2/R3/R4，绝不触碰 R5/R6/R7）
INNERS = [("R2>R3",["R2"],"R3"), ("R2>R4",["R2"],"R4"), ("R3>R2",["R3"],"R2"),
          ("R3>R4",["R3"],"R4"), ("R4>R2",["R4"],"R2"), ("R4>R3",["R4"],"R3"),
          ("R2R3>R4",["R2","R3"],"R4"), ("R2R4>R3",["R2","R4"],"R3"),
          ("R3R4>R2",["R3","R4"],"R2")]
KB_INNER = (["R2","R3"], "R4")          # 第一段平滑窗从这里选，全臂共用
OUTER = [("pos_R5",["R2","R3","R4"],"R5"),
         ("jit_R6",["R2","R3","R4"],"R6"),
         ("jit_R7",["R2","R3","R4"],"R7")]
CANDS=["lr","rf","xgboost","lightgbm"]
KBASE=[1,3,5,10,20]
MARGIN=0.02; MIN_CELL=40
SEEDS=[42,43]
CKPT="/home/lmy/cic_probe/_ckpt_sweep"

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
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def _fp():
    return hashlib.sha256(repr((sorted(CANDS),KBASE,MARGIN,MIN_CELL,"sweep9")).encode()).hexdigest()[:8]

def ck_load(seed,tag):
    d={}
    try:
        with open(f"{CKPT}/{tag}_s{seed}_{_fp()}.jsonl") as f:
            for ln in f:
                r=json.loads(ln); d[(r["i"],r["j"])]=r["v"]
    except FileNotFoundError: pass
    return d

def ck_append(seed,tag,i,j,v):
    os.makedirs(CKPT, exist_ok=True)
    with open(f"{CKPT}/{tag}_s{seed}_{_fp()}.jsonl","a") as f:
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

def main():
    t0=time.time()
    with threadpool_limits(1):
        df, d = load183()
        cols = TC.feature_columns(df)
        L5   = list(d.enc.classes_)
        TX3  = sorted(set(TX(c) for c in L5))
        G3   = [c for c in L5 if TX(c)=="Gateway"]
        i5   = {c:n for n,c in enumerate(L5)}
        GIDX = [i5[c] for c in G3]
        fams = TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none", np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f, keep))
        MD={m:k for m,k in MASKS}
        print(f"183 列  5 类 {L5}   网关内 {G3}",flush=True)
        print(f"掩码 {len(MASKS)}  模型 {CANDS}  inner 切法 {len(INNERS)} 种  seed {SEEDS}",flush=True)

        def block(rounds, mix_gateway=False):
            s=df[df["round"].isin(rounds)].copy()
            s["tx"]=[TX(x) for x in s["label"]]
            s=s.sort_values(["tx","window_start"], kind="mergesort") if mix_gateway \
              else s.sort_values(["label","window_start"], kind="mergesort")
            return (np.asarray(TC.clean_x(s,cols),dtype=float),
                    np.array([i5[x] for x in s["label"]]),
                    np.asarray(s["tx"]))

        def gw_proba(Xtr, ytr5, Xte):
            loc={c:k for k,c in enumerate(GIDX)}
            yloc=np.array([loc[v] for v in ytr5])
            m=TC.make_model("xgboost",len(GIDX)); m.fit(Xtr,yloc)
            P=m.predict_proba(Xte); out=np.zeros((len(Xte), len(L5)), dtype=float)
            for k,c in enumerate(GIDX): out[:,c]=P[:,k]
            return out

        def gw_block(rounds):
            Xr,yr,_ = block(rounds)
            m = np.isin(yr, GIDX)
            return Xr[m], yr[m]

        rows=[]; picks=[]
        for seed in SEEDS:
            TC.SEED=seed
            # ---- 第一段平滑窗（固定 inner，全臂共用）----
            Xa,ya5,_  = block(KB_INNER[0])
            Xb,yb5,gb = block([KB_INNER[1]], mix_gateway=True)
            ya3=np.array([TX3.index(TX(L5[c])) for c in ya5])
            yb3=np.array([TX3.index(TX(L5[c])) for c in yb5])
            m1=TC.make_model("xgboost",len(TX3)); m1.fit(Xa,ya3)
            P1=m1.predict_proba(Xb)
            kb=max(((k, f1_score(yb3, cmeanM(P1,gb,k).argmax(1), average="macro"))
                    for k in KBASE), key=lambda x:x[1])[0]
            print(f"\n=== seed{seed}  第一段平滑窗 kb={kb}（固定，全臂共用）===",flush=True)

            # ---- 9 种 inner 各自导出 config ----
            CFGS={}
            for tag, isrc, itgt in INNERS:
                Xs_,ys_ = gw_block(isrc); Xt_,yt_ = gw_block([itgt])
                oo=np.argsort(-gw_proba(Xs_,ys_,Xt_),axis=1); p1_,p2_=oo[:,0],oo[:,1]
                cand=sorted(set(map(tuple,np.sort(np.c_[p1_,p2_],axis=1))))
                done=ck_load(seed,tag); cfg={}; incs=[]
                for (i,j) in cand:
                    if i==j: continue
                    if (i,j) in done:
                        nD,inc,acc,mn,nm = done[(i,j)]
                    else:
                        D=(((p1_==i)&(p2_==j))|((p1_==j)&(p2_==i)))&np.isin(yt_,[i,j])
                        if D.sum()<MIN_CELL: continue
                        inc=float((p1_[D]==yt_[D]).mean())
                        ms=np.isin(ys_,[i,j])
                        if len(np.unique(ys_[ms]))<2: continue
                        y1=(ys_[ms]==j).astype(int); best=(None,None,-1.0)
                        for mn_,keep in MASKS:
                            for nm_ in CANDS:
                                try:
                                    q=fit_w(MK(nm_),nm_,Xs_[ms][:,keep],y1).predict_proba(Xt_[:,keep])[:,1]
                                except Exception: continue
                                a_=float((np.where(q[D]>=0.5,j,i)==yt_[D]).mean())
                                if a_>best[2]: best=(mn_,nm_,a_)
                        if best[1] is None: continue
                        nD,acc,mn,nm = int(D.sum()),best[2],best[0],best[1]
                        ck_append(seed,tag,i,j,[nD,inc,acc,mn,nm])
                    incs.append(inc)
                    if acc>inc+MARGIN:
                        cfg[(i,j)]=(mn,nm,acc)
                        picks.append({"seed":seed,"inner":tag,"pair":f"{L5[i]}|{L5[j]}",
                                      "n":nD,"inc":inc,"acc":acc,"mask":mn,"model":nm})
                mi = min(incs) if incs else float("nan")
                CFGS[tag]=(cfg, mi, len(cfg))
                pr=" ".join(f"{L5[i][:8]}|{L5[j][:8]}" for (i,j) in cfg)
                print(f"  {tag:9s} |S|={len(isrc)}  评估 {len(incs)} 对  "
                      f"min在位者={mi:.4f}  过闸 {len(cfg)} 对  {pr}",flush=True)

            # ---- outer：基础预测按 (unit,seed) 算一次，9 个 cfg 复用 ----
            for uname, src, tgt in OUTER:
                Xs,ys5,_  = block(src)
                Xt,yt5,gt = block([tgt], mix_gateway=True)
                ys3=np.array([TX3.index(TX(L5[c])) for c in ys5])
                mflat=TC.make_model("xgboost",len(L5)); mflat.fit(Xs,ys5)
                f_flat=f1_score(yt5,mflat.predict_proba(Xt).argmax(1),
                                average="macro",labels=np.arange(len(L5)))
                s1=TC.make_model("xgboost",len(TX3)); s1.fit(Xs,ys3)
                p1=cmeanM(s1.predict_proba(Xt),gt,kb).argmax(1)
                gsel = p1==TX3.index("Gateway")
                base=np.empty(len(yt5),dtype=int)
                for n,t in enumerate(TX3):
                    if t!="Gateway": base[p1==n]=i5[t]
                gm_s=np.isin(ys5,GIDX)
                if gsel.any():
                    og=np.argsort(-gw_proba(Xs[gm_s],ys5[gm_s],Xt[gsel]),axis=1)
                    g1,g2=og[:,0],og[:,1]
                    base[gsel]=g1
                f_h0=f1_score(yt5,base,average="macro",labels=np.arange(len(L5)))
                gmask=np.isin(yt5,GIDX)
                gwf=lambda p: f1_score(yt5[gmask],p[gmask],average="macro",labels=GIDX)
                # 逐 cfg 应用（每个 (pair,mask,model) 只在 outer 源上重训一次，跨 cfg 缓存）
                fitcache={}
                for tag,(cfg,mi,ng) in CFGS.items():
                    pred=base.copy()
                    if gsel.any() and cfg:
                        pg=g1.copy()
                        for (i,j),(mn,nm,_a) in cfg.items():
                            key=(i,j,mn,nm)
                            if key not in fitcache:
                                keep=MD[mn]; ms=np.isin(ys5,[i,j])
                                if len(np.unique(ys5[ms]))<2: fitcache[key]=None
                                else:
                                    pm=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys5[ms]==j).astype(int))
                                    fitcache[key]=pm.predict_proba(Xt[gsel][:,keep])[:,1]
                            q=fitcache[key]
                            if q is None: continue
                            sel=((g1==i)&(g2==j))|((g1==j)&(g2==i))
                            if sel.any(): pg[sel]=np.where(q[sel]>=0.5,j,i)
                        pred[gsel]=pg
                    f_=f1_score(yt5,pred,average="macro",labels=np.arange(len(L5)))
                    rows.append({"unit":uname,"seed":seed,"inner":tag,
                                 "S":1 if ">"==tag[2] else len(tag.split(">")[0])//2,
                                 "flat":f_flat,"hier":f_h0,"cfg":f_,"delta":f_-f_h0,
                                 "min_inc":mi,"n_gate":ng,"gw_hier":gwf(base),"gw_cfg":gwf(pred)})
                cur={r["inner"]:r["cfg"] for r in rows
                     if r["unit"]==uname and r["seed"]==seed}
                line="  ".join(f"{t}={cur[t]:.4f}" for t,_,_ in INNERS)
                print(f"  {uname} seed{seed}  flat={f_flat:.4f} 分层={f_h0:.4f}  {line}",flush=True)

        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/inner_sweep.csv",index=False)
        P=pd.DataFrame(picks); P.to_csv("/home/lmy/cic_probe/inner_sweep_picks.csv",index=False)

        print("\n=== 每种 inner 切法：三单元 × 2 seed 平均 ===",flush=True)
        agg=R.groupby("inner").agg(S=("S","first"), min_inc=("min_inc","mean"),
                                   n_gate=("n_gate","mean"), hier=("hier","mean"),
                                   cfg=("cfg","mean"), delta=("delta","mean")).round(4)
        agg=agg.sort_values("delta",ascending=False)
        print(agg.to_string(),flush=True)

        print("\n=== 逐单元 Δ（看增益是否只在 pos_R5）===",flush=True)
        print(R.pivot_table(index="inner",columns="unit",values="delta").round(4).to_string(),flush=True)

        print("\n=== 预注册判据 ===",flush=True)
        s1_=agg[agg.S==1]["delta"]; s2_=agg[agg.S==2]["delta"]
        rho,pv=spearmanr(agg["min_inc"], agg["delta"])
        print(f"  Spearman ρ(min_inc, Δ) = {rho:+.3f}  p={pv:.4f}   （H1 要求 ρ≤−0.7）",flush=True)
        print(f"  |S|=1 组 Δ: min={s1_.min():+.4f} max={s1_.max():+.4f} 均值={s1_.mean():+.4f}",flush=True)
        print(f"  |S|=2 组 Δ: min={s2_.min():+.4f} max={s2_.max():+.4f} 均值={s2_.mean():+.4f}",flush=True)
        sep = s1_.min() > s2_.max()
        print(f"  两组是否不重叠（H2 要求 True）: {sep}",flush=True)
        n_pos = (agg["delta"]>0.01).sum()
        print(f"  Δ>0.01 的切法数: {n_pos}/9   （H3 的标志是仅 1 个且 ρ 不显著）",flush=True)
        if rho<=-0.7 and pv<0.05:   verdict="H1 —— min_inc 可预测，得到设计时可算的选择规则"
        elif sep:                    verdict="H2 —— |S|=1 是硬规则"
        elif n_pos<=1:               verdict="H3 —— 运气；+0.0188 需退回"
        else:                        verdict="都不干净，需要看逐格数据"
        print(f"\n  >>> 判定: {verdict}",flush=True)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
