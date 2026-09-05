"""【探索性,非协议】自采数据上的新方法（P0-1）。

与 UNSW/CIC 版的关键差别，以及为什么：

1. **主判据 k=1，不做时间平滑。**
   自采的 `Light_T1`/`Light_XM`/`Sensor` 共用同一个小米网关 MAC
   （见 three-labels-share-one-transmitter）。真实部署里三台同时在线，
   从网关 MAC 看到的是混合流 —— 按设备分组等于用答案分组，不合法。
   数据集"一台设备抓一小时"的结构让我们能这么分，但那是采集方式的产物。
   并报 k>1 版本，但明确标注"依赖单设备抓包结构，不可部署"，不进主表。

2. **不做型号级合并。** 网关三类是功能不同的三种设备，合并等于放弃任务
   （与 CIC 的 11 台同款插座相反）。

3. inner / outer 划分（与 UNSW 的 inner 23→30 / outer 23→12 同构）：
       inner  : R2+R3 → R4        （只用源轮次，绝不触碰 R5/R6/R7）
       outer  : R2+R3+R4 → R5 / R6 / R7   （pos_R5 / jit_R6 / jit_R7）

4. 特征池：183 列（94 列缓存左连接 89 列新特征，键 label/round/source_file/window_id），
   与 override_183 同一个池，便于和它的 +0.0554 直接比。

配置 = (特征掩码, 模型)，在 inner 上按【可判决区准确率】选，
闸门 = 相对在位者有增益（MARGIN）。outer 完全 held-out。
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

INNER_SRC, INNER_TGT = ["R2","R3"], "R4"
OUTER = [("pos_R5", ["R2","R3","R4"], "R5"),
         ("jit_R6", ["R2","R3","R4"], "R6"),
         ("jit_R7", ["R2","R3","R4"], "R7")]
CANDS=["lr","rf","xgboost","lightgbm"]
KS_MAIN=[1]                      # 主判据：不平滑
KS_AUX=[1,3,5,10]                # 并报（不可部署，仅参考）
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

def cmean(p,g,k):
    if k<=1: return p
    o=np.empty_like(p)
    for u in np.unique(g):
        i=np.where(g==u)[0]; v=p[i]; c=np.cumsum(np.insert(v,0,0.0))
        for n in range(len(v)):
            lo=max(0,n-k+1); o[i[n]]=(c[n+1]-c[lo])/(n+1-lo)
    return o

def cmeanM(M,g,k):
    if k<=1: return M
    o=np.empty_like(M)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=M[i]; C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

def _fp():
    return hashlib.sha256(repr((sorted(CANDS),KS_MAIN,KS_AUX,MARGIN,MIN_CELL)).encode()).hexdigest()[:8]

def ck_load(tag,seed):
    d={}
    try:
        with open(f"{CKPT}/self_{tag}_s{seed}_{_fp()}.jsonl") as f:
            for ln in f:
                r=json.loads(ln); d[(r["i"],r["j"])]=r["v"]
    except FileNotFoundError: pass
    return d

def ck_append(tag,seed,i,j,v):
    os.makedirs(CKPT, exist_ok=True)
    with open(f"{CKPT}/self_{tag}_s{seed}_{_fp()}.jsonl","a") as f:
        f.write(json.dumps({"i":int(i),"j":int(j),"v":v})+"\n"); f.flush(); os.fsync(f.fileno())

def load183():
    a=hashlib.md5(TC.CACHE.read_bytes()).hexdigest()
    b=hashlib.md5(open(NEW89,"rb").read()).hexdigest()
    assert a==MD5_A, f"94 列缓存 md5 不符 {a}"
    assert b==MD5_B, f"89 列新特征 md5 不符 {b}"
    d=TC.Data()
    new=pd.read_csv(NEW89)
    df=d.df.merge(new, on=KEYS, how="inner")
    assert len(df)==len(d.df), f"连接后行数变了 {len(df)} vs {len(d.df)}"
    return df, d

def main():
    t0=time.time()
    with threadpool_limits(1):
        df, d = load183()
        cols = TC.feature_columns(df)
        classes = list(d.enc.classes_)
        fams = TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
        MASKS=[("none", np.arange(len(cols)))]
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f, keep))
        print(f"183 列池：{len(cols)} 列  {len(classes)} 类 {classes}", flush=True)
        print(f"掩码候选 {len(MASKS)}   模型 {CANDS}   主判据 k={KS_MAIN}  并报 k={KS_AUX}", flush=True)
        print(f"inner {INNER_SRC}→{INNER_TGT}   outer {[o[0] for o in OUTER]}", flush=True)

        def block(rounds):
            s=df[df["round"].isin(rounds)]
            return (np.asarray(TC.clean_x(s,cols),dtype=float),
                    d.enc.transform(s["label"]), np.asarray(s["label"]))

        Xa,ya,_  = block(INNER_SRC)
        Xb,yb,gb = block([INNER_TGT])
        rows=[]
        for seed in SEEDS:
            TC.SEED=seed
            bm=TC.make_model("xgboost",len(classes)); bm.fit(Xa,ya)
            Pb=bm.predict_proba(Xb); ob=np.argsort(-Pb,axis=1); b1,b2=ob[:,0],ob[:,1]
            cand=sorted(set(map(tuple,np.sort(np.c_[b1,b2],axis=1))))
            done=ck_load("inner",seed)
            if done: print(f"  [断点] seed{seed} 已有 {len(done)} 对",flush=True)
            cfg={}; audit=[]
            for (i,j) in cand:
                if (i,j) in done:
                    nD,inc,acc,mname,nm = done[(i,j)]
                    audit.append((i,j,nD,inc,acc,mname,nm))
                    if acc>inc+MARGIN: cfg[(i,j)]=(mname,nm,acc)
                    continue
                D=(((b1==i)&(b2==j))|((b1==j)&(b2==i)))&np.isin(yb,[i,j])
                if D.sum()<MIN_CELL: continue
                inc=float((b1[D]==yb[D]).mean())
                ma=np.isin(ya,[i,j])
                if len(np.unique(ya[ma]))<2: continue
                y1=(ya[ma]==j).astype(int); best=(None,None,-1.0)
                for mname,keep in MASKS:
                    for nm in CANDS:
                        try:
                            m=fit_w(MK(nm),nm,Xa[ma][:,keep],y1)
                            q=m.predict_proba(Xb[:,keep])[:,1]
                        except Exception: continue
                        acc=float((np.where(q[D]>=0.5,j,i)==yb[D]).mean())
                        if acc>best[2]: best=(mname,nm,acc)
                if best[1] is None: continue
                audit.append((i,j,int(D.sum()),inc,best[2],best[0],best[1]))
                ck_append("inner",seed,i,j,[int(D.sum()),inc,best[2],best[0],best[1]])
                if best[2]>inc+MARGIN: cfg[(i,j)]=best
            print(f"\n  seed{seed}  inner 候选 {len(cand)} 对，过闸 {len(cfg)} 对",flush=True)
            for (i,j),(mname,nm,a) in sorted(cfg.items(), key=lambda x:-x[1][2]):
                rec=[r for r in audit if r[0]==i and r[1]==j][0]
                print(f"      {classes[i]}|{classes[j]:10s} 可判决区 {rec[2]:5d}  "
                      f"在位者 {rec[3]:.4f} → 配置 {a:.4f} ({a-rec[3]:+.4f})  {nm}/掩{mname}",flush=True)
            MD={m:k for m,k in MASKS}
            for uname, src, tgt in OUTER:
                Xs,ys,_  = block(src)
                Xt,yt,gt = block([tgt])
                om=TC.make_model("xgboost",len(classes)); om.fit(Xs,ys)
                Pt=om.predict_proba(Xt); ot=np.argsort(-Pt,axis=1); t1,t2=ot[:,0],ot[:,1]
                f_base=f1_score(yt,t1,average="macro")
                pred=t1.copy(); nov=0
                for (i,j),(mname,nm,a) in cfg.items():
                    keep=MD[mname]; ms=np.isin(ys,[i,j])
                    if len(np.unique(ys[ms]))<2: continue
                    mm=((t1==i)&(t2==j))|((t1==j)&(t2==i))
                    if not mm.any(): continue
                    pm=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys[ms]==j).astype(int))
                    q=pm.predict_proba(Xt[:,keep])[:,1]
                    pred[mm]=np.where(q[mm]>=0.5,j,i); nov+=int(mm.sum())
                f_cfg=f1_score(yt,pred,average="macro")
                F0=f1_score(yt,t1,average=None,labels=np.arange(len(classes)))
                F1v=f1_score(yt,pred,average=None,labels=np.arange(len(classes)))
                print(f"    {uname} seed{seed}  base={f_base:.4f}  +定点修={f_cfg:.4f}  "
                      f"Δ={f_cfg-f_base:+.4f}   动手 {nov} 窗",flush=True)
                print("        逐类  " + "  ".join(
                    f"{classes[c][:9]} {F0[c]:.3f}→{F1v[c]:.3f}" for c in range(len(classes))),flush=True)
                rows.append({"unit":uname,"seed":seed,"base":f_base,"cfg":f_cfg,
                             "delta":f_cfg-f_base,"n_gate":len(cfg),"n_win":nov})
        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/cfg_self.csv",index=False)
        print("\n=== 汇总（主判据 k=1，无平滑）===",flush=True)
        print(R.groupby("unit")[["base","cfg","delta"]].mean().round(4).to_string(),flush=True)
        print(f"\n三单元总均值  base={R.base.mean():.4f}  +定点修={R.cfg.mean():.4f}  "
              f"Δ={R.delta.mean():+.4f}",flush=True)
        print(f"（参照：override_183 的 flat_183 = 0.8072，override_183 = 0.8210）",flush=True)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
