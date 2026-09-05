"""【探索性,非协议】v6 = v5 + 三处针对具体病灶的修正。不做网格搜索。

相对 v5 的改动，每一条都有具体依据：
  ① 时长上限 10 → 40
     依据：救援测试里 YutronPlug1|2 用 xgboost，k=10 是 0.9156、k=40 是 0.9667。
           v5 把最有效的那段切掉了。
  ② 逐类对二分类器用 balanced 样本权重
     依据：型号级合并后 GosundPlugSocket 有 6 倍窗口，GlobeLamp 的 F1 = 0.000
           （从未被判出来过）。真不可分应是 0.2–0.3，0.000 更像不平衡的产物。
  ③ 把逐类对特征掩码加回配置维度（v5 里被砍掉了）
     依据：全局删某族有害，不代表逐类对上有害；型号级从没试过。

配置 = (特征掩码, 模型, 观测时长)，全部在 inner 上按【可判决区准确率】选，
闸门 = 相对已平滑的在位者有增益。outer 完全 held-out。
"""
from __future__ import annotations
import os, sys, time, re, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
CKPT_DIR="/home/lmy/cic_probe/_ckpt"

def _cfg_fingerprint():
    """搜索空间指纹。改了 KS/掩码/模型/balanced，指纹变，旧检查点自动失效。"""
    import hashlib
    sig = repr((sorted(CANDS), sorted(KS), sorted(KBASE), MARGIN, MIN_CELL,
                BALANCED, USE_MASKS))
    return hashlib.sha256(sig.encode()).hexdigest()[:8]

def _ck_path(tag, seed):
    os.makedirs(CKPT_DIR, exist_ok=True)
    safe="".join(c if c.isalnum() else "_" for c in tag)[:40]
    return f"{CKPT_DIR}/v6_{safe}_s{seed}_{_cfg_fingerprint()}.jsonl"

def ck_load(tag, seed):
    """返回 {(i,j): [n_D, inc, cfgacc, mask, model, k]}"""
    d={}
    try:
        with open(_ck_path(tag,seed)) as f:
            for ln in f:
                r=json.loads(ln)
                d[(r["i"],r["j"])]=r["v"]
    except FileNotFoundError:
        pass
    return d

def ck_append(tag, seed, i, j, v):
    with open(_ck_path(tag,seed),"a") as f:
        f.write(json.dumps({"i":int(i),"j":int(j),"v":v})+"\n"); f.flush(); os.fsync(f.fileno())
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC

CANDS=["lr","rf","xgboost"]
KS=[1,3,5,10,20,40]          # ① 扩到 40
KBASE=[1,3,5,10,20]
MARGIN=0.02; MIN_CELL=40; SEEDS=(42,43)
BALANCED=True                # ②
USE_MASKS=True               # ③

def TYPE(s):
    if re.match(r"GosundESP.*(Plug|Socket)$", s): return "GosundPlugSocket"
    if re.match(r"TeckinPlug\d$", s):             return "TeckinPlug"
    if re.match(r"YutronPlug\d$", s):             return "YutronPlug"
    if re.match(r"AmazonAlexaEchoDot\d$", s):     return "AmazonAlexaEchoDot"
    return s

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return TC.make_model(n,2)

def fit_w(m, name, X, y):
    """② balanced 样本权重；pipeline 需要带步骤前缀。"""
    if not BALANCED:
        m.fit(X,y); return m
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

def task(tag, fA, dayA, fB, dayB, fC, dayC, type_level=True):
    dfA=pd.read_csv(fA,low_memory=False); dfB=pd.read_csv(fB,low_memory=False)
    dfC=pd.read_csv(fC,low_memory=False)
    cols=P.feature_columns(dfA)
    devs=sorted(set(IID.day_gate(dfA,dayA))&set(IID.day_gate(dfB,dayB))&set(IID.day_gate(dfC,dayC)))
    LAB=(lambda s: TYPE(s)) if type_level else (lambda s: s)
    classes=sorted({LAB(d) for d in devs}); le=LabelEncoder().fit(classes)
    fams=TC.derive_families(cols); idx={c:n for n,c in enumerate(cols)}
    # ③ 掩码候选：不掩 + 各删一族
    MASKS=[("none", np.arange(len(cols)))]
    if USE_MASKS:
        for f,v in fams.items():
            keep=np.array([idx[c] for c in cols if c not in set(v)])
            if len(keep)>=5: MASKS.append((f, keep))
    print(f"\n{'='*98}\n{tag}   inner {dayA}→{dayB}   outer {dayA}→{dayC}",flush=True)
    print(f"  实例 {len(devs)} → 类 {len(classes)}   掩码候选 {len(MASKS)}   "
          f"模型 {CANDS}   时长 {KS}   balanced={BALANCED}",flush=True)
    rows=[]
    for seed in SEEDS:
        t0=time.time(); TC.SEED=seed
        def prep(df,sort=False):
            d=df[df.label.isin(devs)]
            if sort: d=d.sort_values(["label","window_start_epoch"])
            d=P.sample_balanced(d,max_rows=IID.MAX_ROWS,random_state=seed)
            if sort: d=d.sort_values(["label","window_start_epoch"])
            return (np.asarray(P.clean_x(d,cols),dtype=float),
                    le.transform([LAB(x) for x in d.label]), np.asarray(d.label))
        Xa,ya,_=prep(dfA); Xb,yb,gb=prep(dfB,True); Xt,yt,gt=prep(dfC,True)
        bm=TC.make_model("xgboost",len(classes)); bm.fit(Xa,ya)
        Pb=bm.predict_proba(Xb)
        kb=max(((k,f1_score(yb,cmeanM(Pb,gb,k).argmax(1),average="macro")) for k in KBASE),
               key=lambda x:x[1])[0]
        Pbs=cmeanM(Pb,gb,kb); ob=np.argsort(-Pbs,axis=1); b1,b2=ob[:,0],ob[:,1]
        cand=sorted(set(map(tuple,np.sort(np.c_[b1,b2],axis=1))))
        cfg={}; audit=[]
        done=ck_load(tag, seed)
        if done: print(f"      [断点] 已有 {len(done)} 对的结果，跳过重算",flush=True)
        for n_pair,(i,j) in enumerate(cand):
            if (i,j) in done:
                nD,inc,acc,mname,nm,k = done[(i,j)]
                audit.append((i,j,nD,inc,acc,mname,nm,k))
                if acc>inc+MARGIN: cfg[(i,j)]=(mname,nm,k,acc)
                continue
            D=(((b1==i)&(b2==j))|((b1==j)&(b2==i)))&np.isin(yb,[i,j])
            if D.sum()<MIN_CELL: continue
            inc=float((b1[D]==yb[D]).mean())
            ma=np.isin(ya,[i,j])
            if len(np.unique(ya[ma]))<2: continue
            y1=(ya[ma]==j).astype(int); best=(None,None,0,-1.0)
            for mname,keep in MASKS:
                for nm in CANDS:
                    try:
                        m=fit_w(MK(nm), nm, Xa[ma][:,keep], y1)
                        q=m.predict_proba(Xb[:,keep])[:,1]
                    except Exception: continue
                    for k in KS:
                        qq=cmean(q,gb,k)
                        acc=float((np.where(qq[D]>=0.5,j,i)==yb[D]).mean())
                        if acc>best[3]: best=(mname,nm,k,acc)
            if best[1] is None: continue
            audit.append((i,j,int(D.sum()),inc,best[3],best[0],best[1],best[2]))
            ck_append(tag, seed, i, j, [int(D.sum()), inc, best[3], best[0], best[1], best[2]])
            if best[3]>inc+MARGIN: cfg[(i,j)]=best
            if (n_pair+1)%20==0:
                print(f"      {n_pair+1}/{len(cand)} 对  {time.time()-t0:.0f}s",flush=True)
        Pt=bm.predict_proba(Xt); Pts=cmeanM(Pt,gt,kb); ot=np.argsort(-Pts,axis=1)
        t1,t2=ot[:,0],ot[:,1]
        f_base=f1_score(yt,Pt.argmax(1),average="macro"); f_sm=f1_score(yt,t1,average="macro")
        pred=t1.copy(); nov=0; used={}
        MD={m:k for m,k in MASKS}
        for (i,j),(mname,nm,k,a) in cfg.items():
            keep=MD[mname]; ms=np.isin(ya,[i,j])
            pm=fit_w(MK(nm), nm, Xa[ms][:,keep], (ya[ms]==j).astype(int))
            q=cmean(pm.predict_proba(Xt[:,keep])[:,1],gt,k)
            mm=((t1==i)&(t2==j))|((t1==j)&(t2==i))
            if not mm.any(): continue
            pred[mm]=np.where(q[mm]>=0.5,j,i); nov+=int(mm.sum())
            used[f"{nm}/{mname}/k{k}"]=used.get(f"{nm}/{mname}/k{k}",0)+1
        f_v6=f1_score(yt,pred,average="macro")
        print(f"  seed{seed} k_base={kb}  base={f_base:.4f} 平滑={f_sm:.4f} v6={f_v6:.4f}  "
              f"Δ(v6−平滑)={f_v6-f_sm:+.4f}   过闸 {len(cfg)}/{len(audit)} 对 / {nov} 窗  "
              f"{time.time()-t0:.0f}s",flush=True)
        print(f"      选中配置: {used}",flush=True)
        Ad=pd.DataFrame(audit,columns=["i","j","n_D","inc","cfg","mask","model","k"])
        Ad["gain"]=Ad.cfg-Ad.inc
        for r in Ad.sort_values("gain",ascending=False).head(6).itertuples():
            print(f"      {le.classes_[r.i][:20]}|{le.classes_[r.j][:20]:20s} 可判决区 {r.n_D:5d}  "
                  f"在位者 {r.inc:.4f} → 配置 {r.cfg:.4f} ({r.gain:+.4f})  "
                  f"{r.model}/掩{r.mask}/k{r.k}",flush=True)
        F=f1_score(yt,pred,average=None,labels=np.arange(len(classes)))
        Fs=f1_score(yt,t1,average=None,labels=np.arange(len(classes)))
        for c in np.argsort(Fs)[:5]:
            print(f"      最差类 {le.classes_[c][:24]:24s} 平滑 {Fs[c]:.4f} → v6 {F[c]:.4f}",flush=True)
        rows.append({"seed":seed,"base":f_base,"smooth":f_sm,"v6":f_v6,"n_gate":len(cfg)})
    R=pd.DataFrame(rows)
    print(f"  → 均值 base={R.base.mean():.4f} 平滑={R.smooth.mean():.4f} v6={R.v6.mean():.4f}  "
          f"Δ(v6−平滑)={(R.v6-R.smooth).mean():+.4f}",flush=True)
    return R

if __name__=="__main__":
    with threadpool_limits(1):
        T0=time.time(); out=[]
        out.append(task("CIC 型号级 v6",
                        "/home/lmy/cic_probe/idle_1102.csv","2021_11_02_Idle",
                        "/home/lmy/cic_probe/active_1103.csv","2021_11_03_Active",
                        "/home/lmy/cic_probe/active_1108.csv","2021_11_08_Active",
                        type_level=True))
        U=REPO+"/results/unsw_features_full/features_day_%s.csv"
        out.append(task("UNSW v6（对照，确认新增维度不伤已成立的结果）",
                        U%"16-09-23","16-09-23", U%"16-09-30","16-09-30",
                        U%"16-10-12","16-10-12", type_level=False))
        pd.concat(out).to_csv("/home/lmy/cic_probe/cfg_v6.csv",index=False)
        print(f"\n总耗时 {time.time()-T0:.0f}s",flush=True)
