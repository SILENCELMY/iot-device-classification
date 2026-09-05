"""R3∩R1 会不会砍掉 CIC 那个 0.0000→0.8111 的战果？

`cic_transfer.py` 下 outer 上 `GlobeLamp|GosundSocket` 区 n=7701、在位者 **0.0000**。
在位者恒 0 意味着 top1 从不正确。两种可能：
  (a) top1 是【对内另一个】—— 那 R3∩R1 保留这些窗，战果不受影响
  (b) top1 是【第三类】     —— 那 R3∩R1 把它们全砍掉，战果归零
必须查清才能冻结区定义。CIC 的 1102/1103/1108 都是已烧天，可查。

同时并报三种区在 CIC outer 上的：构成、两种记账、端到端 —— 与 UNSW 选参天同口径。
"""
import sys, time, re
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
import pilot_rf_loro as P, run_unsw_iid_reference as IID
EXP=REPO+"/results/feature_expansion_20260905"
DAYS={"src":("2021_11_02_Idle","/home/lmy/cic_probe/idle_1102.csv"),
      "inner":("2021_11_03_Active","/home/lmy/cic_probe/active_1103.csv"),
      "outer":("2021_11_08_Active","/home/lmy/cic_probe/active_1108.csv")}
NJ=12; SEED=42; MARGIN=0.02; MIN_CELL=40; KBASE=[1,3,5,10,20]

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s
def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    return RandomForestClassifier(n_estimators=200,random_state=SEED,
                                  class_weight="balanced",n_jobs=NJ)
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
        i=np.where(g==u)[0]; V=M[i]
        C=np.vstack([np.zeros(V.shape[1]),np.cumsum(V,axis=0)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(C[n+1]-C[lo])/(n+1-lo)
    return o

t0=time.time()
LH=pd.read_csv(EXP+"/lenhist_cic_w10.csv"); lc=[c for c in LH.columns if c.startswith("lenhist_")]
D={}
for tag,(day,csv) in DAYS.items():
    d=pd.read_csv(csv,low_memory=False)
    d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
    D[tag]=d.sort_values(["device","window_id"]).reset_index(drop=True)
base=[c for c in P.feature_columns(D["src"]) if not c.startswith("lenhist_")]
cols=base+lc
devs=sorted(set.intersection(*[set(IID.day_gate(D[t],DAYS[t][0])) for t in DAYS]))
classes=sorted({TYPE(x) for x in devs}); le=LabelEncoder().fit(classes)
def XYG(tag):
    d=D[tag]; d=d[d.device.isin(devs)]
    return (np.asarray(P.clean_x(d,cols),dtype=np.float32),
            le.transform([TYPE(x) for x in d.device]), d.device.to_numpy())
Xs,ys,_=XYG("src"); Xi,yi,gi=XYG("inner"); Xo,yo,go=XYG("outer")
print(f"{len(classes)} 类  训练 {len(ys)} inner {len(yi)} outer {len(yo)}  {time.time()-t0:.0f}s",flush=True)

rf=RandomForestClassifier(n_estimators=300,random_state=SEED,class_weight="balanced",
                          n_jobs=NJ).fit(Xs,ys)
Pi=rf.predict_proba(Xi); Po=rf.predict_proba(Xo); L=np.arange(len(classes))
kb=max(((k,f1_score(yi,cmeanM(Pi,gi,k).argmax(1),average="macro",labels=L))
        for k in KBASE),key=lambda x:x[1])[0]
Pi_s=cmeanM(Pi,gi,kb); Po_s=cmeanM(Po,go,kb)
f_sm=f1_score(yo,Po_s.argmax(1),average="macro",labels=L)
ii=np.argsort(-Pi_s,axis=1); i1,i2,i3=ii[:,0],ii[:,1],ii[:,2]
oo=np.argsort(-Po_s,axis=1); o1,o2,o3=oo[:,0],oo[:,1],oo[:,2]
print(f"平滑 kb={kb}   outer macro（第一段后）={f_sm:.4f}",flush=True)

fams={}
for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
idx={c:n for n,c in enumerate(cols)}
MASKS=[("none",np.arange(len(cols)))]
for f,v in fams.items():
    keep=np.array([idx[c] for c in cols if c not in set(v)])
    if len(keep)>=5: MASKS.append((f,keep))

def reg(name,t1,t2,t3,i,j):
    both=((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j))
    if name=="R3":    return both
    if name=="R3∩R1": return both&((t1==i)|(t1==j))
    return ((t1==i)&(t2==j))|((t1==j)&(t2==i))

err=Counter()
for a,b in zip(yi,i1):
    if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
hard=[p for p,_ in err.most_common(6)]
print(f"\n困难对（inner 错误量前 6）：" + ", ".join(
      f"{classes[i]}|{classes[j]}({err[(i,j)]})" for i,j in hard),flush=True)

for i,j in hard:
    ni,nj=classes[i],classes[j]
    print(f"\n{'='*106}\n{ni} | {nj}",flush=True)
    ms=np.isin(ys,[i,j])
    if len(np.unique(ys[ms]))<2: print("  源天缺一类，跳过",flush=True); continue
    y1=(ys[ms]==j).astype(int); cache={}
    for rn in ("R3","R3∩R1","R2"):
        # inner 上选配置（整区记账），outer 上施加
        Si=reg(rn,i1,i2,i3,i,j)
        if Si.sum()<MIN_CELL: print(f"  {rn:7s} inner 区太小 n={int(Si.sum())}",flush=True); continue
        inc_i=float((i1[Si]==yi[Si]).mean()); best=(None,None,-1.0)
        for mn,keep in MASKS:
            for nm in ("lr","rf"):
                key=(mn,nm)
                if key not in cache:
                    try: cache[key]=fit_w(MK(nm),nm,Xs[ms][:,keep],y1)
                    except Exception: cache[key]=None
                m_=cache[key]
                if m_ is None: continue
                q=m_.predict_proba(Xi[:,keep])[:,1]
                new=i1.copy(); new[Si]=np.where(q[Si]>=0.5,j,i)
                a=float((new[Si]==yi[Si]).mean())
                if a>best[2]: best=(mn,nm,a)
        gated=best[2]>inc_i+MARGIN
        mn,nm,_=best
        So=reg(rn,o1,o2,o3,i,j); n=int(So.sum())
        n_i=int((So&(yo==i)).sum()); n_j=int((So&(yo==j)).sum()); n_3=n-n_i-n_j
        if n==0:
            print(f"  {rn:7s} inner {inc_i:.4f}→{best[2]:.4f} {'放行' if gated else '拒绝'}"
                  f"   outer 区 n=0  【配置无处施加】",flush=True); continue
        q=fit_w(MK(nm),nm,Xs[ms][:,keep_ := dict(MASKS)[mn]],y1).predict_proba(Xo[:,keep_])[:,1]
        newo=o1.copy(); newo[So]=np.where(q[So]>=0.5,j,i)
        inc_pair=float((o1[So&np.isin(yo,[i,j])]==yo[So&np.isin(yo,[i,j])]).mean()) if (n_i+n_j) else np.nan
        aft_pair=float((newo[So&np.isin(yo,[i,j])]==yo[So&np.isin(yo,[i,j])]).mean()) if (n_i+n_j) else np.nan
        inc_full=float((o1[So]==yo[So]).mean()); aft_full=float((newo[So]==yo[So]).mean())
        f_e=f1_score(yo,newo,average="macro",labels=L)
        print(f"  {rn:7s} inner {inc_i:.4f}→{best[2]:.4f} ({nm}/掩{mn}) "
              f"{'放行' if gated else '拒绝'}",flush=True)
        print(f"          outer 区 n={n:6d}  构成: {ni[:14]} {n_i:5d} / {nj[:14]} {n_j:5d} / "
              f"第三类 {n_3:5d}（{n_3/n:5.1%}）",flush=True)
        print(f"          只算{{i,j}}窗  {inc_pair:.4f} → {aft_pair:.4f} "
              f"({aft_pair-inc_pair:+.4f})",flush=True)
        print(f"          算整个区    {inc_full:.4f} → {aft_full:.4f} "
              f"({aft_full-inc_full:+.4f})",flush=True)
        print(f"          端到端 {f_sm:.4f} → {f_e:.4f} ({f_e-f_sm:+.4f})",flush=True)
print(f"\n关键判读：GlobeLamp|GosundSocket 在 R3 下 outer n=7701 在位者 0.0000。",flush=True)
print(f"  若 R3∩R1 的 outer n 仍然很大 → top1 是对内另一个，R3∩R1 安全，战果保住；",flush=True)
print(f"  若 R3∩R1 的 outer n 塌到接近 0 → top1 是第三类，R3∩R1 会砍掉 CIC 的战果。",flush=True)
print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
