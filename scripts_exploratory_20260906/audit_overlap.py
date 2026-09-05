"""审计缺陷 1：多对施加时区重叠有多严重（CIC，已烧天）。

`pred[sel]=...` 逐对写入，无冲突消解；后写覆盖先写，且逐对报的 acc 在累积后的
pred 上量 ⇒ 依赖 dict 顺序。UNSW 只过闸 1 对没暴露，CIC 过闸 3–4 对且共享类。

量：过闸各对的区两两交集大小、被写多次的窗数、以及【换施加顺序】端到端变不变。
"""
import sys, re, itertools
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C
import pilot_rf_loro as P, run_unsw_iid_reference as IID

REPO="/home/lmy/iot-device-classification"
EXP=REPO+"/results/feature_expansion_20260905"
DAYS={"src":("2021_11_02_Idle","/home/lmy/cic_probe/idle_1102.csv"),
      "inner":("2021_11_03_Active","/home/lmy/cic_probe/active_1103.csv"),
      "outer":("2021_11_08_Active","/home/lmy/cic_probe/active_1108.csv")}
SEED=42
def TYPE(s):
    if re.match(r"GosundESP.*Plug$",s):return "GosundPlug"
    if re.match(r"GosundESP.*Socket$",s):return "GosundSocket"
    if re.match(r"TeckinPlug\d$",s):return "Teckin"
    if re.match(r"YutronPlug\d$",s):return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$",s):return "EchoDot"
    return s

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
def XYG(t):
    d=D[t]; d=d[d.device.isin(devs)]
    return (np.asarray(P.clean_x(d,cols),dtype=np.float32),
            le.transform([TYPE(x) for x in d.device]), d.device.to_numpy())
Xs,ys,_=XYG("src"); Xi,yi,gi=XYG("inner"); Xo,yo,go=XYG("outer")
rf=RandomForestClassifier(n_estimators=300,random_state=SEED,class_weight="balanced",
                          n_jobs=12).fit(Xs,ys)
Pi=rf.predict_proba(Xi); Po=rf.predict_proba(Xo); L=np.arange(len(classes))
kb=max(((k,f1_score(yi,C.cmeanM(Pi,gi,k).argmax(1),average="macro",labels=L))
        for k in C.KBASE),key=lambda x:x[1])[0]
Pi_s=C.cmeanM(Pi,gi,kb); Po_s=C.cmeanM(Po,go,kb)
ii=np.argsort(-Pi_s,axis=1); i1,i2,i3=ii[:,0],ii[:,1],ii[:,2]
oo=np.argsort(-Po_s,axis=1); o1,o2,o3=oo[:,0],oo[:,1],oo[:,2]
f_sm=f1_score(yo,o1,average="macro",labels=L)
print(f"CIC {len(classes)} 类  平滑 kb={kb}  outer 第一段后 macro={f_sm:.4f}",flush=True)

fams={}
for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
idx={c:n for n,c in enumerate(cols)}
MASKS=[("none",np.arange(len(cols)))]
for f,v in fams.items():
    keep=np.array([idx[c] for c in cols if c not in set(v)])
    if len(keep)>=5: MASKS.append((f,keep))
MD=dict(MASKS)
def region(rn,t1,t2,t3,i,j):
    both=((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j))
    if rn=="R3": return both
    if rn=="R3nR1": return both&((t1==i)|(t1==j))
    return ((t1==i)&(t2==j))|((t1==j)&(t2==i))

err=Counter()
for a,b in zip(yi,i1):
    if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
hard=[p for p,_ in err.most_common(8)]
cfgs={}
for i,j in hard:
    ms=np.isin(ys,[i,j])
    if len(np.unique(ys[ms]))<2: continue
    y1=(ys[ms]==j).astype(int); cache={}
    for mn,keep in MASKS:
        for nm in ("lr","rf"):
            try: cache[(mn,nm)]=C.fit_w(C.MK(nm,SEED),nm,Xs[ms][:,keep],y1
                                        ).predict_proba(Xi[:,keep])[:,1]
            except Exception: cache[(mn,nm)]=None
    best=(None,None,None,-1.0,-1.0)
    for rn in C.REGIONS:
        sel=region(rn,i1,i2,i3,i,j)
        if sel.sum()<C.MIN_CELL: continue
        inc=float((i1[sel]==yi[sel]).mean()); loc=(None,None,-1.0)
        for (mn,nm),q in cache.items():
            if q is None: continue
            mix=C.blend(Pi_s,q,i,j)
            new=i1.copy(); new[sel]=np.where(mix[sel]>=0.5,j,i)
            a=float((new[sel]==yi[sel]).mean())
            if a>loc[2]: loc=(mn,nm,a)
        if loc[2]-inc>best[3]-best[4]: best=(rn,loc[0],loc[1],loc[2],inc)
    rn,mn,nm,a,inc=best
    if rn and a>inc+C.MARGIN:
        cfgs[(i,j)]=(rn,mn,nm)
        print(f"  过闸 {classes[i]}|{classes[j]}  [{rn}] {inc:.4f}→{a:.4f} ({nm}/掩{mn})",flush=True)

print(f"\n过闸 {len(cfgs)} 对",flush=True)
sels={}; qs={}
for (i,j),(rn,mn,nm) in cfgs.items():
    keep=MD[mn]; ms=np.isin(ys,[i,j])
    q=C.fit_w(C.MK(nm,SEED),nm,Xs[ms][:,keep],(ys[ms]==j).astype(int)
              ).predict_proba(Xo[:,keep])[:,1]
    sels[(i,j)]=region(rn,o1,o2,o3,i,j); qs[(i,j)]=q

print(f"\n=== 缺陷 1：区重叠 ===",flush=True)
ks=list(sels)
for a,b in itertools.combinations(ks,2):
    ov=int((sels[a]&sels[b]).sum())
    na,nb=int(sels[a].sum()),int(sels[b].sum())
    print(f"  {classes[a[0]]}|{classes[a[1]]}  ∩  {classes[b[0]]}|{classes[b[1]]}"
          f"   n={na} / {nb}   交集 {ov}  ({ov/min(na,nb):.1%} of 较小者)",flush=True)
cover=np.zeros(len(yo),dtype=int)
for s in sels.values(): cover+=s.astype(int)
print(f"\n  被写 ≥2 次的窗：{int((cover>=2).sum())}   ≥3 次：{int((cover>=3).sum())}"
      f"   总被覆盖窗 {int((cover>=1).sum())}",flush=True)

print(f"\n=== 缺陷 1 的后果：换施加顺序，端到端变不变 ===",flush=True)
res=[]
for perm in itertools.permutations(ks):
    pred=o1.copy()
    for k in perm:
        i,j=k; sel=sels[k]; mix=C.blend(Po_s,qs[k],i,j)
        pred[sel]=np.where(mix[sel]>=0.5,j,i)
    res.append(f1_score(yo,pred,average="macro",labels=L))
res=np.array(res)
print(f"  {len(res)} 种顺序：min={res.min():.4f}  max={res.max():.4f}  "
      f"极差={res.max()-res.min():.4f}",flush=True)
print(f"  对照：第一段后 {f_sm:.4f}，第二段增益区间 "
      f"[{res.min()-f_sm:+.4f}, {res.max()-f_sm:+.4f}]",flush=True)
print(f"\n判读：极差若与第二段增益同量级 → 缺陷 1 使结果【不确定】，必须加冲突消解。",flush=True)
