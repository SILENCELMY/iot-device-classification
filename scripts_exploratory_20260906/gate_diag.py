"""闸门区定义诊断 —— 只用选参天 16-09-30，保留集不碰。

烟测暴露：闸门在 `Dm = sel & (真类∈{i,j})` 上量收益，却对整个 `sel` 施加。
R3 区含真类是第三类的窗，翻掉它们的伤害闸门看不见 ⇒ 区上 +0.18 而端到端 −0.0085。

量三件事，为选区定义与判据提供依据（全部在选参天上）：
  1 区的构成    sel 里真类是 i / j / 第三类 各多少
  2 两种记账    "只算 {i,j} 窗" vs "算整个 sel"（后者是我们实际作用的对象）
  3 三种施加面  R3 / R3∩R1（top1∈{i,j}） / R2，各自的构成、两种记账、端到端 Δ
"""
import sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C
import pilot_rf_loro as P

TUNE="16-09-30"; SEED=42
LH=pd.read_csv(C.LH_MAIN); lc=[c for c in LH.columns if c.startswith("lenhist_")]
def load(day):
    d=pd.read_csv(C.FULL%day,low_memory=False,encoding="utf-8-sig")
    d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
    return d.sort_values(["device","window_id"]).reset_index(drop=True)
Dtr=load(C.TRAIN); Dtu=load(TUNE)
devs=sorted(C.gate_day(C.TRAIN) & C.gate_day(TUNE))
base=[c for c in P.feature_columns(Dtr) if not c.startswith("lenhist_")]
cols=base+lc; le=LabelEncoder().fit(devs)
def XY(d):
    d=d[d.device.isin(devs)]
    return (np.asarray(P.clean_x(d,cols),dtype=np.float32),
            le.transform(d.device), d.device.to_numpy())
Xs,ys,_=XY(Dtr); Xu,yu,gu=XY(Dtu)
print(f"{len(devs)} 类   训练 {len(ys)}  选参 {len(yu)}   {len(cols)} 列",flush=True)

rf=RandomForestClassifier(n_estimators=300,random_state=SEED,
                          class_weight="balanced",n_jobs=12).fit(Xs,ys)
Pu=rf.predict_proba(Xu); L=np.arange(len(devs))
kb=max(((k,f1_score(yu,C.cmeanM(Pu,gu,k).argmax(1),average="macro",labels=L))
        for k in C.KBASE),key=lambda x:x[1])[0]
Ps=C.cmeanM(Pu,gu,kb); oi=np.argsort(-Ps,axis=1); t1,t2,t3=oi[:,0],oi[:,1],oi[:,2]
f_sm=f1_score(yu,t1,average="macro",labels=L)
print(f"平滑 kb={kb}   选参天 macro（第一段后）={f_sm:.4f}",flush=True)

err=Counter()
for a,b in zip(yu,t1):
    if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
hard=[p for p,_ in err.most_common(3)]

fams={}
for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
idx={c:n for n,c in enumerate(cols)}
MASKS=[("none",np.arange(len(cols)))]
for f,v in fams.items():
    keep=np.array([idx[c] for c in cols if c not in set(v)])
    if len(keep)>=5: MASKS.append((f,keep))

REGS={"R3":      lambda i,j: ((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j)),
      "R3∩R1":   lambda i,j: (((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j)))
                             &((t1==i)|(t1==j)),
      "R2":      lambda i,j: ((t1==i)&(t2==j))|((t1==j)&(t2==i))}

for i,j in hard:
    ni,nj=le.classes_[i],le.classes_[j]
    print(f"\n{'='*104}\n{ni} | {nj}   （选参天错 {err[(i,j)]}）",flush=True)
    ms=np.isin(ys,[i,j]); y1=(ys[ms]==j).astype(int); cache={}
    for rn,fn in REGS.items():
        sel=fn(i,j); n=int(sel.sum())
        if n<C.MIN_CELL: print(f"  {rn:7s} 区太小 n={n}",flush=True); continue
        in_pair=np.isin(yu,[i,j])
        n_i=int((sel&(yu==i)).sum()); n_j=int((sel&(yu==j)).sum())
        n_3=n-n_i-n_j
        inc_pair=float((t1[sel&in_pair]==yu[sel&in_pair]).mean())
        inc_full=float((t1[sel]==yu[sel]).mean())
        best=(None,None,-1,-1,-1)
        for mn,keep in MASKS:
            for nm in ("lr","rf"):
                key=(mn,nm)
                if key not in cache:
                    try: cache[key]=C.fit_w(C.MK(nm,SEED),nm,Xs[ms][:,keep],y1
                                            ).predict_proba(Xu[:,keep])[:,1]
                    except Exception: cache[key]=None
                q=cache[key]
                if q is None: continue
                new=t1.copy(); new[sel]=np.where(q[sel]>=0.5,j,i)
                a_pair=float((new[sel&in_pair]==yu[sel&in_pair]).mean())
                a_full=float((new[sel]==yu[sel]).mean())
                f_e2e=f1_score(yu,new,average="macro",labels=L)
                # 按【整区记账】挑最好的，这才是我们作用的对象
                if a_full>best[3]: best=(mn,nm,a_pair,a_full,f_e2e)
        mn,nm,a_pair,a_full,f_e2e=best
        print(f"  {rn:7s} 区 n={n:6d}  构成: 真类={ni[:12]} {n_i:5d} / {nj[:12]} {n_j:5d} / "
              f"第三类 {n_3:5d}（{n_3/n:5.1%}）",flush=True)
        print(f"          只算{{i,j}}窗   在位者 {inc_pair:.4f} → 修后 {a_pair:.4f} "
              f"({a_pair-inc_pair:+.4f})   ← 旧闸门看的是这个",flush=True)
        print(f"          算整个区     在位者 {inc_full:.4f} → 修后 {a_full:.4f} "
              f"({a_full-inc_full:+.4f})   ← 我们实际作用的对象",flush=True)
        print(f"          端到端 macro {f_sm:.4f} → {f_e2e:.4f} ({f_e2e-f_sm:+.4f})"
              f"   最好配置 {nm}/掩{mn}",flush=True)
        s1="放行" if a_pair>inc_pair+C.MARGIN else "拒绝"
        s2="放行" if a_full>inc_full+C.MARGIN else "拒绝"
        ok="✓一致" if (s2=="放行")==(f_e2e>f_sm) else "✗"
        print(f"          旧闸门 {s1}   新闸门(整区) {s2}   端到端实际 "
              f"{'涨' if f_e2e>f_sm else '跌'}   {ok}",flush=True)
print(f"\n判读：新闸门（整区记账）的放行/拒绝若与端到端涨跌一致 → 记账修好了；",flush=True)
print(f"      并比较三种施加面在端到端上的表现，据此定 C3 的区定义。",flush=True)
