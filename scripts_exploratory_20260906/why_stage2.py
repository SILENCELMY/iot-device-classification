"""第二段为什么在 UNSW 上拿不到东西 —— 只用已烧天。

轻度验证显示：Belkin 对在选参天 在位者 0.5097→修复 0.6093（+0.0996），
到施加天 在位者 0.6304→修复 0.6068（−0.0237）。
**修复本身几乎不变（0.6093 vs 0.6068），变的是在位者。**

假设 H：施加天的在位者已经贴近该对在 10s 窗上的天花板，所以没东西可捞；
       选参天在位者偏低，闸门把"那天基线弱"误当成了增益。

三个查法（全部在已烧天上）：
  Q1 天花板   在【施加天】上把整个配置空间搜一遍（14 掩码 × 2 模型 × 3 区），
              取最好的区上准确率 = 该对在 10s 上的可达上界。
              若 ≈ 在位者 0.6304 → H 成立，第二段确实到顶了。
  Q2 逐天     在位者、修复、天花板 在 16-09-30 / 16-10-11 / 16-10-12 上分别是多少。
              看在位者的天间波动 vs 修复的天间波动，谁不稳一目了然。
  Q3 融合     不做硬替换，改成把专用二分类器的概率与基模型【限制到{i,j}】的概率
              加权平均（w∈{0,0.25,0.5,0.75,1}）。
              若某个 w 在【三天上同时】≥ 各天的在位者 → 第二段有救，问题在"替换"而非"信息"。
"""
import sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C
import pilot_rf_loro as P

TRAIN="16-09-23"; DAYS=["16-09-30","16-10-11","16-10-12"]; SEED=42
LH=pd.read_csv(C.LH_MAIN); lc=[c for c in LH.columns if c.startswith("lenhist_")]
def load(day):
    d=pd.read_csv(C.FULL%day,low_memory=False,encoding="utf-8-sig")
    d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
    return d.sort_values(["device","window_id"]).reset_index(drop=True)
D={d:load(d) for d in [TRAIN]+DAYS}
base=[c for c in P.feature_columns(D[TRAIN]) if not c.startswith("lenhist_")]
cols=base+lc
devs=sorted(set.intersection(*[C.gate_day(d) for d in [TRAIN]+DAYS]))
le=LabelEncoder().fit(devs)
print(f"{len(devs)} 类   {len(cols)} 列   训练 {TRAIN}   考察 {DAYS}",flush=True)

def XYG(day):
    d=D[day]; d=d[d.device.isin(devs)]
    return (np.asarray(P.clean_x(d,cols),dtype=np.float32),
            le.transform(d.device), d.device.to_numpy())
Xs,ys,_=XYG(TRAIN)
rf=RandomForestClassifier(n_estimators=300,random_state=SEED,class_weight="balanced",
                          n_jobs=12).fit(Xs,ys)
i=int(np.where(le.classes_=="BelkinWemoMotion")[0][0])
j=int(np.where(le.classes_=="BelkinWemoSwitch")[0][0])
print(f"考察对：{le.classes_[i]} | {le.classes_[j]}\n",flush=True)

fams={}
for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
idx={c:n for n,c in enumerate(cols)}
MASKS=[("none",np.arange(len(cols)))]
for f,v in fams.items():
    keep=np.array([idx[c] for c in cols if c not in set(v)])
    if len(keep)>=5: MASKS.append((f,keep))
ms=np.isin(ys,[i,j]); y1=(ys[ms]==j).astype(int)
BIN={}
for mn,keep in MASKS:
    for nm in ("lr","rf"):
        try: BIN[(mn,nm)]=C.fit_w(C.MK(nm,SEED),nm,Xs[ms][:,keep],y1)
        except Exception: BIN[(mn,nm)]=None
print(f"专用二分类器 {len(BIN)} 个（14 掩码 × 2 模型），全部训练于 {TRAIN}\n",flush=True)

def region(rn,t1,t2,t3):
    both=((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j))
    if rn=="R3":    return both
    if rn=="R3nR1": return both&((t1==i)|(t1==j))
    return ((t1==i)&(t2==j))|((t1==j)&(t2==i))

rows=[]; blend_rows=[]
for day in DAYS:
    Xt,yt,gt=XYG(day)
    Pm=rf.predict_proba(Xt)
    L=np.arange(len(devs))
    kb=3 if day=="16-09-30" else 3
    Ps=C.cmeanM(Pm,gt,kb)
    oo=np.argsort(-Ps,axis=1); t1,t2,t3=oo[:,0],oo[:,1],oo[:,2]
    for rn in ("R2","R3nR1","R3"):
        sel=region(rn,t1,t2,t3); n=int(sel.sum())
        if n<C.MIN_CELL: continue
        inc=float((t1[sel]==yt[sel]).mean())
        best=-1; bkey=None
        for (mn,nm),m_ in BIN.items():
            if m_ is None: continue
            keep=dict(MASKS)[mn]
            q=m_.predict_proba(Xt[:,keep])[:,1]
            new=t1.copy(); new[sel]=np.where(q[sel]>=0.5,j,i)
            a=float((new[sel]==yt[sel]).mean())
            if a>best: best,bkey=a,(mn,nm)
        rows.append(dict(day=day,region=rn,n=n,inc=inc,ceil=best,
                         head=best-inc,cfg=f"{bkey[1]}/掩{bkey[0]}"))
        print(f"  {day}  {rn:6s} 区 n={n:6d}  在位者={inc:.4f}  "
              f"天花板={best:.4f}（{bkey[1]}/掩{bkey[0]}）  余量={best-inc:+.4f}",flush=True)
    # Q3 融合：用轻度验证选中的配置（lr/掩interarrival, R2）
    keep=dict(MASKS)["interarrival"]; m_=BIN[("interarrival","lr")]
    q=m_.predict_proba(Xt[:,keep])[:,1]
    sub=Ps[:,[i,j]]; pbase=sub[:,1]/np.clip(sub.sum(1),1e-12,None)   # 基模型限制到{i,j}
    sel=region("R2",t1,t2,t3)
    inc=float((t1[sel]==yt[sel]).mean())
    for w in (0.0,0.25,0.5,0.75,1.0):
        mix=(1-w)*pbase+w*q
        new=t1.copy(); new[sel]=np.where(mix[sel]>=0.5,j,i)
        blend_rows.append(dict(day=day,w=w,acc=float((new[sel]==yt[sel]).mean()),inc=inc))
    print(flush=True)

R=pd.DataFrame(rows)
print(f"\n{'='*90}\n=== Q1/Q2  在位者 vs 天花板（区上准确率）===",flush=True)
print(R.pivot_table(index="region",columns="day",values="inc").round(4).to_string(),flush=True)
print("\n天花板：",flush=True)
print(R.pivot_table(index="region",columns="day",values="ceil").round(4).to_string(),flush=True)
print("\n余量（天花板 − 在位者）：",flush=True)
print(R.pivot_table(index="region",columns="day",values="head").round(4).to_string(),flush=True)
print("\n判读 H：施加天余量若 ≈0 而选参天余量大 → 第二段在 UNSW 上确实到顶，",flush=True)
print("        闸门量到的是'那天基线弱'，不是可回收的信息。",flush=True)

B=pd.DataFrame(blend_rows)
print(f"\n{'='*90}\n=== Q3  融合而非替换（R2 区，lr/掩interarrival）===",flush=True)
print(B.pivot_table(index="w",columns="day",values="acc").round(4).to_string(),flush=True)
print("\n各天在位者（w=0 行即纯基模型限制到两类）：",flush=True)
print(B.groupby("day").inc.first().round(4).to_string(),flush=True)
print("\n判读：若存在一个 w 在【三天上同时】不低于各天在位者 → 问题在'硬替换'，第二段可救；",flush=True)
print("      若最好的 w 就是 0（纯基模型）→ 专用二分类器没带来新信息。",flush=True)
