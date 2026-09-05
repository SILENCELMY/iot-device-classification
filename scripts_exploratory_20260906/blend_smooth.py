"""【探索性，不进主线，不改协议】审计缺陷 2：融合两侧平滑口径不一致。

现状（已确证的 C3 用的就是这个形态）：
    pbase = 平滑后的基模型概率，限制到 {i,j} 归一     ← 【已平滑】
    q     = 专用二分类器的逐窗原始概率                ← 【未平滑】
    mix   = 0.5*pbase + 0.5*q
两侧方差尺度不同就等权平均。推断：噪声大的一侧被给了过高的有效权重，
故融合比它本可达到的更抖 ⇒ 报出的数偏低（保守）。**该推断未经实测，本脚本验它。**

对拍两臂（其余完全相同）：
    A  专用侧【原始】   —— 现状，等于已确证的形态
    B  专用侧【同口径平滑】—— 与基模型同一个 kb、同一个分组做因果滑动平均
逐 w ∈ {0,.25,.5,.75,1} × 三天（选参 16-09-30 / 施加 16-10-11 / 16-10-12）
报：区上准确率 + 端到端 macro-F1。

判据：
  B 的峰值 > A 的峰值，且 w=0.5 仍在峰附近 → 修法有效且不需重新调参
  B ≈ A                                    → 不对称无实质影响，缺陷 2 降为记号问题
  B < A                                    → 推断反了，如实记，且【不得】按结果改判据
"""
import sys
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C
import pilot_rf_loro as P

TUNE="16-09-30"; APPLY=["16-10-11","16-10-12"]; SEED=42
WS=[0.0,0.25,0.5,0.75,1.0]

def cmean1(v,g,k):
    """一维概率的因果滑动平均 —— 与 cmeanM 同口径（同 k、同分组、同因果性）。"""
    if k<=1: return v
    o=np.empty_like(v)
    for u in np.unique(g):
        i=np.where(g==u)[0]; V=v[i]
        Cs=np.concatenate([[0.0],np.cumsum(V)])
        for n in range(len(i)):
            lo=max(0,n-k+1); o[i[n]]=(Cs[n+1]-Cs[lo])/(n+1-lo)
    return o

LH=pd.read_csv(C.LH_MAIN); lc=[c for c in LH.columns if c.startswith("lenhist_")]
def load(day):
    d=pd.read_csv(C.FULL%day,low_memory=False,encoding="utf-8-sig")
    d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
    return d.sort_values(["device","window_id"]).reset_index(drop=True)
DAYS=[C.TRAIN,TUNE]+APPLY
D={d:load(d) for d in DAYS}
base=[c for c in P.feature_columns(D[C.TRAIN]) if not c.startswith("lenhist_")]
cols=base+lc
devs=sorted(set.intersection(*[C.gate_day(d) for d in DAYS]))
le=LabelEncoder().fit(devs); L=np.arange(len(devs))
def XYG(day):
    d=D[day]; d=d[d.device.isin(devs)]
    return (np.asarray(P.clean_x(d,cols),dtype=np.float32),
            le.transform(d.device), d.device.to_numpy())
Xs,ys,_=XYG(C.TRAIN)
rf=RandomForestClassifier(n_estimators=300,random_state=SEED,class_weight="balanced",
                          n_jobs=12).fit(Xs,ys)
print(f"{len(devs)} 类  {len(cols)} 列  训练 {C.TRAIN}",flush=True)

# kb 在选参天上选（与 C3 同法）
Xu,yu,gu=XYG(TUNE); Pu=rf.predict_proba(Xu)
kb=max(((k,f1_score(yu,C.cmeanM(Pu,gu,k).argmax(1),average="macro",labels=L))
        for k in C.KBASE),key=lambda x:x[1])[0]
print(f"平滑窗 kb={kb}（在 {TUNE} 上选，与 C3 同法）\n",flush=True)

i=int(np.where(le.classes_=="BelkinWemoMotion")[0][0])
j=int(np.where(le.classes_=="BelkinWemoSwitch")[0][0])
# 专用二分类器：用 C3 确证里 Belkin 实际选中的配置族
ms=np.isin(ys,[i,j]); y1=(ys[ms]==j).astype(int)
fams={}
for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
idx={c:n for n,c in enumerate(cols)}
keep=np.array([idx[c] for c in cols if not c.startswith("lenhist_")])   # 掩 lenhist
spec=C.fit_w(C.MK("lr",SEED),"lr",Xs[ms][:,keep],y1)
print(f"专用二分类器 lr/掩lenhist（C3 确证中 Belkin 实际选中的配置）\n",flush=True)

rows=[]
for day in [TUNE]+APPLY:
    Xt,yt,gt=XYG(day)
    Pm=rf.predict_proba(Xt); Ps=C.cmeanM(Pm,gt,kb)
    t1=Ps.argmax(1); t2=np.argsort(-Ps,axis=1)[:,1]
    sel=((t1==i)&(t2==j))|((t1==j)&(t2==i))            # R2，C3 确证中选中的区
    inc=float((t1[sel]==yt[sel]).mean())
    f_sm=f1_score(yt,t1,average="macro",labels=L)
    sub=Ps[:,[i,j]]; pbase=sub[:,1]/np.clip(sub.sum(1),1e-12,None)
    q_raw=spec.predict_proba(Xt[:,keep])[:,1]
    q_sm =cmean1(q_raw,gt,kb)                          # 与基模型同口径
    for arm,q in (("A 专用原始",q_raw),("B 专用同口径平滑",q_sm)):
        for w in WS:
            mix=(1-w)*pbase+w*q
            new=t1.copy(); new[sel]=np.where(mix[sel]>=0.5,j,i)
            rows.append(dict(day=day,arm=arm,w=w,
                             区上=float((new[sel]==yt[sel]).mean()),
                             端到端=f1_score(yt,new,average="macro",labels=L)))
    rows.append(dict(day=day,arm="在位者",w=np.nan,区上=inc,端到端=f_sm))
    print(f"  {day}  区 n={int(sel.sum()):6d}  在位者 区上={inc:.4f}  端到端={f_sm:.4f}",flush=True)

R=pd.DataFrame(rows); pd.set_option("display.width",200)
R.to_csv("/home/lmy/cic_probe/blend_smooth.csv",index=False)
for metric in ("区上","端到端"):
    print(f"\n{'='*84}\n=== {metric} ===",flush=True)
    T=R[R.arm!="在位者"].pivot_table(index=["arm","w"],columns="day",values=metric)
    print(T.round(4).to_string(),flush=True)
    print("\n在位者：",flush=True)
    print(R[R.arm=="在位者"].set_index("day")[metric].round(4).to_string(),flush=True)

print(f"\n{'='*84}\n=== 判读 ===",flush=True)
for day in [TUNE]+APPLY:
    inc=float(R[(R.arm=="在位者")&(R.day==day)].区上.iloc[0])
    a=R[(R.arm=="A 专用原始")&(R.day==day)]
    b=R[(R.arm=="B 专用同口径平滑")&(R.day==day)]
    aw=a.loc[a.区上.idxmax()]; bw=b.loc[b.区上.idxmax()]
    a5=float(a[a.w==0.5].区上.iloc[0]); b5=float(b[b.w==0.5].区上.iloc[0])
    print(f"  {day}  在位者 {inc:.4f}",flush=True)
    print(f"        A 峰 w={aw.w:.2f} {aw.区上:.4f}（w=0.5 时 {a5:.4f}, {a5-inc:+.4f}）",flush=True)
    print(f"        B 峰 w={bw.w:.2f} {bw.区上:.4f}（w=0.5 时 {b5:.4f}, {b5-inc:+.4f}）"
          f"   B−A@0.5 = {b5-a5:+.4f}",flush=True)
print("\n  B>A 且 w=0.5 仍在峰附近 → 修法有效且不需重新调参；",flush=True)
print("  B≈A → 缺陷 2 降为记号问题；  B<A → 推断反了，如实记，不得按结果改判据。",flush=True)
