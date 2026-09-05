"""【探索性,非协议】CIC 迁移测试：新特征池下，inner 上的大幅修复能不能迁到 outer？

**为什么这是关键**：`region_fix` 显示 CIC 困难对在 inner（1102Idle→1103Active）上修得动，
且幅度很大：
    GosundPlug|GosundSocket  在位者 0.5222 → 配置 0.8162
    GlobeLamp|GosundPlug     在位者 0.4517 → 配置 0.9938
    GlobeLamp|GosundSocket   在位者 0.5929 → 配置 1.0000
    Teckin|Yutron            在位者 0.6182 → 配置 0.6125   ← 连 inner 都修不动
但旧特征池下 outer 只有 −0.0003（记忆：修得出来、迁不过去）。
**本次 inner 用的是新的 130 列池（61 base + 69 lenhist），迁移是否改善从未测过。**

设计：
  base    RF on 130 列，训练于 1102Idle
  inner   1103Active —— 导出逐类对配置（区定义 R2 与 R3 各一套）
  outer   1108Active —— 原样施加，量端到端与逐对
  臂      base / +平滑 / +定点修(R2) / +定点修(R3)

平滑按【设备 MAC】分组做因果滑动平均（CIC 每台自有 MAC，合法），k 在 inner 上选。
**注意**：今天测出第一段应改为「因果特征聚合」（比概率平均高 0.17），
但那需要在长窗上重抽特征，本脚本仍用概率平均 —— 本测试的对象是**第二段的迁移**，
不是第一段的最优形态。这一点须在报告里注明。
"""
from __future__ import annotations
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
NJ=12; SEED=42; MARGIN=0.02; MIN_CELL=40; TOPN=8; KBASE=[1,3,5,10,20]

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,C=1.0))
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

def main():
    t0=time.time()
    LH=pd.read_csv(EXP+"/lenhist_cic_w10.csv")
    lc=[c for c in LH.columns if c.startswith("lenhist_")]
    D={}
    for tag,(day,csv) in DAYS.items():
        d=pd.read_csv(csv,low_memory=False)
        d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
        D[tag]=d.sort_values(["device","window_id"]).reset_index(drop=True)
    base=[c for c in P.feature_columns(D["src"]) if not c.startswith("lenhist_")]
    cols=base+lc
    devs=sorted(set.intersection(*[set(IID.day_gate(D[t],DAYS[t][0])) for t in DAYS]))
    classes=sorted({TYPE(x) for x in devs}); le=LabelEncoder().fit(classes)
    print(f"设备 {len(devs)} → 类 {len(classes)}   {len(cols)} 列",flush=True)

    def XYG(tag):
        d=D[tag]; d=d[d.device.isin(devs)]
        return (np.asarray(P.clean_x(d,cols),dtype=float),
                le.transform([TYPE(x) for x in d.device]),
                d["device"].to_numpy())
    Xs,ys,_=XYG("src"); Xi,yi,gi=XYG("inner"); Xo,yo,go=XYG("outer")
    print(f"训练 {len(ys)}  inner {len(yi)}  outer {len(yo)}   {time.time()-t0:.0f}s",flush=True)

    rf=RandomForestClassifier(n_estimators=300,random_state=SEED,
                              class_weight="balanced",n_jobs=NJ)
    rf.fit(Xs,ys)
    Pi=rf.predict_proba(Xi); Po=rf.predict_proba(Xo)
    L=np.arange(len(classes))
    kb=max(((k,f1_score(yi,cmeanM(Pi,gi,k).argmax(1),average="macro",labels=L))
            for k in KBASE),key=lambda x:x[1])[0]
    Pi_s=cmeanM(Pi,gi,kb); Po_s=cmeanM(Po,go,kb)
    f_base=f1_score(yo,Po.argmax(1),average="macro",labels=L)
    f_sm  =f1_score(yo,Po_s.argmax(1),average="macro",labels=L)
    print(f"\n平滑窗 kb={kb}（inner 选）  outer: base={f_base:.4f}  +平滑={f_sm:.4f}",flush=True)

    oi=np.argsort(-Pi_s,axis=1); i1,i2,i3=oi[:,0],oi[:,1],oi[:,2]
    oo_=np.argsort(-Po_s,axis=1); o1,o2,o3=oo_[:,0],oo_[:,1],oo_[:,2]
    err=Counter()
    for a,b in zip(yi,i1):
        if a!=b: err[tuple(sorted((int(a),int(b))))]+=1
    hard=[p for p,_ in err.most_common(TOPN)]

    fams={}
    for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
    idx={c:n for n,c in enumerate(cols)}
    MASKS=[("none",np.arange(len(cols)))]
    for f,v in fams.items():
        keep=np.array([idx[c] for c in cols if c not in set(v)])
        if len(keep)>=5: MASKS.append((f,keep))
    MD=dict(MASKS); CANDS=["lr","rf"]

    REGS={"R2":lambda a,b,i,j:(((a==i)&(b==j))|((a==j)&(b==i))),
          "R3":lambda a,b,i,j:None}
    def reg(name,t1,t2,t3,i,j):
        if name=="R2": return ((t1==i)&(t2==j))|((t1==j)&(t2==i))
        return (((t1==i)|(t2==i)|(t3==i))&((t1==j)|(t2==j)|(t3==j)))

    cfgs={"R2":{},"R3":{}}
    print(f"\n=== inner 上导出配置（{len(hard)} 个困难对）===",flush=True)
    for i,j in hard:
        ms=np.isin(ys,[i,j])
        if len(np.unique(ys[ms]))<2: continue
        y1=(ys[ms]==j).astype(int); cache={}
        for rn in ("R2","R3"):
            Dm=reg(rn,i1,i2,i3,i,j)&np.isin(yi,[i,j])
            if Dm.sum()<MIN_CELL: continue
            inc=float((i1[Dm]==yi[Dm]).mean()); best=(None,None,-1.0)
            for mn,keep in MASKS:
                for nm in CANDS:
                    k_=(mn,nm)
                    if k_ not in cache:
                        try: cache[k_]=fit_w(MK(nm),nm,Xs[ms][:,keep],y1)
                        except Exception: cache[k_]=None
                    m_=cache[k_]
                    if m_ is None: continue
                    q=m_.predict_proba(Xi[:,keep])[:,1]
                    acc=float((np.where(q[Dm]>=0.5,j,i)==yi[Dm]).mean())
                    if acc>best[2]: best=(mn,nm,acc)
            if best[1] and best[2]>inc+MARGIN:
                cfgs[rn][(i,j)]=best
                print(f"  [{rn}] {classes[i]:22s}|{classes[j]:22s} n={int(Dm.sum()):6d} "
                      f"{inc:.4f}→{best[2]:.4f}  {best[1]}/掩{best[0]}",flush=True)
    print(f"\n过闸：R2 {len(cfgs['R2'])} 对   R3 {len(cfgs['R3'])} 对   {time.time()-t0:.0f}s",flush=True)

    print(f"\n=== outer 施加 ===",flush=True)
    res={"base":f_base,"+平滑":f_sm}
    for rn in ("R2","R3"):
        pred=o1.copy(); detail=[]
        for (i,j),(mn,nm,_a) in cfgs[rn].items():
            keep=MD[mn]; ms=np.isin(ys,[i,j])
            q=fit_w(MK(nm),nm,Xs[ms][:,keep],(ys[ms]==j).astype(int)
                    ).predict_proba(Xo[:,keep])[:,1]
            sel=reg(rn,o1,o2,o3,i,j)
            if not sel.any(): continue
            Dm=sel&np.isin(yo,[i,j])
            inc_o=float((o1[Dm]==yo[Dm]).mean()) if Dm.sum() else np.nan
            new=np.where(q[sel]>=0.5,j,i)
            pred[sel]=new
            acc_o=float((pred[Dm]==yo[Dm]).mean()) if Dm.sum() else np.nan
            detail.append((classes[i],classes[j],int(Dm.sum()),inc_o,acc_o))
        f=f1_score(yo,pred,average="macro",labels=L)
        res[f"+定点修({rn})"]=f
        print(f"\n  [{rn}] outer macro={f:.4f}   Δ vs 平滑={f-f_sm:+.4f}",flush=True)
        for a,b,n,x,y_ in detail:
            print(f"      {a:22s}|{b:22s} 区 n={n:6d}  在位者={x:.4f} → 修后={y_:.4f} "
                  f"({y_-x:+.4f})",flush=True)

    print(f"\n{'='*80}\n=== CIC 端到端（outer = 1108Active，型号级 {len(classes)} 类）===",flush=True)
    for k,v in res.items(): print(f"  {k:14s} {v:.4f}",flush=True)
    print("\n判读：+定点修 若明显 > +平滑 → 新特征池下迁移成立，CIC 上方法有作用面；",flush=True)
    print("      若仍 ≈ 0 → 「修得出来、迁不过去」在新池下依然成立，属身份型的确证。",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
