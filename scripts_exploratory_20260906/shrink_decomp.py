"""分离"缩水"的两个来源 —— 3×3 设计，全部已烧天，不碰保留集。

**问题**：同一配置的区上增幅 选参天 +0.0684 → 保留 16 天 +0.0417，缩水四成。
审计文档初稿把它全算给"选择偏差"，**归因过头** —— 留出天的在位者本来就高
0.124（0.6338 vs 0.5097），headroom 更小，这一条当天已由 why_stage2 独立量到。

**设计**：3 个已烧天两两互作导出天/施加天。
    对角线   在 X 天导出、在 X 天施加  → 含选择偏差
    非对角   在 X 天导出、在 Y 天施加  → 对 Y 而言无偏
分离（两个效应在此设计下正交，无需额外假设）：
    选择偏差 = 固定【施加天】，对角 − 非对角均值   （施加天相同 ⇒ headroom 相同）
    headroom = 固定【导出天】，增幅随施加天在位者高低的变化

**便宜的原因**：专用模型全部训练于同一源天 16-09-23，28 个候选只需拟合一次，
三个天只换评估集。

对象：Belkin 对（C3 确证中唯一过闸的对），区固定 R2（确证中选中的区）。
"""
import sys, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C
import pilot_rf_loro as P
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

DAYS=["16-09-30","16-10-11","16-10-12"]; SEED=42
t0=time.time()
LH=pd.read_csv(C.LH_MAIN); lc=[c for c in LH.columns if c.startswith("lenhist_")]
def load(day):
    d=pd.read_csv(C.FULL%day,low_memory=False,encoding="utf-8-sig")
    d=d.merge(LH[LH.day==day][["device","window_id"]+lc],on=["device","window_id"],how="left")
    return d.sort_values(["device","window_id"]).reset_index(drop=True)
ALL=[C.TRAIN]+DAYS; D={d:load(d) for d in ALL}
base=[c for c in P.feature_columns(D[C.TRAIN]) if not c.startswith("lenhist_")]
cols=base+lc
devs=sorted(set.intersection(*[C.gate_day(d) for d in ALL]))
le=LabelEncoder().fit(devs); L=np.arange(len(devs))
def XYG(day):
    d=D[day]; d=d[d.device.isin(devs)]
    return (np.asarray(P.clean_x(d,cols),dtype=np.float32),
            le.transform(d.device), d.device.to_numpy())
Xs,ys,_=XYG(C.TRAIN)
rf=RandomForestClassifier(n_estimators=300,random_state=SEED,class_weight="balanced",
                          n_jobs=12).fit(Xs,ys)
i=int(np.where(le.classes_=="BelkinWemoMotion")[0][0])
j=int(np.where(le.classes_=="BelkinWemoSwitch")[0][0])
print(f"{len(devs)} 类  {len(cols)} 列  对象 {le.classes_[i]}|{le.classes_[j]}  区 R2  "
      f"{time.time()-t0:.0f}s",flush=True)

fams={}
for c in cols: fams.setdefault(c.split("_")[0],[]).append(c)
idx={c:n for n,c in enumerate(cols)}
MASKS=[("none",np.arange(len(cols)))]
for f,v in fams.items():
    keep=np.array([idx[c] for c in cols if c not in set(v)])
    if len(keep)>=5: MASKS.append((f,keep))
MD=dict(MASKS)
ms=np.isin(ys,[i,j]); y1=(ys[ms]==j).astype(int)
SPEC={}
for mn,keep in MASKS:
    for nm in ("lr","rf"):
        try: SPEC[(mn,nm)]=C.fit_w(C.MK(nm,SEED),nm,Xs[ms][:,keep],y1)
        except Exception: SPEC[(mn,nm)]=None
print(f"专用候选 {len(SPEC)} 个（14 掩码 × 2 模型），全部训练于 {C.TRAIN}   "
      f"{time.time()-t0:.0f}s",flush=True)

# kb 与 C3 同法：在 16-09-30 上选
Xu,yu,gu=XYG("16-09-30"); Pu=rf.predict_proba(Xu)
kb=max(((k,f1_score(yu,C.cmeanM(Pu,gu,k).argmax(1),average="macro",labels=L))
        for k in C.KBASE),key=lambda x:x[1])[0]
print(f"平滑窗 kb={kb}\n",flush=True)

ST={}
for day in DAYS:
    Xt,yt,gt=XYG(day); Ps=C.cmeanM(rf.predict_proba(Xt),gt,kb)
    o=np.argsort(-Ps,axis=1); t1,t2=o[:,0],o[:,1]
    sel=((t1==i)&(t2==j))|((t1==j)&(t2==i))
    inc=float((t1[sel]==yt[sel]).mean())
    acc={}
    for k_,m_ in SPEC.items():
        if m_ is None: continue
        q=m_.predict_proba(Xt[:,MD[k_[0]]])[:,1]
        mix=C.blend(Ps,q,i,j)
        new=t1.copy(); new[sel]=np.where(mix[sel]>=0.5,j,i)
        acc[k_]=float((new[sel]==yt[sel]).mean())
    ST[day]=dict(n=int(sel.sum()),inc=inc,acc=acc)
    print(f"  {day}  区 n={int(sel.sum()):6d}  在位者={inc:.4f}   {time.time()-t0:.0f}s",flush=True)

# 3×3：在 X 天导出（取最大），在 Y 天施加
G=pd.DataFrame(index=DAYS,columns=DAYS,dtype=float)
CFG={}
for dx in DAYS:
    best=max(ST[dx]["acc"], key=lambda k_: ST[dx]["acc"][k_])
    CFG[dx]=best
    for dy in DAYS:
        G.loc[dx,dy]=ST[dy]["acc"][best]-ST[dy]["inc"]
pd.set_option("display.width",200)
print(f"\n{'='*88}\n=== 3×3：行=导出天，列=施加天，值=区上增幅 ===",flush=True)
print(G.round(4).to_string(),flush=True)
print("\n各天选出的配置：",flush=True)
for dx in DAYS: print(f"  导出于 {dx} → {CFG[dx][1]}/掩{CFG[dx][0]}",flush=True)
print("\n各天在位者（headroom 的代理）：",flush=True)
for dy in DAYS: print(f"  {dy}  在位者={ST[dy]['inc']:.4f}  区 n={ST[dy]['n']}",flush=True)

print(f"\n{'='*88}\n=== 分离 ===",flush=True)
print("【选择偏差】固定施加天，对角 − 非对角均值（施加天相同 ⇒ headroom 相同）：",flush=True)
sb=[]
for dy in DAYS:
    diag=G.loc[dy,dy]; off=G.loc[[d for d in DAYS if d!=dy],dy].mean()
    sb.append(diag-off)
    print(f"  施加于 {dy}：对角 {diag:+.4f}  非对角均值 {off:+.4f}  "
          f"→ 选择偏差 {diag-off:+.4f}",flush=True)
print(f"  平均选择偏差 = {np.mean(sb):+.4f}",flush=True)
print("\n【headroom 效应】固定导出天，增幅随施加天在位者的变化：",flush=True)
for dx in DAYS:
    row=[(dy,ST[dy]['inc'],G.loc[dx,dy]) for dy in DAYS if dy!=dx]
    txt="   ".join(f"{dy}(在位者{v:.3f}) {g:+.4f}" for dy,v,g in row)
    print(f"  导出于 {dx}：{txt}",flush=True)
incs=np.array([ST[d]["inc"] for d in DAYS])
offd=np.array([G.loc[[d2 for d2 in DAYS if d2!=d],d].mean() for d in DAYS])
if len(DAYS)>2:
    r=np.corrcoef(incs,offd)[0,1]
    print(f"  在位者 vs 无偏增幅 的相关 r = {r:+.3f}（负 ⇒ 在位者越高可捞越少）",flush=True)
print(f"\n对照 C3 确证：选参天 +0.0684 → 保留 16 天 +0.0417（缩水 0.0267）",flush=True)
print(f"若 平均选择偏差 ≈ 0.027 → 缩水主要是选择偏差；",flush=True)
print(f"若 平均选择偏差 << 0.027 → 缩水主要来自 headroom 差异。",flush=True)
print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
