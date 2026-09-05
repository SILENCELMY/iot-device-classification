"""【探索性,非协议】强基线重测：补上包长分布特征后，方法还剩多少贡献？

**为什么必须做**：检索发现 ByteIoT（IEEE 2021）就是"基于包长分布的 IoT 设备识别"，
在 UNSW 上报 99.08%。我们的 61/94/183 列池只有 `len_*` 的矩（均值/标准差/分位数），
**缺了包长的离散分布**——所以我们此前所有基线都是弱基线，方法的增益是在弱基线上量的。
自采上补齐后 outer 5 类 macro +0.155，量级是配置方法（+0.027）的六倍。

本脚本在 UNSW 与 CIC 上做同一件事，回答三个问题：

  Q1  补上包长分布后，基线抬到多少？（对照 ByteIoT 的 99%，但注意口径：我们是跨天）
  Q2  抬完之后，错误还剩多少、集中在哪几对？（若仍集中 → 方法仍有作用面）
  Q3  跨天 vs 同天的落差有没有被这一族消掉？（若消掉 → 我们的问题被解决了，要换题）

口径：与既有工作一致 —— 逐设备 10 s 窗、窗口级 macro-F1、跨天 outer、3 模型 best_base。
UNSW：inner 16-09-23→16-09-30，outer 16-09-23→16-10-12（既有协议）
CIC ：型号级；inner 1102Idle→1103Active，outer 1102Idle→1108Active
"""
from __future__ import annotations
import os, sys, time, re, itertools
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC

EXP=REPO+"/results/feature_expansion_20260905"
MODELS=["rf","xgboost","lightgbm"]; SEEDS=[42,43,44]

def TYPE(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    if re.match(r"TeckinPlug\d$", s):      return "Teckin"
    if re.match(r"YutronPlug\d$", s):      return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s): return "EchoDot"
    return s

def run(tag, files, days, lenh, src_day, tgt_days, type_level=False):
    """files: {day: csv}；lenh: 包长直方图 csv；对每个 tgt_day 报 base 与 base+lenhist。"""
    print(f"\n{'='*92}\n{tag}",flush=True)
    LH=pd.read_csv(lenh)
    lcols=[c for c in LH.columns if c.startswith("lenhist_")]
    D={}
    for day,f in files.items():
        d=pd.read_csv(f,low_memory=False)
        d=d.merge(LH[LH.day==day][["device","window_id"]+lcols], on=["device","window_id"], how="left")
        D[day]=d
    base=[c for c in P.feature_columns(D[src_day]) if not c.startswith("lenhist_")]
    print(f"  base {len(base)} 列   lenhist {len(lcols)} 列",flush=True)

    devs=set(IID.day_gate(D[src_day],src_day))
    for t in tgt_days: devs &= set(IID.day_gate(D[t],t))
    devs=sorted(devs)
    lab=(lambda s: TYPE(s)) if type_level else (lambda s: s)
    classes=sorted({lab(x) for x in devs}); le=LabelEncoder().fit(classes)
    merged={k:v for k,v in Counter(lab(x) for x in devs).items() if v>1}
    print(f"  设备 {len(devs)} 台 → 类 {len(classes)}   合并 {merged if merged else '无'}",flush=True)

    def XY(day):
        d=D[day]; d=d[d.device.isin(devs)]
        return d, le.transform([lab(x) for x in d.device])
    rows=[]; cms={}
    for pname,cols in [("base",base),("base+lenhist",base+lcols)]:
        ds,ys=XY(src_day); Xs=np.asarray(P.clean_x(ds,cols),dtype=float)
        for t in tgt_days:
            dt,yt=XY(t); Xt=np.asarray(P.clean_x(dt,cols),dtype=float)
            for seed in SEEDS:
                TC.SEED=seed
                for mn in MODELS:
                    m=TC.make_model(mn,len(classes))
                    if m is None: continue
                    m.fit(Xs,ys); p=m.predict(Xt)
                    f=f1_score(yt,p,average="macro",labels=np.arange(len(classes)))
                    rows.append({"pool":pname,"tgt":t,"seed":seed,"model":mn,"macro":f})
                    if seed==42 and mn=="rf": cms[(pname,t)]=(yt,p)
            print(f"    {pname:14s} {t} 完成",flush=True)
    R=pd.DataFrame(rows)
    bb=R.loc[R.groupby(["pool","tgt","seed"]).macro.idxmax()]
    print("\n  === 窗口级 macro-F1（best_base）===",flush=True)
    T=bb.pivot_table(index="tgt",columns="pool",values="macro").round(4)
    T["Δ"]=(T["base+lenhist"]-T["base"]).round(4)
    print("  "+T.to_string().replace("\n","\n  "),flush=True)

    # Q2：错误还剩多少、集中在哪几对
    for t in tgt_days:
        for pname in ["base","base+lenhist"]:
            if (pname,t) not in cms: continue
            yt,p=cms[(pname,t)]
            C=confusion_matrix(yt,p,labels=np.arange(len(classes)))
            off=C.copy(); np.fill_diagonal(off,0)
            tot=off.sum()
            if tot==0:
                print(f"\n  {t} {pname}: 零错误",flush=True); continue
            pairs={}
            for i in range(len(classes)):
                for j in range(i+1,len(classes)):
                    v=off[i,j]+off[j,i]
                    if v>0: pairs[(i,j)]=v
            top=sorted(pairs.items(),key=lambda x:-x[1])[:6]
            cum=sum(v for _,v in top)/tot
            print(f"\n  {t} {pname}: 错误 {tot} 个，前 6 对占 {cum:.1%}",flush=True)
            for (i,j),v in top:
                print(f"     {classes[i]:26s} {classes[j]:26s} {v:6d}  ({v/tot:5.1%})",flush=True)
    return R

def main():
    t0=time.time()
    with threadpool_limits(1):
        U=REPO+"/results/unsw_pilot/four_day/features_unsw_w10_4day.csv"
        # 四天在同一个文件里，按 day 切
        ud=pd.read_csv(U,low_memory=False)
        tmp={}
        for day in ["16-09-23","16-09-30","16-10-12"]:
            f=f"/home/lmy/cic_probe/_unsw_{day}.csv"
            if not os.path.exists(f): ud[ud.day==day].to_csv(f,index=False)
            tmp[day]=f
        del ud
        run("UNSW（10 类，跨天）", tmp, None, EXP+"/lenhist_unsw_w10.csv",
            "16-09-23", ["16-09-30","16-10-12"])

        cf={"2021_11_02_Idle":"/home/lmy/cic_probe/idle_1102.csv",
            "2021_11_03_Active":"/home/lmy/cic_probe/active_1103.csv",
            "2021_11_08_Active":"/home/lmy/cic_probe/active_1108.csv"}
        run("CIC（型号级，跨天）", cf, None, EXP+"/lenhist_cic_w10.csv",
            "2021_11_02_Idle", ["2021_11_03_Active","2021_11_08_Active"], type_level=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
