"""为什么 lenhist 在自采上 +0.1008、在 UNSW/CIC 上 ≈0 —— 量每窗包数与直方图占用率。

**猜想**：不是"环境不同"，是【10 秒窗里包够不够】。
直方图特征要有信息，窗内必须有足够多的包把箱子填出可分的形状；
包太少时直方图退化成几个孤立计数，而矩（均值/分位数）已经把那点信息取完。

三个量，逐数据集：
  1 每窗包数分布           packet_count 的分位数
  2 直方图非零箱数          69 列 lenhist 里 cnt_* 非零的个数（该窗填了几个箱）
  3 占用率                 非零箱数 / 目标箱总数
并按【类】拆开，看困难类是不是特别稀疏。

判据：若自采的每窗包数与占用率显著高于 UNSW/CIC → 机制解释成立，
      "候选族在某粒度上能不能用"可以【事前算】，不必试。
"""
import sys
import numpy as np, pandas as pd
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
EXP=REPO+"/results/feature_expansion_20260905"

JOBS=[
 ("自采", REPO+"/results/g0_environment_grid_strict59_ra_r4/raw_all/features_raw_all_w10.csv",
          REPO+"/results/feature_expansion_20260905/features_lenhist_w10.csv", "label"),
 ("UNSW", REPO+"/results/unsw_features_full/features_day_16-09-30.csv",
          EXP+"/lenhist_unsw_w10.csv", "device"),
 ("CIC",  "/home/lmy/cic_probe/active_1103.csv",
          EXP+"/lenhist_cic_w10.csv", "device"),
]
DAY={"UNSW":"16-09-30","CIC":"2021_11_03_Active"}

rows=[]; percls=[]
for name,fcsv,lcsv,labcol in JOBS:
    f=pd.read_csv(fcsv,low_memory=False,encoding="utf-8-sig")
    f.columns=[c.lstrip("﻿") for c in f.columns]
    L=pd.read_csv(lcsv)
    cnt=[c for c in L.columns if c.startswith("lenhist_cnt_")]
    if name=="自采":
        keys=["label","round","source_file","window_id"]
        d=f.merge(L[keys+cnt],on=keys,how="inner")
    else:
        L=L[L.day==DAY[name]]
        d=f.merge(L[["device","window_id"]+cnt],on=["device","window_id"],how="inner")
    C=d[cnt].to_numpy()
    nz=(C>0).sum(1)                       # 该窗填了几个箱
    pk=d["packet_count"].to_numpy()
    occ=nz/len(cnt)
    inhist=C.sum(1)                       # 落进目标箱的包数
    rows.append(dict(数据集=name, 窗数=len(d), 箱数=len(cnt),
        包数_p10=np.percentile(pk,10), 包数_中位=np.median(pk), 包数_p90=np.percentile(pk,90),
        非零箱_中位=np.median(nz), 占用率_中位=np.median(occ),
        落箱率_中位=np.median(inhist/np.clip(pk,1,None)),
        窗内仅1箱的占比=float((nz<=1).mean())))
    g=d.groupby(labcol).apply(lambda x: pd.Series({
        "窗数":len(x), "包数中位":float(x["packet_count"].median()),
        "非零箱中位":float((x[cnt].to_numpy()>0).sum(1).mean())}),include_groups=False)
    g["数据集"]=name; percls.append(g)

R=pd.DataFrame(rows)
pd.set_option("display.width",200)
print("=== 每窗包数 与 直方图占用率（10 秒窗）===",flush=True)
print(R.round(3).to_string(index=False),flush=True)
print("\n=== 逐类（包数中位 / 非零箱中位）===",flush=True)
P=pd.concat(percls)
for nm in ("自采","UNSW","CIC"):
    sub=P[P.数据集==nm].sort_values("非零箱中位")
    print(f"\n--- {nm} ---",flush=True)
    print(sub[["窗数","包数中位","非零箱中位"]].head(12).round(2).to_string(),flush=True)
print("\n判读：自采若每窗包数与占用率显著高 → 'lenhist 在自采决定性、在另两个惰性' 有机制解释，",flush=True)
print("      且该族能否用可【事前算】：窗内包数不足以填出直方图形状时，它必然退化。",flush=True)
