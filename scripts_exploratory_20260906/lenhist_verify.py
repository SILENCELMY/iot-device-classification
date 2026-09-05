"""【探索性,非协议】帧长直方图 +0.155 的证伪性检查 —— 先当它是 bug 或泄漏来查。

lenhist_test 测到 183 → 183+lenhist 在三个 outer 单元上 +0.155（pos_R5 +0.18），
比配置方法（+0.027）大一个数量级。基线数与 verify_flat 逐位对得上，inner/outer 同涨，
所以不是基线塌了也不是源域过拟合。但量级异常，**先查泄漏，再谈收益**。

四组检查，每组都能单独否掉这个结果：

  A 增益来自哪几列   cnt_* / frac_* / 汇总列(nuniq,entropy,top1_len,top1_frac,cover) 分开加
                     若全靠 top1_len 一列 → 更像捕获指纹而非设备指纹
  B 尺度无关性       只用 frac_*（比例）还成立吗？只有 cnt_* 有效 = 在数包数
  C 单独可用性       69 列单独用能到多少？远超 183 列 = 压倒性信号，需解释
  D 逐类 F1          修的是困难簇还是别的地方？困难簇不涨 = 与我们的问题无关

**泄漏的具体假设**（要能被 A/B 否掉）：
  H-leak1  某些长度只在某个 capture 出现，模型在认 capture 而非认设备
           → 反证：R5 是训练从未见过的 capture，若能迁移就不是认 capture
  H-leak2  计数与窗口包数强相关，等于重复了已有的 packet_count
           → 用 B（只用比例）检验
  H-leak3  目标长度的选择泄漏了目标域
           → 已排除：top32 只由 R2/R3/R4 的类间/类内方差比导出，代码在 lenhist_extract.py
"""
from __future__ import annotations
import os, sys, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import run_two_channel as TC

NEW89=REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
LENH =REPO+"/results/feature_expansion_20260905/features_lenhist_w10.csv"
KEYS=["label","round","source_file","window_id"]
GATEWAY=["Light_T1","Light_XM","Sensor"]
OUTER=[("pos_R5",["R2","R3","R4"],"R5"),
       ("jit_R6",["R2","R3","R4"],"R6"),
       ("jit_R7",["R2","R3","R4"],"R7")]
MODELS=["rf","xgboost","lightgbm"]; SEEDS=[42,43,44]

def main():
    t0=time.time()
    with threadpool_limits(1):
        d=TC.Data(); new=pd.read_csv(NEW89); lh=pd.read_csv(LENH)
        df=d.df.merge(new,on=KEYS,how="inner").merge(lh,on=KEYS,how="inner")
        assert len(df)==len(d.df)
        base=[c for c in TC.feature_columns(df) if not c.startswith("lenhist_")]
        cnt =[c for c in df.columns if c.startswith("lenhist_cnt_")]
        frac=[c for c in df.columns if c.startswith("lenhist_frac_")]
        summ=[c for c in df.columns if c.startswith("lenhist_") and c not in cnt+frac]
        top1=[c for c in df.columns if c=="lenhist_top1_len"]
        print(f"183 池 {len(base)}   cnt {len(cnt)}   frac {len(frac)}   汇总 {len(summ)}: {summ}",flush=True)
        POOLS={
          "183":            base,
          "183+cnt":        base+cnt,
          "183+frac":       base+frac,          # B：尺度无关
          "183+汇总":        base+summ,
          "183+top1_len":   base+top1,          # A：单列
          "183+全部":        base+cnt+frac+summ,
          "只lenhist":       cnt+frac+summ,      # C：单独可用性
          "只frac":          frac,
        }
        L5=sorted(df.label.unique()); i5={c:n for n,c in enumerate(L5)}
        GIDX=[i5[c] for c in GATEWAY]
        rows=[]; percls={}
        for uname,src,tgt in OUTER:
            s=df[df["round"].isin(src)].sort_values(["label","window_start"],kind="mergesort")
            t=df[df["round"]==tgt].sort_values(["label","window_start"],kind="mergesort")
            ys=np.array([i5[x] for x in s.label]); yt=np.array([i5[x] for x in t.label])
            gm=np.isin(yt,GIDX)
            for pname,cols in POOLS.items():
                Xs=np.asarray(TC.clean_x(s,cols),dtype=float)
                Xt=np.asarray(TC.clean_x(t,cols),dtype=float)
                for seed in SEEDS:
                    TC.SEED=seed
                    for mn in MODELS:
                        m=TC.make_model(mn,len(L5))
                        if m is None: continue
                        m.fit(Xs,ys); p=m.predict(Xt)
                        rows.append({"unit":uname,"pool":pname,"seed":seed,"model":mn,
                          "macro5":f1_score(yt,p,average="macro",labels=np.arange(len(L5))),
                          "gw3":f1_score(yt[gm],p[gm],average="macro",labels=GIDX)})
                        if seed==42 and mn=="rf":
                            percls[(uname,pname)]=f1_score(yt,p,average=None,
                                                           labels=np.arange(len(L5)))
            print(f"  {uname} 完成 {time.time()-t0:.0f}s",flush=True)

    R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/lenhist_verify.csv",index=False)
    bb=R.loc[R.groupby(["unit","pool","seed"]).macro5.idxmax()]
    print("\n=== A/B/C：各列组的贡献（best_base，5 类 macro）===",flush=True)
    P=bb.pivot_table(index="pool",columns="unit",values="macro5").round(4)
    P["均值"]=P.mean(axis=1).round(4)
    P["Δvs183"]=(P["均值"]-P.loc["183","均值"]).round(4)
    print(P.sort_values("均值",ascending=False).to_string(),flush=True)

    print("\n=== 网关三类 macro ===",flush=True)
    b3=R.loc[R.groupby(["unit","pool","seed"]).gw3.idxmax()]
    Q=b3.pivot_table(index="pool",columns="unit",values="gw3").round(4)
    Q["均值"]=Q.mean(axis=1).round(4)
    print(Q.sort_values("均值",ascending=False).to_string(),flush=True)

    print("\n=== D 逐类 F1（rf, seed42, pos_R5）===",flush=True)
    print("  " + "  ".join(f"{c[:9]:>9s}" for c in L5),flush=True)
    for pname in ["183","183+frac","183+全部","只lenhist"]:
        k=("pos_R5",pname)
        if k in percls:
            print(f"  " + "  ".join(f"{v:9.4f}" for v in percls[k]) + f"   {pname}",flush=True)

    print("\n=== 判读 ===",flush=True)
    g=lambda p: P.loc[p,"均值"] if p in P.index else float("nan")
    print(f"  只用比例列 183+frac = {g('183+frac'):.4f}（Δ={g('183+frac')-g('183'):+.4f}）",flush=True)
    print(f"    → 与 183+全部（{g('183+全部'):.4f}）接近 ⇒ 不是在数包数，H-leak2 否掉",flush=True)
    print(f"  只加 top1_len       = {g('183+top1_len'):.4f}（Δ={g('183+top1_len')-g('183'):+.4f}）",flush=True)
    print(f"    → 若单列就拿到大部分增益 ⇒ 更像捕获指纹，需进一步查",flush=True)
    print(f"  只用 lenhist        = {g('只lenhist'):.4f}",flush=True)
    print(f"    → 若已超过 183 池 ⇒ 帧长离散结构是压倒性信号，须在论文里正面解释",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
