"""【探索性,非协议】验证归因：cfg_self_hier 的 flat=0.784（xgboost 单模型）
与 override_183 的 flat_183=0.8072（best_base 三模型取最大）之间 0.0235 的差距，
是否确实由"单模型 vs 三模型取最优"造成。

做法：同一份 183 列池、同样的 R2R3R4 → R5/R6/R7、同样 5 seed，
分别跑 rf / xgboost / lightgbm，并按 override_183 的 best_base 口径
逐 (单元,seed) 取三者 macro 最大，看均值是否回到 0.8072 附近。

若回到 → 归因成立，可比性是实测的而非推断的。
若不回到 → 我的 inner/outer 设置与 override_183 有别的不一致，所有 Δ 需重新审视。
"""
from __future__ import annotations
import os, sys, time, hashlib
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import run_two_channel as TC

NEW89 = REPO+"/results/feature_expansion_20260904/features_new89_w10.csv"
KEYS  = ["label","round","source_file","window_id"]
OUTER = [("pos_R5", ["R2","R3","R4"], "R5"),
         ("jit_R6", ["R2","R3","R4"], "R6"),
         ("jit_R7", ["R2","R3","R4"], "R7")]
MODELS=["rf","xgboost","lightgbm"]
SEEDS=[42,43,44,45,46]

def main():
    t0=time.time()
    with threadpool_limits(1):
        d=TC.Data(); new=pd.read_csv(NEW89)
        df=d.df.merge(new, on=KEYS, how="inner")
        assert len(df)==len(d.df)
        cols=TC.feature_columns(df); L=list(d.enc.classes_)
        print(f"183 列池 {len(cols)} 列  {len(L)} 类  {len(df)} 行",flush=True)
        def block(rounds):
            s=df[df["round"].isin(rounds)]
            return np.asarray(TC.clean_x(s,cols),dtype=float), d.enc.transform(s["label"])
        rows=[]
        for uname,src,tgt in OUTER:
            Xs,ys=block(src); Xt,yt=block([tgt])
            for seed in SEEDS:
                TC.SEED=seed
                for m in MODELS:
                    clf=TC.make_model(m,len(L)); clf.fit(Xs,ys)
                    f=f1_score(yt,clf.predict(Xt),average="macro",labels=np.arange(len(L)))
                    rows.append({"unit":uname,"seed":seed,"model":m,"macro_f1":f})
            print(f"  {uname} 完成  {time.time()-t0:.0f}s",flush=True)
        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/verify_flat.csv",index=False)
        print("\n=== 逐模型均值（三单元 × 5 seed）===",flush=True)
        print(R.groupby("model").macro_f1.mean().round(4).to_string(),flush=True)
        print("\n=== 逐单元 × 逐模型 ===",flush=True)
        print(R.pivot_table(index="unit",columns="model",values="macro_f1").round(4).to_string(),flush=True)
        bb=R.loc[R.groupby(["unit","seed"]).macro_f1.idxmax()]
        print("\n=== best_base（逐 单元,seed 取三模型最大，override_183 口径）===",flush=True)
        print(bb.groupby("unit").macro_f1.mean().round(4).to_string(),flush=True)
        print(f"\nbest_base 三单元均值 = {bb.macro_f1.mean():.4f}",flush=True)
        print(f"xgboost 单模型均值   = {R[R.model=='xgboost'].macro_f1.mean():.4f}",flush=True)
        print(f"参照：override_183 的 flat_183 = 0.8072",flush=True)
        print(f"      cfg_self_hier 的 flat    ≈ 0.784（xgboost 单模型）",flush=True)
        print(f"\n差距 best_base − xgboost = {bb.macro_f1.mean()-R[R.model=='xgboost'].macro_f1.mean():+.4f}",flush=True)
        print(f"总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
