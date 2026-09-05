"""【探索性,非协议】UNSW 跨日迁移：错误落在哪。与 CIC 版逐字同构，数字可比。"""
import os, sys, itertools, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
FEAT=REPO+"/results/unsw_features_full/features_day_%s.csv"
PAIRS=[("16-09-23","16-09-30"),("16-09-30","16-10-12"),("16-09-23","16-10-12")]
with threadpool_limits(1):
    t0=time.time(); TC.SEED=42
    for SD,TD in PAIRS:
        src=pd.read_csv(FEAT%SD,low_memory=False); tgt=pd.read_csv(FEAT%TD,low_memory=False)
        cols=P.feature_columns(src)
        devs=sorted(set(IID.day_gate(src,SD)) & set(IID.day_gate(tgt,TD)))
        le=LabelEncoder().fit(devs)
        s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
        t=P.sample_balanced(tgt[tgt.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
        Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
        ys=le.transform(s.label); yt=le.transform(t.label); names=le.classes_
        full=np.arange(len(cols))
        m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys); pr=m.predict(Xt)
        print(f"\n{'='*84}\n{SD} → {TD}   {len(devs)} 类  {len(cols)} 列  "
              f"macro-F1={f1_score(yt,pr,average='macro'):.4f}  acc={(pr==yt).mean():.4f}  n={len(yt)}",flush=True)
        # 逐类对跨域 AUC
        auc={}
        ysn=np.asarray(s.label); ytn=np.asarray(t.label)
        for a,b in itertools.combinations(devs,2):
            v=TC.pair_auc(Xs,ysn,Xt,ytn,a,b,full)
            if v is not None: auc[tuple(sorted((a,b)))]=v
        C=confusion_matrix(yt,pr,labels=np.arange(len(devs)))
        err={}
        for i in range(len(devs)):
            for j in range(len(devs)):
                if i!=j:
                    k=tuple(sorted((names[i],names[j]))); err[k]=err.get(k,0)+C[i,j]
        E=pd.Series(err).sort_values(ascending=False); tot=E.sum()
        print(f"总错误 {tot}（{tot/len(yt)*100:.1f}%）  可测类对 {len(auc)}  反转对(<0.5) {sum(v<0.5 for v in auc.values())}",flush=True)
        print(f"错误集中度：top5 {E.head(5).sum()/tot*100:.1f}%  top10 {E.head(10).sum()/tot*100:.1f}%  "
              f"top20 {E.head(20).sum()/tot*100:.1f}%",flush=True)
        A=pd.DataFrame({"err":E}); A["cross_auc"]=[auc.get(k,np.nan) for k in A.index]; A=A.dropna()
        hi=A[A.cross_auc>=0.95]; lo=A[A.cross_auc<0.70]
        print(f"  跨域 AUC>=0.95：{len(hi):3d} 对，承载 {hi.err.sum()/tot*100:5.1f}% 的错误",flush=True)
        print(f"  跨域 AUC<0.70 ：{len(lo):3d} 对，承载 {lo.err.sum()/tot*100:5.1f}% 的错误",flush=True)
        print(f"  Spearman ρ(错误数, 跨域AUC) = {A.err.corr(A.cross_auc,method='spearman'):+.4f}",flush=True)
        top=A.head(10).copy(); top.index=[f"{a[:24]}|{b[:24]}" for a,b in top.index]
        print(top.assign(占总错误=lambda d:(d.err/tot*100).round(1)).round(4).to_string(),flush=True)
        A.to_csv(f"/home/lmy/cic_probe/unsw_err_{SD}_to_{TD}.csv")
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
