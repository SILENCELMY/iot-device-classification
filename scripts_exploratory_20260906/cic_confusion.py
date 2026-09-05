"""【探索性,非协议】31% 的错误到底落在哪：逐类对可分性 vs 多类 argmax 错误质量。只读 1102。"""
import os, sys
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
with threadpool_limits(1):
    TC.SEED=42
    src=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
    tgt=pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
    cols=P.feature_columns(src)
    devs=sorted(set(IID.day_gate(src,"2021_11_02_Idle")) & set(IID.day_gate(tgt,"2021_11_02_Active")))
    le=LabelEncoder().fit(devs)
    s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    t=P.sample_balanced(tgt[tgt.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=42)
    Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
    ys=le.transform(s.label); yt=le.transform(t.label)
    m=TC.make_model("xgboost",len(devs)); m.fit(Xs,ys); pr=m.predict(Xt)
    print(f"macro-F1={f1_score(yt,pr,average='macro'):.4f}  acc={(pr==yt).mean():.4f}  n={len(yt)}",flush=True)
    C=confusion_matrix(yt,pr,labels=np.arange(len(devs)))
    names=le.classes_
    # 错误质量按无序类对聚合
    err={}
    for i in range(len(devs)):
        for j in range(len(devs)):
            if i==j: continue
            k=tuple(sorted((names[i],names[j])))
            err[k]=err.get(k,0)+C[i,j]
    E=pd.Series(err).sort_values(ascending=False)
    tot=E.sum()
    print(f"\n总错误 {tot} 个窗口（{tot/len(yt)*100:.1f}%）；有错的类对 {int((E>0).sum())}/{len(E)}",flush=True)
    print(f"错误质量集中度：top5 {E.head(5).sum()/tot*100:.1f}%  top10 {E.head(10).sum()/tot*100:.1f}%  "
          f"top20 {E.head(20).sum()/tot*100:.1f}%  top50 {E.head(50).sum()/tot*100:.1f}%",flush=True)
    # 关联到逐类对跨域 AUC
    car=pd.read_csv("/home/lmy/cic_probe/carrier.csv")
    auc={tuple(sorted(p.split("|"))):v for p,v in zip(car.pair,car.cross_full)}
    A=pd.DataFrame({"err":E})
    A["cross_auc"]=[auc.get(k,np.nan) for k in A.index]
    A=A.dropna()
    print(f"\n=== 错误最多的 15 个类对，及其逐类对跨域 AUC ===",flush=True)
    top=A.head(15).copy(); top.index=[f"{a[:22]}|{b[:22]}" for a,b in top.index]
    print(top.assign(占总错误=lambda d:(d.err/tot*100).round(1)).round(4).to_string(),flush=True)
    hi=A[A.cross_auc>=0.95]; lo=A[A.cross_auc<0.7]
    print(f"\n跨域 AUC>=0.95 的类对：{len(hi)} 对，承载 {hi.err.sum()/tot*100:.1f}% 的错误",flush=True)
    print(f"跨域 AUC<0.70 的类对：{len(lo)} 对，承载 {lo.err.sum()/tot*100:.1f}% 的错误",flush=True)
    print(f"\nSpearman ρ(错误数, 跨域AUC) = {A.err.corr(A.cross_auc,method='spearman'):+.4f}",flush=True)
    # 每类被判错的去向数：拥挤度
    row_err=(C.sum(1)-np.diag(C))
    spread=[(names[i], row_err[i], int((C[i]>0).sum())-1, int(C[i].argmax()!=i)) for i in range(len(devs))]
    S=pd.DataFrame(spread,columns=["类","错误窗口","被判成几个别的类","众数就判错"]).sort_values("错误窗口",ascending=False)
    print(f"\n=== 错误最多的 10 个类：错误是集中到 1 个对手还是散开 ===",flush=True)
    print(S.head(10).to_string(index=False),flush=True)
    print(f"\n平均每个出错类散布到 {S[S.错误窗口>0]['被判成几个别的类'].mean():.1f} 个不同的错误目标",flush=True)
