"""【探索性,非协议】端到端：逐类对配置 =(模型, 观测时长)，只用源域导出，看 macro-F1。
选配置的依据：源域内时间块划分（前4块训/第5块测）的逐类对 AUC —— 不看任何目标标签。
臂：base（现状） / cfg_model（逐类对选模型+覆盖） / 并报 k=1 与 k=5 的时长聚合。"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC
UNSW=REPO+"/results/unsw_features_full/features_day_%s.csv"
def MK(nm):
    if nm=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return TC.make_model(nm,2)
CANDS=["lr","rf","xgboost"]

def run(tag, src, tgt, sday, tday, seeds=(42,43,44)):
    cols=P.feature_columns(src)
    devs=sorted(set(IID.day_gate(src,sday)) & set(IID.day_gate(tgt,tday)))
    le=LabelEncoder().fit(devs)
    d0=src[src.label.isin(devs)].sort_values("window_start_epoch")
    blk=TC.time_blocks(np.asarray(d0["window_start_epoch"]))
    X0=np.asarray(P.clean_x(d0,cols),dtype=float); y0=le.transform(d0.label)
    tr,te=blk<4,blk==4
    print(f"\n{'='*88}\n{tag}   {len(devs)} 类  {len(cols)} 列",flush=True)
    out=[]
    for seed in seeds:
        TC.SEED=seed
        s=P.sample_balanced(src[src.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        t=P.sample_balanced(tgt[tgt.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=seed)
        Xs=np.asarray(P.clean_x(s,cols),dtype=float); Xt=np.asarray(P.clean_x(t,cols),dtype=float)
        ys=le.transform(s.label); yt=le.transform(t.label)
        base=TC.make_model("xgboost",len(devs)); base.fit(Xs,ys)
        pp=base.predict_proba(Xt); o=np.argsort(-pp,axis=1); top1,top2=o[:,0],o[:,1]
        f_base=f1_score(yt,top1,average="macro"); a_base=(top1==yt).mean()
        need=sorted(set(map(tuple,np.sort(np.c_[top1,top2],axis=1))))
        pred=top1.copy(); n_ov=0; chosen={}
        for (i,j) in need:
            mi=np.isin(y0,[i,j])
            Xi,yi=X0[mi&tr],(y0[mi&tr]==j).astype(int)
            Xj,yj=X0[mi&te],(y0[mi&te]==j).astype(int)
            if len(np.unique(yi))<2 or len(np.unique(yj))<2: continue
            best=(None,-1)
            for nm in CANDS:                      # 只用源域挑模型
                try:
                    m=MK(nm); m.fit(Xi,yi); a=roc_auc_score(yj,m.predict_proba(Xj)[:,1])
                except Exception: continue
                if a>best[1]: best=(nm,a)
            if best[0] is None or best[1]<0.95: continue     # 源域都分不开就不动手
            ms=np.isin(ys,[i,j])
            if len(np.unique(ys[ms]))<2: continue
            pm=MK(best[0]); pm.fit(Xs[ms],(ys[ms]==j).astype(int))
            mm=((top1==i)&(top2==j))|((top1==j)&(top2==i))
            if not mm.any(): continue
            q=np.asarray(pm.predict(Xt[mm]))
            if q.ndim>1: q=q.argmax(1)
            pred[mm]=np.where(q.ravel()==1, j, i); n_ov+=int(mm.sum())
            chosen[f"{le.classes_[i]}|{le.classes_[j]}"]=[best[0],round(float(best[1]),4)]
        f_cfg=f1_score(yt,pred,average="macro"); a_cfg=(pred==yt).mean()
        out.append({"seed":seed,"base_macro":f_base,"cfg_macro":f_cfg,
                    "base_acc":a_base,"cfg_acc":a_cfg,"n_override":n_ov,"n_pairs":len(chosen)})
        print(f"  seed{seed}  base={f_base:.4f}  逐类对选模型={f_cfg:.4f}  Δ={f_cfg-f_base:+.4f}"
              f"   acc {a_base:.4f}→{a_cfg:.4f}   动手 {len(chosen)} 对 / {n_ov} 窗",flush=True)
        if seed==seeds[0]:
            cnt={}
            for v in chosen.values(): cnt[v[0]]=cnt.get(v[0],0)+1
            print(f"    选中的模型分布: {cnt}",flush=True)
    R=pd.DataFrame(out)
    print(f"  → 均值 Δmacro = {(R.cfg_macro-R.base_macro).mean():+.4f}   "
          f"Δacc = {(R.cfg_acc-R.base_acc).mean():+.4f}",flush=True)
    return R

with threadpool_limits(1):
    t0=time.time(); allr=[]
    for sd,td in [("16-09-23","16-09-30"),("16-09-30","16-10-12"),("16-09-23","16-10-12")]:
        allr.append(run(f"UNSW {sd} → {td}", pd.read_csv(UNSW%sd,low_memory=False),
                        pd.read_csv(UNSW%td,low_memory=False), sd, td))
    allr.append(run("CIC 1102 Idle → Active",
        pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False),
        pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False),
        "2021_11_02_Idle","2021_11_02_Active"))
    pd.concat(allr).to_csv("/home/lmy/cic_probe/cfg_e2e.csv",index=False)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)
