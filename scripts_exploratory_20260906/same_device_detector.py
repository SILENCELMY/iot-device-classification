"""【探索性,非协议】同设备检测器：用"域内可分 + 跨捕获不可分 + 修不动"识别过细的标签。

签名（跑前定义，不因结果调整）：
  ① in_dom  = max(Idle 内, Active 内) 时间块留出 AUC，取模型最优        >= 0.90
  ② cross   = 1102Idle → 1108Idle（同状态、只换捕获）上，(模型 × 时长)
              搜出的最好【逐对准确率】                                   <= 0.60
  两条同时满足 → 判为"数据不支持这个标签区分"

真值：CIC 设备命名。同型号对 = 同一 model 前缀。
  粗：Gosund 全系 6 台一族 → C(6,2)=15 对；Teckin 1；Yutron 1；EchoDot 1  合计 18/630
  细：GosundPlug 4 台 C(4,2)=6；GosundSocket 2 台 1；其余同上            合计 10/630

安全阀检查：自采网关三类（同域 0.86 / 跨轮次 0.68）不应触发——本脚本不含自采，
但阈值 cross<=0.60 已使 0.68 落在触发区之外。
"""
from __future__ import annotations
import os, sys, time, re, itertools
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

REPO="/home/lmy/iot-device-classification"
sys.path.insert(0, REPO+"/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO+"/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO+"/results/two_channel_20260903")
import pilot_rf_loro as P, run_unsw_iid_reference as IID, run_two_channel as TC

SEED=42; MODELS=["lr","rf"]; HORIZONS=[1,3,10]
IN_DOM_GATE=0.90; CROSS_GATE=0.60

def MK(n):
    if n=="lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return TC.make_model(n,2)

def cmean(p,g,k):
    if k<=1: return p
    o=np.empty_like(p)
    for u in np.unique(g):
        i=np.where(g==u)[0]; v=p[i]; c=np.cumsum(np.insert(v,0,0.0))
        for n in range(len(v)):
            lo=max(0,n-k+1); o[i[n]]=(c[n+1]-c[lo])/(n+1-lo)
    return o

def model_of(s):
    if re.match(r"GosundESP.*(Plug|Socket)$", s): return "Gosund"
    if re.match(r"TeckinPlug\d$", s):             return "Teckin"
    if re.match(r"YutronPlug\d$", s):             return "Yutron"
    if re.match(r"AmazonAlexaEchoDot\d$", s):     return "EchoDot"
    return s
def model_of_fine(s):
    if re.match(r"GosundESP.*Plug$", s):   return "GosundPlug"
    if re.match(r"GosundESP.*Socket$", s): return "GosundSocket"
    return model_of(s)

def main():
    t0=time.time()
    with threadpool_limits(1):
        TC.SEED=SEED
        I02=pd.read_csv("/home/lmy/cic_probe/idle_1102.csv",low_memory=False)
        A02=pd.read_csv("/home/lmy/cic_probe/active_1102.csv",low_memory=False)
        I08=pd.read_csv("/home/lmy/cic_probe/idle_1108.csv",low_memory=False)
        cols=P.feature_columns(I02)
        devs=sorted(set(IID.day_gate(I02,"2021_11_02_Idle"))
                    & set(IID.day_gate(A02,"2021_11_02_Active"))
                    & set(IID.day_gate(I08,"2021_11_08_Idle")))
        print(f"{len(devs)} 台设备  {len(cols)} 列  {len(devs)*(len(devs)-1)//2} 对",flush=True)

        # 同域两侧：时间块留出
        def indom(df, tag):
            d=df[df.label.isin(devs)].sort_values("window_start_epoch")
            blk=TC.time_blocks(np.asarray(d["window_start_epoch"]))
            X=np.asarray(P.clean_x(d,cols),dtype=float); y=np.asarray(d.label)
            return X, y, blk<4, blk==4
        XI,yI,trI,teI = indom(I02,"I02")
        XA,yA,trA,teA = indom(A02,"A02")
        # 跨捕获：1102Idle → 1108Idle（同状态，只换捕获）
        s=P.sample_balanced(I02[I02.label.isin(devs)],max_rows=IID.MAX_ROWS,random_state=SEED)
        t=I08[I08.label.isin(devs)].sort_values(["label","window_start_epoch"])
        t=P.sample_balanced(t,max_rows=IID.MAX_ROWS,random_state=SEED).sort_values(["label","window_start_epoch"])
        XS=np.asarray(P.clean_x(s,cols),dtype=float); yS=np.asarray(s.label)
        XT=np.asarray(P.clean_x(t,cols),dtype=float); yT=np.asarray(t.label); gT=np.asarray(t.label)

        rows=[]; pairs=list(itertools.combinations(devs,2))
        for n,(a,b) in enumerate(pairs):
            rec={"a":a,"b":b}
            # ① 同域
            best_in=0.0
            for X,y,tr,te,tag in ((XI,yI,trI,teI,"I"),(XA,yA,trA,teA,"A")):
                mtr=np.isin(y,[a,b])&tr; mte=np.isin(y,[a,b])&te
                if mtr.sum()<40 or mte.sum()<20: continue
                y1=(y[mtr]==b).astype(int); y2=(y[mte]==b).astype(int)
                if len(np.unique(y1))<2 or len(np.unique(y2))<2: continue
                for nm in MODELS:
                    try:
                        m=MK(nm); m.fit(X[mtr],y1)
                        v=roc_auc_score(y2, m.predict_proba(X[mte])[:,1])
                    except Exception: continue
                    best_in=max(best_in, v)
            rec["in_dom"]=best_in
            # ② 跨捕获：最好的逐对准确率
            ms=np.isin(yS,[a,b]); mt=np.isin(yT,[a,b])
            best_acc=0.0; best_auc=0.5
            if ms.sum()>=40 and mt.sum()>=20:
                y1=(yS[ms]==b).astype(int); y2=(yT[mt]==b).astype(int)
                if len(np.unique(y1))>1 and len(np.unique(y2))>1:
                    for nm in MODELS:
                        try:
                            m=MK(nm); m.fit(XS[ms],y1); q=m.predict_proba(XT[mt])[:,1]
                        except Exception: continue
                        for k in HORIZONS:
                            qq=cmean(q,gT[mt],k)
                            acc=float(((qq>=0.5).astype(int)==y2).mean())
                            if acc>best_acc:
                                best_acc=acc; best_auc=roc_auc_score(y2,qq)
            rec["cross_acc"]=best_acc; rec["cross_auc"]=best_auc
            rec["flag"]= bool(best_in>=IN_DOM_GATE and best_acc<=CROSS_GATE)
            rec["same_coarse"]= model_of(a)==model_of(b)
            rec["same_fine"]=   model_of_fine(a)==model_of_fine(b)
            rows.append(rec)
            if (n+1)%100==0: print(f"  {n+1}/{len(pairs)}  {time.time()-t0:.0f}s",flush=True)

        R=pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/same_device.csv",index=False)
        for gt,name in (("same_coarse","粗（Gosund 全系一族）"),("same_fine","细（Plug/Socket 分开）")):
            tp=int((R.flag & R[gt]).sum()); fp=int((R.flag & ~R[gt]).sum())
            fn=int((~R.flag & R[gt]).sum()); tn=int((~R.flag & ~R[gt]).sum())
            pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
            f1=2*pr*rc/max(pr+rc,1e-12)
            print(f"\n=== 真值 {name}：同型号对 {int(R[gt].sum())}/{len(R)} ===",flush=True)
            print(f"  触发 {int(R.flag.sum())} 对   TP={tp}  FP={fp}  FN={fn}  TN={tn}",flush=True)
            print(f"  精确率 {pr:.3f}   召回率 {rc:.3f}   F1 {f1:.3f}",flush=True)
        print("\n=== 触发但真值判为不同型号（假阳性）===",flush=True)
        fpr=R[R.flag & ~R.same_coarse].sort_values("in_dom",ascending=False)
        print(fpr[["a","b","in_dom","cross_acc","cross_auc"]].head(15).to_string(index=False),flush=True)
        print("\n=== 真值同型号但未触发（漏检）===",flush=True)
        fnr=R[~R.flag & R.same_coarse].sort_values("cross_acc")
        print(fnr[["a","b","in_dom","cross_acc","cross_auc"]].head(15).to_string(index=False),flush=True)
        print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
