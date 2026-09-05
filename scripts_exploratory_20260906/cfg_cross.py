"""【探索性,非协议】逐类对配置导出的正确形态：用【inner 跨环境】而非【源域内部】选模型。

设计（与方法的 |S|>=2 硬要求同构）：
  inner  : 在 A→B 这个有标签的环境对上，逐类对挑出"能扛跨环境"的模型
  outer  : 在完全held-out的 A→C 上应用该配置，只看 macro-F1
对照臂：
  base          现状（单一 xgboost 多分类，无覆盖）
  cfg_srcint    源域内时间块选模型（cfg_e2e 那一版，预测会失效）
  cfg_inner     inner 跨环境选模型（本版）
  cfg_inner_k5  同上 + 观测时长 k=5 聚合（仅对被选中的类对）
"""
from __future__ import annotations
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

REPO = "/home/lmy/iot-device-classification"
sys.path.insert(0, REPO + "/code/scripts/analysis/unsw_pilot")
sys.path.insert(0, REPO + "/results/unsw_iid_reference_20260902")
sys.path.insert(0, REPO + "/results/two_channel_20260903")
import pilot_rf_loro as P            # noqa: E402
import run_unsw_iid_reference as IID # noqa: E402
import run_two_channel as TC         # noqa: E402

UNSW = REPO + "/results/unsw_features_full/features_day_%s.csv"
CANDS = ["lr", "rf", "xgboost"]
GATE = 0.95          # inner 跨环境 AUC 达不到就不动手
SEEDS = (42, 43)


def MK(nm):
    if nm == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return TC.make_model(nm, 2)


def frame(path_or_df):
    return path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, low_memory=False)


def pair_scores(Xa, ya01, Xb, yb01, nm):
    m = MK(nm); m.fit(Xa, ya01)
    return roc_auc_score(yb01, m.predict_proba(Xb)[:, 1])


def derive_inner(dfA, dayA, dfB, dayB, devs, cols, le, seed):
    """在 A→B 上逐类对挑模型。只对 inner 里真实出现过的 top-2 类对导出（不碰 outer）。
    返回 {(i,j): (model_name, inner_auc)}。"""
    TC.SEED = seed
    a = P.sample_balanced(dfA[dfA.label.isin(devs)], max_rows=IID.MAX_ROWS, random_state=seed)
    b = P.sample_balanced(dfB[dfB.label.isin(devs)], max_rows=IID.MAX_ROWS, random_state=seed)
    Xa = np.asarray(P.clean_x(a, cols), dtype=float); ya = le.transform(a.label)
    Xb = np.asarray(P.clean_x(b, cols), dtype=float); yb = le.transform(b.label)
    bm = TC.make_model("xgboost", len(devs)); bm.fit(Xa, ya)
    oo = np.argsort(-bm.predict_proba(Xb), axis=1)
    cand = sorted(set(map(tuple, np.sort(np.c_[oo[:, 0], oo[:, 1]], axis=1))))
    print(f"    inner 候选类对 {len(cand)}（全组合 {len(devs)*(len(devs)-1)//2}）", flush=True)
    cfg = {}
    for (i, j) in cand:
            ma = np.isin(ya, [i, j]); mb = np.isin(yb, [i, j])
            if len(np.unique(ya[ma])) < 2 or len(np.unique(yb[mb])) < 2:
                continue
            ya01 = (ya[ma] == j).astype(int); yb01 = (yb[mb] == j).astype(int)
            best = (None, -1.0)
            for nm in CANDS:
                try:
                    v = pair_scores(Xa[ma], ya01, Xb[mb], yb01, nm)
                except Exception:
                    continue
                if v > best[1]:
                    best = (nm, v)
            if best[0] is not None:
                cfg[(i, j)] = best
    return cfg


def derive_srcint(dfA, dayA, devs, cols, le, seed):
    """对照：只用源域内时间块划分挑模型。"""
    TC.SEED = seed
    d0 = dfA[dfA.label.isin(devs)].sort_values("window_start_epoch")
    blk = TC.time_blocks(np.asarray(d0["window_start_epoch"]))
    X0 = np.asarray(P.clean_x(d0, cols), dtype=float); y0 = le.transform(d0.label)
    tr, te = blk < 4, blk == 4
    bm = TC.make_model("xgboost", len(devs)); bm.fit(X0[tr], y0[tr])
    oo = np.argsort(-bm.predict_proba(X0[te]), axis=1)
    cand = sorted(set(map(tuple, np.sort(np.c_[oo[:, 0], oo[:, 1]], axis=1))))
    cfg = {}
    for (i, j) in cand:
            mi = np.isin(y0, [i, j])
            yi = (y0[mi & tr] == j).astype(int); yj = (y0[mi & te] == j).astype(int)
            if len(np.unique(yi)) < 2 or len(np.unique(yj)) < 2:
                continue
            best = (None, -1.0)
            for nm in CANDS:
                try:
                    v = pair_scores(X0[mi & tr], yi, X0[mi & te], yj, nm)
                except Exception:
                    continue
                if v > best[1]:
                    best = (nm, v)
            if best[0] is not None:
                cfg[(i, j)] = best
    return cfg


def agg_pred(p, k):
    """把连续 k 个窗的概率取滑动均值（因果：只用当前及之前的窗）。"""
    if k <= 1:
        return p
    c = np.cumsum(np.insert(p, 0, 0.0))
    out = np.empty_like(p)
    for n in range(len(p)):
        lo = max(0, n - k + 1)
        out[n] = (c[n + 1] - c[lo]) / (n + 1 - lo)
    return out


def evaluate(dfS, dayS, dfT, dayT, devs, cols, le, seed, cfgs, ks=(1, 5)):
    TC.SEED = seed
    s = P.sample_balanced(dfS[dfS.label.isin(devs)], max_rows=IID.MAX_ROWS, random_state=seed)
    t = dfT[dfT.label.isin(devs)].sort_values(["label", "window_start_epoch"])
    t = P.sample_balanced(t, max_rows=IID.MAX_ROWS, random_state=seed)
    t = t.sort_values(["label", "window_start_epoch"])
    Xs = np.asarray(P.clean_x(s, cols), dtype=float); ys = le.transform(s.label)
    Xt = np.asarray(P.clean_x(t, cols), dtype=float); yt = le.transform(t.label)
    base = TC.make_model("xgboost", len(devs)); base.fit(Xs, ys)
    pp = base.predict_proba(Xt); o = np.argsort(-pp, axis=1)
    top1, top2 = o[:, 0], o[:, 1]
    res = {"base": f1_score(yt, top1, average="macro"), "base_acc": (top1 == yt).mean()}
    for cname, cfg in cfgs.items():
        for k in ks:
            pred = top1.copy(); nov = 0; used = {}
            for (i, j), (nm, auc) in cfg.items():
                if auc < GATE:
                    continue
                ms = np.isin(ys, [i, j])
                if len(np.unique(ys[ms])) < 2:
                    continue
                mm = ((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i))
                if not mm.any():
                    continue
                pm = MK(nm); pm.fit(Xs[ms], (ys[ms] == j).astype(int))
                q = pm.predict_proba(Xt[mm])[:, 1]
                if k > 1:
                    q = agg_pred(q, k)          # 目标域时间已排序，逐类对内近似连续
                pred[mm] = np.where(q >= 0.5, j, i); nov += int(mm.sum())
                used[nm] = used.get(nm, 0) + 1
            key = f"{cname}_k{k}"
            res[key] = f1_score(yt, pred, average="macro")
            res[key + "_acc"] = (pred == yt).mean()
            res[key + "_n"] = nov
            res[key + "_models"] = json.dumps(used, ensure_ascii=False)
    return res


def task(tag, A, dayA, B, dayB, C, dayC):
    dfA, dfB, dfC = frame(A), frame(B), frame(C)
    cols = P.feature_columns(dfA)
    devs = sorted(set(IID.day_gate(dfA, dayA)) & set(IID.day_gate(dfB, dayB)) & set(IID.day_gate(dfC, dayC)))
    le = LabelEncoder().fit(devs)
    print(f"\n{'='*92}\n{tag}\n  inner: {dayA} → {dayB}   outer: {dayA} → {dayC}   "
          f"{len(devs)} 类  {len(cols)} 列", flush=True)
    rows = []
    for seed in SEEDS:
        c_in = derive_inner(dfA, dayA, dfB, dayB, devs, cols, le, seed)
        c_si = derive_srcint(dfA, dayA, devs, cols, le, seed)
        r = evaluate(dfA, dayA, dfC, dayC, devs, cols, le, seed,
                     {"cfg_inner": c_in, "cfg_srcint": c_si})
        r["seed"] = seed; rows.append(r)
        print(f"  seed{seed}  base={r['base']:.4f}"
              f"   srcint_k1={r['cfg_srcint_k1']:.4f} ({r['cfg_srcint_k1']-r['base']:+.4f})"
              f"   inner_k1={r['cfg_inner_k1']:.4f} ({r['cfg_inner_k1']-r['base']:+.4f})"
              f"   inner_k5={r['cfg_inner_k5']:.4f} ({r['cfg_inner_k5']-r['base']:+.4f})", flush=True)
        if seed == SEEDS[0]:
            print(f"    inner 选中模型: {r['cfg_inner_k1_models']}   动手 {r['cfg_inner_k1_n']} 窗", flush=True)
            print(f"    srcint 选中模型: {r['cfg_srcint_k1_models']}  动手 {r['cfg_srcint_k1_n']} 窗", flush=True)
    R = pd.DataFrame(rows)
    for c in ("cfg_srcint_k1", "cfg_inner_k1", "cfg_inner_k5"):
        print(f"  → {c:14s} 均值 Δmacro = {(R[c]-R['base']).mean():+.4f}", flush=True)
    return R


if __name__ == "__main__":
    with threadpool_limits(1):
        t0 = time.time(); out = []
        out.append(task("UNSW", UNSW % "16-09-23", "16-09-23", UNSW % "16-09-30", "16-09-30",
                        UNSW % "16-10-12", "16-10-12"))
        out.append(task("CIC（inner 1102，outer 1108）",
                        "/home/lmy/cic_probe/idle_1102.csv", "2021_11_02_Idle",
                        "/home/lmy/cic_probe/active_1102.csv", "2021_11_02_Active",
                        "/home/lmy/cic_probe/active_1108.csv", "2021_11_08_Active"))
        pd.concat(out).to_csv("/home/lmy/cic_probe/cfg_cross.csv", index=False)
        print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
