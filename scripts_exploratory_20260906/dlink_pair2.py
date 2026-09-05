"""【探索性,非协议】D-Link 逐对可分性 —— 修正度量。

**修正的错误**：上一版把"对内准确率"算成"8 类预测在这两类窗上的准确率"，
那是被整体 8 类失败主导的量，不回答"这两类之间能不能分开"。
（同一类错误今天第三次：pair_auc 用 LR、闸门盲点看错模型 —— 都是量的对象与问题不一致。）

**正确度量**（两个都报，答不同问题）：
  pair_argmax   在真类 ∈{i,j} 的窗上，把概率限制到这两类取 argmax → 该对本身可不可分
  pair_binary   直接训一个 i-vs-j 二分类器（源域），在目标域该对的窗上评 → 上界
两者都用【跨天】划分。分组按"是否同型号"。
"""
from __future__ import annotations
import sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

REPO = "/home/lmy/iot-device-classification"
sys.path.insert(0, REPO + "/code/scripts/analysis/unsw_pilot")
import pilot_rf_loro as P

FEAT = Path(REPO + "/results/dlink_cams")
NJ, SEED = 12, 42
def TYPE(d): return "DayCam" if d.startswith("D-LinkDayCam") else "Cam"

def main():
    t0 = time.time()
    f = sorted(FEAT.glob("features_dlink_w10_*.csv"))[-1]
    d = pd.read_csv(f, low_memory=False)
    cols = P.feature_columns(d)
    days = sorted(d.day.unique()); k = len(days) // 2
    tr, te = days[:k], days[k:]
    s = d[d.day.isin(tr)]; t = d[d.day.isin(te)]
    devs = sorted(set(s.device) & set(t.device)); le = LabelEncoder().fit(devs)
    Xs = np.asarray(P.clean_x(s, cols), dtype=float); ys = le.transform(s.device)
    Xt = np.asarray(P.clean_x(t, cols), dtype=float); yt = le.transform(t.device)
    print(f"{len(devs)} 台  训练 {len(ys)}  测试 {len(yt)}   {time.time()-t0:.0f}s", flush=True)

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                class_weight="balanced", n_jobs=NJ)
    rf.fit(Xs, ys); Q = rf.predict_proba(Xt)
    print(f"8 类基模型完成  macro={f1_score(yt, Q.argmax(1), average='macro'):.4f}"
          f"   {time.time()-t0:.0f}s", flush=True)

    rows = []
    for i, j in itertools.combinations(range(len(devs)), 2):
        pm = np.isin(yt, [i, j])
        if pm.sum() < 100: continue
        # ① 概率限制到两类取 argmax
        sub = Q[pm][:, [i, j]]
        pred = np.where(sub[:, 1] > sub[:, 0], j, i)
        a_arg = float((pred == yt[pm]).mean())
        # ② 专门训一个二分类器（上界）
        ms = np.isin(ys, [i, j])
        b = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                   class_weight="balanced", n_jobs=NJ)
        b.fit(Xs[ms], (ys[ms] == j).astype(int))
        pb = b.predict(Xt[pm])
        a_bin = float((np.where(pb == 1, j, i) == yt[pm]).mean())
        rows.append({"a": devs[i], "b": devs[j], "同型号": TYPE(devs[i]) == TYPE(devs[j]),
                     "n": int(pm.sum()), "pair_argmax": a_arg, "pair_binary": a_bin})
        print(f"  {devs[i][-10:]:>10s}|{devs[j][-10:]:<10s} 同型号={TYPE(devs[i])==TYPE(devs[j])!s:5s} "
              f"n={pm.sum():7d}  argmax={a_arg:.4f}  专用二分类={a_bin:.4f}", flush=True)

    R = pd.DataFrame(rows); R.to_csv("/home/lmy/cic_probe/dlink_pair2.csv", index=False)
    print(f"\n{'='*80}\n=== 同型号 vs 跨型号（修正后的度量）===", flush=True)
    print(R.groupby("同型号")[["pair_argmax", "pair_binary"]].agg(
        ["count", "mean", "min", "max"]).round(4).to_string(), flush=True)
    print("\n=== 最难的 8 对（按专用二分类器）===", flush=True)
    print(R.nsmallest(8, "pair_binary").to_string(index=False), flush=True)
    print("\n判读：专用二分类器仍低 → 该对在这套特征下确实不可分（身份型）；", flush=True)
    print("      argmax 低而专用二分类器高 → 是多类竞争问题，逐对机制正好治它。", flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
