"""【探索性,非协议】X 与 Y 到底差在哪？

D-Link 6 台同型号 DCS-930L 的混淆结构分裂成两个互不可分簇：
    X = {DayCam1, DayCam2, DayCam4, DayCam5}   组内 6 对全部 ≈0.50（精确随机）
    Y = {DayCam3, DayCam6}                      组内 1 对 = 0.507
    X 对 Y  8 对全部 ≈0.77
而这个划分【不跟批次、型号、MAC 段走】—— DayCam2(3d:3e)∈X 与 DayCam3(3d:3f)∈Y 相邻；
DayCam4/5(42:8f)∈X 与 DayCam6(42:8f)∈Y 同批。数据集里没有标注这个分组。

本脚本查特征层：训一个 X-vs-Y 二分类器，看哪些特征承担区分，并逐组报均值。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

REPO = "/home/lmy/iot-device-classification"
sys.path.insert(0, REPO + "/code/scripts/analysis/unsw_pilot")
import pilot_rf_loro as P

X = ["D-LinkDayCam1", "D-LinkDayCam2", "D-LinkDayCam4", "D-LinkDayCam5"]
Y = ["D-LinkDayCam3", "D-LinkDayCam6"]

d = pd.read_csv(sorted(Path(REPO + "/results/dlink_cams").glob("features_dlink_w10_*.csv"))[-1],
                low_memory=False)
d = d[d.device.isin(X + Y)]
cols = P.feature_columns(d)
days = sorted(d.day.unique()); k = len(days) // 2
s = d[d.day.isin(days[:k])]; t = d[d.day.isin(days[k:])]
Xs = np.asarray(P.clean_x(s, cols), dtype=float); ys = s.device.isin(Y).astype(int).to_numpy()
Xt = np.asarray(P.clean_x(t, cols), dtype=float); yt = t.device.isin(Y).astype(int).to_numpy()
print(f"训练 {len(ys)} (Y占{ys.mean():.2f})  测试 {len(yt)}", flush=True)

rf = RandomForestClassifier(n_estimators=300, random_state=42,
                            class_weight="balanced", n_jobs=12)
rf.fit(Xs, ys)
p = rf.predict(Xt)
print(f"\nX vs Y 跨天二分类  准确率={float((p==yt).mean()):.4f}  "
      f"F1={f1_score(yt,p):.4f}", flush=True)

imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
print("\n=== 承担区分的前 15 个特征 ===", flush=True)
gx = d[d.device.isin(X)]; gy = d[d.device.isin(Y)]
rows = []
for c in imp.head(15).index:
    a, b = gx[c].median(), gy[c].median()
    rows.append({"特征": c, "重要性": round(imp[c], 4),
                 "X中位": round(float(a), 3), "Y中位": round(float(b), 3),
                 "比值": round(float(b / a), 3) if a not in (0,) and not pd.isna(a) else None})
print(pd.DataFrame(rows).to_string(index=False), flush=True)

print("\n=== 逐设备关键量中位数 ===", flush=True)
key = ["packet_count", "byte_count"] + [c for c in imp.head(6).index if c not in ("packet_count","byte_count")]
print(d.groupby("device")[key].median().round(3).to_string(), flush=True)
