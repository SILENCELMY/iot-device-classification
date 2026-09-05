"""【探索性,非协议】D-Link 摄像头：8 台实例级跨天，同型号组是身份型的最硬测试。

为什么这个数据集值得跑：
  6 台【同型号同批次】DCS-930L（DayCam4/5/6 的 MAC 连号 b0:c5:54:42:8f:xx）
  + 2 台同型号 DCS-936L（Cam1/Cam2 连号 b2:c5:54:44:0f:xx）
  → 同型号对共 16 个（DayCam C(6,2)=15 + Cam 1），比 CIC 的 Gosund 组更大更纯
  → 51 天共同覆盖，跨天泛化的样本远超 UNSW 的 16 天

两个任务：
  实例级 8 类   身份型的直接测试（同硬件同固件能不能分开）
  型号级 2 类   Cam vs DayCam，对照用，预期很容易

判据（按 confusion-matrix-is-the-instrument 的口径，不看全体 macro）：
  逐对【对内准确率】—— 该对自己的窗上二选一，0.5 = 随机
  同型号对 vs 跨型号对 分组报，看难度是否跟着"是否同型号"走
"""
from __future__ import annotations
import sys, time, itertools
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix

REPO = "/home/lmy/iot-device-classification"
sys.path.insert(0, REPO + "/code/scripts/analysis/unsw_pilot")
import pilot_rf_loro as P

FEAT = Path(REPO + "/results/dlink_cams")
NJ, SEED = 12, 42
DAYCAM = [f"D-LinkDayCam{i}" for i in range(1, 7)]
CAM = ["D-LinkCam1", "D-LinkCam2"]
def TYPE(d): return "DayCam" if d.startswith("D-LinkDayCam") else "Cam"

def main():
    f = sorted(FEAT.glob("features_dlink_w10_*.csv"))
    if not f: sys.exit("没有特征表")
    d = pd.read_csv(f[-1], low_memory=False)
    print(f"读入 {f[-1].name}   {len(d)} 行", flush=True)
    cols = P.feature_columns(d)
    days = sorted(d.day.unique())
    print(f"{len(cols)} 特征列   {len(days)} 天：{days[0]} .. {days[-1]}", flush=True)
    print("\n=== 逐设备逐天窗口数 ===", flush=True)
    piv = d.groupby(["device", "day"]).size().unstack(fill_value=0)
    print(piv.to_string(), flush=True)

    # 跨天划分：前一半训练，后一半测试
    k = len(days) // 2
    tr_days, te_days = days[:k], days[k:]
    print(f"\n训练日 {tr_days}\n测试日 {te_days}", flush=True)

    for name, lab in [("实例级 8 类", lambda x: x), ("型号级 2 类", TYPE)]:
        s = d[d.day.isin(tr_days)]; t = d[d.day.isin(te_days)]
        ys_raw = [lab(x) for x in s.device]; yt_raw = [lab(x) for x in t.device]
        cls = sorted(set(ys_raw) & set(yt_raw)); le = LabelEncoder().fit(cls)
        ms = np.isin(ys_raw, cls); mt = np.isin(yt_raw, cls)
        Xs = np.asarray(P.clean_x(s[ms], cols), dtype=float)
        Xt = np.asarray(P.clean_x(t[mt], cols), dtype=float)
        ys = le.transform(np.array(ys_raw)[ms]); yt = le.transform(np.array(yt_raw)[mt])
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                    class_weight="balanced", n_jobs=NJ)
        rf.fit(Xs, ys); p = rf.predict(Xt)
        mac = f1_score(yt, p, average="macro", labels=np.arange(len(cls)))
        print(f"\n{'='*80}\n{name}：{len(cls)} 类  训练 {len(ys)}  测试 {len(yt)}  "
              f"跨天 macro-F1 = {mac:.4f}", flush=True)
        if len(cls) > 2:
            C = confusion_matrix(yt, p, labels=np.arange(len(cls)))
            off = C.copy(); np.fill_diagonal(off, 0); tot = off.sum()
            rows = []
            for i, j in itertools.combinations(range(len(cls)), 2):
                pm = np.isin(yt, [i, j])
                acc = float((p[pm] == yt[pm]).mean()) if pm.sum() else np.nan
                same = TYPE(cls[i]) == TYPE(cls[j])
                rows.append({"a": cls[i], "b": cls[j], "同型号": same,
                             "错误": int(off[i, j] + off[j, i]),
                             "对内准确率": acc, "n": int(pm.sum())})
            R = pd.DataFrame(rows).sort_values("对内准确率")
            print(f"\n  错误总数 {tot}，前 6 对占 "
                  f"{R.nlargest(6,'错误')['错误'].sum()/max(tot,1):.1%}", flush=True)
            print("\n  === 最难的 8 对（按对内准确率）===", flush=True)
            print("  " + R.head(8).to_string(index=False).replace("\n", "\n  "), flush=True)
            print("\n  === 同型号 vs 跨型号 ===", flush=True)
            print("  " + R.groupby("同型号")["对内准确率"].describe()[
                ["count", "mean", "min", "max"]].round(4).to_string().replace("\n", "\n  "),
                flush=True)
            R.to_csv("/home/lmy/cic_probe/dlink_pairs.csv", index=False)
    print(f"\n判读：若同型号对的对内准确率显著低于跨型号对 → 身份型困难在第四个数据集上复现；",
          flush=True)
    print("      且 6 台同型号 DCS-930L 是我们手上最大最纯的同硬件同固件组。", flush=True)

if __name__ == "__main__":
    main()
