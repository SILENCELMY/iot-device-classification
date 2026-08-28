import sys; sys.path.insert(0, 'code/scripts/core')
import numpy as np, pandas as pd
from robust_iot_research import SimpleStackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

rng = np.random.default_rng(0)
n = 300
x = pd.DataFrame(rng.normal(size=(n,4)), columns=[f"f{i}" for i in range(4)])
y = rng.integers(0,3,n); rounds = rng.integers(0,5,n); ws = rng.uniform(0,1000,n)
def mk(mode):
    return SimpleStackingClassifier(
        estimators=[("dt", DecisionTreeClassifier(random_state=42))],
        final_estimator=LogisticRegression(max_iter=500),
        cv=5, random_state=42, oof_mode=mode)
ok = True
def check(label, cond):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + label); ok = ok and cond

print("=== 1. 多轮 grouped 必须等价于 GroupKFold(group=round) ===")
m = mk("grouped"); folds = list(m._splitter(x, y, rounds, ws))
gk = list(GroupKFold(n_splits=5).split(x, y, groups=rounds))
check("折数 == GroupKFold 折数 (无时间块污染)", len(folds) == len(gk))
check("每折索引与 GroupKFold 逐折一致",
      all(np.array_equal(a[1], b[1]) for a, b in zip(folds, gk)))
# 关键:同一 round 的样本绝不跨 train/val 同时出现
leak = any(set(rounds[tr]) & set(rounds[va]) for tr, va in folds)
check("无轮次泄漏 (train/val 的 round 集合不相交)", not leak)

print("=== 2. 单轮 grouped 必须是时间块,且不打散相邻窗口 ===")
one = np.zeros(n, dtype=int)
folds1 = list(mk("grouped")._splitter(x, y, one, ws))
check("折数 <= cv", len(folds1) <= 5)
# 每折的 val 应是 window_start 上的连续区间
contig = True
for tr, va in folds1:
    lo, hi = ws[va].min(), ws[va].max()
    # 训练集里不应有落在该区间内部的样本
    inside = ((ws[tr] > lo) & (ws[tr] < hi)).sum()
    if inside > 0: contig = False
check("每折 val 是 window_start 上的连续时间区间", contig)
check("覆盖全部样本", len(np.unique(np.concatenate([v for _, v in folds1]))) == n)

print("=== 3. random 臂必须逐位复现历史 ===")
m3 = mk("random"); m3.fit(x, y)
sp = StratifiedKFold(n_splits=max(2,min(5,min(np.bincount(y)))), shuffle=True, random_state=42)
o = np.zeros((n,3))
for tr, va in sp.split(x, y):
    o[va] = DecisionTreeClassifier(random_state=42).fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[va])
check("random 臂 OOF 与历史实现逐位一致", np.allclose(m3.oof_meta_, o))

print("=== 4. OOF 概率合法性 ===")
m.fit(x, y, train_round=rounds, window_start=ws)
check("oof_meta_ 每基模型块行和 = 1", np.allclose(m.oof_meta_.sum(axis=1), 1.0))
check("oof_meta_ 无全零行 (每样本都被预测过)", (m.oof_meta_.sum(axis=1) > 0).all())

print("=== 5. 防御 ===")
try: mk("bogus"); check("非法 oof_mode 报错", False)
except ValueError: check("非法 oof_mode 报错", True)
try: mk("grouped").fit(x, y); check("缺分组信息报错", False)
except ValueError: check("缺分组信息报错", True)

print("\n" + ("全部通过" if ok else "存在失败项"))
sys.exit(0 if ok else 1)
