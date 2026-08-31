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

print("=== 6. 折内缺类必须回填而非报错/错位 ===")
# 构造：类别 2 只出现在 round 0。GroupKFold 留出 round 0 那一折，其训练集完全没有类别 2,
# fold_model.predict_proba 只返回 2 列，必须映射回全局 3 列、缺失列留 0。
y_mc = y.copy()
y_mc[rounds != 0] = np.where(y_mc[rounds != 0] == 2, 0, y_mc[rounds != 0])
present_by_round = {r: sorted(set(y_mc[rounds == r])) for r in sorted(set(rounds))}
check("构造成立：类别 2 仅存在于部分轮次",
      sum(2 in v for v in present_by_round.values()) < len(present_by_round))
m6 = mk("grouped")
try:
    m6.fit(x, y_mc, train_round=rounds, window_start=ws)
    fit_ok = True
except Exception as e:
    fit_ok = False
    print(f"        raised: {type(e).__name__}: {e}")
check("折内缺类时 fit 不报错", fit_ok)
if fit_ok:
    check("oof_meta_ 列数 = 全局类别数", m6.oof_meta_.shape[1] == len(np.unique(y_mc)))
    check("每行仍恰好一个基模型块、行和 = 1", np.allclose(m6.oof_meta_.sum(axis=1), 1.0))
    # 缺类折的样本：该类列应为 0 而非把别的类的概率错位填进去
    check("无全零行", (m6.oof_meta_.sum(axis=1) > 0).all())
    # 列位正确性：用单折手算对照
    gk_single = list(GroupKFold(n_splits=len(set(rounds))).split(x, y_mc, groups=rounds))
    ok_cols = True
    glob = np.unique(y_mc)
    ref = np.zeros((n, len(glob)))
    for tr, va in gk_single:
        fm = DecisionTreeClassifier(random_state=42).fit(x.iloc[tr], y_mc[tr])
        ci = np.searchsorted(glob, np.asarray(fm.classes_))
        ref[np.ix_(va, ci)] = fm.predict_proba(x.iloc[va])
    m6b = mk("grouped"); m6b.cv = len(set(rounds))
    m6b.fit(x, y_mc, train_round=rounds, window_start=ws)
    check("OOF 列位与手算对照一致", np.allclose(m6b.oof_meta_, ref))

print("=== 7. 协议 §19.2：划分索引持久化 ===")
# `evaluate_task` 必须在任务输出目录写 train_idx.npy / test_idx.npy / split_metadata.json。
# 两个分支都要覆盖：single_round（分层随机划分）与 fixed_split（按轮次划分）。
import argparse, json, tempfile
from pathlib import Path
import robust_iot_research as R

_rng = np.random.default_rng(7)
_labels = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]
_rows = []
for _r in ("R2", "R3"):
    for _lab in _labels:
        for _w in range(24):
            _rows.append({
                "label": _lab, "round": _r, "traffic": "full", "filter_mode": "raw_all",
                "source_file": f"/tmp/{_lab}_{_r}.pcapng", "window_id": _w,
                "window_start": float(_w * 10), "window_end": float(_w * 10 + 10),
                "packet_count": float(_rng.integers(10, 200)),
                "byte_count": float(_rng.integers(1000, 9000)),
                "len_mean": float(_rng.normal(_labels.index(_lab) * 3, 1.0)),
                "len_std": float(abs(_rng.normal(5, 1.0))),
            })
_feat = pd.DataFrame(_rows)

_ns = argparse.Namespace(
    test_size=0.3, random_state=42, n_jobs=1, max_rows=0,
    feature_mode="all", disable_feature_selection=True,
)
_cases = [
    ("single_round_R2", {"name": "single_round_R2", "type": "single_round", "rounds": ["R2"]}, 120),
    ("loro_R2_to_R3", {"name": "loro_R2_to_R3", "type": "fixed_split",
                       "train_rounds": ["R2"], "test_rounds": ["R3"]}, 240),
]
with tempfile.TemporaryDirectory() as _tmp:
    _root = Path(_tmp)
    for _tname, _task, _expected_union in _cases:
        R.evaluate_task(_feat, _task, "raw_all", ["rf"], _ns, {}, _root,
                        _labels, [], [])
        _tdir = _root / "raw_all" / _task["name"]
        _tri, _tei, _meta = _tdir / "train_idx.npy", _tdir / "test_idx.npy", _tdir / "split_metadata.json"
        check(f"[{_tname}] 三个划分文件全部存在",
              _tri.exists() and _tei.exists() and _meta.exists())
        check(f"[{_tname}] 无 error.json（任务正常跑完）",
              not list(_tdir.rglob("error.json")))
        _a, _b = np.load(_tri), np.load(_tei)
        check(f"[{_tname}] 索引可加载且非空", _a.size > 0 and _b.size > 0)
        check(f"[{_tname}] train/test 无交集", len(np.intersect1d(_a, _b)) == 0)
        check(f"[{_tname}] 并集大小 = {_expected_union}",
              len(np.union1d(_a, _b)) == _expected_union)
        # 索引必须真的指向特征表：用它还原的行数与标签分布要对得上
        check(f"[{_tname}] 索引落在特征表行索引内",
              bool(pd.Index(np.concatenate([_a, _b])).isin(_feat.index).all()))
        _md = json.loads(_meta.read_text(encoding="utf-8"))
        check(f"[{_tname}] metadata 记录任务定义", _md["task"] == _task)
        check(f"[{_tname}] metadata 记录种子 = 42", _md["split"]["random_state"] == 42)
        check(f"[{_tname}] metadata 记录划分方式", bool(_md["split"]["method"]))
        check(f"[{_tname}] metadata 计数与索引一致",
              _md["counts"]["train"] == len(_a) and _md["counts"]["test"] == len(_b)
              and _md["counts"]["overlap"] == 0)
        _restored = _feat.loc[_a]
        check(f"[{_tname}] features.loc[train_idx] 可精确还原训练集",
              len(_restored) == len(_a)
              and _restored["label"].value_counts().to_dict() == _md["label_counts"]["train"])

    # environment report 的 §19.2 字段
    R.save_environment_report(_root, ["rf"], _ns)
    _env = json.loads((_root / "environment_report.json").read_text(encoding="utf-8"))
    check("environment_report 含 git 段", "git" in _env and "head" in _env["git"]["repo_root"])
    check("environment_report 含完整命令行 argv",
          _env["command_line"]["argv"] == list(sys.argv))
    check("environment_report 含关键包版本",
          all(_env["versions"].get(k) for k in ("python", "numpy", "pandas", "scikit-learn")))
    check("environment_report 含种子", _env["random_state"] == 42)
    check("environment_report 保留原有字段（默认行为不变）",
          _env["requested_models"] == ["rf"] and "available_optional_modules" in _env)

print("\n" + ("全部通过" if ok else "存在失败项"))
sys.exit(0 if ok else 1)
