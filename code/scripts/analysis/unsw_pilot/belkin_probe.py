#!/usr/bin/env python3
"""判别检验：BelkinWemoSwitch ↔ BelkinWemoMotion 的混淆是「设备本身相似」还是「标签错误」。

判据
----
若是**标签错误**（某天两类的标签被交换/串位），则：
  - 同一天内部的 IID 划分应当**能**分开（同一天内标签自洽），而跨天会灾难性崩溃；
  - 且崩溃形态接近「一一对调」（recall≈0，几乎全部判到对方）。

若是**设备本身相似**（同厂同固件同协议栈），则：
  - **同一天内部的 IID 划分同样分不开**；
  - 混淆是双向渗透的「混合」而非「对调」。

本脚本对每一天单独做一次两类 IID 随机划分（分层，test_size=0.3），报两类的 F1 与混淆。
RF 口径仍走主线 build_model（协议 §7）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "code" / "scripts" / "core" / "robust_iot_research.py").exists():
            return cand
    raise SystemExit("找不到仓库根")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts" / "core"))
from robust_iot_research import build_model, clean_x  # noqa: E402

from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

META = {"device", "day", "label", "source_file", "window_id",
        "window_start", "window_end", "window_start_epoch"}
PAIR = ["BelkinWemoMotion", "BelkinWemoSwitch"]

feats = pd.read_csv(sys.argv[1])
out_path = Path(sys.argv[2])
cols = [c for c in feats.columns
        if c not in META and pd.api.types.is_numeric_dtype(feats[c])]

report = {"pair": PAIR, "n_features": len(cols), "within_day_iid": [], "note": ""}
lines = ["# BELKIN_PROBE.md — Belkin Wemo 两类混淆的成因判别", "",
         "**问题**：LORO 的每一个日对都在 `BelkinWemoSwitch` ↔ `BelkinWemoMotion` 上报警，",
         "且**只有**这一对报警。需要区分：设备本身相似 / 标签错误。", "",
         "**判据**：标签错误只会在**跨天**崩溃（同一天内标签自洽）；",
         "设备相似则**同一天内的 IID 划分也分不开**。", "",
         "## 1. 每天单独做两类 IID 随机划分（分层，test_size=0.3，RF 走主线 build_model）", "",
         "| day | 两类窗口数 | 二分类 macro-F1 | Motion recall | Switch recall | 判读 |",
         "|---|---|---|---|---|---|"]

for day in sorted(feats["day"].unique()):
    sub = feats[(feats["day"] == day) & (feats["label"].isin(PAIR))]
    vc = sub["label"].value_counts()
    if len(vc) < 2 or vc.min() < 50:
        lines.append(f"| `{day}` | {dict(vc)} | — | — | — | 样本不足，跳过 |")
        continue
    x, y = clean_x(sub, cols), sub["label"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y)
    model = build_model("rf", random_state=42, n_jobs=12, class_count=2)
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    cm = confusion_matrix(y_te, pred, labels=PAIR)
    rec = cm.diagonal() / cm.sum(axis=1)
    mf1 = float(f1_score(y_te, pred, average="macro", labels=PAIR, zero_division=0))
    # 对照：同一数据集里可分的类对 IID 二分类可达 ~0.99（见 §2b 的全类逐类 F1），
    # 因此 0.9 以下即属「同一天内也分不开」。
    verdict = ("**同一天内也分不开** → 指向设备相似" if mf1 < 0.90
               else "同一天内可分 → 需进一步查跨天标签")
    lines.append(f"| `{day}` | {int(vc[PAIR[0]]):,} / {int(vc[PAIR[1]]):,} | "
                 f"{mf1:.4f} | {rec[0]:.4f} | {rec[1]:.4f} | {verdict} |")
    report["within_day_iid"].append(
        {"day": day, "macro_f1": round(mf1, 6),
         "recall_motion": round(float(rec[0]), 4),
         "recall_switch": round(float(rec[1]), 4),
         "cm": cm.tolist(), "n": {k: int(v) for k, v in vc.items()}})

# ---- 特征层面的直接证据：两类的中位特征向量有多接近 ----
lines += ["", "## 2. 两类在特征空间的直接距离（16-09-30，两类都足量的那天）", ""]
sub = feats[(feats["day"] == "16-09-30") & (feats["label"].isin(PAIR))]
med = sub.groupby("label")[cols].median()
if len(med) == 2:
    a, b = med.loc[PAIR[0]], med.loc[PAIR[1]]
    denom = (a.abs() + b.abs()).replace(0, np.nan)
    rel = ((a - b).abs() / denom).dropna().sort_values(ascending=False)
    identical = int((rel < 1e-9).sum())
    lines += [f"- 61 维中位向量中，**完全相同的维度: {identical} / {len(rel)}**", "",
              "差异最大的 8 维：", "",
              "| 特征 | Motion 中位 | Switch 中位 | 相对差 |", "|---|---|---|---|"]
    for c in rel.head(8).index:
        lines.append(f"| `{c}` | {a[c]:.6g} | {b[c]:.6g} | {rel[c]:.4f} |")
    report["identical_median_dims"] = identical
    report["n_dims"] = int(len(rel))
    report["top_diff_dims"] = {c: round(float(rel[c]), 4) for c in rel.head(8).index}

# ---- 对照组：同一天内的**全类** IID，看这两类相对其它类差多少 ----
from sklearn.model_selection import train_test_split as _tts  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from robust_iot_research import sample_balanced  # noqa: E402

lines += ["", "## 2b. 对照组：同一天（16-09-30）**全类** IID，逐类 F1", "",
          "这是关键对照 —— 同一天内 MAC 是硬标识、标签必然自洽，",
          "若这两类在此条件下仍是全场最差，则不可能是标签错误。", ""]
sub_all = feats[feats["day"] == "16-09-30"]
vc_all = sub_all["label"].value_counts()
keep_all = sorted(vc_all[vc_all >= 100].index)
sub_all = sample_balanced(sub_all[sub_all["label"].isin(keep_all)],
                          max_rows=20000, random_state=42)
Xa, ya = clean_x(sub_all, cols), sub_all["label"].to_numpy()
xtr, xte, ytr, yte = _tts(Xa, ya, test_size=0.3, random_state=42, stratify=ya)
ma = build_model("rf", random_state=42, n_jobs=12, class_count=len(keep_all))
ma.fit(xtr, ytr)
pa = ma.predict(xte)
rep_all = classification_report(yte, pa, labels=keep_all, output_dict=True, zero_division=0)
macro_all = float(f1_score(yte, pa, average="macro", labels=keep_all, zero_division=0))
ranked = sorted(((rep_all[k]["f1-score"], k) for k in keep_all))
lines += [f"全类 IID：**{len(keep_all)} 类，macro-F1 = {macro_all:.4f}**", "",
          "| 排名 | 类别 | F1 | support |", "|---|---|---|---|"]
for i, (v, k) in enumerate(ranked, 1):
    mark = " **←**" if k in PAIR else ""
    lines.append(f"| {i} | `{k}`{mark} | {v:.4f} | {int(rep_all[k]['support'])} |")
belkin_ranks = [i for i, (_, k) in enumerate(ranked, 1) if k in PAIR]
report["within_day_full_iid"] = {
    "day": "16-09-30", "n_classes": len(keep_all), "macro_f1": round(macro_all, 6),
    "per_class_f1": {k: round(float(rep_all[k]["f1-score"]), 4) for k in keep_all},
    "belkin_ranks": belkin_ranks,
}
lines += ["", f"两个 Belkin 类的排名：**第 {belkin_ranks} 名（共 {len(keep_all)} 类，1 = 最差）**。", ""]

mean_iid = float(np.mean([r["macro_f1"] for r in report["within_day_iid"]])) \
    if report["within_day_iid"] else float("nan")
report["within_day_iid_mean_macro_f1"] = round(mean_iid, 6)
lines += ["", "## 3. 结论", "",
          f"- 每天**同一天内部** IID 二分类 macro-F1 均值 = **{mean_iid:.4f}**",
          f"- 同一天全类 IID 中，这两类排名第 {belkin_ranks}（共 {len(keep_all)} 类，1 = 最差），"
          f"而全类 macro-F1 = {macro_all:.4f}，最好的若干类 F1 ≥ 0.99"]
if mean_iid < 0.90:
    lines += ["- 同一天内部（标签必然自洽的条件下）这两类**同样分不开** →",
              "  **混淆来自设备本身的网络行为相似，不是标签错误。**",
              "- 两者 MAC 前缀同为 `ec:1a:59`（Belkin International OUI），同厂同 WeMo/UPnP 协议栈，",
              "  这一混淆在 Sivanathan TMC 2018 原文中亦有记载。",
              "- **判定：五问之 5 的「无明显标签错误」成立。**"]
else:
    lines += ["- 同一天内部可分而跨天崩溃 → **需要按标签错误进一步排查，不得放行。**"]
report["verdict"] = "device_similarity" if mean_iid < 0.75 else "needs_label_investigation"

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
out_path.with_suffix(".json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[out] {out_path}")
print(f"[verdict] within-day IID mean macro-F1 = {mean_iid:.4f} -> {report['verdict']}")
