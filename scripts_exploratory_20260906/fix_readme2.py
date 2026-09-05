# -*- coding: utf-8 -*-
"""保留 1（flat 可比性）已由 verify_flat.py 实测销掉，更新 README。"""
import pathlib, sys

P = pathlib.Path("/home/lmy/iot-device-classification/results/"
                 "config_derivation_exploratory_20260905/README.md")
t = P.read_text(encoding="utf-8")

OLD = """1. **`flat 0.7854` 与 `flat_183 = 0.8072` 差 0.0218**，归因于"单模型 vs `best_base` 三模型"，
   **尚未实测**（`verify_flat.py` 待跑）。在此之前"可比"只是推断。"""

NEW = """1. ~~**`flat 0.7854` 与 `flat_183 = 0.8072` 差 0.0218** 归因于"单模型 vs `best_base` 三模型"，
   尚未实测~~ —— **已由 `verify_flat.py` 实测销掉**（同池同划分同 5 seed，逐模型重跑）：

```
              lightgbm      rf   xgboost   best_base
jit_R6          0.8232  0.7645    0.8122      0.8232
jit_R7          0.8686  0.8513    0.8676      0.8694
pos_R5          0.6800  0.7276    0.6716      0.7276
均值            0.7906  0.7811    0.7838   →  0.8067
                                     override_183 的 flat_183 = 0.8072（差 0.0005）
```

归因坐实：差距就是"单模型 vs 三模型取最优"（+0.0229），**不是 inner/outer 设置的不一致**。
可比性从推断变成实测，记账可以在同一杆秤上重开：

```
两段式（单模型）         0.8333
best_base 扁平基线       0.8067    +0.0266
override_183（旧方法）   0.8210    +0.0123
pos_R5 单看：两段式 0.7812 vs best_base 0.7276    +0.0536
```

顺带是 `always-rf-is-a-finding-not-a-baseline` 的又一次验证：**位移最大的 pos_R5 上
rf（0.7276）把 xgboost（0.6716）甩开 0.056**，而两个抖动单元上 rf 最差。位移越大越该退到简单模型。"""

if OLD not in t:
    sys.exit("未找到保留 1 段落")
P.write_text(t.replace(OLD, NEW), encoding="utf-8")
print("保留 1 已销")
