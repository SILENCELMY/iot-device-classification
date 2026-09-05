# -*- coding: utf-8 -*-
"""把 README 里"双 inner 的功劳归给第二环境验证"这段错误归因换成消融实测结论。"""
import pathlib, sys

P = pathlib.Path("/home/lmy/iot-device-classification/results/"
                 "config_derivation_exploratory_20260905/README.md")
t = P.read_text(encoding="utf-8")

OLD = """## 一处自我更正

曾把"双 inner 优于单 inner"解释为 winner's curse 对策（剔除了噪声配置）。
**seed45 两臂都只过 1 对，差距照旧 +0.008 vs +0.061 —— 该解释被证伪。**
真正的差别是**两个臂看到了不同的世界**：

```
单 inner（R2+R3→R4，|S|=2）  Light_T1|Sensor 在位者 0.8384 —— 认为没毛病
双 inner（R2→R3，|S|=1）      Light_T1|Sensor 在位者 0.0029 —— 认为几乎全判反
```

而 outer 证明后者判对。但该臂同时改了两件事（inner 的 `|S|`，与第二环境验证），
**功劳归属未定**，需消融（`cfg_ablation.py`）。
另注：这与 `rssi-ablation-is-the-origin` 记录的"`|S|=1` 有误导性、`|S|≥2` 是硬要求"
方向相反 —— 两者测的量不同（那次是 `d_auc`，这次是可判决区准确率 + 端到端 macro），
**不算直接矛盾，但必须调和后才能把任何一条当规则。**"""

NEW = """## 归因消融：功劳全在 inner 的 `|S|`，第二个环境的验证贡献恰好为 0

上表里叫"双 inner"的那个臂同时改了两件事（inner 的 `|S|`，与第二环境验证）。
`cfg_ablation.py` 加了第三个臂 `cfg_A`（只在 A=R2→R3 上选，**不做 B 验证**）来拆开：

```
                                          三单元 × 2 seed 均值   Δ vs 分层   过闸对数
分层（不修）                                    0.8141              —          —
cfg_S   R2+R3→R4       |S|=2，无验证            0.8191          +0.0050       2 对
cfg_A   R2→R3          |S|=1，无验证            0.8329          +0.0188       1 对
cfg_AB  R2→R3 + R3→R2  |S|=1，有验证            0.8329          +0.0188       1 对
```

**`cfg_A` 与 `cfg_AB` 逐格四位小数完全相同** —— B 从未拒绝过任何提名。
**该臂不应叫"双 inner 验证"，应叫 `|S|=1` inner。**

此前对这个臂给过两次解释，**两次都错**：先归因 winner's curse 对策（被 seed45 证伪），
再归因第二环境验证（被本次消融证伪）。

注意差距不是"修得多"：`|S|=2` 过 **2 对**只得 +0.0050，`|S|=1` 过 **1 对**得 +0.0188 ——
`|S|=1` 找对了 `Light_T1|Sensor`，`|S|=2` 找的两对不是关键那对。
**闸门的灵敏度取决于 inner 怎么切，不取决于闸门本身。**

### 为什么 `|S|=1` 更灵敏（机制猜想，扫描中）

轮次几何（目录名实证）：`R2/R3/R4 = normal`（同位置同操作）、`R5 = positionB`、
`R6/R7 = jitter`。**没有任何 inner 任务里含位置漂移**，所以"inner 复现 outer 的
位移 regime"这个说法不成立（曾这样解释过，随即被目录名否掉）。

剩下的猜想是**放大**：R2/R3/R4 之间只差时间与噪声。训两轮 → 模型见到两份 rssi 实现、
自动把它降权 → 在位者在 R4 上正常（0.8384）→ 闸门看不见病；训一轮 → 模型死抓这一轮的
rssi → 在位者在 R3 上崩（0.0029）→ 闸门看见病。outer 训三轮降权更多，但 R5 的位置位移
远大于 normal 轮次间漂移，**照样崩**。即 `|S|=1` 是**用小位移把同一个病放大到可见的
压力测试**，不是同分布代理。

`inner_sweep.py` 扫 9 种 inner 切法（6 个 `|S|=1` + 3 个 `|S|=2`）检验之，
**预注册**（结果前写死）：

| | 判据 | 结论 |
|---|---|---|
| H1 原则 | `min_inc`（inner 暴露的最低在位者）与 outer Δ 的 Spearman ρ ≤ −0.7 | 得到**设计时可算**（不看 outer）的 inner 选择规则 |
| H2 规则 | 6 个 `\\|S\\|=1` 的 Δ 全部高于 3 个 `\\|S\\|=2` | `\\|S\\|=1` 是硬规则，仍缺解释 |
| H3 运气 | 只有 R2→R3 能用，ρ 不显著 | **+0.0188 是 9 选 1 抽中的，方法真实增益退回 ≈ +0.005** |

### 这是目前最大的未解问题

`|S|` 一个参数值 **+0.0138**，而定点修相对分层的总增益才 **+0.0188** —— 占 73%，
且没有理论。且与 `rssi-ablation-is-the-origin` 的"`|S|=1` 有误导性、`|S|≥2` 是硬要求"
方向相反（两者测的量不同：`d_auc` vs 可判决区准确率 + 端到端 macro）。
**在调和之前方法不能冻结** —— "为什么 inner 只用一个源环境"目前答不上来。"""

if OLD not in t:
    sys.exit("未找到待替换段落，README 可能已改动")
P.write_text(t.replace(OLD, NEW), encoding="utf-8")
print("README 已更新")

# 表头里的"单/双 inner"也标注一下真实含义
t2 = P.read_text(encoding="utf-8")
h_old = "          flat     分层      +单inner   +双inner"
h_new = ("          flat     分层      |S|=2修   |S|=1修"
         "      （原名「单/双inner」，见下方归因消融）")
if h_old in t2:
    P.write_text(t2.replace(h_old, h_new), encoding="utf-8")
    print("表头已标注")
