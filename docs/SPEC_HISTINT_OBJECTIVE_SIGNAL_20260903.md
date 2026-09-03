# 规格：历史池内「目标函数单位」不稳定度信号 `S_obj`（运行前冻结）

**日期**：2026-09-03
**性质**：诊断性对照测量。**不改动任何已冻结协议，不重开任何已冻结判定。**
**上游**：`docs/PROTOCOL_TWO_CHANNEL_20260903.md`（sha256 `dc298198e70957dcd4d1b445900ce85e6ae7bbfac60cac4c8c546515594c0e86`）
与其判定书 `results/two_channel_20260903/VERDICT.md`（判定 `PROCEDURE_BEATS_PRACTICE`）。

---

## 1. 为什么做这一件

协议 §2.3 的坐标 1 **提议阶段**用的代理量是逐类对 AUC：

```
S(F) = Σ_{(任务, 无序类对)} ΔAUC_tgt(F)      冻结实测：rssi +5.2440，次名 subwin +0.0156
```

而**接受阶段**用的是真实目标 macro-F1。协议 §2.3 已原文记录这两者会给出不同答案——
删 `rssi` 在代理量上 `W = −0.2951` 超过上限（不过），在 macro-F1 上 9 个任务无一变差（过）。
既然两者在接受阶段已被实测到不一致，**提议阶段的函数形式选择（AUC 而非 macro-F1）需要一个独立的支撑或反证**。

本规格测的就是这一件：**把不稳定度直接放到目标函数自己的单位上重算一次，看排序是否一致。**
这不是对坐标 1 可部署性的检验——坐标 1 的 `S(F)` 只在历史池 H 内求和（9 个任务源与目标
全部落在 `{R2,R3,R4}` 内），对 H 之外的新轮次本就零标签需求，UNSW 锚3「族代理量 `S(F)` 在 H 内」
即依赖这一点。自采那轮之所以是样本内演示，原因是评估目标与算信号的池重合（协议 §3.3 已披露），
与「是否读目标标签」是两个不同的问题。**本规格不声称修复任何可部署性问题。**

## 2. 信号定义（运行前定死）

历史池 `H = {R2, R3, R4}`。内层留一轮次任务取协议 `TASKS` 中 `|S|=2` 的三个，逐字沿用不另定义：

```
g0_R2_R3_to_R4    g0_R2_R4_to_R3    g0_R3_R4_to_R2
```

对特征集 `c`：

```
LORO_r(c)   目标为 r 的内层任务上的跨轮次 macro-F1
JOINT(c)    H 内联合时间块 macro-F1（定义见 §3，本规格新增）
G_r(c)      = JOINT(c) − LORO_r(c)          「没见过轮次 r」造成的损失
S_obj(F)    = Σ_{r∈H} [ G_r(c_full) − G_r(c_full \ F) ]
```

`S_obj(F) > 0` = 删掉族 `F` 之后「联合与跨轮次的差」缩小 = `F` 是不稳定族
（见过该轮次时做功、迁移时不做功）。

**恒等分解（并报，用于归因）**：

```
S_obj(F) = ΔLORO(F) − 3 × ΔJOINT(F)
  ΔLORO(F)  = Σ_r [ LORO_r(¬F) − LORO_r(c_full) ]     迁移收益
  ΔJOINT(F) =       JOINT(¬F)  − JOINT(c_full)        天花板代价
```

该恒等式在实现中逐族断言（容差 1e-12），不成立即中止。

## 3. `JOINT` 的定义（本规格唯一新增定义）

协议与 `run_two_channel.py` **均未计算任何 IID 量**（`passline.json` 无判据 5 条目，
`VERDICT.md` 判据 5 行的数值列为「—」），故此处必须自行定义并声明为新增：

```
对 H 中每一个轮次，用 time_blocks(window_start, k=5)（沿用协议实现，N_TIME_BLOCKS=5）
划出 5 个时间块；训练集 = 各轮次的块 0..3 并集，测试集 = 各轮次的块 4 并集。
不随机打散相邻窗口（协议 §9.1 纪律）。
```

**交叉参照（只报不设门槛）**：`c_full` 下的 `JOINT` 与 `MAINLINE_20260903.md` §2 的
「联合三轮时间块 0.8936」并列报出。两者代码路径不同，**不要求相等**；差异过大时如实报出并
在结论中标注，不据此调整定义。

## 4. 聚合口径

族导出沿用协议 `derive_families`（按列名首个下划线段分组、成员 <2 并入 `singletons`），
自采 94 列 → 10 组，不另定义。基模型集 = `rf` / `xgboost` / `lightgbm`（**不含 stacking**——
本信号测特征不稳定度，元层是坐标 2 的事）。

| 口径 | 定义 | 地位 |
|---|---|---|
| `max_base` | 逐任务取三个基模型 macro-F1 的最大值 | **主口径**（与已登记的 `TE` 定义一致；从特征信号中移除模型选择噪声） |
| `best_base_src` | 按源域 OOF macro-F1 选出的基模型（协议 `coord2_diag` 的 `best_base_src`） | 并报，供审计口径选择 |

两个口径的全部数字都必须报出。主判据只在 `max_base` 上评估。

## 5. 预注册判据（运行前定死，不因结果调整）

**主判据（二值）**：`S_obj(F)` 在 `max_base` 口径下 **`rssi` 排第一**
→ `OBJECTIVE_SIGNAL_AGREES`；否则 `OBJECTIVE_SIGNAL_DISAGREES`。

**只报不设门槛**（理由同 UNSW 协议锚3：`S(F)` 与 `S_obj(F)` 单位不同，
用 AUC 口径的数字去卡 macro-F1 口径的量会反卡自己）：

- 头名/次名比值（冻结 `S(F)` 精确值为 5.2440/0.0156 = **336.2**；
  `EXPERIMENT_REGISTRY.md` 与 UNSW 协议中的「328」系用四舍五入后的 0.016 计算，两者皆非错）；
- `S_obj` 与冻结 `S(F)` 在 10 族上的 Spearman 秩相关；
- 逐族 `ΔLORO(F)`、`ΔJOINT(F)` 分解；
- `best_base_src` 口径下的同一排序；
- 逐任务、逐族原始 macro-F1。

**三分支处置（运行前定死）**：

| 分支 | 处置 |
|---|---|
| `rssi` 第一**且为唯一正值族** | AUC 代理在提议阶段是忠实代理；可进正文作提议阶段形式选择的支撑 |
| `rssi` 第一**但存在其它正值族** | 目标函数单位下还有别的不稳定族、AUC 代理漏了它；该族须另立协议评估，**不在本规格内做任何删除动作** |
| `rssi` **非第一** | 提议阶段的形式选择须重新论证；`TWO-CHANNEL-SELF` 判定**不动**（其接受阶段本就用 macro-F1，删 `rssi` 的决定由真实目标背书），但正文**不得**把 AUC 代理表述为一般性的不稳定度量 |

**本规格不执行任何删除动作、不改变任何配置、不产生新的流程终点。**

## 6. 硬门

1. 复用冻结实现：`import run_two_channel`，直接调用其 `Data` / `derive_families` /
   `make_model` / `fit_eval` / `coord2_diag` / `time_blocks` / `TASKS`；
   `feature_columns` / `clean_x` / `build_model` / `fit_label_encoder` 经其转引，**一律不重实现**。
2. 恒等式 `S_obj = ΔLORO − 3·ΔJOINT` 逐族断言，容差 1e-12。
3. 线程钉死 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`，模型 `n_jobs=1`。
4. **双跑 md5 逐字节一致**（除 `provenance.json` 的时间戳字段外全部产物）。
5. 解释器 `~/anaconda3/envs/iotcls/bin/python`；禁网禁代理。
6. 种子 42，与协议同。

## 7. 产物

`results/histint_objective_signal_20260903/`：
`s_obj.csv`（逐族两口径 + 分解）、`raw_scores.csv`（逐配置逐任务逐模型 macro-F1）、
`joint_reference.json`（`JOINT` 逐配置值与 0.8936 对照）、`passline.json`（主判据与并报量）、
`provenance.json`、`VERDICT.md`。

**预估成本**：配置 11 个（基线 + 10 族剔除）× (3 内层任务 + 1 联合) × 3 基模型 = 132 次拟合；
按 `TWO-CHANNEL-SELF` 实测 9 任务 4 模型 225 s 折算约 14 min/单跑，双跑约 28 min。
