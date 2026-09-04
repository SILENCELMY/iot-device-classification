# 协议：类组分解判决（分层）——保住易类同时提升困难簇（运行前冻结）

**日期**：2026-09-04
**性质**：确认性实验，用于检验 2026-09-04 单种子探索性探针的结论。**不改动任何已冻结协议，不重开任何已冻结判定。**
**上游**：`docs/PROTOCOL_TWO_CHANNEL_20260903.md` sha256
`dc298198e70957dcd4d1b445900ce85e6ae7bbfac60cac4c8c546515594c0e86`；
`docs/PROTOCOL_R567_94DIM_RETEST_20260904.md` sha256
`6ab54aea29288e7ec157b75f8d7fb6c450b424c2de659008653e3f3ae15db64c`（判定 `REGIME_DEPENDENT`，
其逐类分解是本协议的动机）。

---

## 1. 动机，以及**已经看过什么**（探索→确认，必须披露）

`R567-94DIM-RETEST` 的逐类分解给出：删 `rssi` 在八个单元上**全部**损伤 `Camera`
（−0.0239 ～ −0.3494），收益**全部**来自困难簇；`Socket` 恒不变（160 条 F1 中 159 条为 1.000）。
即坐标 1 是一笔「用 Camera 换困难簇」的交易，而反转是**类对级**而非族级的
（冻结产物 `diag_coord1.csv` 本就逐 `(task, pair, family)` 记录，验收时用 `d_auc_sum` 把类对那一维加和掉了）。

**本协议要检验的假设**：把判决按类组拆开——易类层用含 `rssi` 的特征集、困难簇层用删 `rssi` 的特征集——
可以同时保住易类与提升困难簇。

**【必须披露：本协议冻结前已运行过一次单种子探索性探针】**
`/tmp/hier_seed42.csv`（seed 42，窗口级，簇成员由作者按先验手设，硬路由）。已看到的量：
六单元窗口级 macro 与逐类 F1、`stage1_acc`。已看到的结论：固定 `xgboost+xgboost` 时六单元均值
0.8349 vs 平铺 oracle-max 0.7826；逐类均值 `Camera −0.0048 / T1 +0.0222 / XM +0.0236 /
Sensor +0.0136 / Socket −0.0002`；`stage1_acc` 0.959–0.984。

**因此本协议的判据一律设在探针未覆盖的维度上**：① 五种子稳定性；② 簇成员由源域**程序性**导出
而非手设；③ 两层模型由源域侧规则选出而非事后选；④ 设备级三档曲线。
探针的单种子数字**不作为本协议的门槛依据**，只在结论列作对照记录。

## 2. 单元、配置、模型（运行前定死）

| 单元 | 类型 | 源域 | 目标 | 用途 |
|---|---|---|---|---|
| `inner_to_R2` | inner | R3+R4 | R2 | 源域侧：簇导出 + 模型对选择（目标在源域内，标签合法可见） |
| `inner_to_R3` | inner | R2+R4 | R3 | 同上 |
| `inner_to_R4` | inner | R2+R3 | R4 | 同上 |
| `pos_R5` | outer | R2+R3+R4 | R5 | **主判据单元**，目标从未参与任何决策 |
| `jit_R6` | outer | R2+R3+R4 | R6 | 主判据单元 |
| `jit_R7` | outer | R2+R3+R4 | R7 | 主判据单元 |

特征集：`full94`（94 列）、`drop_rssi`（86 列，`TC.derive_families` 的 `rssi` 族 8 列）。
基模型：`rf` / `xgboost` / `lightgbm`（`stacking` **不进本协议** —— 是否堆叠属
`PIPELINE-DROPRSSI-E2E` 的射程，两者不混）。种子 **42–46**。

**三条臂**：`flat_full94`、`flat_drop_rssi`（各 3 模型）、`hier`（模型对由 §4 选出）。

## 3. 簇成员的程序性导出（源域，运行前定死）

1. 取三个 inner 单元、`full94`、§4 选出的第一层模型的预测，**逐种子**聚合三个单元的混淆矩阵；
2. 归一化为行条件概率，复用 `results/g0_environment_grid/diag_grid_override.py` 的
   `cluster_from_cm`，**τ = 0.1**（与 `UNSW-META-MISMATCH` 冻结取值一致，不另设）；
3. 取规模 ≥2 的最大连通分量为 `CLUSTER`；其余类为易类层的独立类别。
4. **若导出结果不是 `{Light_T1, Light_XM, Sensor}`，一律照导出结果执行并在结论列如实记差异。**
   若五个种子导出的簇不一致，取五种子**交集**，并记录不一致情形。
5. 若 `CLUSTER` 规模 <2，判定 `INVALID_RUN_STOP`（无簇可分层，假设不适用）。

## 4. 两层模型对的源域侧选择（运行前定死）

对 (m1, m2) ∈ {rf, xgboost, lightgbm}²，在**三个 inner 单元**上按 §5 的分层实现计算 macro-F1，
取五种子均值再取三单元均值，选 argmax。**只用 inner，不看 outer。**
选出的 (m1\*, m2\*) 用于 outer 三单元。`flat` 两臂的模型同样只按 inner 均值选出各自的 m\*。

## 5. 分层实现（新增代码，声明如下）

- 第一层：把训练标签中属 `CLUSTER` 的类合并为一个类别，其余类各自保留，得连续编号的
  `(|非簇类| + 1)` 分类问题；在 `full94` 上用 m1\* 训练。
- 第二层：只取训练集中属 `CLUSTER` 的行，`|CLUSTER|` 分类；在 `drop_rssi` 上用 m2\* 训练。
- 判决：第一层输出为 `CLUSTER` 的样本进第二层，否则直接输出（**硬路由**）。
- 软路由（`P(类)=P(簇)×P(类|簇)`）**不进本协议**，另立。

## 6. 主判据（运行前定死，门槛不因结果调整）

在三个 outer 单元上，逐类比较 `hier`（m1\*+m2\*）与 `flat_best`
（两条平铺臂中同类 F1 较高者，各臂用 §4 选出的 m\*），全部取五种子均值。

- **条件 1（无退化）**：`hier` 的 `Camera` 与 `Socket` ≥ `flat_best` 同类 − **0.01**。
  格数 = 3 单元 × 2 类 = **6 格**。（若 §3 导出的簇不同，则「易类」= 非簇类，格数随之调整并记录。）
- **条件 2（有提升）**：`hier` 的 `CLUSTER` 三类 F1 均值 > `flat_best` 的同三类均值。
  格数 = 3 单元 = **3 格**。

| 分支 | 条件 | 处置 |
|---|---|---|
| `HIER_DOMINATES` | 条件 1 ≥ 5/6 **且** 条件 2 = 3/3 | 假设成立：类组分解可同时保住易类与提升困难簇。可作为坐标 3 写入流程，并据此陈述「V1 配置空间已被跳出」 |
| `HIER_TRADES` | 条件 2 ≥ 2/3 但条件 1 < 5/6 | 提升是**又一次交易**而非双赢。只能报为「另一条交易曲线」，不得称保住两边 |
| `HIER_FAILS` | 条件 2 ≤ 1/3 | 假设不成立。须如实报，并记录探针的单种子结论未复现 |

## 7. 并报（无论方向，不得省略）

- 逐单元、逐臂、逐模型、逐种子窗口级 macro-F1 与逐类 F1（均值 ± 样本标准差）；
- **设备级三档**（N=1 / 6 / 30，即 10 s / 1 min / 5 min），定义与
  `results/g0_environment_grid/diag_window_aggregation.py` 完全一致；
- `stage1_acc`（第一层准确率）与逐设备众数裕度；
- §3 导出的簇成员（逐种子）与是否等于 `{Light_T1, Light_XM, Sensor}`；
- §4 选出的 (m1\*, m2\*) 与 inner 上 9 个组合的完整排名；
- 与 V1 空间 oracle（两条平铺臂逐单元取 max）的差值；
- 探针（seed 42、手设簇）数字作**只读对照**列出，不参与判据。

## 8. 硬门

1. 复用冻结实现：`import run_two_channel`（`Data` / `derive_families` / `make_model` / `clean_x`）；
   簇导出复用 `diag_grid_override.py` 的 `cluster_from_cm`，不重实现。
2. **新增代码只有三处，且必须声明**：(a) §5 的分层训练与硬路由；(b) 标签合并/还原的连续编号映射
   （xgboost 要求标签连续）；(c) 块投票（定义同 `diag_window_aggregation.py`）。
3. **决策隔离断言**：簇导出函数与模型对选择函数的签名只接受 inner 单元的结果，
   以 `assert` 保证与 outer 无交集。R5/R6/R7 只用于打分。
4. 种子切换经 `run_two_channel.SEED` 重绑定，不改冻结文件。
5. 线程钉死 `OMP=MKL=OPENBLAS=1`，模型 `n_jobs=1`。
6. **双跑 md5 逐字节一致**（除 `provenance.json` 的 `_volatile` 段）；
   **双跑之间不得产生任何 commit**。
7. 解释器 `~/anaconda3/envs/iotcls/bin/python`；禁网禁代理。

## 9. 产物与成本

`results/hier_classpair_20260904/`：`per_unit.csv`（逐单元逐臂逐模型逐种子，窗口级 + 三档设备级 + 逐类）、
`cluster_derivation.json`、`model_selection.csv`（inner 上 9 组合排名）、`passline.json`、
`device_margin.csv`、`provenance.json`、`VERDICT.md`。

**预估成本**：inner 三单元需 9 个分层组合（各 2 次拟合）+ 6 次平铺 = 24 次/单元/种子；
outer 三单元只跑选中组合（2 次）+ 6 次平铺 = 8 次/单元/种子。
合计 (3×24 + 3×8) × 5 = **480 次拟合**。按 R567 实测 ~10.9 s/次折算约
**87 min/单跑、174 min/双跑**（分层的两次拟合各只用部分数据，实际应快于此估计）。

## 10. 本协议不做什么

- 不测软路由；不测 `strict59_ra`（独立线）；不改任何特征定义；不重抽 pcap；
- 不测是否堆叠（属 `PIPELINE-DROPRSSI-E2E`）；不做窗长消融；
- 不改动 `TWO-CHANNEL-SELF`、`R567-94DIM-RETEST`、`PIPELINE-DROPRSSI-E2E` 的任何判定。
