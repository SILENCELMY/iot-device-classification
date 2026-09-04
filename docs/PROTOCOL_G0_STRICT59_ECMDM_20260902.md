# G0 strict59 表示下 EC-MDM 重裁定协议（运行前冻结）

**冻结日期**：2026-09-02
**状态**：`PROTOCOL_FROZEN_BEFORE_NUMERIC_RUN`
**性质**：承接 `MULTI_MODEL_BRIDGE_SUPPORTED`（2026-09-02 01:37:31 完成），在无线不变表示下重新裁定
EC-MDM 的位置。不修改任何既有冻结判定，不恢复 CPD 的机制/预测/部署身份，不恢复 Route A/B/C 判定，
不把 oracle 上界改称部署收益。
**裁定方**：主线侧。执行方只实现与运行，不作裁定。

## 0. 触发事实（本协议的唯一动因，全部产自本协议之前）

1. bridge 5 种子聚合：等权 `ΔLORO = +0.1056`、`Δ最差LORO = +0.1644`，5/5 模型改善，
   RF 历史对照最大误差 `0.000e+00`，模型单元 350/350。出处
   `results/feature_window_ablation/run_multi_model_bridge_20260902_server/VERDICT.md`。
2. 同一运行的 `stacking_diagnostic.csv`，5 种子均值 `gain_vs_best_base`：LORO 三任务均值
   `−0.1700 → −0.0093`；其 full94 臂逐项复现 E1-FULL 的 B 臂（旗舰 `loro_R2_R4_to_R3` −0.1057、
   `loro_R2_R3_to_R4` −0.2236、`loro_R3_R4_to_R2` −0.1807、`joint_R2_R3_R4` −0.1037），
   即两臂差异不来自口径变更。IID 三任务两臂 `gain` 均在 −0.0069~−0.0003，效应只在 OOD 上。
3. M2 的被测量——固定源 Stacking 元层 `S` 与 oracle 目标重拟合元层 `F` 之差，六环境等权可恢复超额
   `0.1833663976`——是在 full94 表示下测得的。M3 在 UNSW 61 维（天然无 802.11 字段）上的同类量为
   `excess_MMG_equal = 0.0120705`，相差约 15 倍。

因此需要一次**同数据集、同任务、同模型、同划分、同 OOF 口径、只换表示**的对照，
以确定 EC-MDM 是表示无关的结构结论，还是 full94 条件下的结论。UNSW 与自采差异不止表示
（数据集、设备、任务构造、类别数均不同），故 M3 不能替代本对照。

## 1. 唯一问题

在删除 33 项 802.11/radiotap 专属字段与 2 项 UNSW 侧恒零列后的 59 维表示下，
M1/M2 所定义的元层失配（`MMG` 及其 `I/G/C/F` 嵌套恢复阶梯）是否仍然存在、
量级是否仍具实质、以及 `C` 是否仍为首个充分阶段。

本实验不增加特征、不改窗口、不调参、不做特征选择、不计算 CPD、不实现任何新适配器族。

## 2. 冻结输入与表示

- 行集、标签、任务构造、划分与模型超参全部继承现有 G0
  （`code/scripts/core/environment_grid_experiment.py` 常量：`FILTER_MODE=raw_all`、
  `WINDOW_SECONDS=10.0`、`SEED=42`、`MODELS=[rf,xgboost,lightgbm,stacking]`、
  `ENVIRONMENTS=[R2..R7]`、`MAX_SOURCES=3`）。
- `full94` 臂 = 现有 `results/g0_environment_grid/`，**不重跑、不改动**，仅作复现门参照。
- `strict59` 臂列名单**逐字继承**
  `results/feature_window_ablation/run_multi_model_bridge_20260902_server/feature_arms.json`
  的 `strict59` 条目。运行前记录该文件 sha256 与列数；列数不等于 59 即 `INVALID_RUN_STOP`。
- 特征缓存物化方式：自 `results/robust_v2/raw_all/features_raw_all_w10.csv` 取列子集。
  硬性输入审计，任一不满足即停：行数 = 11303；meta 列齐全
  （`round/traffic/filter_mode/source_file/window_id/window_start/window_end/label`）；
  59 个数值列 0 NaN / 0 inf；列序与 `feature_arms.json` 逐字一致。
- **不得改动 `code/` 下任何冻结文件。** G0 以 wrapper 启动：`import environment_grid_experiment as G`
  后覆盖 `G.CACHE_SRC` 与 `G.OUT_ROOT`，再调其入口。先例 = E1-G0-GRID 从 G0 生成器 import 任务定义。
- 输出 root = `results/g0_environment_grid_strict59/`。因 G0 内部为 `feature_mode="all"` 且
  `disable_feature_selection=True`，逐任务子目录名仍为 `all_features`，与 M1/M1-R/M2 的硬编码路径
  一致——**不得为此改写任何下游脚本**。

## 3. 冻结执行顺序与依赖链

严格串行，每步过门才进下一步：

```text
S1  物化 strict59 特征缓存 + 输入审计（§2）
S2  G0(strict59)：162 任务 × 4 模型 × 单种子 42
S3  M1(strict59)   → MMG
S4  M1-R(strict59) → 匹配对照 predictions（M2 的硬复现门输入）
S5  M2(strict59)   → I/G/C/F 阶梯与 ER
S6  full94 复现臂：同一管道跑 S3–S5 于现有 G0 root
```

S6 是**判否门而非补充**。其输出必须复现既有冻结值（§5），否则本次运行整体作废、
strict59 数值不得读取、不产生任何科学结论。

## 4. 资源与确定性

- 线程钉死 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
  （与 M2 provenance 既有常量一致）。G0 的 `--n-jobs` 记入 provenance，同一臂内不得变更。
- **不与任何其他计算作业并行。** E1-G0-GRID 已记 stacking 跨线程拓扑复现界约 `2e-3`，
  D12 R3 曾因 stacking 非确定性在 26/148 行出字节差 fail-stop；本协议以串行独占规避。
- 双跑门：S2 的逐任务 `metrics.json` 与 S5 的判定性产物，双跑 md5 逐字节一致。
  不一致即 `INVALID_RUN_STOP`。
- 全程禁网、禁代理、不安装依赖。解释器 = `~/anaconda3/envs/iotcls/bin/python`。

## 5. 冻结复现门（S6，先于任何 strict59 读数）

full94 臂必须复现下列既有冻结值。容差取 M2 自身常量 `reproduction_max_abs = 1e-12`：

| 量 | 既有冻结值 | 出处 |
|---|---:|---|
| M1 `MMG_ood_equal` | 0.20292761623438427 | `results/meta_mismatch_exploratory/m1/M1_RESULTS_NOTE.md` |
| M1 `MMG_ood(e) > 0` 环境数 | 6/6 | 同上 |
| M2 完整 `F` 超额恢复 | 0.1833663976 | `independent/meta_mismatch/M2_FACT_CARD.md` §2 |
| M2 `ER_C_equal` | 0.8909628 | 同上 |
| M2 `ER_C(e) >= 0.50` 环境数 | 6/6 | 同上 |

任一超差 → 状态 `INVALID_RUN_STOP`，写失败报告，**不读 strict59 臂**。

## 6. 冻结判据（三门，运行前定死；strict59 臂，六目标环境等权）

1. **存在门**：`excess_F_equal >= 0.005` 且 `excess_F(e) > 0` 在 ≥4/6 环境。
   阈值**逐字沿用** M3 的冻结外部存在门，不新造常量。
2. **实质门**：`excess_F_equal >= 0.02`。锚点两条且均为项目既有冻结量——§11.3 第 3 块的真 5 种子
   噪声尺度 `0.0102–0.0163` macro-F1（低于此即不可操作），以及 bridge 协议自用的 `0.02` 门槛量级。
3. **结构门**：`C` 仍为首个充分阶段。沿用 M2 原冻结门 `ER_equal >= 0.80`、`ER(e) >= 0.50`
   且 ≥4/6 环境；且 `I`、`G` 均不得先于 `C` 达标。

逐环境值无论方向一律并报。`ER` 的分子分母同时缩小属预期可能，必须同时报 `excess_F_equal`
的绝对量与各级 `ER`，不得只报比值。

## 7. 三分支处置（运行前写死，不得事后选择）

| 分支 | 条件 | 状态码 | 处置 |
|---|---|---|---|
| A | 三门全过 | `EC_MDM_REPRESENTATION_ROBUST` | EC-MDM 升为表示无关结论，与表示发现并列为两个互补贡献进正文；§11.2 弱点⑤的 scope 修订按此执行 |
| B | 存在门过、实质门不过 | `EC_MDM_MAGNITUDE_REPRESENTATION_CONDITIONAL` | 结构主张（类别条件为充分参数化）保留；量级主张降为「含无线专属特征的表示下」的条件结论；EC-MDM 定位为后备结论（被迫使用非不变特征时的修法），正文承重结构归表示发现 |
| C | 存在门不过 | `EC_MDM_NOT_SUPPORTED_UNDER_INVARIANT_REPRESENTATION` | EC-MDM 只作 full94 表示下的现象描述，不进正文承重结构；§11.1「不得记为明确失败」改写为「在含无线专属特征的表示下成立」，原文保留加注 |

存在门与实质门均过但结构门失败 → 状态 `EC_MDM_EXISTS_STRUCTURE_NOT_REPLICATED`，
按 B 处置，并单独记录哪一级先达标。

三分支之外不得新增分支；不得因结果不利而改判据、改阈值、改环境集或改表示。

## 8. 明文不做

- 不重算 Route A / Route B / D8 投票基线 / G1 / E2 / D12。S2 产物**声明可复用**于后续此类重算，
  但每次须另立冻结协议、另行登记，不得由本协议顺带产生任何相关结论。
- 不改窗口长度、不加任何特征列、不调参、不做特征选择、不实现 `59+C` 或任何新适配器族。
- 不计算或恢复 CPD；不恢复 CPD 的机制、预测或部署身份。
- 不重开 E1 / E2 / G1 / Route B / D12 / M1–M6 的任何既有判定。
- `F` 阶段使用目标标签属 M2 原设计的 oracle ceiling 定义内用途，**不得**改称部署收益或
  少样本方法收益。
- 不动 `dataset/`；不写入 `results/` 既有任何子目录；不改 `code/` 与 `docs/` 既有文件
  （本协议为新增文件）。

## 9. 交付物

`results/g0_strict59_ecmdm/`：

```text
input_audit.json        §2 全部硬性审计的实测值
reproduction_gate.json  S6 四项复现门逐项偏差
mmg_table.csv           M1 的 MMG，逐环境 + 等权
er_ladder.csv           I/G/C/F 的 excess 与 ER，逐环境 + 等权
per_environment.csv     三门的逐环境判定明细
passline.json           三门实测值与门槛并列
acceptance.json         双跑 md5、输入审计、线程变量、单元计数
provenance.json         git、argv、包版本、逐任务折叠记录、耗时
VERDICT.md              状态码 + §7 处置 + 解释边界
```

S2 的模型产物落 `results/g0_environment_grid_strict59/`，按协议 §19.7 白名单入库
（`*.csv` / `*.json` / `*.md`；`*.joblib` 与 `scratch/` 不入库）。

## 10. 登记与提交顺序

1. 本协议先提交（本文件的 commit 必须先于任何 strict59 数值产生）。
2. 在 `docs/EXPERIMENT_REGISTRY.md` 登记一行 `G0-STRICT59-ECMDM`（运行前登记，
   含本协议 sha256 与 implementation commit），跑完补「结论」列。
3. 实现完成后由主线侧作实现审阅，追加 `RUN_AUTHORIZED` 方可正式运行。
4. 判定书写入 `results/g0_strict59_ecmdm/VERDICT.md`，并按 §7 分支同步
   `docs/CROSS_LINE_DISCUSSION_20260830.md` 的处置记录。历史文档一律加注降格、保留原文。

## 11. 解释边界

本协议无论落在哪一分支，都不构成：CPD 机制或预测身份的恢复；无标签部署方法的成立；
Route A/B/C 判定的改变；`oracle` 可恢复上界与部署收益的等同；以及「表示发现已外部复现」的主张
（外部复现须另立 UNSW 侧协议）。strict59 对同轮上界的代价（bridge 实测 `ΔSingle`
`−0.0699 ~ −0.1075`、`ΔJoint` `−0.0788 ~ +0.0308`）必须在任何引用表示发现的场合同时报告。



