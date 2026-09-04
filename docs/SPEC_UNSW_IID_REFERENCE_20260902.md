# UNSW 设备身份粒度日内 IID 参照规格（运行前冻结）

**冻结日期**：2026-09-02
**状态**：`SPEC_FROZEN_BEFORE_NUMERIC_RUN`
**性质**：补一个**缺失的参照测量**——UNSW 在设备身份粒度上的日内时间块 IID macro-F1。
现有落盘只有跨天值（`results/unsw_pilot/loro/loro_summary.csv`，6 有序日对 0.8164–0.8674），
日内参照在设备身份粒度上从未测过；D12 的 IID 参照是**类别**粒度（6 类）。
本规格不修改任何既有冻结判定，不构成 §19 RQ1 的替代或外部复现主张。

## 1. 唯一问题

在**设备身份**为标签、日内时间块划分下，UNSW 的 macro-F1 是多少；
它与同测试日、同类集的跨天 LORO 值之差是否为正。

## 2. 冻结输入与构造

- 特征 = `results/unsw_pilot/four_day/features_unsw_w10_4day.csv`（pilot 已落盘，不重新提取）。
- 日集 = pilot LORO 所用三天：`16-09-23` / `16-09-30` / `16-10-12`（`16-10-11` 不入，
  与 pilot LORO 口径一致）。
- 标签 = `label` 列（设备身份），非 `category`。
- 特征列 = 直接调用 `code/scripts/analysis/unsw_pilot/pilot_rf_loro.py` 的
  `feature_columns()` 与 `META_COLUMNS`，不重实现。
- 模型 = 同一脚本路径的主线 `build_model("rf", random_state=42, class_count=len(keep))`；
  不调参、不做特征选择、不换模型。
- 平衡 = 主线 `sample_balanced(max_rows=20000, random_state=42)`，训练侧与测试侧各自施加。
- **划分 = 设备内时间块**：每设备按 `(window_start_epoch, window_id)` 升序，
  前 70% 训练 / 后 30% 测试。口径逐字对齐 D12 IID
  （`device_internal_sort(window_start_epoch,window_id), first_70pct_train, last_30pct_test`）。
  **不使用随机划分**——§11.3 第 2 块已量化相邻窗口泄漏（自采 R6 达 +0.155~+0.206）。

## 3. 冻结任务集

**主口径 = 6 个配对 IID。** 对 pilot 的每个有序对 `A→B`，构造一个 IID 任务：
测试日 = `B`，类集 = `keep(A,B)` = 两天各自 ≥100 窗的设备交集（与该 LORO 对逐字相同），
训练/测试均取自 `B` 的设备内时间块。于是 `IID(B | A→B)` 与 `LORO(A→B)`
**同测试日、同类集、同门槛、同平衡、同模型、同种子**，唯一差异是训练数据来自 `B` 自身前段
还是来自 `A`。

**并列参照 = 3 个逐日 IID**：类集 = 该日单独 ≥100 窗的全部设备，只作描述。

## 4. 冻结判据（运行前定死）

配对差 `drop(A→B) = macro_f1(IID(B|A→B)) − macro_f1(LORO(A→B))`。

- **存在跨时段下降**：`mean(drop) > 0` 且 ≥5/6 对为正。
- 否则为 `NO_CROSS_DAY_DROP_AT_DEVICE_GRANULARITY`。
- 不做显著性检验（n=6；§15 不报未校正 p 值）。逐对值无论方向一律并报。

## 5. 解释边界

- 幅度与自采并列报告（自采 `strict59` gap +0.120、`full94` +0.264），
  **不作统一化或"同一机制"声明**。
- 三天、单种子、单测试床，属参照测量；不构成外部复现，不改 D12 判定，
  不恢复 CPD / EC-MDM 的任何身份。
- 不得因结果不利而换天、换门槛、换类集、换粒度或改判据。
- 与 D12 类别粒度结果的差异只作为**口径依赖的观察**记录，不倒推 D12 结论。

## 6. 交付物与门

`results/unsw_iid_reference_20260902/`：`iid_summary.csv`、`paired_comparison.csv`、
逐任务 `cm_*.csv` 与 `per_class_*.csv`、`provenance.json`（git、argv、包版本、耗时）、`NOTE.md`。

硬门：① 每任务实际类集与对应 LORO 对逐字一致；② 特征 0 NaN / 0 inf；
③ 训练/测试窗口不重叠（设备内时间块边界检查）；④ 双跑 macro_f1 逐位一致。
任一失败即停，不写结论。全程禁网、禁代理。
