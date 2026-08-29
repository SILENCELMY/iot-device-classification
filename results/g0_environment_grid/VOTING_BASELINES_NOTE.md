# VOTING_BASELINES_NOTE.md —— 投票基线（协议 §7 基线 4–6）

**性质**：产物说明。只记口径定义、输入路径、平局规则、校准数据边界、自检结果与产物字典，
**不做任何结果解读**。
**协议依据**：`docs/experiment_protocol_final.md` §7（基线 2、4、5、6）、§9.1、§10、§19.2。
**执行口径**：`docs/EXECUTION_PLAN_20260829.md` 决策 D8。
**生成脚本**：`code/scripts/analysis/voting_baselines.py`（口径的完整定义在该脚本 docstring）
**生成日期**：2026-08-29（末次运行 UTC 04:47；两次独立运行的两个 CSV 逐字节一致，md5 相同）
**运行记录**：`results/g0_environment_grid/voting_baselines_run_metadata.json`（§19.2 五要素）

---

## 1. 性质与范围

本批产物是对 `results/g0_environment_grid/` 已落盘结果的**纯后处理**：只读
`predictions.csv` / `pred_proba.csv` / `metrics.json` / `stacking/oof_meta.csv`，
**不重新训练任何模型**、不读取原始特征表、不使用任何随机数。

范围 = G0 网格全部 **162 个任务**（150 个 OOD + 6 个 `iid_random` + 6 个 `iid_time_block`），
特征集 `all_features`，`filter_mode = raw_all`。

协议 §7 的 source-only 基线中，本批新算的是 **4 hard voting**、**5 等权 soft voting**、
**6 校准后 soft voting** ——这三条在此之前从未在任何结果目录中被计算过（D8 的缺口）。
同表并列的 `rf` / `xgboost` / `lightgbm` / `stacking` / `best_base_posthoc` **不是新结果**，
是用**同一套指标函数**从已落盘产物重算的对照行，只为让投票行与对照行可直接相减；
它们的 5-class macro-F1 已由第 5 节的硬门自检逐一对齐各自 `metrics.json`。

---

## 2. 输入的结构事实（已逐任务校验）

| 事实 | 状态 |
|---|---|
| `predictions.csv` 含 `true_label` + `predicted_label` | 是（162 × 4 模型全有） |
| `pred_proba.csv` 列名自带类别名 `proba_<cls>`，5 列齐全 | 是；全 162 任务表头逐字一致 |
| `pred_proba.csv` 行和 = 1 | 是（`atol=1e-6`） |
| `argmax(pred_proba)` ≡ `predicted_label` | 是（648 个模型-任务组合逐行 100% 一致） |
| 同任务 4 个模型的行数、行序（`source_file, round, window_id, window_start`）、`true_label` 一致 | 是 |
| `oof_meta.csv` **含训练标签列 `true_label`** | **是** —— 故**不需要**从 `features_raw_all_w10.csv` 重建训练标签 |
| `oof_meta.csv` 行数 ≡ `metrics.json` 的 `train_samples` | 是（162/162） |
| `oof_<model>_<cls>` 15 列齐全、每模型行和 = 1 | 是 |
| 5 个类别在每个任务的训练集与测试集中均出现 | 是（162/162） |

> D8 规格中"若 `oof_meta` 不含训练标签则按任务定义重建"的分支**未被触发**。

**类别轴**：`['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']`，与
`code/configs/research_experiments.json` 的 `labels`（即 `robust_iot_research.metric_summary`
的 `labels` 参数）逐字一致。脚本按**列名**取概率列，不依赖列位置；缺列即报错退出。

**网格元数据**（`task_type` / `n_sources` / `target_env` / `grid_kind` / `split_mode`）
唯一来源为 `summary_metrics.csv`，脚本**不重新推导**（§11 唯一实现纪律），
只与各任务 `metrics.json` 的 `train_rounds` / `test_rounds` / `task_type` 交叉校验。

---

## 3. 三条投票基线的口径

### 3.1 hard voting（§7.4）

三个基模型 `predictions.csv` 的 `predicted_label` 多数票。3 票 5 类只有两种局面：

- 某类得 2 或 3 票 → 胜者唯一，概率不参与；
- 三个模型各投一类（1-1-1）→ 三个候选并列。

**平局规则（确定性，执行细节）**：并列（各 1 票）的类中取**三模型平均概率**
（未校准的原始 `pred_proba` 等权平均，即 §7.5 用的同一张概率表）最大者；
若该平均概率仍浮点意义下完全相等，取类别轴上**字典序最靠前**者。

实现为 `argmax(votes + 0.5 * mean_proba)`：`votes ∈ {0,1,2,3}` 相差至少 1，
而 `0.5 * mean_proba ∈ [0, 0.5]`，票数项永远压过概率项，故该式与上述规则逐字等价；
numpy `argmax` 在完全相等时返回最小下标，即字典序最靠前的类。

**实测平局频率**（全 162 任务共 289 362 个测试窗口）：
一致（3-0）83.070%，多数（2-1）16.234%，**平局（1-1-1）0.696%（2 013 个窗口）**。
平局规则会实际影响约 0.7% 的预测，不是空条款。

### 3.2 等权 soft voting（§7.5）

三个基模型 `pred_proba.csv` 的概率矩阵**等权平均**后 `argmax`。
`argmax` 平局同样由 numpy 取最小下标解决（= 类别轴字典序最靠前）。

### 3.3 校准后 soft voting（§7.6）

协议 §7.6 允许 temperature 或 isotonic，**D8 选 temperature**（执行细节，非协议条款）。

**伪 logit（已知近似）**：G0 落盘的是概率而非 logit，取 `z = log(p + 1e-12)` 作伪 logit。
softmax 对逐行常数不变，故当 `p` 是行和为 1 的 softmax 输出时该近似是精确的；
但 RF 的概率是叶节点投票频率、并非任何 softmax 的输出，对它"log p 即 logit"只是工作假设。
`1e-12` 偏置把 `p = 0` 映到 −27.63 而非 −inf，这本身也改变了该项与其它项的间距。
两点均记为**已知近似**。

**参数化**：令 `w = 1/T`，则 `softmax(z/T) = softmax(w·log p) = p^w / Σ_c p_c^w`（幂缩放）。
`w = 1`（`T = 1`）**还原**未校准概率，故 §7.6 在 `T = 1` 处退化为 §7.5；
`T > 1` 平滑（治过自信），`T < 1` 锐化。

> **订正（2026-08-29 落盘后自核）**：本节初稿写的“精确还原（实测 `max|Δ| = 3.6e-12`）”
> **不可复现**，该数已作废。按全 162 任务 × 3 基模型的测试概率表重测，订正为两个数：
>
> - 对**行归一化**后的概率：`max|Δ| = 4.0e-12` —— 即 `1e-12` 偏置自身的残差（上界约 `5e-12`）；
> - 对 `pred_proba.csv` 中**原样存储**的概率：`max|Δ| = 8.9e-08` —— 主导项**不是**偏置，
>   而是落盘概率的**存储精度舍入**：实测测试概率表 `max|行和 − 1| = 8.9e-08`
>   （OOF 表同为 `8.9e-08`），而 softmax 往返会把这点行和偏差重新归一化掉。
>
> 故严格表述是：`T = 1` 把未校准概率还原到 **`~1e-7`**（受落盘精度限制），并非逐位相等。
> §7.5 平均的是原样存储的概率，§7.6 在 `T = 1` 时平均的是同一批被重新归一化的概率，
> 二者逐元素相差 `≤ 8.9e-08` —— 远低于任何能改变 `argmax` 的量级（精确并列除外），
> 故“§7.6 在 `T = 1` 处退化为 §7.5”在结论层面成立。生成脚本 docstring §4 与
> `fit_temperature()` 内的“精确还原”一语，按本条读作“还原到 `~1e-7`”
> （**脚本未改动**：该措辞不影响任何计算，且脚本需与已落盘产物保持溯源上的逐字节对应）。

**拟合**：以训练标签的多类交叉熵（NLL）为目标，对 `u = log w` 在 `T ∈ [0.01, 100]`
对应区间上做有界 Brent 一维搜索（`scipy.optimize.minimize_scalar(method="bounded",
xatol=1e-10)`）。NLL 对 `w` 凸（固定伪 logit 上的单权重 softmax 回归），
经 `w = e^u` 单调重参数化后保持单峰，故有界 Brent 确定且稳定。
**每个任务 × 每个基模型独立拟合一个标量 T，共 162 × 3 = 486 个**，全部落盘。

**施加与合成**：把各自的 T 施加于该基模型的测试 `pred_proba` 后，
三张校准概率表**等权平均**再 `argmax`（与 §7.5 唯一的差别就是校准这一步）。

---

## 4. 校准的数据边界（审计说明）

协议 §7「校准只在源域内做」/ §8.2 / §9.2 的执行落点：

- 温度拟合**只**读 `<task>/all_features/stacking/oof_meta.csv` 的两样东西：
  - **源域 OOF 概率** `oof_<model>_<cls>`（由 §9.1 分组 CV 生成，无相邻窗口泄漏）；
  - 同文件的 **`true_label` = 训练轮次标签**。
- **测试集标签只在最后一步打分时出现**，绝不进入温度拟合。
- 代码上由 `fit_temperature(oof_proba, y_oof_idx)` 的签名结构性保证：
  该函数只接受 OOF 概率矩阵与训练标签，函数体内无任何测试集对象可达。
- 逐任务逐模型的 `temperature` / `w` / `nll_uncalibrated` / `nll_calibrated` /
  `n_oof` / `optimizer_converged` / `bound_hit` 全部落盘为
  `voting_calibration_temperatures.csv`，供审计复核。

**校准拟合的实测边界状态**：486 个拟合**全部收敛**，**0 个触界**（`bound_hit = 0`），
`n_oof` 逐行等于该任务的 `train_samples`。

---

## 5. 管道自检（硬门，D8 验收）

出任何产物**之前**，对 162 任务 × 4 模型（rf / xgboost / lightgbm / stacking）
从 `predictions.csv` 重推并与 `metrics.json` 逐一比对，容差 **1e-6（绝对）**：

| 比对项 | 检查数 | 失败 | 实测最大绝对偏差 |
|---|---|---|---|
| 5-class macro-F1 vs `macro_f1` | 648 | 0 | **0.0** |
| accuracy vs `accuracy` | 648 | 0 | **0.0** |
| 5 个逐类 F1 vs `per_class_f1` | 648×5 | 0 | **0.0** |

D8 只要求三个基模型的 macro-F1（162×3 = 486 项），此处**加严**为 4 个模型 ×
3 类比对项（macro-F1 / accuracy / 逐类 F1）——后两项是 4-class macro-F1 与最差类 F1
的直接依据，故一并纳入硬门。**648/648 全过，偏差恒为 0.0（位相同，非仅在容差内）。**

任一项超容差时脚本立即退出且**不写任何产物**（单遍计算全在内存，硬门在写盘之前）。

---

## 6. 指标口径（协议 §10）

| 列 | 口径 |
|---|---|
| `macro_f1_5class` | §10 主指标 1。`labels` 固定为上述 5 类、`average='macro'`、`zero_division=0`，与 `robust_iot_research.metric_summary` 逐字同口径 |
| `macro_f1_4class` | §10 主指标 2（去 Socket）。在**完整测试集**上算 5 个逐类 F1，再对 4 个非 Socket 类取无权平均。等价于 sklearn 传 `labels=` 非 Socket 四类 + `average='macro'`——sklearn 的 `labels` 只选取参与平均的类、**不删样本**，故真值为 Socket 的样本仍作为其它类的 FP 计入。**不过滤测试样本** |
| `worst_class_f1` | §10 主指标 3。5 个逐类 F1 的最小值 |
| `worst_class_f1_4class` | 同上但只在 4 个非 Socket 类上取最小。§10 未指明"最差类"取 5 类还是 4 类读法，故**两个都给、不替读者选**；Socket 恒为 1.0 时二者相等 |
| `accuracy` | §10 辅助指标 |
| `f1_<class>` | 5 个逐类 F1，使"最差类"的两种读法可被逐行复核 |
| `gain_vs_best_base` | §10 主指标 4 的一般化：本行 `macro_f1_5class` − 同任务三个基模型 `macro_f1_5class` 的最大值。`method='stacking'` 行即 §10 定义的 stacking gain |

§10 主指标 5（selection regret）与 6（运行开销）不在 D8 范围内：前者需要选择器（G1），
后者需要训练/推理计时，均非后处理可得。

---

## 7. 产物字典

### 7.1 `voting_baselines.csv` —— 1 296 行 = 162 任务 × 8 个 method

`method` / `method_kind` 取值：

| `method` | `method_kind` | 协议 | 性质 |
|---|---|---|---|
| `rf` / `xgboost` / `lightgbm` | `base_model` | §7.1 及其余两个基模型 | 对照行，重算自已落盘产物 |
| `stacking` | `stacking` | §7.3 | 对照行，重算自已落盘产物 |
| `best_base_posthoc` | `reference` | §7.2 事后上界 | 按 `macro_f1_5class` 选最佳基模型，并列取模型名字典序最靠前；所选模型记于 `selected_base_model` 列（**仅这 162 行**有值；其余 1 134 行该列写为空串，pandas 读回为 `NaN` —— 是“不适用”的占位，**不是**缺失数据。全表其余各列无任何 NaN） |
| `hard_voting` | `voting` | §7.4 | **本批新算** |
| `soft_voting_equal` | `voting` | §7.5 | **本批新算** |
| `soft_voting_calibrated` | `voting` | §7.6 | **本批新算** |

> `best_base_posthoc` 与三个 `base_model` 行在数值上重复（它就是其中之一）。
> 对全表做无条件平均会重复计数，聚合时请先按 `method_kind` 过滤。

其余列：`task` / `task_type` / `grid_kind` / `split_mode` / `n_sources` / `target_env` /
`train_rounds` / `test_rounds` / `n_train` / `n_test` / `n_oof`，以及第 6 节的指标列。

### 7.2 `voting_calibration_temperatures.csv` —— 486 行 = 162 任务 × 3 基模型

`task` / `model` / `temperature` / `w_inv_temperature` / `nll_uncalibrated` /
`nll_calibrated` / `n_oof` / `optimizer_converged` / `bound_hit`。
`nll_*` 均为**源域 OOF 上**的 NLL（拟合目标本身），不是测试集上的量。

### 7.3 `voting_baselines_run_metadata.json`

§19.2 要素：git commit（`b9a4ddb`，工作区 dirty）、完整命令行、解释器与
numpy / pandas / scipy / scikit-learn 版本、输入输出路径、类别轴、校准配置、
自检统计、耗时（15.6 s）。`random_seed = null`：全流程无随机数。

---

## 8. 复现

```bash
# 全量（单进程；服务器有训练任务在跑时不会抢 CPU）
~/anaconda3/envs/iotcls/bin/python code/scripts/analysis/voting_baselines.py

# 只跑硬门自检
~/anaconda3/envs/iotcls/bin/python code/scripts/analysis/voting_baselines.py --selfcheck-only

# smoke：前 4 个任务，算但不写盘
~/anaconda3/envs/iotcls/bin/python code/scripts/analysis/voting_baselines.py --limit 4 --dry-run
```

脚本在 `import numpy` **之前**把 `OMP_/OPENBLAS_/MKL_/NUMEXPR_/VECLIB_ NUM_THREADS`
钉为 1（`setdefault`，可被外部环境覆盖）。全流程单进程、无进程池，
实测全量耗时 15.4–15.6 s。无随机数：连续两次独立全量运行产出的
`voting_baselines.csv` 与 `voting_calibration_temperatures.csv` **md5 相同**。

---

## 9. 本批未做的事

- **不含**历史 110 条主线任务：`results/robust_v2/**/stacking/` 无 `oof_meta.csv` /
  `pred_proba.csv`，需 §20.3 重打分（D7 backlog）后才能补投票基线。本批只覆盖 G0 网格。
- **不做**显著性检验、不做趋势解读（§8.6：网格不用于显著性结论）。
- **不含** §10 主指标 5（selection regret）与 6（运行开销）——见第 6 节末。
- **未**修改任何既有脚本、未改动任何既有结果文件；本批只新增上述三个产物与
  `code/scripts/analysis/voting_baselines.py`。
