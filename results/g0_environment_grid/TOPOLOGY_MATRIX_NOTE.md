# TOPOLOGY_MATRIX_NOTE.md —— 同质环境×环境拓扑矩阵（G0 `|S|=1`）

**性质**：产物说明。只记口径定义、输入路径、变体、覆盖率与验证结果，**不做任何结果解读**。
**协议依据**：`docs/experiment_protocol_final.md` §8.5 第 5 条、§4.2、§4.3、§8.4、§11、§20.2。
**执行口径**：`docs/EXECUTION_PLAN_20260829.md` 决策 D4。
**生成脚本**：`code/scripts/analysis/six_env_confusion_similarity.py`
**生成日期**：2026-08-29

---

## 1. 口径定义

### 1.1 矩阵语义（D4 已定）

6×6 矩阵，**行 = 源环境 `i`，列 = 目标环境 `j`**，`i ≠ j`，共 30 个有序对
（协议 §8.5.5：`|S|=1` 的 30 个有序对构成同质的环境×环境拓扑矩阵）：

```
cell[i, j] = cpd_core.cpd_y(ref = CM(g0_iid_R{i}_{variant}),
                            tgt = CM(g0_R{i}_to_R{j}))
```

参照系按**源环境 `i`** 取，即每一行共用一个域内参照。
对角线 `i == j` 在 G0 中无对应的 `|S|=1` 任务，一律留空（NaN）。

`CPD_dir` 矩阵语义相同，只把 `cpd_y` 换成 `cpd_dir`（`min_err = 20`）。

### 1.2 指标定义与唯一实现

`CPD_y`、`CPD_dir` 的定义见协议 §4.2 / §4.3，此处不复述。
**所有 CPD 计算只经 `code/scripts/analysis/cpd_core.py`**（协议 §11 唯一实现），
本脚本不含任何私有 CPD 公式副本。

- `CPD_y` 与 `CPD_dir` 均需目标环境真实标签，**不可用于部署前**（协议 §4.1）。
- `CPD_dir` 的逐行准入门槛 `n_err ≥ 20` 取 `cpd_core.DEFAULT_MIN_ERR`（协议 §4.3），
  脚本**不提供放宽该门槛的入口**；`ref` 或 `tgt` 任一侧不达标的行整行剔除，
  无任何行达标的格置 NaN（不强算）。

### 1.3 类别轴

`['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']`
（`docs/CPD_DEFINITIONS.md` §1.2 的一致约定）。脚本读入每个 CSV 时校验轴序，
不一致即报错退出。

### 1.4 与已废弃口径的关系

本产物**替代**已废弃的 `results/robust_v2/report/six_env_off_diag_frobenius_rf.csv`
（协议 §4.4：旧 `env_mapping` 把 R2/R3/R4 指向 IID 模型、R5/R6/R7 指向 OOD 模型，矩阵不同质）。
旧文件本次**未移动、未修改**：`test_cpd_core.py::test_hist_0_1521_six_env_pairwise`
仍以它复现历史值 0.1521。改造后的脚本不再读写 `results/robust_v2/` 下的任何路径。

---

## 2. 输入路径

一律读 G0 落盘的**原始计数**混淆矩阵，不重训、不重新推理：

| 角色 | 路径 |
|---|---|
| 有序对（30） | `results/g0_environment_grid/raw_all/g0_R{i}_to_R{j}/all_features/rf/confusion_matrix.csv` |
| IID 参照（6×2 变体） | `results/g0_environment_grid/raw_all/g0_iid_R{i}_{variant}/all_features/rf/confusion_matrix.csv` |
| macro-F1 核对基准 | `results/g0_environment_grid/env_topology_matrix_rf.csv` |

模型 `rf`、特征集 `all_features`；G0 全量为 seed=42 单种子（协议 §14，`environment_grid_experiment.py`）。
环境集合 `R2..R7`。

---

## 3. 参照系变体

协议 §8.4 要求 `single_round` 的随机分层划分只能称「session 内上界」，
并必须并列报告按 `window_start` 分块的时间划分数值。因此 IID 参照出两个变体，
**两个矩阵并列，不合并、不取优**：

| 变体 | 参照任务 | 定位 | 产物 |
|---|---|---|---|
| `time_block` | `g0_iid_R{i}_time_block` | **primary**（诚实域内参照） | `env_topology_cpd_y_ref_time_block_rf.csv` |
| `random` | `g0_iid_R{i}_random` | **secondary**（与历史 IID 参照口径可比） | `env_topology_cpd_y_ref_random_rf.csv` |

两个变体均为支撑材料（协议 §8.6：网格不用于显著性结论）。

---

## 4. 产物清单

| 文件 | 内容 |
|---|---|
| `env_topology_cpd_y_ref_time_block_rf.csv` | 6×6 `CPD_y`，primary 参照。30 格全部有值 |
| `env_topology_cpd_y_ref_random_rf.csv` | 6×6 `CPD_y`，secondary 参照。30 格全部有值 |
| `env_topology_macro_f1_from_cm_rf.csv` | 由同一批 30 个 CM 重算的 5-class macro-F1 核对表 |
| `env_topology_cpd_dir_ref_time_block_rf.csv` | 6×6 `CPD_dir`，primary 参照；未达准入的格为空（NaN） |
| `env_topology_cpd_dir_coverage_rf.csv` | `CPD_dir` 逐对逐行准入明细（30 行 × 21 列） |

CSV 格式：首列 `source_env` = 源环境（行），其余 6 列为目标环境；UTF-8 with BOM；
对角线为空。

`env_topology_cpd_dir_coverage_rf.csv` 的列：
`source_env, target_env, ref_task, tgt_task, min_err, cpd_dir, is_defined,
n_included_rows, included_classes, excluded_classes, exclusion_reasons,
n_err_ref_<5 类>, n_err_tgt_<5 类>`。
`exclusion_reasons` 逐类标注剔除来源：`ref_below` / `tgt_below` / `both_below`。

---

## 5. `CPD_dir` 覆盖率（协议 §4.3，未放宽门槛、未强算）

参照系 = `g0_iid_R{i}_time_block`，`min_err = 20`。

| 项 | 值 |
|---|---|
| 有序对总数 | 30 |
| `CPD_dir` 有定义（≥1 行达标）的对 | **14 / 30** |
| 完全未定义（0 行达标，置 NaN）的对 | **16 / 30** |
| 5 行全部纳入的对 | **0 / 30** |

纳入行数分布（对数）：

| 纳入行数 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 对数 | 16 | 4 | 1 | 9 | 0 | 0 |

行槽位口径（30 对 × 5 类 = 150 个槽位）的原因分布：

| 状态 | 槽位数 |
|---|---:|
| 纳入 | 33 |
| 剔除：`ref` 侧 `n_err < 20` | 69 |
| 剔除：`tgt` 侧 `n_err < 20` | 2 |
| 剔除：两侧均 `n_err < 20` | 46 |

逐类别剔除原因（每类 30 个槽位）：

| 类别 | `both_below` | `ref_below` | `tgt_below` | 纳入 |
|---|---:|---:|---:|---:|
| Camera | 10 | 20 | 0 | 0 |
| Light_T1 | 2 | 13 | 1 | 14 |
| Light_XM | 1 | 19 | 0 | 10 |
| Sensor | 3 | 17 | 1 | 9 |
| Socket | 30 | 0 | 0 | 0 |

有定义的 14 个格分布在源环境 R4（4 格）、R6（5 格）、R7（5 格）；
源环境 R2 / R3 / R5 的全部 15 格未定义。

IID 参照逐行误分类计数（决定 `ref_below` 的直接输入）：

| 参照任务（`time_block`） | Camera | Light_T1 | Light_XM | Sensor | Socket |
|---|---:|---:|---:|---:|---:|
| `g0_iid_R2_time_block` | 0 | 11 | 13 | 1 | 0 |
| `g0_iid_R3_time_block` | 1 | 0 | 10 | 13 | 1 |
| `g0_iid_R4_time_block` | 10 | 38 | 4 | 16 | 0 |
| `g0_iid_R5_time_block` | 1 | 8 | 14 | 4 | 0 |
| `g0_iid_R6_time_block` | 3 | 48 | 73 | 66 | 0 |
| `g0_iid_R7_time_block` | 0 | 37 | 32 | 23 | 0 |

`random` 变体的 IID 参照逐行误分类计数（本次未用于 `CPD_dir` 产物，仅备查）：

| 参照任务（`random`） | Camera | Light_T1 | Light_XM | Sensor | Socket |
|---|---:|---:|---:|---:|---:|
| `g0_iid_R2_random` | 0 | 12 | 11 | 0 | 0 |
| `g0_iid_R3_random` | 2 | 0 | 11 | 9 | 0 |
| `g0_iid_R4_random` | 1 | 8 | 8 | 14 | 0 |
| `g0_iid_R5_random` | 2 | 4 | 9 | 8 | 0 |
| `g0_iid_R6_random` | 4 | 46 | 32 | 16 | 0 |
| `g0_iid_R7_random` | 0 | 20 | 29 | 19 | 0 |

---

## 6. 验证结果

### 6.1 macro-F1 核对（D4 硬门，容差 1e-6）

由 30 个 `|S|=1` 混淆矩阵重算 5-class macro-F1，与 G0 落盘的
`env_topology_matrix_rf.csv` 逐格比对：

| 项 | 结果 |
|---|---|
| 比对格数 | 30 / 30 |
| 最大绝对偏差 | **1.110e-16** |
| 判定 | **通过**（< 1e-6） |

重算口径与 `robust_iot_research.metric_summary`（第 914-920 行）严格一致：
把计数矩阵展开回 `(y_true, y_pred)` 后调用同一个
`precision_recall_fscore_support(labels=CLASS_ORDER, average='macro', zero_division=0)`，
不另写第二份 F1 公式。核对不通过时脚本以退出码 2 终止且不写任何输出文件。

对角线在重算矩阵与落盘矩阵两侧均为空，逐格核对包含该一致性检查。

### 6.2 回归测试

| 测试 | 结果 |
|---|---|
| `code/scripts/analysis/test_cpd_core.py` | **15 / 15 通过**（退出码 0），含 `test_hist_0_1521_six_env_pairwise`（历史值 0.1521 六环境重建） |
| `code/scripts/analysis/test_oof_modes.py` | **全部通过**（退出码 0），6 组共 17 项 |

---

## 7. 复现

```bash
cd ~/iot-device-classification
~/anaconda3/bin/python3 code/scripts/analysis/six_env_confusion_similarity.py
# 只核对不落盘：加 --dry-run
```

默认参数：`--results-root results/g0_environment_grid/raw_all`、
`--output-dir results/g0_environment_grid`、`--model rf`、`--feature-set all_features`。

**环境**：Python 3.10.9（`~/anaconda3/bin/python3`）、numpy 1.23.5、pandas 1.5.3、
scikit-learn 1.2.1。
**代码版本**：仓库 HEAD `042fa039b4679327b27333123d2257156ee83966`
（`six_env_confusion_similarity.py` 的本次改造尚未提交）。

本脚本只读混淆矩阵 CSV，不重训模型，因此结果与训练侧库版本无关。

## 环境追记（2026-08-29，审阅方）

本目录 5 个拓扑 CSV 已在 **canonical 分析环境 iotcls**（`code/requirements-lock.txt` 所锁，
numpy 2.4.6）下重生成。与初版（anaconda base，numpy 1.23.5）相比仅 ULP 级差异
（CPD_y time_block 变体 9/30 格，max |Δ| = 2.22e-16）；根因是 numpy 1.23→2.4 之间
`np.linalg.norm(x, "fro")` 内部归约路径变化。宏 F1 核对硬门（1e-6）两版均通过；
同解释器下逐位可复现。此后凡要求与本目录 CSV **逐位**一致的交叉验证（如 E2 验收门 b），
须在 iotcls 下执行。
