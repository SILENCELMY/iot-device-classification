# 历史 110 条 `pred_proba` 重打分记录（2026-08-29）

**性质**：只记事实与方法，不作解读。
**对应条款**：协议 §20.3（复用既有模型补概率，不重写训练逻辑）、§19.2（provenance 落盘）；
执行计划 `docs/EXECUTION_PLAN_20260829.md` D7 节。
**范围**：`results/robust_v2/raw_all/` 下 11 任务 × 2 特征集 × 5 模型 = 110 个
`(task, feature_set, model)` 组合。本文件不涉及任何其它结果根。

---

## 1. 产物

| 项 | 路径 |
|---|---|
| 逐组合概率（110 个，新增文件，未覆盖任何既有文件） | `results/robust_v2/raw_all/<task>/<feature_set>/<model>/pred_proba_rescored_20260829.csv` |
| 逐条状态汇总 | `results/robust_v2/rescore_20260829/rescore_summary_20260829.csv` |
| provenance（§19.2） | `results/robust_v2/rescore_20260829/rescore_manifest_20260829.json` |
| 生成脚本 | `code/scripts/utils/rescore_historical.py` |

列布局与 `evaluate_model` 写的 `pred_proba.csv` 一致：
`source_file, round, window_id, window_start, proba_Camera, proba_Light_T1, proba_Light_XM,
proba_Sensor, proba_Socket, true_label`（110 个文件的概率列名与列序完全相同）。

## 2. 方法（生成侧）

- 不重训。逐组合加载既有 `model.joblib` 与 `feature_columns.json`，用
  `robust_iot_research.task_data` / `feature_columns` / `clean_x` / `fit_label_encoder`
  重建测试集后调用 `predict_proba`；划分、特征、标签编码逻辑无本地副本。
- 任务定义唯一来源 `code/configs/research_experiments.json`。
- 运行参数（`robust_iot_research.parse_args` 的默认值）：
  `test_size=0.3, random_state=42, max_rows=0, n_jobs=1`。
- 概率列位按 `model.classes_` 回填，与 `evaluate_model` 同一处理。
- 生成侧四道硬门（任一不满足即跳过该组合、不写文件）：
  1. `joblib.load` 无异常且无告警；
  2. 重建测试集与 `predictions.csv` 逐行对齐（`source_file` / `window_id` /
     `window_start` / `true_label`）；
  3. `argmax(pred_proba)` 与 `predictions.csv` 的 `predicted_label` 逐行一致；
  4. 概率行和为 1（`atol=1e-6`，取协议 §22.1 P1 容差）。
- 未做：OOF 重建。折内子模型从未持久化，`model.joblib` 只有全量训练集上重拟合的
  `final_estimator_` / `named_estimators_`；本批文件只含 `pred_proba`，不含任何 OOF 数值。

## 3. 独立核对（2026-08-30）

核对脚本不复用生成脚本的任何函数或硬门，只读盘上两份 CSV 作对照
（`/tmp/d7_verify/verify_rescore.py`、`/tmp/d7_verify/spotcheck_cells.py`，未入库）。

### 3.1 全量 110 组合

| 量 | 值 |
|---|---|
| 组合数 | 110（= 盘上模型目录数，无遗漏、无多余） |
| 有 `predictions.csv` 且有重打分文件 | 110 / 110 |
| 逐行比较总行数 | 185 890 |
| `argmax` 与 `predicted_label` 不一致行数 | 0 |
| 整体一致率 | 1.000000000000 |
| 行对齐四字段（`source_file`/`window_id`/`window_start`/`true_label`）全部一致 | 110 / 110 |
| 概率含 NaN 的行 | 0 |
| 全局最小概率值 | 0.0 |
| 由重打分 `argmax` 复算的 accuracy 与 `metrics.json` 的差 | 最大 0.0（110 / 110 相等） |

按模型族（每族 22 组合、37 178 行）：

| 模型 | 不一致行数 | `max abs(rowsum-1)` | 并列最大概率(tie)行数 |
|---|---|---|---|
| rf | 0 | 2.220446e-16 | 46 |
| extra_trees | 0 | 2.220446e-16 | 55 |
| lightgbm | 0 | 7.771561e-16 | 0 |
| stacking | 0 | 7.771561e-16 | 0 |
| xgboost | 0 | 8.871939e-08 | 0 |

按特征集：`all_features` 55 组合 / 92 945 行 / 0 不一致（`n_features` = 94）；
`selected_features` 55 组合 / 92 945 行 / 0 不一致（`n_features` ∈ {20, 30, 40, 60}）。

按任务（每任务 10 组合，行数为单组合测试集行数）：
`single_round_R2` 545、`single_round_R3` 552、`single_round_R4` 556、
`loro_R2_R4_to_R3` 1 837、`loro_R2_R3_to_R4` 1 853、`loro_R3_R4_to_R2` 1 816、
`joint_R2_R3_R4` 1 652、`position_R2_R3_R4_to_R5` 1 816、
`jitter_R2_R3_R4_to_R6` 1 988、`jitter_R2_R3_R4_to_R7` 1 993、
`jitter_R2_R3_R4_to_R6_R7` 3 981；各任务不一致行数均为 0。

并列最大概率的 101 行全部出现在 rf / extra_trees（树投票比例相同），此时 `argmax`
取首个最大列；这 101 行的硬标签也与 `predictions.csv` 一致。

### 3.2 抽查：逐格比概率（不只比 argmax）

重新加载 `model.joblib`、重建测试集、重算 `predict_proba`，与盘上 CSV 逐个数值比较：

| task | feature_set | model | 行数 | `max abs(Δp)` | argmax 不一致 | 硬标签不一致 |
|---|---|---|---|---|---|---|
| loro_R2_R4_to_R3 | all_features | rf | 1 837 | 0.000e+00 | 0 | 0 |
| loro_R2_R4_to_R3 | selected_features | stacking | 1 837 | 1.110e-16 | 0 | 0 |
| single_round_R2 | selected_features | xgboost | 545 | 1.110e-16 | 0 | 0 |
| single_round_R4 | all_features | lightgbm | 556 | 1.110e-16 | 0 | 0 |
| joint_R2_R3_R4 | all_features | extra_trees | 1 652 | 0.000e+00 | 0 | 0 |
| joint_R2_R3_R4 | selected_features | rf | 1 652 | 0.000e+00 | 0 | 0 |
| jitter_R2_R3_R4_to_R6_R7 | all_features | stacking | 3 981 | 1.110e-16 | 0 | 0 |
| position_R2_R3_R4_to_R5 | selected_features | xgboost | 1 816 | 1.110e-16 | 0 | 0 |

覆盖三种任务类型（`fixed_split` / `single_round` / `joint_validation`）、五个模型族、
两个特征集。八个组合的 `joblib.load` 均无告警。

## 4. 跳过清单

**空**：110 个组合全部通过四道硬门并落盘，生成侧 `rescore_summary_20260829.csv`
的 `status` 列 110 行全为 `ok`、`n_mismatch` 全为 0、`load_warnings` 全为空。

三条边界事实：

1. `code/configs/research_experiments.json` 有 12 个 `evaluation_tasks`，第 12 个
   `filtered_R1_single_round` 在 `results/robust_v2/` 下无结果目录（该结果根只有
   `raw_all` 一个 filter mode），因此从未进入这 110 个组合的范围；
2. 本批只覆盖 `results/robust_v2/`，其它结果根（`g0_environment_grid`、
   `e1_oof_arms*`、`gpu_capacity_full_20260703` 等）不在范围内；
3. `oof_meta.csv` 未补（见 §2 末条）。

## 5. 环境与 provenance

生成运行（`rescore_manifest_20260829.json` 原样记录）：
`generated_utc` 2026-08-29T12:52:37+00:00；
`git.head` `d4473238aa5a0bc1cf29552c50a09c0f8d90b0e5`，`dirty` true；
解释器 `/home/lmy/anaconda3/envs/iotcls/bin/python3`；
python 3.11.15 / numpy 2.4.6 / pandas 3.0.3 / scikit-learn 1.9.0 / scipy 1.17.1 /
joblib 1.5.3 / xgboost 3.2.0 / lightgbm 4.6.0；
平台 Linux-5.4.0-189-generic-x86_64-with-glibc2.31。

独立核对运行：2026-08-30，同一解释器与同一组包版本，仓库 HEAD
`0f33137df3fdd7a98e190f645cbdcf16f8a15a1c`（工作区 dirty）。
