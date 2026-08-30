# G1_RESULTS_NOTE.md —— 判定由审阅方按 §17.1 作出，本文档不含解读

**性质**：产物说明与数表转录。只记口径、输入、验收证据与数值；
**不含任何解读、机制语言、判定或结论性表述**——§17.1 五条判据的通过 /
不通过判定由审阅方作出。

**协议依据**：`docs/experiment_protocol_final.md` §17.1 / §9.2 / §4.2 / §10 / §11 / §15 / §19.2。
**执行口径**：`docs/EXECUTION_PLAN_20260829.md` 决策 **D10**（v1.4，先写后看） + **D10 追记 v1.5**（commit `d447323`，运行前修订条件 4 口径）。
**生成脚本**：`code/scripts/analysis/g1_route_a.py`（口径完整定义见该脚本 docstring）。
**随机性**：无（全流程确定性，零重训，`random_seed = null`）。
**运行记录**：`provenance.json`（§19.2 五要素）。

---

## 1. 口径（逐条对应 D10）

- 任务域 = G0 网格的 150 个 OOD 任务；任务定义唯一来源 = `environment_grid_experiment.build_task_grid()`。
- 外层 fold `e_out ∈ {R2..R7}`：外层评估任务 = `target_env == e_out` 的 25 个，每任务只评一次；内层池 = 严格全含任务（`e_out` 既不作源也不作目标）= 5 目标 × 14 源组合 = 70。
- 候选 = `rf` / `stacking`（G0 落盘 = §9.1 grouped OOF，B 口径）/ `soft_voting`（变体 `soft_voting_equal` 或 `soft_voting_calibrated`，由内层与阈值联合选定）。
- **候选 F1 一律读落盘值，本次运行不重算任何 F1**：`rf` / `stacking` 取 `metrics.json::macro_f1`；两个 soft 变体取 `voting_baselines.csv::macro_f1_5class`。指标 = 5-class macro-F1（§10 主指标 1）。
- `UDS` 只经 `cpd_core.uds`（§11）：源侧 = `stacking/oof_meta.csv` 三基模型 OOF 概率 `argmax`；目标侧 = 三基模型 `predictions.csv` 的 `predicted_label`；6 个有序模型对取均值。
- 选择器 = `UDS` 单调双阈值，风险序 `[stacking, soft_voting, rf]`：`UDS ≤ t1 → stacking`；`t1 < UDS ≤ t2 → soft_voting`；`UDS > t2 → rf`。
- `regret = F1(该任务三候选中最大) − F1(所选)`；聚合主口径 = 环境等权，任务级并报。
- 条件 4 的 regret 门槛（D10 追记 v1.5）：primary `τ_repro = 0.002`（取自登记表 E1-G0-GRID 行实测的跨线程拓扑 stacking 复现界）；`τ = 0` 的原严值计数以 `_tau0_parallel` 后缀并列报告。

## 2. 验收硬门

| 门 | 内容 | 规模 | 结果 |
|---|---|---|---|
| `1a_candidate_f1_bitwise_vs_disk` | rf / stacking 的 metrics.json::macro_f1 与 voting_baselines.csv::macro_f1_5class 逐位比对 | 300 | n_mismatch = 0 |
| `1b_grid_metadata_three_way` | 任务名 / target_env / n_sources / train_rounds / test_rounds 在生成器、metrics.json、voting_baselines.csv 三源一致 | 450 | n_mismatch = 0 |
| `1c_soft_voting_macro_internal_consistency` | 两个 soft 变体的 macro_f1_5class 与其 5 个逐类 F1 的算术平均之差 | 300 | max_abs_dev = 0.000e+00（tol 1e-12） |
| `1d_f1_table_complete` | 150 任务 × 4 个落盘 F1 全部存在且非缺失 | 600 | 150 行 × 4 列 = 600 个 F1，缺失 0 |
| `3_fold_structure_assertions` | §9.2 fold 自检：外层 6×25 覆盖 150 不重不漏；内层 6×70；逐 fold 断言 e_out 不出现在任何内层任务的 train_rounds / test_rounds | 903 | 全部 assert 通过 |
| `4_leakage_static_audit` | 选择路径 12 个函数的签名无标签参数、函数体无标签词元命中 | 12 | n_label_tokens = 0 |
| `5_tiebreak_reading_ambiguity` | D10 并列打破 ② '更保守（阈值更靠 RF 侧）' 的两种读法（(t1,t2) 更大 / 更小）必须给出逐位一致的外层**结果**（OUTCOME_COLS：选择、F1、oracle、regret 及其派生）；阈值参数本身的差异并列记录、不构成失败 | 20 | 外层结果列一致 = True；选择改变的外层任务 = 0/150；参数不同的 fold = ['R3', 'R5'] |

> 双跑 md5 一致性（验收门 ②）由脚本外的两次独立运行判定，证据记入 `provenance.json` 的 `outputs` 与运行报告。

## 3. 各 fold 内层选中的 (变体, t1, t2)

| fold `e_out` | 变体 | t1 | t2 | t1=t2 | 内层平均 regret（环境等权） | 内层平均 regret（任务级） | 内层最差环境 regret | 内层最差环境 | 网格点数 | 目标值并列数 | 最差环境后并列数 | 打破阶段 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 | `soft_voting_equal` | 0.137592 | 0.213784 | 否 | 0.023792 | 0.023792 | 0.032724 | R4 | 4830 | 1 | 1 | `unique_at_objective` |
| R3 | `soft_voting_calibrated` | 0.098037 | 0.242142 | 否 | 0.008134 | 0.008134 | 0.016536 | R6 | 4830 | 2 | 2 | `stage2_threshold_larger` |
| R4 | `soft_voting_calibrated` | 0.129935 | 0.212721 | 否 | 0.021591 | 0.021591 | 0.030825 | R6 | 4830 | 1 | 1 | `unique_at_objective` |
| R5 | `soft_voting_equal` | 0.104193 | 0.201665 | 否 | 0.015564 | 0.015564 | 0.025073 | R6 | 4830 | 2 | 2 | `stage2_threshold_larger` |
| R6 | `soft_voting_equal` | 0.086050 | 0.130770 | 否 | 0.018522 | 0.018522 | 0.029377 | R7 | 4830 | 1 | 1 | `unique_at_objective` |
| R7 | `soft_voting_calibrated` | 0.160731 | 0.174061 | 否 | 0.021891 | 0.021891 | 0.032715 | R6 | 4830 | 1 | 1 | `unique_at_objective` |

## 4. UDS 分布（逐 fold）

| fold `e_out` | 内层 n | 内层 UDS min | median | max | 外层 n | 外层 UDS min | median | max | 内层选择 stacking / soft / rf |
|---|---|---|---|---|---|---|---|---|---|
| R2 | 70 | 0.048974 | 0.571985 | 1.144370 | 25 | 0.112627 | 0.389225 | 1.012504 | 2 / 6 / 62 |
| R3 | 70 | 0.048974 | 0.561386 | 1.098147 | 25 | 0.123270 | 0.286189 | 1.019751 | 2 / 13 / 55 |
| R4 | 70 | 0.048974 | 0.531302 | 1.144370 | 25 | 0.164976 | 0.541335 | 1.129869 | 3 / 11 / 56 |
| R5 | 70 | 0.048974 | 0.538554 | 1.129869 | 25 | 0.156486 | 0.498595 | 1.144370 | 2 / 10 / 58 |
| R6 | 70 | 0.059474 | 0.467293 | 1.144370 | 25 | 0.151914 | 0.458256 | 1.021871 | 1 / 1 / 68 |
| R7 | 70 | 0.112627 | 0.475761 | 1.021871 | 25 | 0.048974 | 0.425443 | 1.098147 | 2 / 2 / 66 |

## 5. 逐环境外层结果

| 目标环境 | n | 变体 | 选择器平均 regret | always-RF | always-Stacking | always-soft | 选择 stacking / soft / rf | 最大单任务 regret | win/loss/tie vs RF | win/loss/tie vs Stacking |
|---|---|---|---|---|---|---|---|---|---|---|
| R2 | 25 | `soft_voting_equal` | 0.016680 | 0.011371 | 0.084224 | 0.072757 | 1 / 3 / 21 | 0.096309 | 0/4/21 | 18/6/1 |
| R3 | 25 | `soft_voting_calibrated` | 0.022936 | 0.011292 | 0.058838 | 0.062773 | 0 / 8 / 17 | 0.114465 | 1/7/17 | 15/10/0 |
| R4 | 25 | `soft_voting_calibrated` | 0.018450 | 0.020298 | 0.079668 | 0.058193 | 0 / 5 / 20 | 0.147019 | 5/0/20 | 17/8/0 |
| R5 | 25 | `soft_voting_equal` | 0.022159 | 0.019866 | 0.069562 | 0.055287 | 0 / 2 / 23 | 0.071598 | 0/2/23 | 18/7/0 |
| R6 | 25 | `soft_voting_equal` | 0.031823 | 0.031823 | 0.090815 | 0.010907 | 0 / 0 / 25 | 0.096733 | 0/0/25 | 20/5/0 |
| R7 | 25 | `soft_voting_calibrated` | 0.036115 | 0.026865 | 0.092412 | 0.015635 | 5 / 1 / 19 | 0.175106 | 2/3/20 | 15/5/5 |

## 6. §17.1 五条判据的原始比较量（不含判定）

| 判据 | 量 | 数值 | 文本 |
|---|---|---|---|
| scope | `n_outer_tasks` | 150.000000 |  |
| scope | `n_outer_folds` | 6.000000 |  |
| scope | `f1_metric` |  | 5-class macro-F1 (§10 主指标 1；落盘值，未重算) |
| scope | `aggregation_primary` |  | 环境等权（先每环境均值再 6 环境平均，§15.1） |
| cond1 | `mean_regret_selector_env_equal` | 0.024694 |  |
| cond1 | `mean_regret_selector_task_level` | 0.024694 |  |
| cond2 | `mean_regret_selector_env_R2` | 0.016680 |  |
| cond2 | `mean_regret_selector_env_R3` | 0.022936 |  |
| cond2 | `mean_regret_selector_env_R4` | 0.018450 |  |
| cond2 | `mean_regret_selector_env_R5` | 0.022159 |  |
| cond2 | `mean_regret_selector_env_R6` | 0.031823 |  |
| cond2 | `mean_regret_selector_env_R7` | 0.036115 |  |
| cond2 | `worst_env_mean_regret_selector` | 0.036115 |  |
| cond2 | `worst_env_name_selector` |  | R7 |
| cond2 | `max_single_task_regret_selector` | 0.175106 |  |
| cond3 | `mean_regret_always_rf_env_equal` | 0.020253 |  |
| cond3 | `mean_regret_always_stacking_env_equal` | 0.079253 |  |
| cond3 | `mean_regret_always_soft_voting_env_equal` | 0.045925 |  |
| cond3 | `mean_regret_always_rf_task_level` | 0.020253 |  |
| cond3 | `mean_regret_always_stacking_task_level` | 0.079253 |  |
| cond3 | `mean_regret_always_soft_voting_task_level` | 0.045925 |  |
| cond3 | `delta_selector_minus_always_rf_env_equal` | 0.004441 |  |
| cond3 | `delta_selector_minus_always_stacking_env_equal` | -0.054560 |  |
| cond3 | `worst_env_mean_regret_always_rf` | 0.031823 |  |
| cond3 | `worst_env_mean_regret_always_stacking` | 0.092412 |  |
| cond3 | `n_tasks_selector_f1_gt_always_rf` | 8.000000 |  |
| cond3 | `n_tasks_selector_f1_lt_always_rf` | 16.000000 |  |
| cond3 | `n_tasks_selector_f1_gt_always_stacking` | 103.000000 |  |
| cond3 | `n_tasks_selector_f1_lt_always_stacking` | 41.000000 |  |
| cond4 | `n_tasks_selected_nonrf` | 25.000000 |  |
| cond4 | `n_tasks_selected_stacking` | 6.000000 |  |
| cond4 | `n_tasks_selected_soft_voting` | 19.000000 |  |
| cond4 | `n_tasks_selected_rf` | 125.000000 |  |
| cond4 | `tau_repro` | 0.002000 | D10 追记 v1.5：条件 4 primary 口径的 regret 门槛（= 登记表 E1-G0-GRID 实测复现界） |
| cond4 | `n_tasks_nonrf_and_regret_le_tau_repro_primary` | 7.000000 |  |
| cond4 | `n_tasks_nonrf_and_regret_zero_tau0_parallel` | 6.000000 |  |
| cond4 | `n_tasks_nonrf_and_f1_ge_rf_lenient` | 9.000000 |  |
| cond4 | `n_envs_with_any_nonrf_selection` | 5.000000 |  |
| cond4 | `n_envs_with_tau_repro_hit_primary` | 2.000000 |  |
| cond4 | `n_envs_with_tau0_hit_parallel` | 2.000000 |  |
| cond4 | `n_envs_with_lenient_hit` | 3.000000 |  |
| cond4 | `envs_with_tau_repro_hit_primary` |  | R4;R7 |
| cond4 | `envs_with_tau0_hit_parallel` |  | R4;R7 |
| cond4 | `oracle_distribution_rf_only` | 63.000000 |  |
| cond4 | `oracle_distribution_stacking_only` | 37.000000 |  |
| cond4 | `oracle_distribution_soft_voting_only` | 49.000000 |  |
| cond4 | `oracle_distribution_ties` | 1.000000 |  |
| cond4 | `n_tasks_top2_candidate_gap_lt_tau_repro_selected_variant` | 10.000000 |  |
| cond4 | `median_top2_candidate_gap_selected_variant` | 0.026729 |  |
| cond4 | `min_top2_candidate_gap_selected_variant` | 0.000000 |  |
| cond4 | `n_tasks_top2_candidate_gap_lt_tau_repro_calibrated` | 12.000000 |  |
| cond4 | `median_top2_candidate_gap_calibrated` | 0.027092 |  |
| cond4 | `min_top2_candidate_gap_calibrated` | 0.000000 |  |
| cond4 | `n_tasks_top2_candidate_gap_lt_tau_repro_equal` | 8.000000 |  |
| cond4 | `median_top2_candidate_gap_equal` | 0.028917 |  |
| cond4 | `min_top2_candidate_gap_equal` | 0.000000 |  |
| cond5 | `uds_signature` |  | uds(pred_src_oof, pred_tgt, *, class_order=None) -> 'float' |
| cond5 | `n_label_tokens_on_selection_path` | 0.000000 |  |
| cond5 | `n_selection_path_functions_audited` | 12.000000 |  |
| cond5 | `audit_document` |  | g1_leakage_audit.md |

## 7. 产物字典

| 文件 | 内容 |
|---|---|
| `g1_task_detail.csv` | 150 个外层任务逐条：fold、UDS、四个落盘 F1、fold 变体、oracle、所选候选、regret 与三条固定策略 regret |
| `g1_env_summary.csv` | 六个目标环境：平均 regret、选择分布、win/loss/tie vs always-RF 与 always-Stacking、UDS 分位 |
| `g1_overall.csv` | §17.1 五条判据的原始比较量（长表：criterion / quantity / value） |
| `g1_fold_params.csv` | 各 fold 内层选中的 (变体, t1, t2)、内层 regret、并列打破记录、UDS 分位 |
| `g1_uds.csv` | 150 任务的 UDS（选择路径唯一输入量） |
| `g1_acceptance.json` | 验收门逐条证据 |
| `g1_leakage_audit.md` | §17.1 条件 5 的结构性 + 静态审计记录 |
| `provenance.json` | §19.2 五要素 + 输入清单 md5 + 输出 md5 |

## 8. 规格歧义的登记（D10 并列打破 ②）

D10 原文 "② 更保守（阈值更靠 RF 侧）" 中，"更保守"（阈值更小 → 更多任务落入 RF）与 "阈值更靠 RF 侧"（阈值数值更大）在本规则下指向相反方向。本次运行以 `(t1, t2)` 字典序**更大**为 primary，并完整跑了另一读法作为硬门。

- 外层结果列（选择 / F1 / oracle / regret 及派生）在两读法下逐位一致：**True**；
- 选择发生改变的外层任务数：**0/150**；
- 被记录的阈值参数存在差异的 fold（两读法参数在 `g1_fold_params.csv` 的 `t1_alt_reading` / `t2_alt_reading` / `soft_variant_alt_reading` 列并列落盘）：
  - `R3`：primary = (`soft_voting_calibrated`, t1=0.098037, t2=0.242142)；alternative = (`soft_voting_calibrated`, t1=0.054224, t2=0.242142)
  - `R5`：primary = (`soft_voting_equal`, t1=0.104193, t2=0.201665)；alternative = (`soft_voting_equal`, t1=0.054224, t2=0.201665)

