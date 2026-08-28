# 鲁棒性评估 V2 复跑：跨场景结果汇总

## 1. 各任务下的 Macro-F1

|  | extra_trees | lightgbm | rf | stacking | xgboost |
|---|---|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 0.7054 | 0.7646 | 0.7489 | 0.7722 | 0.7784 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 0.7039 | 0.7443 | 0.7235 | 0.7583 | 0.7642 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 0.7516 | 0.7890 | 0.7858 | 0.7955 | 0.7970 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 0.7521 | 0.7779 | 0.7643 | 0.7865 | 0.7880 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 0.7968 | 0.8129 | 0.8220 | 0.8185 | 0.8154 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 0.7998 | 0.8108 | 0.8048 | 0.8143 | 0.8118 |
| ('joint_R2_R3_R4', 'all_features') | 0.9003 | 0.9519 | 0.9307 | 0.9495 | 0.9471 |
| ('joint_R2_R3_R4', 'selected_features') | 0.9086 | 0.9453 | 0.9202 | 0.9446 | 0.9410 |
| ('loro_R2_R3_to_R4', 'all_features') | 0.6658 | 0.6665 | 0.6592 | 0.6644 | 0.6619 |
| ('loro_R2_R3_to_R4', 'selected_features') | 0.6557 | 0.6620 | 0.6554 | 0.6589 | 0.6669 |
| ('loro_R2_R4_to_R3', 'all_features') | 0.5888 | 0.5525 | 0.6148 | 0.5465 | 0.5332 |
| ('loro_R2_R4_to_R3', 'selected_features') | 0.5031 | 0.5195 | 0.5368 | 0.5197 | 0.5165 |
| ('loro_R3_R4_to_R2', 'all_features') | 0.7769 | 0.7923 | 0.8098 | 0.7957 | 0.7752 |
| ('loro_R3_R4_to_R2', 'selected_features') | 0.7617 | 0.7964 | 0.8084 | 0.7987 | 0.7776 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 0.6423 | 0.6594 | 0.7012 | 0.6700 | 0.6603 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 0.6088 | 0.6410 | 0.6541 | 0.6379 | 0.6432 |
| ('single_round_R2', 'all_features') | 0.9484 | 0.9557 | 0.9576 | 0.9559 | 0.9522 |
| ('single_round_R2', 'selected_features') | 0.9502 | 0.9338 | 0.9481 | 0.9394 | 0.9356 |
| ('single_round_R3', 'all_features') | 0.9566 | 0.9621 | 0.9602 | 0.9620 | 0.9638 |
| ('single_round_R3', 'selected_features') | 0.9545 | 0.9603 | 0.9655 | 0.9601 | 0.9583 |
| ('single_round_R4', 'all_features') | 0.9320 | 0.9436 | 0.9434 | 0.9436 | 0.9455 |
| ('single_round_R4', 'selected_features') | 0.9455 | 0.9418 | 0.9433 | 0.9509 | 0.9527 |

## 2. 鲁棒性均值：按场景和特征集汇总

|  | 全量特征 | 筛选特征 |
|---|---|---|
| ('cross_jitter', 'extra_trees') | 0.7513 | 0.7520 |
| ('cross_jitter', 'lightgbm') | 0.7888 | 0.7776 |
| ('cross_jitter', 'rf') | 0.7855 | 0.7642 |
| ('cross_jitter', 'stacking') | 0.7954 | 0.7864 |
| ('cross_jitter', 'xgboost') | 0.7969 | 0.7880 |
| ('cross_position', 'extra_trees') | 0.6423 | 0.6088 |
| ('cross_position', 'lightgbm') | 0.6594 | 0.6410 |
| ('cross_position', 'rf') | 0.7012 | 0.6541 |
| ('cross_position', 'stacking') | 0.6700 | 0.6379 |
| ('cross_position', 'xgboost') | 0.6603 | 0.6432 |
| ('joint', 'extra_trees') | 0.9003 | 0.9086 |
| ('joint', 'lightgbm') | 0.9519 | 0.9453 |
| ('joint', 'rf') | 0.9307 | 0.9202 |
| ('joint', 'stacking') | 0.9495 | 0.9446 |
| ('joint', 'xgboost') | 0.9471 | 0.9410 |
| ('loro', 'extra_trees') | 0.6772 | 0.6402 |
| ('loro', 'lightgbm') | 0.6704 | 0.6593 |
| ('loro', 'rf') | 0.6946 | 0.6669 |
| ('loro', 'stacking') | 0.6688 | 0.6591 |
| ('loro', 'xgboost') | 0.6567 | 0.6537 |
| ('single_round', 'extra_trees') | 0.9457 | 0.9501 |
| ('single_round', 'lightgbm') | 0.9538 | 0.9453 |
| ('single_round', 'rf') | 0.9537 | 0.9523 |
| ('single_round', 'stacking') | 0.9539 | 0.9501 |
| ('single_round', 'xgboost') | 0.9538 | 0.9489 |

## 3. Stacking 集成模型与最佳基分类器对比

|  | Stacking Macro-F1 | 最佳基分类器 Macro-F1 | 性能差值 |
|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 0.7722 | 0.7784 | -0.0062 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 0.7583 | 0.7642 | -0.0059 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 0.7955 | 0.7970 | -0.0015 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 0.7865 | 0.7880 | -0.0016 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 0.8185 | 0.8220 | -0.0035 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 0.8143 | 0.8118 | 0.0025 |
| ('joint_R2_R3_R4', 'all_features') | 0.9495 | 0.9519 | -0.0024 |
| ('joint_R2_R3_R4', 'selected_features') | 0.9446 | 0.9453 | -0.0006 |
| ('loro_R2_R3_to_R4', 'all_features') | 0.6644 | 0.6665 | -0.0021 |
| ('loro_R2_R3_to_R4', 'selected_features') | 0.6589 | 0.6669 | -0.0079 |
| ('loro_R2_R4_to_R3', 'all_features') | 0.5465 | 0.6148 | -0.0683 |
| ('loro_R2_R4_to_R3', 'selected_features') | 0.5197 | 0.5368 | -0.0171 |
| ('loro_R3_R4_to_R2', 'all_features') | 0.7957 | 0.8098 | -0.0141 |
| ('loro_R3_R4_to_R2', 'selected_features') | 0.7987 | 0.8084 | -0.0097 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 0.6700 | 0.7012 | -0.0313 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 0.6379 | 0.6541 | -0.0161 |
| ('single_round_R2', 'all_features') | 0.9559 | 0.9576 | -0.0017 |
| ('single_round_R2', 'selected_features') | 0.9394 | 0.9502 | -0.0108 |
| ('single_round_R3', 'all_features') | 0.9620 | 0.9638 | -0.0018 |
| ('single_round_R3', 'selected_features') | 0.9601 | 0.9655 | -0.0054 |
| ('single_round_R4', 'all_features') | 0.9436 | 0.9455 | -0.0019 |
| ('single_round_R4', 'selected_features') | 0.9509 | 0.9527 | -0.0018 |

## 4. 新特征族：Burst 与 Direction 的 Top-10 出现情况

新特征详细统计见同目录下：

```text
new_features_top10_stats.csv
new_features_top10_stats_complete.csv
```

复跑结果显示：11 个任务的 Top-10 重要特征中均出现了新特征，平均每个任务出现 2.73 个新特征；出现频率最高的是上行方向相关特征。

## 5. 关键观察

- **单轮 R2**：最佳 macro-F1 = **0.9576**，来自全量特征。
- **LORO 跨轮次泛化**：平均 macro-F1 = **0.6647**，范围为 0.6552-0.6807，是本项目最核心的跨 session 泛化测试。
- **Jitter 抖动场景**：平均 macro-F1 = **0.7786**，训练轮次为 R2-R4，测试轮次为 R6/R7。

- 新特征族（Burst + Direction）详见第 4 节和新特征统计 CSV。

## 6. 完整性

所有配置的实验单元均已完成。
