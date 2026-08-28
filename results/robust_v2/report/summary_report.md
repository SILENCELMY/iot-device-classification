# 鲁棒性评估 V2：跨场景结果汇总

## 1. 各任务下的 Macro-F1

| 评估任务场景 | 特征子集 | extra_trees | lightgbm | rf | stacking | xgboost |
|---|---|---|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 全量特征 | 0.7054 | 0.7641 | 0.7489 | 0.7731 | 0.7784 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 筛选特征 | 0.7039 | 0.7545 | 0.7235 | 0.7593 | 0.7642 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 全量特征 | 0.7516 | 0.7867 | 0.7858 | 0.7969 | 0.7970 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 筛选特征 | 0.7521 | 0.7819 | 0.7643 | 0.7852 | 0.7880 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 全量特征 | 0.7968 | 0.8088 | 0.8220 | 0.8204 | 0.8154 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 筛选特征 | 0.7998 | 0.8091 | 0.8048 | 0.8109 | 0.8118 |
| ('joint_R2_R3_R4', 'all_features') | 全量特征 | 0.9003 | 0.9519 | 0.9307 | 0.9489 | 0.9471 |
| ('joint_R2_R3_R4', 'selected_features') | 筛选特征 | 0.9086 | 0.9434 | 0.9202 | 0.9428 | 0.9410 |
| ('loro_R2_R3_to_R4', 'all_features') | 全量特征 | 0.6658 | 0.6662 | 0.6592 | 0.6634 | 0.6619 |
| ('loro_R2_R3_to_R4', 'selected_features') | 筛选特征 | 0.6557 | 0.6688 | 0.6554 | 0.6602 | 0.6669 |
| ('loro_R2_R4_to_R3', 'all_features') | 全量特征 | 0.5888 | 0.5527 | 0.6148 | 0.5455 | 0.5332 |
| ('loro_R2_R4_to_R3', 'selected_features') | 筛选特征 | 0.5031 | 0.5207 | 0.5368 | 0.5204 | 0.5165 |
| ('loro_R3_R4_to_R2', 'all_features') | 全量特征 | 0.7769 | 0.7934 | 0.8098 | 0.7905 | 0.7752 |
| ('loro_R3_R4_to_R2', 'selected_features') | 筛选特征 | 0.7617 | 0.7869 | 0.8084 | 0.7918 | 0.7776 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 全量特征 | 0.6423 | 0.6584 | 0.7012 | 0.6678 | 0.6603 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 筛选特征 | 0.6088 | 0.6507 | 0.6541 | 0.6365 | 0.6432 |
| ('single_round_R2', 'all_features') | 全量特征 | 0.9484 | 0.9539 | 0.9576 | 0.9559 | 0.9522 |
| ('single_round_R2', 'selected_features') | 筛选特征 | 0.9502 | 0.9429 | 0.9481 | 0.9432 | 0.9356 |
| ('single_round_R3', 'all_features') | 全量特征 | 0.9566 | 0.9638 | 0.9602 | 0.9620 | 0.9638 |
| ('single_round_R3', 'selected_features') | 筛选特征 | 0.9545 | 0.9566 | 0.9655 | 0.9565 | 0.9583 |
| ('single_round_R4', 'all_features') | 全量特征 | 0.9320 | 0.9436 | 0.9434 | 0.9436 | 0.9455 |
| ('single_round_R4', 'selected_features') | 筛选特征 | 0.9455 | 0.9490 | 0.9433 | 0.9491 | 0.9527 |

## 1.2 各任务下的准确率

| 评估任务场景 | 特征子集 | extra_trees | lightgbm | rf | stacking | xgboost |
|---|---|---|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 全量特征 | 0.7037 | 0.7606 | 0.7455 | 0.7691 | 0.7757 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 筛选特征 | 0.7007 | 0.7495 | 0.7188 | 0.7545 | 0.7606 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 全量特征 | 0.7506 | 0.7850 | 0.7832 | 0.7945 | 0.7950 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 筛选特征 | 0.7506 | 0.7790 | 0.7611 | 0.7822 | 0.7860 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 全量特征 | 0.7973 | 0.8093 | 0.8209 | 0.8199 | 0.8144 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 筛选特征 | 0.8003 | 0.8083 | 0.8033 | 0.8098 | 0.8113 |
| ('joint_R2_R3_R4', 'all_features') | 全量特征 | 0.9007 | 0.9522 | 0.9310 | 0.9492 | 0.9473 |
| ('joint_R2_R3_R4', 'selected_features') | 筛选特征 | 0.9092 | 0.9437 | 0.9207 | 0.9431 | 0.9413 |
| ('loro_R2_R3_to_R4', 'all_features') | 全量特征 | 0.7145 | 0.7156 | 0.7070 | 0.7134 | 0.7124 |
| ('loro_R2_R3_to_R4', 'selected_features') | 筛选特征 | 0.7113 | 0.7242 | 0.7113 | 0.7178 | 0.7232 |
| ('loro_R2_R4_to_R3', 'all_features') | 全量特征 | 0.6102 | 0.6015 | 0.6336 | 0.5895 | 0.5803 |
| ('loro_R2_R4_to_R3', 'selected_features') | 筛选特征 | 0.5661 | 0.5939 | 0.5825 | 0.5868 | 0.5830 |
| ('loro_R3_R4_to_R2', 'all_features') | 全量特征 | 0.7770 | 0.7968 | 0.8084 | 0.7935 | 0.7786 |
| ('loro_R3_R4_to_R2', 'selected_features') | 筛选特征 | 0.7627 | 0.7891 | 0.8084 | 0.7946 | 0.7808 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 全量特征 | 0.6547 | 0.6768 | 0.7115 | 0.6834 | 0.6856 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 筛选特征 | 0.6189 | 0.6746 | 0.6663 | 0.6591 | 0.6685 |
| ('single_round_R2', 'all_features') | 全量特征 | 0.9486 | 0.9541 | 0.9578 | 0.9560 | 0.9523 |
| ('single_round_R2', 'selected_features') | 筛选特征 | 0.9505 | 0.9431 | 0.9486 | 0.9431 | 0.9358 |
| ('single_round_R3', 'all_features') | 全量特征 | 0.9565 | 0.9638 | 0.9601 | 0.9620 | 0.9638 |
| ('single_round_R3', 'selected_features') | 筛选特征 | 0.9547 | 0.9565 | 0.9656 | 0.9565 | 0.9583 |
| ('single_round_R4', 'all_features') | 全量特征 | 0.9335 | 0.9442 | 0.9442 | 0.9442 | 0.9460 |
| ('single_round_R4', 'selected_features') | 筛选特征 | 0.9460 | 0.9496 | 0.9442 | 0.9496 | 0.9532 |

## 1.3 各任务下的宏平均精确率

| 评估任务场景 | 特征子集 | extra_trees | lightgbm | rf | stacking | xgboost |
|---|---|---|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 全量特征 | 0.7249 | 0.7935 | 0.7646 | 0.7868 | 0.7870 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 筛选特征 | 0.7206 | 0.7828 | 0.7399 | 0.7749 | 0.7782 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 全量特征 | 0.7623 | 0.8096 | 0.7950 | 0.8075 | 0.8048 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 筛选特征 | 0.7609 | 0.8008 | 0.7736 | 0.7968 | 0.8000 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 全量特征 | 0.8006 | 0.8257 | 0.8262 | 0.8285 | 0.8228 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 筛选特征 | 0.8030 | 0.8198 | 0.8091 | 0.8189 | 0.8219 |
| ('joint_R2_R3_R4', 'all_features') | 全量特征 | 0.9009 | 0.9521 | 0.9310 | 0.9492 | 0.9476 |
| ('joint_R2_R3_R4', 'selected_features') | 筛选特征 | 0.9088 | 0.9436 | 0.9206 | 0.9432 | 0.9414 |
| ('loro_R2_R3_to_R4', 'all_features') | 全量特征 | 0.7275 | 0.6946 | 0.7277 | 0.7084 | 0.6919 |
| ('loro_R2_R3_to_R4', 'selected_features') | 筛选特征 | 0.6272 | 0.6735 | 0.6520 | 0.6494 | 0.6810 |
| ('loro_R2_R4_to_R3', 'all_features') | 全量特征 | 0.5943 | 0.5428 | 0.6225 | 0.5263 | 0.5073 |
| ('loro_R2_R4_to_R3', 'selected_features') | 筛选特征 | 0.4691 | 0.4861 | 0.5154 | 0.4874 | 0.4822 |
| ('loro_R3_R4_to_R2', 'all_features') | 全量特征 | 0.7834 | 0.7994 | 0.8143 | 0.7965 | 0.7844 |
| ('loro_R3_R4_to_R2', 'selected_features') | 筛选特征 | 0.7680 | 0.7941 | 0.8112 | 0.7982 | 0.7852 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 全量特征 | 0.6412 | 0.6538 | 0.7065 | 0.6679 | 0.6476 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 筛选特征 | 0.6096 | 0.6444 | 0.6594 | 0.6311 | 0.6308 |
| ('single_round_R2', 'all_features') | 全量特征 | 0.9483 | 0.9542 | 0.9575 | 0.9562 | 0.9522 |
| ('single_round_R2', 'selected_features') | 筛选特征 | 0.9502 | 0.9430 | 0.9488 | 0.9436 | 0.9357 |
| ('single_round_R3', 'all_features') | 全量特征 | 0.9569 | 0.9641 | 0.9605 | 0.9623 | 0.9640 |
| ('single_round_R3', 'selected_features') | 筛选特征 | 0.9552 | 0.9569 | 0.9655 | 0.9566 | 0.9585 |
| ('single_round_R4', 'all_features') | 全量特征 | 0.9336 | 0.9446 | 0.9438 | 0.9445 | 0.9470 |
| ('single_round_R4', 'selected_features') | 筛选特征 | 0.9477 | 0.9498 | 0.9447 | 0.9502 | 0.9543 |

## 1.4 各任务下的宏平均召回率

| 评估任务场景 | 特征子集 | extra_trees | lightgbm | rf | stacking | xgboost |
|---|---|---|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 全量特征 | 0.7034 | 0.7612 | 0.7454 | 0.7695 | 0.7760 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 筛选特征 | 0.7003 | 0.7500 | 0.7187 | 0.7548 | 0.7609 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 全量特征 | 0.7496 | 0.7849 | 0.7826 | 0.7943 | 0.7948 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 筛选特征 | 0.7496 | 0.7787 | 0.7604 | 0.7819 | 0.7857 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 全量特征 | 0.7959 | 0.8086 | 0.8198 | 0.8191 | 0.8135 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 筛选特征 | 0.7990 | 0.8075 | 0.8022 | 0.8090 | 0.8105 |
| ('joint_R2_R3_R4', 'all_features') | 全量特征 | 0.9002 | 0.9519 | 0.9307 | 0.9489 | 0.9471 |
| ('joint_R2_R3_R4', 'selected_features') | 筛选特征 | 0.9088 | 0.9434 | 0.9203 | 0.9428 | 0.9410 |
| ('loro_R2_R3_to_R4', 'all_features') | 全量特征 | 0.7130 | 0.7142 | 0.7058 | 0.7122 | 0.7108 |
| ('loro_R2_R3_to_R4', 'selected_features') | 筛选特征 | 0.7098 | 0.7231 | 0.7101 | 0.7167 | 0.7220 |
| ('loro_R2_R4_to_R3', 'all_features') | 全量特征 | 0.6070 | 0.5979 | 0.6305 | 0.5859 | 0.5766 |
| ('loro_R2_R4_to_R3', 'selected_features') | 筛选特征 | 0.5621 | 0.5899 | 0.5786 | 0.5828 | 0.5790 |
| ('loro_R3_R4_to_R2', 'all_features') | 全量特征 | 0.7770 | 0.7969 | 0.8084 | 0.7936 | 0.7787 |
| ('loro_R3_R4_to_R2', 'selected_features') | 筛选特征 | 0.7627 | 0.7891 | 0.8084 | 0.7947 | 0.7809 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 全量特征 | 0.6553 | 0.6773 | 0.7119 | 0.6839 | 0.6862 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 筛选特征 | 0.6196 | 0.6752 | 0.6668 | 0.6598 | 0.6692 |
| ('single_round_R2', 'all_features') | 全量特征 | 0.9486 | 0.9541 | 0.9578 | 0.9560 | 0.9523 |
| ('single_round_R2', 'selected_features') | 筛选特征 | 0.9505 | 0.9431 | 0.9486 | 0.9431 | 0.9358 |
| ('single_round_R3', 'all_features') | 全量特征 | 0.9563 | 0.9637 | 0.9600 | 0.9618 | 0.9636 |
| ('single_round_R3', 'selected_features') | 筛选特征 | 0.9544 | 0.9564 | 0.9655 | 0.9564 | 0.9582 |
| ('single_round_R4', 'all_features') | 全量特征 | 0.9324 | 0.9434 | 0.9435 | 0.9434 | 0.9453 |
| ('single_round_R4', 'selected_features') | 筛选特征 | 0.9452 | 0.9489 | 0.9434 | 0.9489 | 0.9525 |

## 2. 鲁棒性均值：按场景和特征集汇总

| 场景分类与模型 | 全量特征 | 筛选特征 |
|---|---|---|
| ('cross_jitter', 'extra_trees') | 0.7513 | 0.7520 |
| ('cross_jitter', 'lightgbm') | 0.7866 | 0.7818 |
| ('cross_jitter', 'rf') | 0.7855 | 0.7642 |
| ('cross_jitter', 'stacking') | 0.7968 | 0.7851 |
| ('cross_jitter', 'xgboost') | 0.7969 | 0.7880 |
| ('cross_position', 'extra_trees') | 0.6423 | 0.6088 |
| ('cross_position', 'lightgbm') | 0.6584 | 0.6507 |
| ('cross_position', 'rf') | 0.7012 | 0.6541 |
| ('cross_position', 'stacking') | 0.6678 | 0.6365 |
| ('cross_position', 'xgboost') | 0.6603 | 0.6432 |
| ('joint', 'extra_trees') | 0.9003 | 0.9086 |
| ('joint', 'lightgbm') | 0.9519 | 0.9434 |
| ('joint', 'rf') | 0.9307 | 0.9202 |
| ('joint', 'stacking') | 0.9489 | 0.9428 |
| ('joint', 'xgboost') | 0.9471 | 0.9410 |
| ('loro', 'extra_trees') | 0.6772 | 0.6402 |
| ('loro', 'lightgbm') | 0.6708 | 0.6588 |
| ('loro', 'rf') | 0.6946 | 0.6669 |
| ('loro', 'stacking') | 0.6665 | 0.6575 |
| ('loro', 'xgboost') | 0.6567 | 0.6537 |
| ('single_round', 'extra_trees') | 0.9457 | 0.9501 |
| ('single_round', 'lightgbm') | 0.9538 | 0.9495 |
| ('single_round', 'rf') | 0.9537 | 0.9523 |
| ('single_round', 'stacking') | 0.9538 | 0.9496 |
| ('single_round', 'xgboost') | 0.9538 | 0.9489 |

## 3. Stacking 集成模型与最佳基分类器对比

| 评估任务场景与特征模式 | Stacking Macro-F1 | 最佳基分类器 Macro-F1 | 性能差值 |
|---|---|---|---|
| ('jitter_R2_R3_R4_to_R6', 'all_features') | 0.7731 | 0.7784 (xgboost) | -0.0053 |
| ('jitter_R2_R3_R4_to_R6', 'selected_features') | 0.7593 | 0.7642 (xgboost) | -0.0049 |
| ('jitter_R2_R3_R4_to_R6_R7', 'all_features') | 0.7969 | 0.7970 (xgboost) | -0.0001 |
| ('jitter_R2_R3_R4_to_R6_R7', 'selected_features') | 0.7852 | 0.7880 (xgboost) | -0.0028 |
| ('jitter_R2_R3_R4_to_R7', 'all_features') | 0.8204 | 0.8220 (rf) | -0.0015 |
| ('jitter_R2_R3_R4_to_R7', 'selected_features') | 0.8109 | 0.8118 (xgboost) | -0.0009 |
| ('joint_R2_R3_R4', 'all_features') | 0.9489 | 0.9519 (lightgbm) | -0.0030 |
| ('joint_R2_R3_R4', 'selected_features') | 0.9428 | 0.9434 (lightgbm) | -0.0006 |
| ('loro_R2_R3_to_R4', 'all_features') | 0.6634 | 0.6662 (lightgbm) | -0.0028 |
| ('loro_R2_R3_to_R4', 'selected_features') | 0.6602 | 0.6688 (lightgbm) | -0.0086 |
| ('loro_R2_R4_to_R3', 'all_features') | 0.5455 | 0.6148 (rf) | **-0.0693** ⚠️ (严重崩盘) |
| ('loro_R2_R4_to_R3', 'selected_features') | 0.5204 | 0.5368 (rf) | -0.0164 |
| ('loro_R3_R4_to_R2', 'all_features') | 0.7905 | 0.8098 (rf) | -0.0193 |
| ('loro_R3_R4_to_R2', 'selected_features') | 0.7918 | 0.8084 (rf) | -0.0167 |
| ('position_R2_R3_R4_to_R5', 'all_features') | 0.6678 | 0.7012 (rf) | -0.0334 |
| ('position_R2_R3_R4_to_R5', 'selected_features') | 0.6365 | 0.6541 (rf) | -0.0176 |
| ('single_round_R2', 'all_features') | 0.9559 | 0.9576 (rf) | -0.0017 |
| ('single_round_R2', 'selected_features') | 0.9432 | 0.9502 (extra_trees) | -0.0070 |
| ('single_round_R3', 'all_features') | 0.9620 | 0.9638 (lightgbm/xgboost) | -0.0018 |
| ('single_round_R3', 'selected_features') | 0.9565 | 0.9655 (rf) | -0.0090 |
| ('single_round_R4', 'all_features') | 0.9436 | 0.9455 (xgboost) | -0.0019 |
| ('single_round_R4', 'selected_features') | 0.9491 | 0.9527 (xgboost) | -0.0037 |

## 4. 新加入的特征簇表现：Burst 与 Direction

### 4.1 新特征在 Top-10 重要特征中的表现

**核心发现**: ✅ **所有 11 个任务的 top-10 重要特征中都有新特征出现！**

### 4.2 统计摘要

| 指标 | 数值 |
|------|------|
| 任务总数 | 11 |
| 所有任务都有新特征在 top-10 | ✅ 11/11 (100%) |
| 平均每任务 top-10 中新特征数 | 2.73 个 |
| 新特征在 top-10 中总出现次数 | 30 次 |
| 新特征种类数（出现过的） | 5 种 |

### 4.3 最重要的新特征（跨所有任务）

| 排名 | 新特征 | 出现频率 | 占比 | 特征类型 | 物理意义 |
|------|--------|---------|------|---------|----------|
| 🥇 | **up_len_p50** | 10/11 | 90.9% | 方向-上行 | 上行包长度中位数 |
| 🥈 | **up_packet_ratio** | 9/11 | 81.8% | 方向-上行 | 上行包占比 |
| 🥉 | **up_ia_mean** | 7/11 | 63.6% | 方向-上行 | 上行包平均间隔 |
| 4 | up_ia_std | 3/11 | 27.3% | 方向-上行 | 上行包间隔标准差 |
| 5 | up_len_mean | 1/11 | 9.1% | 方向-上行 | 上行包平均长度 |

### 4.4 新特征类别分析

| 特征类别 | Top-10 中出现次数 | 表现 |
|---------|------------------|------|
| **上行方向特征** | **30 次** | ✅ 非常突出 |
| 下行方向特征 | 0 次 | ❌ 未进入 top-10 |
| Burst 结构特征 | 0 次 | ❌ 未进入 top-10 |
| Other/Side 方向特征 | 0 次 | ❌ 未进入 top-10 |

### 4.5 关键洞察

**1. Up 方向特征最具区分性**
- 所有表现突出的新特征都是"上行流量"特征
- 上行流量（设备→服务器）最能体现 IoT 设备的特征
- 摄像头、传感器等设备主要通过上行数据传输来区分

**2. Down 方向特征表现平平**
- 下行流量（服务器→设备）在不同设备间差异较小
- 主要是通用的控制指令，设备特征性不强

**3. Burst 特征未进入 top-10**
- 虽然 burst 特征设计合理，但重要性不如方向特征
- 可能在 top-20 或更靠后的位置
- 未来可以探索 burst 与方向特征的交互

### 4.6 各任务详细数据

| 任务 | Top-10 中新特征数 | 具体新特征 |
|------|-----------------|-----------|
| single_round_R2 | 2/10 | up_len_p50, up_ia_std |
| single_round_R3 | 3/10 | up_len_p50, up_ia_std, up_packet_ratio |
| single_round_R4 | 2/10 | up_len_p50, up_ia_std |
| joint_R2_R3_R4 | 3/10 | up_len_mean, up_ia_mean, up_packet_ratio |
| loro_R2_R3_to_R4 | 3/10 | up_len_p50, up_ia_mean, up_packet_ratio |
| loro_R2_R4_to_R3 | 2/10 | up_len_p50, up_packet_ratio |
| loro_R3_R4_to_R2 | 3/10 | up_len_p50, up_packet_ratio, up_ia_mean |
| position_R2_R3_R4_to_R5 | 3/10 | up_len_p50, up_packet_ratio, up_ia_mean |
| jitter_R2_R3_R4_to_R6 | 3/10 | up_len_p50, up_packet_ratio, up_ia_mean |
| jitter_R2_R3_R4_to_R7 | 3/10 | up_len_p50, up_packet_ratio, up_ia_mean |
| jitter_R2_R3_R4_to_R6_R7 | 3/10 | up_len_p50, up_packet_ratio, up_ia_mean |

**数据来源**: `feature_rankings_all_tasks_complete.csv` (1034 行，包含所有 11 个任务的完整特征排名)

## 5. 核心实验观测结果
- **单轮 IID 基准**: 在 R3 独立训练与测试时，系统取得最高 Macro-F1 分数为 **0.9638**。
- **LORO (留一轮交叉验证) 跨环境测试**: 平均 Macro-F1 大幅下降至 **0.6643** 左右。这是检验跨部署轮次/时段泛化能力的核心测试，表现出明显的跨环境衰减。
- **跨抖动测试 (R6, R7)**: 平均 Macro-F1 为 **0.7788** (模型在正常 R2-R4 训练，在包含数据包抖动的网络会话中评估)。

## 6. 实验完成率
所有设定实验参数已全量运行并分析完毕。
