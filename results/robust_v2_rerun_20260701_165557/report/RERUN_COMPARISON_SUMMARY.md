# 复跑结果对比摘要

本目录是对 `results/robust_v2/` 主线实验的一次独立复跑，输出目录为：

```text
results/robust_v2_rerun_20260701_165557/
```

主线结果目录 `results/robust_v2/` 未被覆盖。

## 完整性

| 项 | 结果 |
|---|---:|
| 新 summary 行数 | 110 |
| 旧 summary 行数 | 110 |
| 对齐后缺失行 | 0 |
| 任务数 | 11 |
| 特征集 | 2 |
| 模型数 | 5 |
| 特征缓存行数 | 11303 |
| pcap 来源文件数 | 30 |

30 个 pcap 来源文件对应 R2-R7 六轮 × 5 个设备；R1 是过滤策略测试轮次，不属于当前 `core` 主线。

复跑特征缓存中的 `source_file` 字段全部指向 `dataset/.../*.pcapng`，且来源文件均存在。

## 新旧差异

| 指标 | 数值 |
|---|---:|
| 最大 macro-F1 绝对差异 | 0.010270 |
| 平均 macro-F1 绝对差异 | 0.001085 |
| 最大 accuracy 绝对差异 | 0.011013 |

差异明细见：

```text
results/robust_v2_rerun_20260701_165557/report/comparison_vs_robust_v2.csv
```

## 最大 macro-F1 差异案例

| 任务 | 特征集 | 模型 | 旧 macro-F1 | 新 macro-F1 | 差值 |
|---|---|---|---:|---:|---:|
| `jitter_R2_R3_R4_to_R6` | `selected_features` | `lightgbm` | 0.754534 | 0.744264 | -0.010270 |
| `position_R2_R3_R4_to_R5` | `selected_features` | `lightgbm` | 0.650690 | 0.641034 | -0.009656 |
| `loro_R3_R4_to_R2` | `selected_features` | `lightgbm` | 0.786876 | 0.796357 | +0.009481 |
| `single_round_R2` | `selected_features` | `lightgbm` | 0.942927 | 0.933790 | -0.009137 |
| `single_round_R4` | `selected_features` | `lightgbm` | 0.949009 | 0.941788 | -0.007221 |

## 关键结论复核

- `loro_R2_R4_to_R3` 的 Stacking 崩溃现象复现。
- 该任务 all_features 下 RF macro-F1 = 0.6148，新 Stacking macro-F1 = 0.5465。
- Position R5 下 RF all_features 仍为 0.7012。
- Jitter R7 下 RF all_features 仍为 0.8220。

总体看，新旧结果高度一致，主要差异集中在 `selected_features + lightgbm/stacking` 的小幅波动。
