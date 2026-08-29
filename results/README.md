# results 目录

这里保存当前主线实验产物。探索性实验和过渡版本已经归档到 `../legacy/results/`。

## 结构

```
results/
├── robust_v2/                         # 传统 ML 主线：110 个实验 + CPD 报告
├── robust_v2_rerun_20260701_165557/   # 从 dataset 真实 pcapng 复跑验证
├── gpu_capacity_full_20260703/        # 深度模型 RF/CNN/Transformer 基线对比
├── cnn_architecture_contrast_20260707/ # 最终 CNN 架构对比，包含 TDR-CNN
├── p0_audit/                           # P0 指标审计产出：`r=-0.630` 的 leave-one-task-out、leave-one-environment-out 与按测试环境聚类 bootstrap，结论文档 `R630_SENSITIVITY_CONCLUSION.md`
├── e1_oof_arms/                        # E1 OOF 折叠三臂对比（协议 §12）：`e1_arms_raw.csv`、`e1_decomposition.csv`、`e1_arms.json`
└── robustness_scaling_20260706_v2/     # 固定 split artifacts 和 scaling 结果
```

## 当前主线

[robust_v2/](robust_v2/) 包含：

- `summary_metrics.csv`：110 条实验汇总
- `raw_all/`：每个任务、特征集、模型的详细结果
- `report/`：论文主线分析报告和图表
- `feature_stability.csv`：特征稳定性统计
- `environment_report.json`：可选依赖检测结果

## 常用文件

| 目的 | 文件 |
|---|---|
| 看所有实验指标 | `robust_v2/summary_metrics.csv` |
| 看每个任务的混淆矩阵 | `robust_v2/raw_all/<task>/<feature_set>/<model>/confusion_matrix.csv` |
| 看预测明细 | `robust_v2/raw_all/<task>/<feature_set>/<model>/predictions.csv` |
| 看论文报告 | `robust_v2/report/` |
| 看深度模型最终组织 | `../docs/PAPER_MODEL_ORGANIZATION.md` |
| 看深度模型方法细节 | `../docs/DEEP_MODEL_METHODS.md` |

数据和模型文件体量较大，通常不建议纳入 git。
