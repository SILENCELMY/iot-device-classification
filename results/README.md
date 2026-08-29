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
├── g0_environment_grid/                # G0 环境网格全量（协议 §8.5 / §8.4，seed 42 单种子）：162 任务 / 648 行 `summary_metrics.csv`、RF macro-F1 6×6 `env_topology_matrix_rf.csv`、逐任务 CM 与 stacking 的 pred_proba / oof_meta
└── robustness_scaling_20260706_v2/     # 固定 split artifacts 和 scaling 结果
```

口径提示：`p0_audit/`、`e1_oof_arms/`、`g0_environment_grid/` 三个目录是 2026-08 之后的**当前口径**产出；
`robust_v2/` 及其 `report/` 下的分析报告为 2026-06/07 历史口径，文件头已加降格或口径标注，引用前先看标注。
按协议 §8.6，`g0_environment_grid/` 的网格结果**不用于显著性结论**（约 150 个任务由 6 次采集重组而来，不是独立样本）。

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
