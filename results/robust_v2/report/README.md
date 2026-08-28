# robust_v2 报告入口

这里保留当前论文主线报告。次要分析、旧索引、路线图和生成日志已归档到 `legacy/docs/archive/result-reports-secondary/`。

## 推荐顺序

1. [summary_report.md](summary_report.md)
2. [CPD_PAPER_LEVEL_ANALYSIS.md](CPD_PAPER_LEVEL_ANALYSIS.md)
3. [CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md](CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md)
4. [STACKING_COLLAPSE_ANALYSIS.md](STACKING_COLLAPSE_ANALYSIS.md)
5. [COMPREHENSIVE_OOD_ANALYSIS.md](COMPREHENSIVE_OOD_ANALYSIS.md)
6. [复跑对比摘要](../../robust_v2_rerun_20260701_165557/report/RERUN_COMPARISON_SUMMARY.md)

## 主线报告

| 文件 | 作用 |
|---|---|
| `summary_report.md` | 自动生成的跨场景结果概览 |
| `CPD_PAPER_LEVEL_ANALYSIS.md` | CPD 定义、相关性和统计显著性 |
| `CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md` | CPD 对 Stacking 集成增益的机制验证 |
| `STACKING_COLLAPSE_ANALYSIS.md` | Stacking 崩溃案例分析 |
| `COMPREHENSIVE_OOD_ANALYSIS.md` | OOD 现象综合整理 |
| `../../robust_v2_rerun_20260701_165557/report/RERUN_COMPARISON_SUMMARY.md` | 2026-07-01 独立复跑验证摘要 |

## 复跑验证

2026-07-01 使用 `dataset/` 下 R2-R7 的 30 个 pcapng 文件独立复跑主线实验，结果保存在：

```text
results/robust_v2_rerun_20260701_165557/
```

复跑结果与当前主线 `results/robust_v2/` 的 110 条指标完整对齐；平均 macro-F1 绝对差异为 `0.001085`，最大 macro-F1 绝对差异为 `0.010270`。核心结论复现，因此当前目录仍作为主线结果保留。

## 图和数据

| 内容 | 文件 |
|---|---|
| CPD vs 性能下降 | `cpd_vs_performance_correlation.png` |
| CPD 与 Stacking 集成增益 | `controlled_cpd_vs_gain_v2.png` |
| 混淆拓扑 | `confusion_topology_graphs.png` |
| 六环境相似度 | `six_env_similarity_matrices_rf.png` |
| 全部实验透视 | `pivot_macro_f1.csv`, `pivot_accuracy.csv` |
| CPD controlled 数据 | `controlled_cpd_data_v2.csv` |
