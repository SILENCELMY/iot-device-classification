# robust_v2 主线结果

`robust_v2/` 是当前项目最重要的实验结果目录。

## 实验规模

- 11 个任务
- 2 个特征集：`all_features`、`selected_features`
- 5 个模型：`rf`、`extra_trees`、`xgboost`、`lightgbm`、`stacking`
- 共 110 条实验结果

## 目录

```
robust_v2/
├── raw_all/
│   ├── features_raw_all_w10.csv
│   └── <task>/<feature_set>/<model>/
├── report/
├── summary_metrics.csv
├── summary_metrics.json
├── feature_stability.csv
└── environment_report.json
```

当前特征缓存 `raw_all/features_raw_all_w10.csv` 是 11303 行、102 列，其中 94 列为数值特征，8 列为元数据。

## 任务

| 类型 | 任务 |
|---|---|
| IID 基线 | `single_round_R2`, `single_round_R3`, `single_round_R4` |
| 跨正常轮次 | `loro_R2_R3_to_R4`, `loro_R2_R4_to_R3`, `loro_R3_R4_to_R2` |
| 联合训练 | `joint_R2_R3_R4` |
| 位置漂移 | `position_R2_R3_R4_to_R5` |
| 抖动漂移 | `jitter_R2_R3_R4_to_R6`, `jitter_R2_R3_R4_to_R7`, `jitter_R2_R3_R4_to_R6_R7` |

## 阅读入口

- 报告索引：[report/README.md](report/README.md)
- 实验汇总：`summary_metrics.csv`
- 简要结果：[report/summary_report.md](report/summary_report.md)

## 重新生成报告

```bash
python code/scripts/utils/generate_robustness_report.py \
  --input results/robust_v2/summary_metrics.csv \
  --output-dir results/robust_v2/report
```
