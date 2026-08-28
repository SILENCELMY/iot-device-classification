# 实验工作流

本文只保留当前主线的运行方式。旧路径、云端上下文和探索性结果已经归档到 `legacy/`。

## 环境准备

建议环境：

- Python 3.11
- `tshark`
- `numpy`、`pandas`、`scikit-learn`、`scipy`、`joblib`
- 可选：`xgboost`、`lightgbm`、`shap`

安装依赖：

```bash
pip install -r code/requirements-core.txt
pip install -r code/requirements-cloud.txt
```

## 主实验入口

统一入口：

```bash
python code/scripts/core/robust_iot_research.py \
  --config code/configs/research_experiments.json \
  --dataset-root dataset \
  --output-root results/robust_v2 \
  --filter-modes raw_all \
  --task-set core \
  --models rf,extra_trees,xgboost,lightgbm,stacking \
  --feature-mode both \
  --n-jobs 16
```

## 冒烟测试

只跑一个任务和一个模型：

```bash
python code/scripts/core/robust_iot_research.py \
  --config code/configs/research_experiments.json \
  --dataset-root dataset \
  --output-root results/smoke_test \
  --tasks single_round_R2 \
  --models rf \
  --feature-mode all \
  --disable-feature-selection \
  --max-rows 300
```

带特征选择的轻量测试：

```bash
python code/scripts/core/robust_iot_research.py \
  --config code/configs/research_experiments.json \
  --dataset-root dataset \
  --output-root results/smoke_test_fs \
  --tasks single_round_R2 \
  --models rf \
  --feature-mode selected \
  --max-rows 300
```

## 主要输出

每次运行会写入：

- `summary_metrics.csv`
- `summary_metrics.json`
- `environment_report.json`
- `feature_stability.csv`
- `feature_rankings_all_tasks.csv`
- `raw_all/features_raw_all_w10.csv`

每个任务、特征集、模型下面会有：

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `predictions.csv`
- `feature_importance.csv`
- `feature_columns.json`
- `model.joblib`

## 结果复用

已有主线结果在 `results/robust_v2/`。如果只是读结论或生成报告，通常不需要重新抽特征或重新训练模型。

重新聚合已有指标：

```bash
python code/scripts/core/aggregate_existing_metrics.py \
  --output-root results/robust_v2 \
  --out results/robust_v2/summary_metrics.csv
```

重新生成鲁棒性汇总报告：

```bash
python code/scripts/utils/generate_robustness_report.py \
  --input results/robust_v2/summary_metrics.csv \
  --output-dir results/robust_v2/report
```
