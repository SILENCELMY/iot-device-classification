# Robust IoT Device Recognition Workflow

This project uses one unified experiment runner for the thesis line:

> Robust IoT device recognition under complex scenarios.

The default experimental feature set is `FULL`. RSSI is treated as one ordinary
feature family in the full feature pool, not as a separate research line.

## Local Lab Host

Use the lab host for capture checks and small validation runs.

Smoke test, RF only:

```powershell
python scripts\robust_iot_research.py `
  --output-root results\robust_iot_smoke `
  --tasks single_round_R2 `
  --models rf `
  --feature-mode all `
  --disable-feature-selection `
  --max-rows 250
```

Smoke test with feature selection:

```powershell
python scripts\robust_iot_research.py `
  --output-root results\robust_iot_smoke_fs `
  --tasks single_round_R2 `
  --models rf `
  --feature-mode selected `
  --max-rows 250
```

## Cloud Server

Use the cloud server for full feature extraction, feature selection, and model
search. Recommended environment:

- Linux
- CUDA-ready NVIDIA driver
- Python 3.11 conda environment
- `tshark`
- `numpy pandas scipy scikit-learn imbalanced-learn joblib`
- Optional but recommended: `xgboost lightgbm shap`

Full core experiment:

```bash
python scripts/robust_iot_research.py \
  --output-root results/robust_iot_core \
  --filter-modes raw_all \
  --task-set core \
  --models rf,xgboost,lightgbm,stacking \
  --feature-mode both \
  --n-jobs 16
```

Filter strategy experiment:

```bash
python scripts/robust_iot_research.py \
  --output-root results/robust_iot_filter_strategy \
  --filter-modes raw_all,data_only,data_non_null \
  --task-set filter \
  --models rf,xgboost,lightgbm,stacking \
  --feature-mode both \
  --n-jobs 16
```

All configured experiments:

```bash
python scripts/robust_iot_research.py \
  --output-root results/robust_iot_all \
  --filter-modes raw_all,data_only,data_non_null \
  --task-set all \
  --models rf,xgboost,lightgbm,stacking \
  --feature-mode both \
  --n-jobs 16
```

## Outputs

Each run writes:

- `summary_metrics.csv`
- `summary_metrics.json`
- `environment_report.json`
- `feature_stability.csv` when feature selection is enabled
- per-task/per-model:
  - `metrics.json`
  - `classification_report.csv`
  - `confusion_matrix.csv`
  - `predictions.csv`
  - `feature_importance.csv`
  - `feature_columns.json`
  - `model.joblib`

Feature caches are stored under each filter mode:

- `features_raw_all_w10.csv`
- `features_data_only_w10.csv`
- `features_data_non_null_w10.csv`

## Data Sync Policy

Keep code and data separate:

- Git tracks scripts, configs, and docs.
- `dataset/`, `results/`, model files, and large feature CSVs are ignored.
- Sync data/results to the cloud with archive, `scp`, or `rsync`.

