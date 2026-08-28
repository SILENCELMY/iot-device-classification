# code 目录

这里保存当前实验代码和配置。项目根目录不是 git 仓库，`code/` 自身是一个 git 仓库。

## 结构

```
code/
├── configs/
│   └── research_experiments.json
├── scripts/
│   ├── core/
│   ├── analysis/
│   ├── feature/
│   ├── utils/
│   └── legacy/
├── requirements-core.txt
└── requirements-cloud.txt
```

## 主入口

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

## 依赖

```bash
pip install -r code/requirements-core.txt
pip install -r code/requirements-cloud.txt
```

`requirements-cloud.txt` 包含 XGBoost、LightGBM、SHAP 等可选实验依赖。

## 子目录说明

| 目录 | 作用 |
|---|---|
| `scripts/core/` | 主实验框架和已有指标聚合 |
| `scripts/analysis/` | CPD、误差相关、元特征漂移、CORAL 等分析 |
| `scripts/feature/` | 新特征重要性分析 |
| `scripts/utils/` | 报告生成和指标聚合 |
| `scripts/legacy/` | 早期脚本，保留追溯，不建议作为新入口 |

旧版子目录 README 已归档到 `../legacy/docs/archive/code-readmes/`。
