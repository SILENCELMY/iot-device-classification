# scripts 目录

这里按功能放置 Python 脚本。新工作优先使用 `core/` 和 `analysis/`，`legacy/` 只用于追溯。

## 当前入口

| 目录 | 主要文件 | 作用 |
|---|---|---|
| `core/` | `robust_iot_research.py` | 抽特征、训练、评估、保存结果 |
| `core/` | `aggregate_existing_metrics.py` | 从已有 `metrics.json` 重建汇总表 |
| `analysis/` | `cpd_comprehensive_analysis.py` | CPD 综合分析 |
| `analysis/` | `controlled_cpd_experiment_v2.py` | CPD 与 Stacking gain 的机制验证 |
| `analysis/` | `domain_generalization.py` | CORAL 等领域泛化实验 |
| `feature/` | `analyze_new_features.py` | 新特征出现频率和重要性 |
| `utils/` | `generate_robustness_report.py` | 生成汇总报告 |

## 路径约定

推荐从项目根目录运行脚本：

```bash
cd <project-root>
python code/scripts/core/robust_iot_research.py --help
```

工具脚本默认从项目根目录解析 `results/robust_v2`，需要换位置时优先使用脚本提供的 `--results-root`、`--output-dir` 或 `--input` 参数。

## 输出约定

主实验输出到：

```text
results/<run_name>/<filter_mode>/<task>/<feature_set>/<model>/
```

当前主线结果是：

```text
results/robust_v2/
```
