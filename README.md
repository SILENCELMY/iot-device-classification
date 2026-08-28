# IoT 设备鲁棒识别研究

本项目研究 **IoT 设备流量识别模型在跨场景条件下为什么会失效**。当前主线不是普通分类器调参，而是从混淆矩阵结构出发，提出并验证 **Confusion Pattern Drift (CPD)**：跨场景后模型的“错法”发生漂移，进而破坏集成学习的互补性，导致 Stacking 失效。

## 一句话总结

IoT 设备识别在单轮 IID 场景下可以达到约 0.94-0.96 macro-F1，但一旦跨时间轮次、位置或操作状态，性能明显下降；本项目的核心发现是：性能下降不只是特征分布变了，而是 **类别之间的混淆拓扑变了**。

## 当前结论

机制链条：

```text
特征漂移
  -> 混淆模式漂移（CPD）
  -> 误差相关性坍缩
  -> 元学习器失配
  -> 集成模型失效
```

关键数值：

| 指标 | 数值 |
|---|---:|
| 实验规模 | 11 任务 × 2 特征集 × 5 模型 = 110 结果 |
| 特征缓存 | 11303 样本，94 数值特征 + 8 meta 列 |
| Single-round 平均表现 | 约 0.94-0.96 macro-F1 |
| LORO 跨轮次平均表现 | 约 0.65-0.69 macro-F1 |
| Position R5 最佳表现 | RF all_features macro-F1 = 0.7012 |
| Jitter R7 最佳表现 | RF all_features macro-F1 = 0.8220 |
| CPD vs F1 下降 Pearson r | 0.9499 |
| CPD vs Stacking gain Pearson r | -0.630 |
| 最严重崩溃任务 | `loro_R2_R4_to_R3` |

复跑验证：

- 2026-07-01 已从 `dataset/` 原始 pcapng 独立复跑主线实验。
- 新结果目录：`results/robust_v2_rerun_20260701_165557/`
- 新旧 110 条结果完整对齐，平均 macro-F1 绝对差异为 `0.001085`，最大 macro-F1 绝对差异为 `0.010270`。
- `loro_R2_R4_to_R3` 的 Stacking 崩溃、Position R5、Jitter R7 等关键结论均复现。
- 新特征缓存记录了 30 个 `dataset/.../*.pcapng` 来源文件，正好对应 R2-R7 六轮 × 5 个设备；R1 是 filter 测试轮次，不属于当前 `core` 主线。

最典型案例：`loro_R2_R4_to_R3`

- RF all_features macro-F1 = 0.6148
- Stacking all_features macro-F1 = 0.5455
- Stacking 比最佳基础模型低 0.0693
- 说明 Stacking 在高 CPD 场景下没有稳定收益，反而可能放大错误结构漂移。

## 数据

原始数据在 `dataset/`：

```text
5 类设备 × 7 轮采集 = 35 个 pcapng 文件
```

设备类别：

- `Camera`
- `Light_T1`
- `Light_XM`
- `Sensor`
- `Socket`

轮次含义：

| 轮次 | 场景 | 用途 |
|---|---|---|
| `R1` | filter 测试 | 过滤策略对比，非当前主线 |
| `R2-R4` | normal | IID 基线、LORO、联合训练 |
| `R5` | position B | 位置漂移测试 |
| `R6-R7` | jitter | 操作抖动测试 |

当前主线使用 `raw_all` 过滤模式，特征缓存为：

```text
results/robust_v2/raw_all/features_raw_all_w10.csv
```

复跑缓存为：

```text
results/robust_v2_rerun_20260701_165557/raw_all/features_raw_all_w10.csv
```

复跑缓存中的 `source_file` 字段全部指向 `dataset/` 下的真实 pcapng 文件，没有缺失来源。

## 实验设计

当前主线结果在 `results/robust_v2/`。

任务类型：

| 类型 | 任务 |
|---|---|
| IID 基线 | `single_round_R2`, `single_round_R3`, `single_round_R4` |
| 跨正常轮次 | `loro_R2_R3_to_R4`, `loro_R2_R4_to_R3`, `loro_R3_R4_to_R2` |
| 联合训练 | `joint_R2_R3_R4` |
| 位置漂移 | `position_R2_R3_R4_to_R5` |
| 操作抖动 | `jitter_R2_R3_R4_to_R6`, `jitter_R2_R3_R4_to_R7`, `jitter_R2_R3_R4_to_R6_R7` |

特征集：

- `all_features`：全部数值特征
- `selected_features`：基于 MI、模型重要性、SHAP 可用时的联合特征选择结果

模型：

- `rf`
- `extra_trees`
- `xgboost`
- `lightgbm`
- `stacking`

## 代码结构

```text
code/
├── configs/
│   └── research_experiments.json
├── scripts/
│   ├── core/       # 主实验框架
│   ├── analysis/   # CPD、误差相关、元特征漂移、CORAL
│   ├── feature/    # 新特征分析
│   ├── utils/      # 报告生成和指标聚合
│   └── legacy/     # 早期脚本，只保留追溯
├── requirements-core.txt
└── requirements-cloud.txt
```

最重要的入口：

| 文件 | 作用 |
|---|---|
| `code/scripts/core/robust_iot_research.py` | 抽特征、训练、评估、落盘 |
| `code/configs/research_experiments.json` | 设备、轮次、任务、模型配置 |
| `code/scripts/analysis/cpd_comprehensive_analysis.py` | CPD 综合分析 |
| `code/scripts/analysis/controlled_cpd_experiment_v2.py` | CPD 机制验证 |
| `code/scripts/utils/generate_robustness_report.py` | 生成汇总报告 |

注意：项目根目录不是 git 仓库，`code/` 自身是 git 仓库。

## 结果结构

```text
results/robust_v2/
├── raw_all/
│   ├── features_raw_all_w10.csv
│   └── <task>/<feature_set>/<model>/
│       ├── metrics.json
│       ├── confusion_matrix.csv
│       ├── predictions.csv
│       ├── feature_importance.csv
│       └── model.joblib
├── report/
├── summary_metrics.csv
├── summary_metrics.json
├── feature_stability.csv
└── environment_report.json
```

最常看的文件：

| 目的 | 文件 |
|---|---|
| 全部 110 条实验指标 | `results/robust_v2/summary_metrics.csv` |
| 简要结果表 | `results/robust_v2/report/summary_report.md` |
| CPD 主报告 | `results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md` |
| CPD 机制验证 | `results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md` |
| Stacking 崩溃分析 | `results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md` |
| OOD 综合分析 | `results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md` |
| 复跑对比摘要 | `results/robust_v2_rerun_20260701_165557/report/RERUN_COMPARISON_SUMMARY.md` |

## 阅读路线

如果只想快速接回项目：

1. 读本文件
2. 读 [docs/论文冲刺计划.md](docs/论文冲刺计划.md)
3. 读 [docs/CPD_FINDINGS.md](docs/CPD_FINDINGS.md)
4. 读 [results/robust_v2/report/summary_report.md](results/robust_v2/report/summary_report.md)

如果要写论文或整理章节：

1. [docs/PAPER_MODEL_ORGANIZATION.md](docs/PAPER_MODEL_ORGANIZATION.md)
2. [docs/DEEP_MODEL_METHODS.md](docs/DEEP_MODEL_METHODS.md)
3. [results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md](results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md)
4. [results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md](results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md)
5. [results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md](results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md)
6. [results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md](results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md)

如果要重新跑实验：

1. [docs/WORKFLOW.md](docs/WORKFLOW.md)
2. [code/README.md](code/README.md)
3. `code/scripts/core/robust_iot_research.py --help`

如果要找旧材料：

- [legacy/README.md](legacy/README.md)
- `legacy/`

## 快速运行

环境依赖：

```bash
pip install -r code/requirements-core.txt
pip install -r code/requirements-cloud.txt
```

主实验：

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

轻量测试：

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

重新生成汇总报告：

```bash
python code/scripts/utils/generate_robustness_report.py \
  --input results/robust_v2/summary_metrics.csv \
  --output-dir results/robust_v2/report
```

## 当前文档整理状态

根目录只保留这个总览文件。当前可读文档主要是：

- [docs/WORKFLOW.md](docs/WORKFLOW.md)
- [docs/CPD_FINDINGS.md](docs/CPD_FINDINGS.md)
- [docs/PAPER_MODEL_ORGANIZATION.md](docs/PAPER_MODEL_ORGANIZATION.md)
- [docs/DEEP_MODEL_METHODS.md](docs/DEEP_MODEL_METHODS.md)
- [code/README.md](code/README.md)
- [results/README.md](results/README.md)
- [results/robust_v2/report/README.md](results/robust_v2/report/README.md)

旧版说明、重复报告、英文上下文、探索性实验结果、备份和占位目录已经移动到 `legacy/`，没有删除。

## 后续可做

- 统一脚本里的硬编码路径，减少换机器成本
- 把 CPD 计算、报告生成和可视化进一步模块化
- 将最核心图表和数值整理成论文 Figure/Table 清单
- 针对高 CPD 场景设计更稳健的特征选择或模型适配策略
