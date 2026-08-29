# IoT 设备鲁棒识别研究

> [!CAUTION] **本文件含已降格的历史结论（加注日期 2026-08-29，依据协议 §2.3 / §4.3）**
> 下面「历史基线结论」一节的数值为 **2026-06/07 口径的历史基线**，读取时须带以下限定：
>
> - `CPD vs Stacking gain Pearson r = -0.630` 已于 2026-08-26 的 P0 审计中**降为探索性结果**
>   （LOTO 仅 3/11 次 p<0.05；Spearman 不显著；口径依赖）。出处
>   [`results/p0_audit/R630_SENSITIVITY_CONCLUSION.md`](results/p0_audit/R630_SENSITIVITY_CONCLUSION.md)。
> - `CPD vs F1 下降 Pearson r = 0.9499` 按协议 §4.3 属于**代数上内含误差幅度**
>   （行归一化后每行离对角质量恰等于 `1 − recall_i`，相关性高有一部分是代数必然），
>   按 §2.3 **只作描述性证据**，不得作为「预测指标」主张。
> - 本文件（以及 `results/robust_v2/` 全部历史报告）的 **Stacking 数值均为随机折叠 OOF 口径**，
>   即 E1 的 **A 臂**。协议 §9.1 要求的按轮次分组 OOF（**B 臂**）结果见
>   [`results/e1_oof_arms/`](results/e1_oof_arms/) 与 [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md)
>   的 **E1-FULL** 行；B 臂下崩溃在全部 OOD 任务持续并加深。
> - 下面的「机制链条」是**待 E2 检验的工作假设**（协议 §5 假设 H2），不是已验证结论；
>   协议冻结条款规定相关性结论一律不得表述为因果关系。
>
> 权威口径以 [`docs/experiment_protocol_final.md`](docs/experiment_protocol_final.md)（FROZEN 2026-08-25）为准；
> CPD 命名与口径对照见 [`docs/CPD_DEFINITIONS.md`](docs/CPD_DEFINITIONS.md)。

本项目研究 **IoT 设备流量识别模型在跨场景条件下为什么会失效**。当前主线不是普通分类器调参，而是从混淆矩阵结构出发，提出并验证 **Confusion Pattern Drift (CPD)**：跨场景后模型的“错法”发生漂移，进而破坏集成学习的互补性，导致 Stacking 失效。

## 一句话总结

IoT 设备识别在单轮 IID 场景下可以达到约 0.94-0.96 macro-F1，但一旦跨时间轮次、位置或操作状态，性能明显下降；本项目的核心发现是：性能下降不只是特征分布变了，而是 **类别之间的混淆拓扑变了**。

## 历史基线结论（口径与降格标注见注）

> 本节数值为 2026-06/07 的历史基线，**原文原样保留供溯源**，逐行的口径与降格状态见表中「口径 / 状态」列。

机制链条（**待 E2 检验的工作假设**，非已验证结论；协议 §5 假设 H2、§2.3 主张强度约束）：

```text
特征漂移
  -> 混淆模式漂移（CPD）
  -> 误差相关性坍缩
  -> 元学习器失配
  -> 集成模型失效
```

E2（预定 9/12）正是要检验这条链是不是机制：按协议 §5，H2 的判否条件是「E2 中 CPD 增量解释力消失」。
在 E2 出结果之前，上面这条链只能写成假设，不能写成已验证的因果机制。

关键数值：

| 指标 | 数值 | 口径 / 状态（2026-08-29 加注） |
|---|---:|---|
| 实验规模 | 11 任务 × 2 特征集 × 5 模型 = 110 结果 | 已核对：`results/robust_v2/summary_metrics.csv` 共 110 行 |
| 特征缓存 | 11303 样本，94 数值特征 + 8 meta 列 | 与协议 §3.1 一致（CSV 共 102 列） |
| Single-round 平均表现 | 约 0.94-0.96 macro-F1 | 已核对：all_features 跨 5 模型的单任务均值 R2=0.9536 / R3=0.9613 / R4=0.9416，总均值 0.9522 |
| LORO 跨轮次平均表现 | 约 0.65-0.69 macro-F1 | 已核对：all_features 跨 5 模型总均值 0.6732；单任务均值 0.5670 / 0.6633 / 0.7892（区间比「0.65-0.69」宽） |
| Position R5 最佳表现 | RF all_features macro-F1 = 0.7012 | 已核对：0.701222 |
| Jitter R7 最佳表现 | RF all_features macro-F1 = 0.8220 | 已核对：0.821972 |
| CPD vs F1 下降 Pearson r | 0.9499 | **§4.3 代数耦合**：`CPD_y` 行归一化后每行离对角质量恰等于 `1 − recall_i`，代数上内含误差幅度，相关性高有一部分是代数必然；按 §2.3 **只作描述性证据**，不得作为预测指标主张 |
| CPD vs Stacking gain Pearson r | -0.630 | **已降格为探索性结果**（P0 审计，2026-08-26）：LOTO 仅 3/11 次 p<0.05，剔除 `loro_R2_R4_to_R3` 后 r=-0.37 (p=0.30)；Spearman ρ=-0.50 (p=0.1173) 本就不显著；按测试环境聚类 bootstrap 95% CI [-0.97, -0.08] 极宽。出处 `results/p0_audit/` |
| 最严重崩溃任务 | `loro_R2_R4_to_R3` | E1-FULL 复核仍成立；且在协议 §9.1 的分组 OOF（B 臂）下崩溃进一步加深，见下方案例段 |

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

> [!IMPORTANT] **口径注（2026-08-29，协议 §9.1 / §12）**
> 上面三个数字是 **E1 A 臂（随机折叠 OOF）、seed 42** 口径，与
> `results/robust_v2/summary_metrics.csv` 逐位一致（0.614816 / 0.545482 / −0.069334）。
> 协议 §9.1 要求 OOF 按训练轮次分组（E1 **B 臂**）。E1-FULL（11 任务 × 5 种子 × 三臂）实测本任务：
>
> | 臂 | OOF 折叠口径 | Stacking macro-F1（5 种子均值 ± 标准差） | gain = Stacking − 最佳基模型 |
> |---|---|---:|---:|
> | A | 随机 5 折（历史口径） | 0.5452 ± 0.0018 | −0.0662 ± 0.0080 |
> | A′ | 随机，折数对齐 B（2 折） | 0.5533 ± 0.0035 | −0.0581 ± 0.0084 |
> | **B** | **按轮次分组（§9.1，2 折）** | **0.5057 ± 0.0162** | **−0.1057 ± 0.0181** |
>
> 即 B 臂下崩溃幅度约为 A 臂的 **1.6 倍**（seed 42 单点：0.4907 / −0.1241，约 1.8 倍）。
> A→A′ 的折数效应小而偏正（本任务 +0.0081，11 任务总均值 +0.0004），加深主要来自**分组效应**。
> 数据出处 `results/e1_oof_arms/e1_arms_raw.csv` 与 `e1_decomposition.csv`；
> 登记见 `docs/EXPERIMENT_REGISTRY.md` 的 **E1-FULL** 行。**引用本案例时以 E1 的 B 臂为准。**

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

### 仓库结构与检出方式

**（2026-08-29 更新）项目根目录现在就是唯一的 git 仓库**，远端为
`https://github.com/SILENCELMY/iot-device-classification.git`，唯一分支 `main`。
早先「项目根目录不是 git 仓库、`code/` 自身是另一个 git 仓库」的说法已经过时：
`code/` 已通过 `git subtree` 并入根仓库（合并提交 `d24755d`，源提交 `1fa5bcc`），
不再是独立仓库，也不再需要单独 clone。

因此：

```bash
git clone https://github.com/SILENCELMY/iot-device-classification.git
cd iot-device-classification
# code/ 已在检出内容中，下文所有 code/scripts/... 路径可直接使用
```

未纳入版本控制的目录（见 `.gitignore`，依据协议 §19.7）：

| 目录 | 状态 |
|---|---|
| `dataset/` | 原始 pcapng，权限锁定不可分发，只在服务器本地 |
| `legacy/` | 4.7 GB 被取代的历史结果，留在服务器磁盘但不入库 |
| `results/**` | 白名单入库：只跟踪 `*.csv` / `*.json` / `*.md`；`*.joblib` 等大文件与 `**/scratch/` 不入库 |

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
| 简要结果表（stacking 列为 A 臂口径） | `results/robust_v2/report/summary_report.md` |
| CPD 历史分析（2026-06 口径，已加注降格） | `results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md` |
| CPD 机制验证（已降格为探索性，见文件头标注） | `results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md` |
| Stacking 崩溃分析（A 臂口径，以 E1 为准） | `results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md` |
| OOD 综合分析（历史口径，已加注降格） | `results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md` |
| 复跑对比摘要 | `results/robust_v2_rerun_20260701_165557/report/RERUN_COMPARISON_SUMMARY.md` |
| **E1 三臂 OOF 对比（当前口径，优先看这个）** | `results/e1_oof_arms/e1_arms_raw.csv`、`e1_decomposition.csv` |
| **`r=-0.630` 降格审计** | `results/p0_audit/R630_SENSITIVITY_CONCLUSION.md` |

## 阅读路线

> [!NOTE] 阅读顺序建议：**先读协议再读历史报告**。
> [`docs/experiment_protocol_final.md`](docs/experiment_protocol_final.md)（FROZEN）是唯一权威口径；
> `results/robust_v2/report/` 下的四份报告均为 2026-06 历史口径，文件头已加降格/口径标注。

如果只想快速接回项目：

1. 读本文件
2. 读 [docs/experiment_protocol_final.md](docs/experiment_protocol_final.md) 的 §2（主张强度）、§4（CPD 定义）、§9.1（OOF 口径）
3. 读 [docs/EXECUTION_PLAN_20260829.md](docs/EXECUTION_PLAN_20260829.md) 与 [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md)（当前进度与每次实验的登记）
4. 读 [docs/论文冲刺计划.md](docs/论文冲刺计划.md)
5. 读 [docs/CPD_FINDINGS.md](docs/CPD_FINDINGS.md)
6. 读 [results/robust_v2/report/summary_report.md](results/robust_v2/report/summary_report.md)

如果要写论文或整理章节：

1. [docs/PAPER_MODEL_ORGANIZATION.md](docs/PAPER_MODEL_ORGANIZATION.md)
2. [docs/DEEP_MODEL_METHODS.md](docs/DEEP_MODEL_METHODS.md)
3. [results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md](results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md)
4. [results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md](results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md)
5. [results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md](results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md)
6. [results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md](results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md)

如果要重新跑实验：

1. [docs/WORKFLOW.md](docs/WORKFLOW.md)
2. [code/README.md](code/README.md)（`code/` 已并入本仓库，此链接在检出内容中有效）
3. `code/scripts/core/robust_iot_research.py --help`
4. `code/requirements-lock.txt`（协议 §19.5：干净环境复现用的完整版本锁定）

如果要找旧材料：

- `legacy/`（**未纳入版本控制**，见上文「仓库结构与检出方式」：只存在于服务器
  `~/iot-device-classification/legacy/`，clone 出来的检出里没有这个目录，
  因此这里不给仓库内链接）

## 快速运行

环境依赖：

```bash
pip install -r code/requirements-core.txt
pip install -r code/requirements-cloud.txt
```

若要**严格复现**服务器上已落盘的结果（协议 §19.5），改用完整版本锁定：

```bash
# Python 3.11.15（conda 环境 iotcls），与生成现有传统 ML 结果的运行环境逐位一致
pip install -r code/requirements-lock.txt
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
  --results-root results/robust_v2 \
  --output-dir results/robust_v2/report
```

（`--results-root` 是必填参数，2026-08-29 补上：原命令缺该参数会直接报错退出。
重新生成会覆盖 `summary_report.md`，其中的「口径说明」块已写进生成器，不会丢失；
但**当前文件的中文小节标题是手工翻译的，重新生成会变回英文标题**，注意确认是否可接受。）

## 当前文档整理状态

根目录只保留这个总览文件。当前可读文档主要是：

权威 / 当前有效（优先）：

- [docs/experiment_protocol_final.md](docs/experiment_protocol_final.md)（**FROZEN 2026-08-25，唯一权威口径**）
- [docs/EXECUTION_PLAN_20260829.md](docs/EXECUTION_PLAN_20260829.md)（执行层，FROZEN v1.0）
- [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md)（实验登记表，活跃维护）
- [docs/CPD_DEFINITIONS.md](docs/CPD_DEFINITIONS.md)（`CPD_y` / `CPD_dir` / `UDS` 口径对照，P0 交付物）
- [results/p0_audit/R630_SENSITIVITY_CONCLUSION.md](results/p0_audit/R630_SENSITIVITY_CONCLUSION.md)（P0 交付物）

工作文档 / 论文素材：

- [docs/WORKFLOW.md](docs/WORKFLOW.md)
- [docs/CPD_FINDINGS.md](docs/CPD_FINDINGS.md)（部分结论已降格，文件头有标注）
- [docs/PAPER_MODEL_ORGANIZATION.md](docs/PAPER_MODEL_ORGANIZATION.md)
- [docs/DEEP_MODEL_METHODS.md](docs/DEEP_MODEL_METHODS.md)
- [code/README.md](code/README.md)
- [results/README.md](results/README.md)
- [results/robust_v2/report/README.md](results/robust_v2/report/README.md)

旧版说明、重复报告、英文上下文、探索性实验结果、备份和占位目录已经移动到 `legacy/`，没有删除；
按协议 §19.7，`legacy/` 不纳入版本控制（只在服务器磁盘上）。

## 后续可做

- 统一脚本里的硬编码路径，减少换机器成本
- 把 CPD 计算、报告生成和可视化进一步模块化
- 将最核心图表和数值整理成论文 Figure/Table 清单
- 针对高 CPD 场景设计更稳健的特征选择或模型适配策略
