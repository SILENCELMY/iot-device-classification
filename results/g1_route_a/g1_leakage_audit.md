# g1_leakage_audit.md —— §17.1 条件 5 无泄漏审计（记录，不含解读）

审计对象：G1 路线 A 的**选择路径**——从落盘预测到 `UDS`、再到双阈值决策的全部代码。
审计不判定判据通过与否，只登记结构性事实与静态扫描结果。

## 1. 结构性约束：决策函数不收标签

| 函数 | 签名 | 禁用参数名命中 | 标签词元命中 |
|---|---|---|---|
| `g1_route_a._read_columns_no_label` | `_read_columns_no_label(path: 'Path', usecols: 'list[str]') -> 'pd.DataFrame'` | （无） | 0 |
| `g1_route_a.load_target_predictions` | `load_target_predictions(task_name: 'str') -> 'dict[str, np.ndarray]'` | （无） | 0 |
| `g1_route_a.load_source_oof_predictions` | `load_source_oof_predictions(task_name: 'str') -> 'dict[str, np.ndarray]'` | （无） | 0 |
| `g1_route_a.compute_uds` | `compute_uds(task_name: 'str') -> 'float'` | （无） | 0 |
| `g1_route_a.select_candidate` | `select_candidate(uds_value: 'float', t1: 'float', t2: 'float') -> 'str'` | （无） | 0 |
| `g1_route_a.select_vector` | `select_vector(uds_values: 'np.ndarray', t1: 'float', t2: 'float') -> 'np.ndarray'` | （无） | 0 |
| `g1_route_a.threshold_candidates` | `threshold_candidates(uds_values: 'np.ndarray') -> 'np.ndarray'` | （无） | 0 |
| `cpd_core.uds` | `uds(pred_src_oof, pred_tgt, *, class_order=None) -> 'float'` | （无） | 0 |
| `cpd_core.disagreement_matrix` | `disagreement_matrix(pred_a, pred_b, class_order) -> 'np.ndarray'` | （无） | 0 |
| `cpd_core.off` | `off(cm) -> 'np.ndarray'` | （无） | 0 |
| `cpd_core.normalize_cm` | `normalize_cm(cm) -> 'np.ndarray'` | （无） | 0 |
| `cpd_core._as_pred_mapping` | `_as_pred_mapping(preds, what: 'str') -> 'dict[str, np.ndarray]'` | （无） | 0 |

`cpd_core.uds` 签名：`uds(pred_src_oof, pred_tgt, *, class_order=None) -> 'float'`——只有两个预测入参与一个类别轴关键字参数，无任何标签入参（与 `test_cpd_core.py::test_uds_signature_takes_no_labels` 的 `inspect.signature` 断言同源）。

扫描词元集合：`true_label` / `y_true` / `y_test` / 单字母 `y` / `.correct` / `ground_truth` / `labels=`。选择路径 12 个函数的函数体命中总数 = **0**。

## 2. 读取层的结构性守卫

选择路径的每一次 CSV 读取都经 `_read_columns_no_label(path, usecols)`：

- 读取**前**检查请求列名是否命中黑名单 `FORBIDDEN_COLS = ['true_label', 'correct']`，命中即抛 `AssertionError`；
- `pandas.read_csv(..., usecols=...)` 只物化被请求的列，标签列不进入内存；
- 读取**后**再查一次读回列集合，并要求与请求列集合逐项相等。

实际请求的列：

| 输入文件 | 请求列 | 该文件中被排除的标签派生列 |
|---|---|---|
| `raw_all/<task>/all_features/{lightgbm,rf,xgboost}/predictions.csv` | `predicted_label`（1 列） | `true_label`、`correct` |
| `raw_all/<task>/all_features/stacking/oof_meta.csv` | `oof_<model>_<class>`（15 列概率） | `true_label` |

覆盖范围：150 个任务 × （3 个 `predictions.csv` + 1 个 `oof_meta.csv`） = 600 次读取，全部经该守卫。

## 3. 全文件静态扫描（含审计代码自身的命中）

对 `code/scripts/analysis/g1_route_a.py` 全文逐行扫描 `true_label` / `y_true` / `ground_truth`，命中如下（逐条给出行号与原文）：

| 行号 | 词元 | 原文 |
|---|---|---|
| 130 | `true_label` | `#: 的 `true_label` 是真实标签本身。` |
| 131 | `true_label` | `FORBIDDEN_COLS = ("true_label", "correct")` |
| 834 | `true_label` | `LABEL_TOKENS = (r"true_label", r"y_true", r"y_test", r"\by\b", r"\.correct\b",` |
| 835 | `ground_truth` | `r"ground_truth", r"labels\s*=")` |
| 859 | `true_label` | `if re.fullmatch(r"y\|y_true\|labels?\|true_label\|targets?", p)]` |
| 878 | `true_label` | `for tok in ("true_label", "y_true", "ground_truth"):` |
| 1083 | `true_label` | `lines.append("扫描词元集合：`true_label` / `y_true` / `y_test` / 单字母 `y` / `.correct` /"` |
| 1084 | `ground_truth` | `" `ground_truth` / `labels=`。选择路径 12 个函数的函数体命中总数 = "` |
| 1101 | `true_label` | `" `predicted_label`（1 列） \| `true_label`、`correct` \|")` |
| 1103 | `true_label` | `" `oof_<model>_<class>`（15 列概率） \| `true_label` \|")` |
| 1110 | `true_label` | `lines.append("对 `code/scripts/analysis/g1_route_a.py` 全文逐行扫描 `true_label` /"` |
| 1111 | `y_true` | `" `y_true` / `ground_truth`，命中如下（逐条给出行号与原文）：")` |

以上命中全部位于**黑名单常量、守卫代码、审计代码与文档字符串**中，没有任何一处构成对标签列的读取（`FORBIDDEN_COLS` 的用途是拒绝读取）。

## 4. 信息使用边界（事实登记）

| 环节 | 使用的信息 | 是否接触目标环境标签 |
|---|---|---|
| `UDS` 计算 | 源域 OOF 预测 + 目标域测试预测 | 否 |
| 外层决策（`select_candidate`） | 该任务 `UDS` + fold 的 `(t1, t2)` | 否 |
| 内层阈值/变体学习 | 内层 70 个任务的落盘 F1（`e_out` 既不作源也不作目标） | 否（不含 `e_out`） |
| 外层 regret 计算 | 外层任务的落盘 F1 | 是——属**评估**，不属决策路径 |

外层任务的落盘 F1 只在选择发生**之后**用于计算 regret；`select_candidate` 的入参中不含 F1，其调用点也不向其传入任何标签或 F1 派生量。

