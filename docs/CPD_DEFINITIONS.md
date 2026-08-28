# CPD_DEFINITIONS.md

**用途**：`CPD_y` / `CPD_dir` / `UDS` 三个指标的唯一定义、标签需求、历史口径对照表。
**协议依据**：[experiment_protocol_final.md](experiment_protocol_final.md) 第 4、11、20.1 节。
**创建日期**：2026-08-26（P0 阶段，§22.2 立即执行清单第 1、3 项）
**唯一实现**：[`code/scripts/analysis/cpd_core.py`](../code/scripts/analysis/cpd_core.py)
**回归测试**：[`code/scripts/analysis/test_cpd_core.py`](../code/scripts/analysis/test_cpd_core.py) — 15/15 通过

本文档不引入任何新定义。协议第 4 节是唯一定义来源；本文档只做转录、口径归因与执行发现记录。

---

## 1. 三个指标（唯一合法命名）

| 名称 | 需要目标环境真实标签 | 可用于部署前 | 依赖性质 | Transductive |
|---|---|---|---|---|
| `CPD_y` | **是** | **否** | performance-dependent（行归一化后每行离对角质量恰为 `1 − recall_i`） | 否 |
| `CPD_dir` | **是** | **否** | 已去除误差幅度，但**不得声称完全独立于性能** | 否 |
| `UDS` | **否** | **是** | prediction-only（只用模型预测，不用标签） | **是**（若使用整批无标签目标样本） |

### 1.1 数学定义

设混淆矩阵 `C`，行归一化 `C̃_ij = C_ij / Σ_k C_ik`，`Off(C̃)` 为置零对角后的矩阵。

```
CPD_y(C_ref, C_tgt) = || Off(C̃_ref) − Off(C̃_tgt) ||_F
```

`CPD_dir`：对每行 `i`，将离对角向量再按 L1 归一化为

```
D_ij = Off(C̃)_ij / Σ_{k≠i} Off(C̃)_ik   （即 P(ŷ=j | y=i, ŷ≠i)）
CPD_dir(C_ref, C_tgt) = || D_ref − D_tgt ||_F
```

`UDS`：对模型对 `(m_a, m_b)`，统计"在 `m_a` 预测为类 `c` 的样本上，`m_b` 的预测分布"，
构成分歧矩阵 `G^{ab}`（行归一化）。源域用 OOF 预测、目标域用测试预测，取

```
UDS = mean over (a,b) pairs of || Off(G^{ab}_src) − Off(G^{ab}_tgt) ||_F
```

### 1.2 实现约定（不得更改，历史值复现依赖它们）

| 约定 | 取值 | 出处 |
|---|---|---|
| 行和为 0 时的处理 | 该行除以 1（保持全零行），不产生 NaN | `cpd_comprehensive_analysis.py:88-92` 历史口径 |
| `CPD_dir` 的 `min_err` | 20 | 协议 §4.3 |
| `UDS` 的模型对 | 全部**有序**对 `(a,b)`，`a ≠ b`（`G^{ab} ≠ G^{ba}`，两者都计入） | 协议 §4.2「mean over (a,b) pairs」 |
| 类别轴顺序 | `['Camera','Light_T1','Light_XM','Sensor','Socket']` | 历史脚本一致约定 |

`cpd_core.py` 中 `D_ij` 由原始计数直接算（`D_ij = C_ij / n_err_i`），与"先行归一化再 L1 归一化"
代数等价，但保留了 `n_err_i` 以便施加 `min_err` 门槛。

### 1.3 `CPD_dir` 的动机与限制（协议 §4.3，不得当作可选项砍掉）

**动机**：行归一化后每行离对角质量恰好等于 `1 − recall_i`，因此 `CPD_y` 在**代数上**内含误差幅度。
用它去预测 ΔF1，相关性高有一部分是代数必然。`CPD_dir` 只保留误分类方向。

> 该代数性质已由 `test_off_diag_row_mass_equals_one_minus_recall` 在 5 个真实任务上断言验证（atol 1e-12）。

**限制**：`CPD_dir` 去除了误差**幅度**，但**没有**去除误差**计数**带来的估计噪声——误差极少的行，
方向分布估计方差极大，会系统性抬高高准确率环境的 `CPD_dir`。因此：

- 每行要求 `n_err ≥ 20` 才纳入计算，不足则该行置为缺失并在结果中标注；
- RQ1 必须附置换/随机标签参照，给出 `CPD_dir` 在无真实漂移下的零分布。

### 1.4 `UDS` 无泄漏声明（协议 §17.1 条件 5）

`uds()` 的签名为 `uds(pred_src_oof, pred_tgt, *, class_order=None)`——**没有任何标签入参**，
函数体内也不读取任何真实标签。三重机器可验证证据：

| 证据 | 测试 |
|---|---|
| 签名参数名精确比对（按下划线切词，排除标签 token） | `test_uds_signature_takes_no_labels` |
| 源码中不出现 `y_true` / `true_label` / `ground_truth` | 同上 |
| 行为级：同一批预测 → `UDS == 0`；结果确定性 | `test_uds_is_invariant_to_label_permutation_of_truth` |

**调用方责任**：若把真实标签列混入 `pred_*` 映射，它会被当作"另一个模型"参与分歧统计。
上游必须只传模型预测。

---

## 2. `cpd_core.py` 导出接口

```python
normalize_cm(cm)                          # 行归一化
off(cm)                                   # 置零对角
cpd_y(cm_ref, cm_tgt)                     # → float
cpd_dir(cm_ref, cm_tgt, min_err=20)       # → CpdDirResult（数值 + 逐行纳入/缺失标注）
uds(pred_src_oof, pred_tgt, *, class_order=None)   # → float，不接受标签参数
```

辅助导出：`CpdDirResult`、`dir_matrix`、`disagreement_matrix`、`DEFAULT_MIN_ERR`。

`cpd_dir` 返回 `CpdDirResult` 而非裸 float，因为协议 §4.3 要求"该行置为缺失并**在结果中标注**"：

```python
res = cpd_dir(cm_ref, cm_tgt)
res.value          # float，全行缺失时为 nan
res.is_defined     # 是否有任何一行通过门槛
res.included_rows  # 纳入计算的行索引
res.excluded_rows  # 因 n_err < min_err 被剔除的行索引
res.n_err_ref / res.n_err_tgt   # 两侧逐行原始误分类计数
res.notes          # 人读的剔除说明
```

`ref` 或 `tgt` 任一侧不达标的行**整行剔除**，Frobenius 范数只在共同纳入的行上计算。

---

## 3. 历史口径对照表（协议 §22.2 第 3 项）

### 3.1 各脚本使用的 CPD 口径

已对 `code/scripts/` 下全部 **10 个**计算 CPD 类量的历史脚本做穷尽审计（不含本次新增的
`cpd_core.py` 与 `test_cpd_core.py`）。共存在 **3 种实现**，
其中 2 种与 `cpd_core.cpd_y` 数学等价，1 种是**不同的量**。

| 实现 | 脚本 | 与 `cpd_core.cpd_y` 的最大偏差（45 个任务对） | 结论 |
|---|---|---|---|
| **impl_A** | `cpd_comprehensive_analysis.py`<br>`controlled_cpd_experiment.py`<br>`controlled_cpd_experiment_v2.py` | `1.11e-16` | **等价**（浮点噪声） |
| **impl_B** | `deep_robustness_validation.py`（`normalized_offdiag`, 第 1005-1011 行）<br>`robustness_scaling_experiment.py`<br>`cnn_contrast_search_experiment.py`<br>`topology_sweet_spot_experiment.py`<br>`final_extreme_capacity_experiment.py` | `1.11e-16` | **等价**（浮点噪声） |
| **impl_C** | `confusion_pattern_analysis.py` | **`5.21e-01`** | **不等价，是另一个量** |
| — | `six_env_confusion_similarity.py` | 用 impl_A 数学，但 `env_mapping` 不同质 | 见 §5.2，已废弃 |

**impl_C 的问题**：`confusion_pattern_analysis.py` 在第 64 行定义了 `off_diagonal_pattern`，
但 `compute_pattern_similarity_matrix`（第 70-113 行）**从未调用它**——第 93 行只做行归一化，
第 108 行直接对**含对角线**的归一化矩阵求 Frobenius 距离。因此它的 `frobenius_matrix`
同时包含 recall 差异与方向差异，**不是 `CPD_y`**。

`off_diagonal_pattern` 在整个 `code/` 下无任何调用点（死代码）。

> **受影响产物**：`results/robust_v2/report/confusion_pattern_similarity_rf.png`、
> `class_level_confusion_drift_rf.csv`。这些图表若在论文中出现，其纵轴不得标注为 `CPD_y`。
> 处置建议见 §6.3。

### 3.2 各历史报告使用的基准

| 报告 / 产物 | 基准 | 口径列名 | 实现 |
|---|---|---|---|
| `gpu_capacity_full_20260703/report/cpd_comparison.csv` | 三个 IID CM，取均值 / 最大值 | `cpd_vs_iid_mean` / `cpd_vs_iid_max` | impl_B |
| `robustness_scaling_20260706_v2/report/cpd_comparison.csv` | 同上 | 同上 | impl_B |
| `cnn_architecture_contrast_20260707/report/cpd_comparison.csv` | 同上（深度模型） | 同上 | impl_B |
| `robust_v2/report/controlled_cpd_data.csv` | 单个 `joint_R2_R3_R4` CM | `cpd` | impl_A |
| `robust_v2/report/controlled_cpd_data_v2.csv` | 已废弃六环境 pairwise 均值 | `cpd` | impl_A + 废弃映射 |
| `robust_v2/report/six_env_off_diag_frobenius_rf.csv` | 环境 × 环境 pairwise | — | impl_A + 废弃映射 |
| `robust_v2/report/confusion_pattern_similarity_rf.png` | 任务 × 任务 pairwise | — | **impl_C（含对角线）** |

**注意 `cpd_vs_iid_max` 的语义**：它是"对三个 IID CM 分别算 `CPD_y` 后取**最大值**"，
不是"vs 某个特定环境"。在 `loro_R2_R4_to_R3` 上它恰好等于 vs `single_round_R3` 的值
（0.8933110696945264），因为 R3 正是最大者——这是巧合，不是定义。

---

## 4. 三个历史值的归因（协议 §11）

三个历史值全部来自**同一个任务** `loro_R2_R4_to_R3`（RF、`all_features`）。
其差异 **100% 来自基准选择**，与公式无关——三份历史实现在数学上等价（§3.1）。

| 历史值 | 出处脚本 | 基准 | `cpd_core` 复现值 | 落盘产物核对 |
|---|---|---|---|---|
| **0.8397** | `cpd_comprehensive_analysis.py` | vs 三个 IID CM 的 `CPD_y` **均值** | `0.8396605380838675` | `cpd_comparison.csv` ✓ |
| **0.801** | `controlled_cpd_experiment.py` | vs 单个 `joint_R2_R3_R4` CM | `0.8014316033268614` | `controlled_cpd_data.csv` ✓ |
| **0.1521** | `controlled_cpd_experiment_v2.py` | 已废弃六环境 pairwise 均值 | `0.1521129604852716` | `controlled_cpd_data_v2.csv` ✓ |

全部在 `1e-10` 容差内复现，且与落盘 CSV 交叉核对一致。

### 4.1 分解

**0.8397** = 对三个 IID CM 逐个算 `CPD_y` 后取算术平均：

| 基准 | `CPD_y` vs `loro_R2_R4_to_R3` |
|---|---|
| `single_round_R2` | `0.8182260868552512` |
| `single_round_R3` | `0.8933110696945264` |
| `single_round_R4` | `0.8074444577018252` |
| **mean** | **`0.8396605380838675`** |

**0.801** = `cpd_y(joint_R2_R3_R4, loro_R2_R4_to_R3)` = `0.8014316033268614`（单基准，不取均值）。

**0.1521** = 六环境矩阵中两格的均值：

| 格 | 值 |
|---|---|
| `six_env[R2,R3]` | `0.18904571338687684` |
| `six_env[R4,R3]` | `0.11518020758366636` |
| **mean** | **`0.1521129604852716`** |

### 4.2 为什么 0.1521 小一个量级

**关键**：0.1521 **压根没有用 `loro_R2_R4_to_R3` 这个任务的混淆矩阵**。

`controlled_cpd_experiment_v2.py` 的 `compute_task_cpd(['R2','R4'], ['R3'])` 只查六环境矩阵中
`[R2,R3]` 与 `[R4,R3]` 两格，而这两格按废弃的 `env_mapping` 都指向 **IID 模型**的混淆矩阵
（`single_round_R2` / `single_round_R3` / `single_round_R4`）。

因此 0.1521 度量的是"**两个 IID 环境之间的混淆结构差异**"，而 0.8397 / 0.801 度量的是
"**OOD 任务的混淆结构相对 IID 参照的漂移**"。三者量级关系 `0.1521 < 0.25 < 0.801 < 0.8397`
已由 `test_three_values_share_one_task_and_differ_only_by_baseline` 断言固定。

**这不是三个 CPD 值的分歧，是两个不同问题的答案。** 历史报告若把 0.1521 与 0.8397 并列
讨论"CPD 大小"，属口径混用。

### 4.3 结果根的完备性

`results/robust_v2/` 是**唯一**同时包含 `joint_R2_R3_R4` 与 `jitter_R2_R3_R4_to_R6/R7` 的结果根，
三个历史值的复现全部以它为准。

| 结果根 | 三个 IID | `loro_R2_R4_to_R3` | `joint_R2_R3_R4` | `jitter_*_R6/R7` |
|---|---|---|---|---|
| `robust_v2` | ✓ | ✓ | ✓ | ✓ |
| `gpu_capacity_full_20260703` | ✓ | ✓ | ✗ | ✗ |
| `robustness_scaling_20260706_v2` | ✓ | ✓ | ✗ | ✗ |

共享任务的混淆矩阵在各根之间**逐元素完全相同**（`test_hist_0_8397_is_root_independent` 断言），
因此 0.8397 与结果根无关；0.801 与 0.1521 只能在 `robust_v2` 下复现。

---

## 5. 已废弃的定义（协议 §4.4）

### 5.1 ~~`cpd_env`（无标签、部署前可算）~~ — 明确废弃

混淆矩阵按定义依赖真实标签，任何由真实混淆矩阵计算的环境间距离都需要目标环境标签，
不能宣称部署前可算。V2 的这一表述是**方法学错误**。

部署前可算的无标签指标只有 `UDS`，它不使用混淆矩阵，只使用模型预测之间的分歧。

### 5.2 ~~`six_env_off_diag_frobenius_rf.csv` 六环境矩阵~~ — 明确废弃

**废弃原因**：其 `env_mapping`（`six_env_confusion_similarity.py` 第 33-39 行）把
R2/R3/R4 指向 `single_round_*`（**IID 模型**）、R5/R6/R7 指向 `position_*`/`jitter_*`（**OOD 模型**），
**矩阵不同质**——同一个矩阵里混了两类根本不同的模型状态。

```python
# 已废弃，不得用于任何新分析
env_mapping = {
    'R2': 'single_round_R2',            # IID 模型
    'R3': 'single_round_R3',            # IID 模型
    'R4': 'single_round_R4',            # IID 模型
    'R5': 'position_R2_R3_R4_to_R5',    # OOD 模型
    'R6': 'jitter_R2_R3_R4_to_R6',      # OOD 模型
    'R7': 'jitter_R2_R3_R4_to_R7',      # OOD 模型
}
```

**替代方案**：协议 §8.5 的 G0 网格中 `|S|=1` 的 30 个有序对构成**同质**的环境×环境拓扑矩阵。
`six_env_confusion_similarity.py` 的 `env_mapping` 应改读 G0 的 `|S|=1` 结果（协议 §20.2）。

**旧文件处置**：移入 `legacy/`。在此之前，`test_hist_0_1521_six_env_pairwise` 会重建该矩阵
并与落盘 CSV 逐格核对——**该测试只为复现历史数字，不构成对该口径的认可**。

### 5.3 ~~"无标签 CPD"这一叫法~~ — 统一改称 `UDS`

---

## 6. P0 阶段执行发现

以下三项是 P0 执行中发现的**事实**，不是设计变更。按协议冻结条款，任何据此的设计调整
都必须走 §23 Change Log，并落在五种允许例外之内。**本文档不做任何调整，只记录。**

### 6.1 `min_err = 20` 使 `CPD_dir` 在 IID 参照下完全未定义

逐行原始误分类计数（RF、`all_features`、`results/robust_v2/`）：

| 任务 | Camera | Light_T1 | Light_XM | Sensor | Socket | 达门槛行数 |
|---|---:|---:|---:|---:|---:|---:|
| `single_round_R2` | 0 | 12 | 11 | 0 | 0 | **0/5** |
| `single_round_R3` | 2 | 0 | 11 | 9 | 0 | **0/5** |
| `single_round_R4` | 1 | 8 | 8 | 14 | 0 | **0/5** |
| `joint_R2_R3_R4` | 4 | 28 | 46 | 36 | 0 | 3/5 |
| `loro_R2_R3_to_R4` | 55 | 74 | 359 | 55 | 0 | 4/5 |
| `loro_R2_R4_to_R3` | 3 | 275 | 99 | 296 | 0 | 3/5 |
| `loro_R3_R4_to_R2` | 35 | 140 | 74 | 99 | 0 | 4/5 |
| `position_R2_R3_R4_to_R5` | 63 | 277 | 63 | 121 | 0 | 4/5 |
| `jitter_R2_R3_R4_to_R6` | 113 | 128 | 169 | 96 | 0 | 4/5 |
| `jitter_R2_R3_R4_to_R7` | 37 | 108 | 122 | 90 | 0 | 4/5 |

**后果**：三个 IID `single_round` CM **0/5 行**达到 `n_err ≥ 20`。因此任何以 IID single_round CM
为参照的 `CPD_dir` 是**完全未定义**（`nan`），不是"部分缺失"。这直接影响：

- 协议 §6 的 **X1**：「IID vs OOD 的 `CPD_y` / `CPD_dir`」——`CPD_dir` 半边无法用 IID single_round 作参照；
- 协议 §13 的 **E2 模型 M2** = M0 + `CPD_dir`——逐任务 `CPD_dir` 若以 IID 为参照则全为 `nan`。

**可选参照（均在冻结协议内，不需改设计）**：`joint_R2_R3_R4` 作参照可得 3/5 行。
是否改用它属基准选择，**需由你决定并记入 Change Log**——我没有替你改。

> 该事实已由 `test_cpd_dir_is_undefined_for_iid_single_round_references` 固定成断言：
> 若将来有人改了 `min_err` 或换了 CM，测试会失败，从而强制走 Change Log 而不是静默漂移。

### 6.2 `Socket` 行在**所有** 10 个任务中误分类计数均为 0

`Socket` 永久被 `CPD_dir` 排除。因此 `CPD_dir` 在自采数据上**最多是一个 4 行的量**，
而协议 §3.1 的 5 类设定下 `CPD_y` 有 20 个离对角项（协议 §16.1 已注明"有效仅 12 项"）。

这与协议 §10 的观察一致（Socket 在 110 条结果中 96 条 F1 = 1.000），也与 §10 主指标 2
（4-class macro-F1，去除 Socket）的动机一致。**支持在正文中把 Socket 作为"免费类"处理。**

### 6.3 `confusion_pattern_analysis.py` 的产物不是 `CPD_y`

见 §3.1 impl_C。该脚本的 Frobenius 矩阵含对角线，与 `CPD_y` 最大偏差 0.52。
按协议 §20.2「删私有 `compute_cpd` / `_normalize_cm`，改 import `cpd_core`」，
该脚本改造时**必须同时决定**：

- 若意图是 `CPD_y` → 补上 `off()` 调用（这是修 bug，属冻结例外第 2 种"代码实现与协议不一致"）；
- 若意图是"含 recall 的整体混淆矩阵距离" → 改名为别的量，不得叫 `CPD`。

**未决**，等你判定。当前未改动该脚本。

### 6.4 `r = -0.630` 的显著性是口径依赖的

同样 11 个任务、同样的 `gain_absolute`，只换 CPD 口径：

| CPD 口径 | `r` | `p` |
|---|---:|---:|
| 0.801 口径（基准 = `joint_R2_R3_R4`） | `-0.6295` | `0.0379` |
| 废弃六环境口径（v2） | `-0.2065` | `0.5424` |

完整敏感性分析见 [`results/p0_audit/R630_SENSITIVITY_CONCLUSION.md`](../results/p0_audit/R630_SENSITIVITY_CONCLUSION.md)。
结论：原「统计显著」主张不成立，按协议 §2.3 / §11 降为探索性结果。

这也说明协议 §8.3「禁止 selection-on-test」在本项目中不是形式条款——
口径选择的自由度足以翻转显著性判断，必须在内层完成。

---

## 7. 回归测试

```bash
python3 code/scripts/analysis/test_cpd_core.py     # 规范入口，退出码 0 = 门通过
```

测试也写成 pytest 兼容形式（函数名以 `test_` 开头、断言即校验），但**当前环境
（conda env `iotcls`）未安装 pytest**，所以规范入口是上面的独立运行方式。
若后续装了 pytest，`pytest code/scripts/analysis/test_cpd_core.py -v` 亦可用。

15 项，全部通过：

| # | 测试 | 覆盖 |
|---|---|---|
| 1 | `test_hist_0_8397_vs_iid_mean` | 历史值 0.8397 + 落盘核对 |
| 2 | `test_hist_0_8397_is_root_independent` | 值与结果根无关；CM 逐元素相同 |
| 3 | `test_hist_0_801_vs_joint` | 历史值 0.801 + 落盘核对 |
| 4 | `test_hist_0_1521_six_env_pairwise` | 历史值 0.1521 + 六环境矩阵逐格核对 |
| 5 | `test_three_values_share_one_task_and_differ_only_by_baseline` | 三值同任务、差异只来自基准、量级关系 |
| 6 | `test_off_diag_row_mass_equals_one_minus_recall` | §4.3 代数动机 |
| 7 | `test_cpd_y_basic_properties` | 同一性 / 对称性 / 非负性 |
| 8 | `test_cpd_dir_flags_low_error_rows` | `min_err` 剔除与标注 |
| 9 | `test_cpd_dir_is_undefined_for_iid_single_round_references` | §6.1 / §6.2 发现锁定 |
| 10 | `test_cpd_dir_undefined_when_all_rows_below_threshold` | 全行缺失 → `nan` |
| 11 | `test_cpd_dir_removes_error_magnitude` | 方向不变时 `CPD_dir` 不变 |
| 12 | `test_uds_signature_takes_no_labels` | §17.1 条件 5 无泄漏审计 |
| 13 | `test_uds_is_invariant_to_label_permutation_of_truth` | `UDS` 只依赖预测 |
| 14 | `test_uds_rejects_mismatched_model_sets` | 输入校验 |
| 15 | `test_uds_detects_disagreement_drift` | `UDS` 有效性 |

**环境**（P0 执行时实测）：Python 3.11.15、numpy 2.4.6、pandas 3.0.3、scipy 1.17.1、
sklearn 1.9.0、xgboost 3.2.0、lightgbm 4.6.0。

> 本测试套件只读取**落盘的混淆矩阵 CSV**，不重训模型，因此与上述库版本无关。
> 协议 §22.1 的 9/7 退出条件「原有任务指标回归一致（容差 1e-6）」**需要重训**，
> 而现有结果产出于 2026-06-22，当时的库版本未知——该风险见 §8。

---

## 8. 待决事项

| # | 事项 | 阻塞对象 | 需要的决定 |
|---|---|---|---|
| 1 | `CPD_dir` 在 IID 参照下未定义（§6.1） | X1、E2 的 M2 | 是否改用 `joint_R2_R3_R4` 作参照；记入 Change Log |
| 2 | `confusion_pattern_analysis.py` 是 bug 还是另一个量（§6.3） | §20.2 的脚本改造 | 判定意图 |
| 3 | 库版本与 2026-06-22 结果产出时不一致 | §22.1 的 9/7「容差 1e-6」退出条件 | 是否需要先固定环境（`requirements.txt` / lockfile）再谈回归 |

---

## 9. Change Log

| 日期 | 修改项 | 原因 | 修改前 → 修改后 |
|---|---|---|---|
| 2026-08-26 | 本文档创建 | 协议 §22.2 第 1、3 项 | — |

对协议本身的任何修改记入 [experiment_protocol_final.md](experiment_protocol_final.md) 第 23 节，不记在此处。
