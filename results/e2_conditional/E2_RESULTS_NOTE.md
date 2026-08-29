# E2 —— CPD 条件解释力分层回归：口径与数表

> **判定由审阅方按协议 §13 分支作出，本文档不含解读。**
> 本文件只记录口径、输入、验收证据与数表。任何机制语言、结论性表述、显著性叙事都不在本文件范围内。
> 不报 p 值（§15.3：禁止未校正多重检验）。

生成日期：2026-08-29
产物目录：`results/e2_conditional/`
实现脚本：`code/scripts/analysis/e2_conditional_explanation.py`
执行口径：`docs/EXECUTION_PLAN_20260829.md` 决策 **D9**（先写后看，commit `cac9b3f`）
        + v1.3 **"D9 追记"** 三项裁定 A / B / C

---

## 1. 口径

### 1.1 因变量（DV）

| 项 | 取值 |
|---|---|
| 来源 | `results/e1_oof_arms_g0/e1_arms_raw_all.csv` |
| 列 | `gain_absolute` |
| 主口径臂 | **B 臂**（§9.1 分组 OOF） |
| 敏感性臂 | A 臂（历史随机 OOF 口径） |
| 种子 | 42 |
| 样本 | E1-G0-GRID 的 **150** 个 OOD 网格任务 |

任务定义唯一来源 = `code/scripts/core/environment_grid_experiment.py::build_task_grid`（§11）；
本脚本不重复实现任务生成，也不从任务名反解源集合。

### 1.2 M0（7 维，逐字按 §13）

对每类 `c`：`recall_src_IID(c) − recall_tgt(c)`（5 维）+ 该 5 维差向量的 L2 范数 + 最大值。

- 类别轴序：`Camera, Light_T1, Light_XM, Sensor, Socket`（`docs/CPD_DEFINITIONS.md` §1.2）。
- `recall_src_IID(c)` = 源集合 S 中各环境 G0 IID（**time_block**，§8.4 诚实域内参照，与 D4 同一参照）
  RF 混淆矩阵逐类 recall 的**算术均值**。
- `recall_tgt(c)` = 该任务 RF 测试混淆矩阵逐类 recall。
- 逐类 recall = `C_ii / Σ_j C_ij`；行和为 0 时该行除以 1（沿用 `cpd_core.normalize_cm` 历史口径）。
- 模型口径 = **RF**（§7 基线 1）。敏感性 = rf / xgboost / lightgbm 三基模型逐类 recall 均值。
- **"最大值" = 字面读法（差向量的最大元素）**——见 §5.3 裁定 C。

M0 的 7 个回归项（设计矩阵列序固定）：
`d_recall_Camera, d_recall_Light_T1, d_recall_Light_XM, d_recall_Sensor, d_recall_Socket, d_recall_l2, d_recall_max`

### 1.3 M1 / M2

- **M1 = M0 + `CPD_y`**（全 150 行拟合）
- **M2 = M0 + `CPD_dir`**（限于 `CPD_dir` 有定义的子集；**M2 的 M0 对照 `M0_sub` 在同一子集重新拟合**，否则增量不可比）
- `ref` = 源集合 S 各环境的 IID CM（time_block，RF）；`tgt` = 该任务 RF 测试 CM。
- CPD 计算**只经 `cpd_core`**（§11 唯一实现），本文件不含任何私有 CPD 公式副本。`min_err = 20`（§4.3）。
- **多源参照构造沿用 0.8397 历史口径的同一构造**：对每个参照 CM 分别算指标，再对这些值取算术平均
  （**不是**先平均 CM）。出处：`code/scripts/analysis/test_cpd_core.py:100-101`；
  文档转录见 `docs/CPD_DEFINITIONS.md` §4.1。

### 1.4 估计与观测量（§13 双通道）

1. **标准化 OLS**：X、y 各自 z-score 后带截距最小二乘；标准化系数即 β。
   报 R²、adj R²、增量 R²、Cohen's f² = ΔR² / (1 − R²_full)。
   零方差列的 std 记 1（该列 z 分全 0）；设计阵秩亏时 `lstsq` 给最小范数解并置 `design_rank_deficient=True`。
2. **按目标环境聚类的 bootstrap**（§15.1）：重抽单位 = **6 个目标环境**（有放回），
   被抽中的环境整块进入。**B = 10000**，`numpy.random.default_rng(seed=42)`，percentile 2.5 / 97.5。
   §15.2 禁止点级 bootstrap、禁止把 n=150 当独立样本。
3. **留一目标环境 CV**（§15.5）：6 折（每折 = 1 个目标环境），标准化参数只在训练折上估计，
   报 pooled 样本外 MSE 与逐折 MSE。
4. 每 |S| 分层的描述性均值（**不加回归项**）与逐目标环境完整结果表（§15.4）。

预注册通道之外**不加任何协变量**。

### 1.5 三项裁定的引用

| 裁定 | 内容 | 本次执行的落实 |
|---|---|---|
| **A** | canonical 分析环境 = `code/requirements-lock.txt` 所锁的 **iotcls**；此后"逐位一致"类验收一律限定在 canonical 环境内执行；跨环境 ULP 差异按可复现性事实登记 | 全部产物由 `~/anaconda3/envs/iotcls/bin/python` 生成（python 3.11.15 / numpy 2.4.6 / pandas 3.0.3）；验收门 b 在该环境内判定 |
| **B** | 多源 `CPD_dir` 的 NaN 传播：primary = **lenient**（`nanmean`）；**strict 并列完整报告且必须带秩亏标记**；两变体终身并列，所有表格与本 NOTE 同时呈现两者 | 每张表都带 `cpd_dir_variant` + `cpd_dir_role` 两列（`lenient=primary`、`strict=parallel_reported`）；秩亏标记列见 §5.2；两个变体的 bootstrap 复制明细都落盘 |
| **C** | `d_recall_max` 维持**字面读法**（变化向量的最大元素）；与最大绝对值读法在 16/150 任务上不同的事实记入 provenance | 实现用 `np.max(delta)`；16 个任务逐条列入 `provenance.json → diagnostics.d_recall_max_reading` 与 `rulings.C_d_recall_max_reading` |

`CPD_dir` 两个 NaN 传播变体的定义：

- `strict` —— 逐字用 `np.mean`：任一参照未定义 → 该任务未定义（NaN）；
- `lenient` —— 用 `np.nanmean`：在已定义的参照上取均值，全部未定义才 NaN。

---

## 2. 输入与 sha256

| 输入 | 路径 | sha256 |
|---|---|---|
| DV 表 | `results/e1_oof_arms_g0/e1_arms_raw_all.csv` | `79dd7fe63bc000af439d280af54ba296890c2f49aed67dd8fc08bb4b6172c721` |
| 门 b 交叉验证矩阵 | `results/g0_environment_grid/env_topology_cpd_y_ref_time_block_rf.csv` | `d96056418fbf7f118a2045323ad936d43b2ff2192671184e2be38da3be748b0b` |
| 混淆矩阵（486 个 CSV，汇总摘要） | `results/g0_environment_grid/raw_all/<task>/all_features/<model>/confusion_matrix.csv` | `128eee9ea98ca99968955520054780abeb70221b0840b086c24a4b5e792d42cb` |

混淆矩阵汇总摘要的复算规则：对 486 个文件逐个算 sha256，按仓库相对路径字典序拼成
`<path>:<sha256>\n` 行，再对拼接结果算 sha256（`n_files_hashed = n_files_expected = 486`）。
486 = 150 任务 × 3 基模型 + 6 环境 × 2 IID 划分 × 3 基模型。

其他口径依赖：`cpd_core`（§11 唯一实现）、`environment_grid_experiment.build_task_grid`（§11 任务定义唯一来源）。

**运行环境（裁定 A canonical）**

| 项 | 值 |
|---|---|
| 解释器 | `/home/lmy/anaconda3/envs/iotcls/bin/python` |
| python | 3.11.15 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.1 |
| xgboost / lightgbm | 3.2.0 / 4.6.0 |
| joblib | 1.5.3 |
| platform | Linux-5.4.0-189-generic-x86_64-with-glibc2.31 |
| git HEAD | `00f225f6d5298d758bef346a305c1168dc6adc0b`（working tree dirty） |

bootstrap 复制统计量 md5（primary 变体）：`strict = 8abe42edfa271ed1e37157d1b4586446`、
`lenient = f52f40a425bf87fc46f341efd1e49685`。

---

## 3. 验收四门证据

四门**全部通过**（`e2_acceptance.json → all_passed = true`）。任一门不过则脚本非零退出且不写任何输出。

### 门 a —— M0 特征独立代码路径手算比对（容差 1e-12）

用标准库 `csv` 逐行读、纯 Python 算术求和，刻意不复用脚本的 `load_cm` / `recall_vector` / numpy 聚合。

| 任务 | \|S\| | 源集合 | max abs dev | 容差 | 判定 |
|---|---|---|---|---|---|
| `g0_R2_to_R3` | 1 | R2 | **0.000e+00** | 1e-12 | PASS |
| `g0_R2_R3_R4_to_R5` | 3 | R2;R3;R4 | **0.000e+00** | 1e-12 | PASS |

### 门 b —— `CPD_y` 在 30 个 |S|=1 任务上与 D4 拓扑矩阵逐位一致

两个子检验都是硬门，**均在 canonical 环境（裁定 A）内判定**：

| 子检验 | 对比对象 | 逐位一致 | max\|Δ\| | 判定 |
|---|---|---|---|---|
| b1 | **落盘** `env_topology_cpd_y_ref_time_block_rf.csv` | **30 / 30** | **0.000e+00** | PASS |
| b2 | 由 D4 生成器（`six_env_confusion_similarity`）在**当前解释器**下重算的同一矩阵 | **30 / 30** | **0.000e+00** | PASS |

"逐位一致" = float64 位模式（`struct.pack("<d", ·)`）相同，非数值容差比较。
逐格证据（30 行，含 `repr` 级数值）落盘于 `e2_acceptance.json → gates[1].cells`。
示例：`g0_R2_to_R3` 单元格 `[R2,R3]`，E2 = `1.2544092722335234`，
落盘 = `1.2544092722335234`，D4 重算 = `1.2544092722335234`。

判定解释器：python 3.11.15 / numpy 2.4.6 / pandas 3.0.3 / `/home/lmy/anaconda3/envs/iotcls/bin/python`。

### 门 c —— bootstrap 同种子双跑复制统计量 md5 相同

裁定 B 的并列要求：`strict` 与 `lenient` **两个变体都跑**，两者都必须复现。B = 2000，seed = 42。

| `cpd_dir` 变体 | md5 run1 | md5 run2 | 一致 | 秩亏复制数（M0/M1/M0_sub/M2） |
|---|---|---|---|---|
| strict | `8aac437ef18f33576cd3b57773a30424` | `8aac437ef18f33576cd3b57773a30424` | PASS | 2000 / 2000（0 / 0 / **2000** / **2000**） |
| lenient | `58200076401737522aa579bfe6808878` | `58200076401737522aa579bfe6808878` | PASS | 0 / 2000（0 / 0 / 0 / 0） |

### 门 d —— 行数核对

9 项全部通过：

| 项 | 期望 | 实得 |
|---|---|---|
| `main_table_n` | 150 | 150 |
| `M0_n[strict]` / `M1_n[strict]` | 150 / 150 | 150 / 150 |
| `M0_sub_n[strict]` / `M2_n[strict]` | 26 / 26 | 26 / 26 |
| `M0_n[lenient]` / `M1_n[lenient]` | 150 / 150 | 150 / 150 |
| `M0_sub_n[lenient]` / `M2_n[lenient]` | 119 / 119 | 119 / 119 |

---

## 4. 主表数字（primary 变体：B 臂 DV / time_block IID 参照 / RF recall）

### 4.1 模型拟合与增量（两个 `CPD_dir` 变体并列，裁定 B）

| `cpd_dir` 变体 | 角色 | 模型 | n | R² | adj R² | R² 95% CI | ΔR² | ΔR² 95% CI | Cohen's f² | LOEO-CV pooled MSE |
|---|---|---|---|---|---|---|---|---|---|---|
| **lenient** | **primary** | M0 | 150 | 0.146982 | 0.104932 | [0.101046, 0.345733] | — | — | — | 0.008130 |
| **lenient** | **primary** | M1 | 150 | 0.153618 | 0.105596 | [0.125312, 0.363998] | **+0.006636** | [0.000067, 0.102501] | 0.007840 | 0.008500 |
| **lenient** | **primary** | M0_sub | 119 | 0.103217 | 0.046663 | [0.084860, 0.347045] | — | — | — | 0.009278 |
| **lenient** | **primary** | M2 | 119 | 0.178047 | 0.118269 | [0.161069, 0.465954] | **+0.074830** | [0.023126, 0.213893] | 0.091040 | 0.009638 |
| strict | parallel | M0 | 150 | 0.146982 | 0.104932 | [0.101046, 0.345733] | — | — | — | 0.008130 |
| strict | parallel | M1 | 150 | 0.153618 | 0.105596 | [0.125312, 0.363998] | **+0.006636** | [0.000067, 0.102501] | 0.007840 | 0.008500 |
| strict | parallel | M0_sub | 26 | 0.486963 | 0.287449 | [0.356177, 0.933316] | — | — | — | 0.053019 |
| strict | parallel | M2 | 26 | 0.684145 | 0.535507 | [0.580907, 0.989490] | **+0.197181** | [0.001428, 0.234066] | 0.624277 | 0.028277 |

M0 与 M1 在两个 `cpd_dir` 变体下**数值相同**（两者都在全 150 行拟合，与 `CPD_dir` 无关）；
两行并列呈现只为满足裁定 B 的"所有表格同时呈现两者"要求。

### 4.2 LOEO-CV 的 MSE 改善（相对各自基线）

| `cpd_dir` 变体 | 模型 | 基线 | pooled MSE | 基线 pooled MSE | 绝对改善 | 相对改善 |
|---|---|---|---|---|---|---|
| **lenient** | M1 | M0 | 0.008500 | 0.008130 | **−0.000369** | **−0.045430** |
| **lenient** | M2 | M0_sub | 0.009638 | 0.009278 | **−0.000359** | **−0.038736** |
| strict | M1 | M0 | 0.008500 | 0.008130 | **−0.000369** | **−0.045430** |
| strict | M2 | M0_sub | 0.028277 | 0.053019 | **+0.024742** | **+0.466664** |

（正值 = 加入 CPD 项后样本外 MSE 下降；负值 = 上升。逐折 MSE 见 `e2_loeo_cv.csv` 与 §4.5。）

### 4.3 CPD 项的标准化系数 β（环境聚类 bootstrap 95% CI，B=10000）

| `cpd_dir` 变体 | 角色 | 模型 | 项 | β | 95% CI | n_boot_finite | 该模型秩亏 | 秩亏复制数 /10000 |
|---|---|---|---|---|---|---|---|---|
| **lenient** | **primary** | M1 | `cpd_y` | **−0.245413** | [−0.756975, 0.320216] | 10000 | False | 0 |
| **lenient** | **primary** | M2 | `cpd_dir_lenient` | **−0.295607** | [−0.555496, −0.175361] | 10000 | False | 0 |
| strict | parallel | M1 | `cpd_y` | **−0.245413** | [−0.756975, 0.320216] | 10000 | False | 0 |
| strict | parallel | M2 | `cpd_dir_strict` | **−0.514428** | [−1.625200, −0.125250] | 10000 | **True** | **10000** |

### 4.4 M0 项的标准化系数 β（primary）

M1（n=150，两个 `cpd_dir` 变体相同）:

| 项 | β | 95% CI |
|---|---|---|
| `d_recall_Camera` | −0.284117 | [−0.375495, 0.004644] |
| `d_recall_Light_T1` | 0.074982 | [−0.270901, 0.238637] |
| `d_recall_Light_XM` | −0.158478 | [−0.515811, 0.049817] |
| `d_recall_Sensor` | 0.114151 | [−0.018094, 0.303086] |
| `d_recall_Socket` | −0.067270 | [−0.314199, 0.233011] |
| `d_recall_l2` | 1.291936 | [0.799736, 1.974342] |
| `d_recall_max` | −0.633370 | [−1.156849, −0.197668] |
| `cpd_y` | −0.245413 | [−0.756975, 0.320216] |

M2（两个变体并列）:

| 项 | β (lenient, n=119) | 95% CI | β (strict, n=26) | 95% CI |
|---|---|---|---|---|
| `d_recall_Camera` | −0.314657 | [−0.474618, 0.061912] | −0.213376 | [−1.836380, 1.388233] |
| `d_recall_Light_T1` | −0.259288 | [−0.756293, 0.005989] | −0.764775 | [−2.246267, −0.061998] |
| `d_recall_Light_XM` | −0.127309 | [−0.673382, 0.175183] | −0.570632 | [−2.633699, 0.261726] |
| `d_recall_Sensor` | −0.238681 | [−0.645596, 0.126776] | −0.343812 | [−0.847127, 1.459726] |
| `d_recall_Socket` | −0.167226 | [−0.518942, 0.244116] | **0.000000**（零方差项） | [−0.000000, 0.000000] |
| `d_recall_l2` | 0.057013 | [−0.490903, 0.584804] | 0.156204 | [−0.335960, 1.311063] |
| `d_recall_max` | 0.578186 | [−0.216047, 1.236322] | 0.969321 | [−0.585349, 2.352065] |
| `CPD_dir` 项 | −0.295607 | [−0.555496, −0.175361] | −0.514428 | [−1.625200, −0.125250] |

### 4.5 逐目标环境结果（§15.4，primary）

| 目标环境 | n_tasks | DV 均值 | DV 标准差 | `cpd_y` 均值 | `d_recall_l2` 均值 | strict 有定义 | lenient 有定义 | LOEO MSE M0 | M1 | M0_sub(len) | M2(len) | M0_sub(str) | M2(str) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 | 25 | −0.077382 | 0.101291 | 0.533780 | 0.485088 | 3 | 21 | 0.008393 | 0.008283 | 0.013410 | 0.013685 | 0.006778 | 0.002816 |
| R3 | 25 | −0.048179 | 0.086432 | 0.711030 | 0.623523 | 7 | 22 | 0.008791 | 0.008640 | 0.009216 | 0.007586 | 0.130703 | 0.068968 |
| R4 | 25 | −0.063067 | 0.107178 | 0.672074 | 0.552880 | 3 | 18 | 0.012937 | 0.013606 | 0.011109 | 0.013167 | 0.001114 | 0.001666 |
| R5 | 25 | −0.064288 | 0.087571 | 0.712922 | 0.761732 | 7 | 22 | 0.006964 | 0.006764 | 0.008046 | 0.007745 | 0.031699 | 0.014762 |
| R6 | 25 | −0.092491 | 0.085405 | 0.716278 | 0.764625 | 3 | 18 | 0.006313 | 0.008120 | 0.006154 | 0.006075 | 0.047041 | 0.012572 |
| R7 | 25 | −0.091971 | 0.082497 | 0.689596 | 0.648545 | 3 | 18 | 0.005384 | 0.005585 | 0.007335 | 0.009772 | 0.025629 | 0.032644 |

### 4.6 每 |S| 分层描述性均值（不加回归项，primary）

| \|S\| | n_tasks | DV 均值 | DV 标准差 | DV 最小 | DV 最大 | `d_recall_l2` 均值 | `d_recall_max` 均值 | `cpd_y` 均值 | `cpd_dir` strict 均值 (n) | `cpd_dir` lenient 均值 (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30 | −0.069981 | 0.095476 | −0.353134 | 0.002305 | 0.977112 | 0.697641 | 1.002951 | 1.107676 (14) | 1.107676 (14) |
| 2 | 60 | −0.093138 | 0.105568 | −0.248235 | 0.147019 | 0.677876 | 0.533032 | 0.715782 | 1.000514 (10) | 0.671383 (48) |
| 3 | 60 | −0.054113 | 0.070653 | −0.232086 | 0.056923 | 0.432065 | 0.354789 | 0.464275 | 0.858859 (2) | 0.538807 (57) |

---

## 5. M2 覆盖率（`CPD_dir` 的 §4.3 `n_err ≥ 20` 准入）

逐任务明细见 `e2_m2_coverage.csv`（150 行，含 `per_ref_cpd_dir` / `dir_included_rows` / `n_ref_dir_defined`）。

**总覆盖：strict 26 / 150；lenient 119 / 150。**

### 5.1 按 |S| 与目标环境

| \|S\| | 任务数 | strict 有定义 | lenient 有定义 |
|---|---|---|---|
| 1 | 30 | 14 | 14 |
| 2 | 60 | 10 | 48 |
| 3 | 60 | 2 | 57 |
| **合计** | **150** | **26** | **119** |

|S|=1 时两变体必然相同（单参照，`mean` 与 `nanmean` 等价）。

| 目标环境 | strict 有定义 | lenient 有定义 |
|---|---|---|
| R2 | 3 | 21 |
| R3 | 7 | 22 |
| R4 | 3 | 18 |
| R5 | 7 | 22 |
| R6 | 3 | 18 |
| R7 | 3 | 18 |

### 5.2 秩亏标记列（裁定 B 要求）

| 表 | 秩亏标记列 |
|---|---|
| `e2_regression_main.csv` | `design_rank` / `design_n_params` / `design_rank_deficient` / `zero_variance_terms` / `n_boot_degenerate_replicates_this_model` / `n_boot_replicates` |
| `e2_coefficients.csv` | `design_rank_deficient` / `n_boot_degenerate_replicates_this_model` / `B` |
| `e2_bootstrap_summary.csv` | `n_degenerate_replicates_any_model` / `n_degenerate_replicates_by_model` |

primary 变体的秩状态：

| `cpd_dir` 变体 | 模型 | n | design_rank | design_n_params | 秩亏 | 零方差项 | 秩亏复制数 /10000 |
|---|---|---|---|---|---|---|---|
| lenient | M0 | 150 | 8 | 8 | False | — | 0 |
| lenient | M1 | 150 | 9 | 9 | False | — | 0 |
| lenient | M0_sub | 119 | 8 | 8 | False | — | 0 |
| lenient | M2 | 119 | 9 | 9 | False | — | 0 |
| strict | M0 | 150 | 8 | 8 | False | — | 0 |
| strict | M1 | 150 | 9 | 9 | False | — | 0 |
| strict | M0_sub | 26 | **7** | 8 | **True** | `d_recall_Socket` | **10000** |
| strict | M2 | 26 | **8** | 9 | **True** | `d_recall_Socket` | **10000** |

---

## 6. 敏感性数表

三个敏感性臂各自的完整回归表另存为 `e2_sens_dv_a_arm.csv` / `e2_sens_ref_random.csv` /
`e2_sens_recall_3model.csv`；下表为汇总（两个 `CPD_dir` 变体并列）。

### 6.1 `sens_dv_a_arm` —— DV 换 A 臂（历史随机 OOF 口径）

| `cpd_dir` | 模型 | n | R² | ΔR² | ΔR² 95% CI | f² | CV MSE | CPD 项 β | β 95% CI | 秩亏 |
|---|---|---|---|---|---|---|---|---|---|---|
| lenient | M0 | 150 | 0.127671 | — | — | — | 0.003441 | — | — | False |
| lenient | M1 | 150 | 0.394984 | +0.267314 | [0.119488, 0.395893] | 0.441830 | 0.002621 | −1.557625 | [−2.112383, −1.145163] | False |
| lenient | M0_sub | 119 | 0.210354 | — | — | — | 0.003998 | — | — | False |
| lenient | M2 | 119 | 0.692107 | +0.481753 | [0.297660, 0.566801] | 1.564677 | 0.001620 | −0.750046 | [−0.871292, −0.665744] | False |
| strict | M0_sub | 26 | 0.595710 | — | — | — | 0.022162 | — | — | **True** (`d_recall_Socket`) |
| strict | M2 | 26 | 0.867935 | +0.272225 | [0.001195, 0.426498] | 2.061294 | 0.004519 | −0.604443 | [−1.049964, −0.109956] | **True** (`d_recall_Socket`) |

（M0 / M1 与 `cpd_dir` 变体无关，strict 行数值同 lenient。）

### 6.2 `sens_ref_random` —— IID 参照换 random 划分

| `cpd_dir` | 模型 | n | R² | ΔR² | ΔR² 95% CI | f² | CV MSE | CPD 项 β | β 95% CI | 秩亏 |
|---|---|---|---|---|---|---|---|---|---|---|
| lenient | M0 | 150 | 0.135706 | — | — | — | 0.008115 | — | — | **True** (`d_recall_Socket`) |
| lenient | M1 | 150 | 0.145197 | +0.009492 | [0.000018, 0.102209] | 0.011104 | 0.008425 | −0.407476 | [−1.001200, 0.237007] | **True** (`d_recall_Socket`) |
| lenient | M0_sub | 94 | 0.048165 | — | — | — | 0.011842 | — | — | **True** (`d_recall_Socket`) |
| lenient | M2 | 94 | 0.163251 | +0.115086 | [0.023541, 0.297403] | 0.137540 | 0.014253 | −0.377304 | [−0.644248, −0.201498] | **True** (`d_recall_Socket`) |
| strict | M0_sub | 14 | 0.872679 | — | — | — | 0.017040 | — | — | **True** (`d_recall_Socket`) |
| strict | M2 | 14 | 0.915209 | +0.042530 | [0.000000, 0.102329] | 0.501589 | 0.056641 | −0.771726 | [−10.553310, 5.798843] | **True** (`d_recall_Socket`) |

该臂 `CPD_dir` 覆盖率：strict 14 / 150；lenient 94 / 150。
该臂在**全 150 行**上 `d_recall_Socket` 即为零方差项（M0 / M1 也秩亏），与 primary 臂不同 —— 见 §7 事实 (2) 附注。

### 6.3 `sens_recall_3model` —— M0 的 recall 换三基模型均值（CPD 仍为 RF 口径）

| `cpd_dir` | 模型 | n | R² | ΔR² | ΔR² 95% CI | f² | CV MSE | CPD 项 β | β 95% CI | 秩亏复制数 /10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| lenient | M0 | 150 | 0.145838 | — | — | — | 0.008336 | — | — | 0 |
| lenient | M1 | 150 | 0.145990 | +0.000152 | [0.000004, 0.020545] | 0.000178 | 0.008422 | 0.037592 | [−0.209160, 0.601744] | 0 |
| lenient | M0_sub | 119 | 0.154255 | — | — | — | 0.008754 | — | — | 0 |
| lenient | M2 | 119 | 0.173790 | +0.019535 | [0.000694, 0.083965] | 0.023644 | 0.009031 | −0.174277 | [−0.420604, −0.037477] | 0 |
| strict | M0_sub | 26 | 0.531924 | — | — | — | 0.035430 | — | — | 89 |
| strict | M2 | 26 | 0.728054 | +0.196130 | [0.000011, 0.196136] | 0.721212 | 0.020557 | −0.615326 | [−2.438130, 1.619128] | 89 |

---

## 7. 三处已登记异常（事实陈述）

以下为可复现性 / 效度事实的登记，不含成因判断以外的任何解读。

**(1) 跨环境 ULP 差异（裁定 A）** ——
D4 的 5 个拓扑 CSV 初版产自 anaconda base（numpy 1.23）；与 canonical 环境 iotcls（numpy 2.4）
产出相比，**9 / 30 格**不同，**max 2.22e-16**；根因为 numpy 1.23→2.4 的 fro-norm 归约路径变化。
D4 自身的 1e-6 硬门两版**都过**。裁定 A 据此把 canonical 分析环境固定为 iotcls，
拓扑 CSV 已在该环境下重生成（sha256 见 §2）。本次 E2 全量运行在 iotcls 内完成，
门 b 的 b1（vs 落盘）与 b2（vs D4 生成器当场重算）**均 30/30 逐位一致、max|Δ| = 0.000e+00**。
此后"逐位一致"类验收一律限定在 canonical 环境内执行；跨环境 ULP 差异按可复现性事实登记
（与 E1-G0-GRID 的线程序发现同类）。

**(2) strict 子集的设计阵秩亏（裁定 B 的效度依据）** ——
在 `CPD_dir` strict 子集（n=26）上，`d_recall_Socket` 为**零方差项**：设计阵 `design_rank = 7`（M0_sub）
/ `8`（M2），少于参数数 8 / 9，`design_rank_deficient = True`；
环境聚类 bootstrap 中 **10000 / 10000** 次复制秩亏。对应的 lenient 子集（n=119）为 **0 / 10000**。
`lstsq` 对秩亏设计阵返回最小范数解，strict 侧 M2 的 `d_recall_Socket` 系数因此为 `0.000000`，
CI 为 `[−0.000000, 0.000000]`。
裁定 B 据此定 primary = lenient（并与 §4.3 逐 CM 准入规则一致），strict 并列完整报告并带秩亏标记。
**透明度声明**：该裁定作出时两变体的诊断数值已可见；因此两变体在所有产出中终身并列，
引用任一变体时同时给出另一变体的数值与覆盖率。
*附注*：秩亏不限于 primary 臂的 strict 侧——`sens_ref_random` 臂在全 150 行上 `d_recall_Socket`
即为零方差项（M0 / M1 / M0_sub / M2 四个模型都秩亏，10000/10000 或 9815/10000 复制秩亏）；
`sens_recall_3model` 臂 strict 侧点估计不秩亏，但 bootstrap 中有 89 / 10000 次复制秩亏。
逐模型秩亏计数见 `e2_regression_main.csv` 与 `e2_bootstrap_summary.csv`。

**(3) `d_recall_max` 的两种读法差异（裁定 C）** ——
`d_recall_max` 维持**字面读法**（差向量的最大元素，`np.max(delta)`）。
与"最大绝对值"读法（`np.max(np.abs(delta))`）在 **16 / 150** 个任务上取值不同：

`g0_R2_R4_R6_to_R7`、`g0_R2_R6_R7_to_R4`、`g0_R2_R6_to_R4`、`g0_R2_R6_to_R7`、
`g0_R3_R4_R6_to_R7`、`g0_R3_R6_to_R7`、`g0_R4_R6_R7_to_R2`、`g0_R4_R6_to_R2`、
`g0_R4_R6_to_R7`、`g0_R5_R6_to_R7`、`g0_R6_R7_to_R2`、`g0_R6_to_R2`、
`g0_R6_to_R3`、`g0_R6_to_R4`、`g0_R6_to_R5`、`g0_R6_to_R7`

逐任务两种读法的取值列于 `provenance.json → diagnostics.d_recall_max_reading.tasks_where_readings_differ`
（同时镜像于 `rulings.C_d_recall_max_reading`）。回归中使用的**始终是字面读法**；
最大绝对值读法只作为诊断列 `_diag_d_recall_max_abs` 存在于 `e2_features.csv`，不进入任何设计矩阵。

---

## 8. 产物清单

| 文件 | 大小 | 行/内容 |
|---|---|---|
| `e2_features.csv` | 298,621 B | 4 变体 × 150 任务 = 600 行逐任务特征（含 `per_ref_cpd_y` / `per_ref_cpd_dir` / `_diag_d_recall_max_abs`） |
| `e2_regression_main.csv` | 7,915 B | 4 变体 × 2 `cpd_dir` 变体 × 4 模型 = 32 行；含 `cpd_dir_role` 与全部秩亏标记列 |
| `e2_coefficients.csv` | 27,876 B | 240 行逐项标准化 β + bootstrap CI + 秩亏标记 |
| `e2_loeo_cv.csv` | 10,816 B | 192 行逐折 MSE（4 变体 × 2 × 4 模型 × 6 环境） |
| `e2_by_source_count.csv` | 3,830 B | 4 变体 × 3 个 \|S\| 分层 = 12 行描述性均值 |
| `e2_by_target_env.csv` | 8,621 B | 4 变体 × 6 环境 = 24 行逐环境结果（§15.4） |
| `e2_m2_coverage.csv` | 17,682 B | 150 行 `CPD_dir` 逐任务覆盖明细（两变体） |
| `e2_bootstrap_replicates_primary_lenient.csv` | 4,480,032 B | 10000 行 × 复制统计量（primary / lenient） |
| `e2_bootstrap_replicates_primary_strict.csv` | 4,438,299 B | 10000 行 × 复制统计量（primary / strict） |
| `e2_bootstrap_summary.csv` | 38,055 B | 176 行统计量汇总（均值 / sd / CI / 秩亏计数 / md5） |
| `e2_sens_dv_a_arm.csv` | 2,224 B | 8 行（该臂 × 2 `cpd_dir` × 4 模型） |
| `e2_sens_ref_random.csv` | 2,338 B | 8 行 |
| `e2_sens_recall_3model.csv` | 2,264 B | 8 行 |
| `e2_acceptance.json` | 17,367 B | 四门完整证据（含门 b 的 30 行逐格 `repr` 数值） |
| `provenance.json` | 12,027 B | §19.2 五要素 + `rulings` A/B/C + `diagnostics` |
| `E2_RESULTS_NOTE.md` | 本文件 | 口径 / 输入 / 验收 / 数表 |

目录合计 9.0 MB。

复现命令（canonical 环境）：

```
~/anaconda3/envs/iotcls/bin/python code/scripts/analysis/e2_conditional_explanation.py \
    --bootstrap-b 10000 --seed 42
```
