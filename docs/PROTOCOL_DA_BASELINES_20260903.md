# 协议：域适应基线重测（CORAL 修复 + 已发表基线）

**冻结日期**：2026-09-03
**状态**：`PROTOCOL_FROZEN_BEFORE_NUMERIC_RUN`
**性质**：修复 CORAL 着色矩阵转置缺陷，在主线 G0 口径下重测域适应族基线。
作废 2026-06 那次 CORAL 运行的推断，不删除其原文。
不重开 E1/E2/G1/RouteB/D12/UNSW-META-MISMATCH 任何判定；不恢复 CPD 与 EC-MDM 的机制、预测或部署身份。
**裁定方**：主线侧。

## 0. 为什么必须重测

`code/scripts/analysis/coral_baseline.py` 第 65 行在行向量约定下取了错误的着色矩阵：

```text
transform 计算  X_aligned = X_c @ W @ C
=> Cov(X_aligned) = C.T @ (W.T @ Cs @ W) @ C = C.T @ C
目标            Ct = L_tgt @ L_tgt.T
故必须          C = L_tgt.T      旧代码取 C = L_tgt
```

构造性检验（`code/tests/test_coral_coloring.py`，5 个随机域对，纯代数无抽样噪声）：
`||C.T @ C - Ct||_F / ||Ct||_F` 旧版 0.616–0.938，修复版 8.1e-17–1.1e-16。
SVD 兜底分支（对称平方根）原本就正确，但 `fit` 里的 `+1e-5·I` 使 Cholesky 几乎总成功，
所以落盘结果走的全是有缺陷分支。

落盘的 `results/robust_v2/report/coral_baseline_results.csv` 的 `cov_reduction_pct`
为 **−113.2% / −479.6% / −886.6%**（协方差差异被放大 2.13×/5.80×/9.87×），
而正确实现按构造应 ≈ +100%。因此 2026-06「CORAL 使 macro-F1 暴跌 37.75%，
说明全局协方差对齐不是瓶颈」的推断**不成立**：域适应族在本项目数据上从未被真正检验。

## 1. 唯一问题

在主线 G0 口径的三个旗舰 LORO 任务上，正确实现的全局分布对齐方法能否缩小跨轮次落差。

## 2. 口径（全部逐字沿用主线，不重实现）

```text
特征缓存  results/robust_v2/raw_all/features_raw_all_w10.csv    102 列 = 8 meta + 94 特征
标签      Camera / Light_T1 / Light_XM / Sensor / Socket
任务      LORO 三个：{R2,R3}→R4   {R2,R4}→R3   {R3,R4}→R2
模型      仅 RF，用 build_model("rf", random_state=42, n_jobs=1, class_count=5)
工具      feature_columns / clean_x / fit_label_encoder / build_model 全部 import
指标      目标域 macro-F1（labels 固定为上述 5 类，zero_division=0）
```

**锚定事实（运行前已核实，作为判据 1 的依据）**：旧 CORAL 脚本的无对齐臂
（0.6592117370350504 / 0.6148161314237031 / 0.8097991283932819）与 G0 网格
`raw_all/g0_R2_R3_to_R4|g0_R2_R4_to_R3|g0_R3_R4_to_R2` 的 `rf/metrics.json`
（0.659212 / 0.614816 / 0.809799）逐任务一致，故本协议的无对齐臂必须复现同一数字。

## 3. 六个臂（机械枚举，无运行时自由度）

| 臂 | 预处理 | 对齐 |
|---|---|---|
| `none_raw` | 无 | 无（参照臂 = 主线 RF） |
| `zscore` | 源域拟合 z-score | 无（对照：单纯缩放是否就够） |
| `coral_raw` | 无 | CORAL |
| `coral_zscore` | 源域拟合 z-score | CORAL |
| `sa_raw` | 无 | SA |
| `sa_zscore` | 源域拟合 z-score | SA |

**CORAL**：Sun, Feng & Saenko, *Return of Frustratingly Easy Domain Adaptation*, AAAI 2016
的无监督闭式解。源域白化后按目标域协方差着色，训练在对齐后的源域、预测在**原始**目标域。
（原文件 docstring 误引 Deep CORAL/ECCV 2016，本协议同时改正引用。）

**SA**：Fernando et al., *Unsupervised Visual Domain Adaptation Using Subspace Alignment*, ICCV 2013。
源/目标各自 PCA 得 `Ps, Pt`（d×k），对齐矩阵 `M = Ps.T @ Pt`，
源投影 `Zs = Xs_c @ Ps @ M`，目标投影 `Zt = Xt_c @ Pt`，RF 在 `Zs` 上训练、在 `Zt` 上预测。
**k 由确定性规则定死**：源域累计解释方差首次 ≥ 0.95 的最小 k，上限 `min(94, 50)`，逐任务落盘。

`z-score` 一律**只用源域**的均值与标准差拟合（标准差为 0 的列除以 1.0），
目标域用同一变换，避免用到目标域统计量之外的信息。CORAL/SA 本身允许用无标签目标域特征。

## 4. 判据（运行前定死）

| # | 判据 | 门槛 | 不通过时 |
|---|---|---|---|
| 1 | **锚定门** | `none_raw` 与 G0 网格 RF 逐任务差 ≤ 1e-6 | `CALIBER_MISMATCH_STOP` |
| 2 | **实现正确性门** | 两个 CORAL 臂逐任务 `cov_reduction_pct >= +95%` | `CORAL_IMPLEMENTATION_INVALID_STOP` |
| 3 | **主判据** | 见下三分支 | — |

`cov_reduction_pct = (1 − ||Cov(对齐后源) − Cov(目标)||_F / ||Cov(原源) − Cov(目标)||_F) × 100`。
判据 2 是本协议存在的理由：**没有它，任何"域适应无效"的结论都无法与实现缺陷区分**。

`delta_vs_none = macro_f1(臂) − macro_f1(none_raw)`，在 3 个任务上取均值。

| 分支 | 条件 | 状态码 | 处置 |
|---|---|---|---|
| A | 任一对齐臂均值 `delta_vs_none >= +0.02` | `DA_REDUCES_CROSS_ROUND_GAP` | 该方法进正文作为有效基线；主线必须与之比较而非只与 always-RF 比 |
| B | 全部 `< +0.02`，且至少一臂 `>= −0.005` | `DA_NEUTRAL` | 域适应族已被正确检验且无增益，可作已发表基线写入正文 |
| C | 全部 `< −0.005` | `DA_DOES_NOT_HELP` | 同 B 写入正文，并如实报告负增益幅度 |

分支 B 与 C 都构成论文需要的「已发表基线」，区别只在报告口径。
**任何分支都不得**由此推断"协方差漂移不是瓶颈"——那是 2026-06 的错误推断形式，
本协议只裁定这些**具体方法在本口径下的增益**。

## 5. 硬门（任一失败即停，不写结论）

1. **无泄漏**：训练轮次与目标轮次交集为空；窗口级无重复 `(round, window_id)`。
2. **有限值**：`clean_x` 后 94 维 0 NaN / 0 inf；对齐后同样 0 NaN / 0 inf（白化可能放大，须显式校验）。
3. **锚定**：判据 1，即 `none_raw` 复现 G0 网格 RF。
4. **双跑**：全部判定性产物 md5 逐字节一致。线程钉死
   `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` **且 `n_jobs=1`**。不与其他作业并行。
5. **单元检验**：`test_coral_coloring.py` 在本轮解释器上 `PASS`，输出落盘。
6. 全程禁网、禁代理。解释器 `~/anaconda3/envs/iotcls/bin/python`。

## 6. 明文不做

- 不改窗口长度、不加特征列、不调 RF 超参、不做特征选择、不换任务集。
- 不用 `category` 粒度；不在 UNSW / CIC 上跑本协议（跨数据集幅度不可通约）。
- 不计算或恢复 CPD；不恢复 CPD / EC-MDM 的机制、预测或部署身份。
- 不因结果不利而改判据门槛、改 k 规则、改臂集合。
- 不动 `dataset/`；不改 `code/` 与 `docs/` 既有文件，唯一例外是
  `coral_baseline.py` 第 65 行的缺陷修复与 docstring 引用改正（改动本身落盘 diff）。
- 不删除 `results/robust_v2/report/coral_baseline_*`；改为加注作废并保留原文。

## 7. 交付物

`results/da_baselines_20260903/`：

```text
da_results.csv          逐任务逐臂 macro-F1、delta_vs_none、cov_reduction_pct、SA 的 k
per_arm/                逐任务逐臂混淆矩阵与逐类 F1
unit_test_coral.txt      §5.5 单元检验输出
passline.json           三条判据的实测与门槛并列
acceptance.json         §5 六道硬门逐项
provenance.json         git、argv、包版本、线程变量、coral_baseline.py 修复前后 md5、耗时
VERDICT.md              状态码 + §4 处置 + 解释边界
```

## 8. 登记与顺序

```text
1. 本协议提交（commit 早于任何新数字）
2. docs/EXPERIMENT_REGISTRY.md 登记 DA-BASELINES-G0 一行（运行前登记，含本协议 sha256）
3. 修复 coral_baseline.py + 落地单元检验 → 实现审阅（核对 n_jobs=1、k 规则、无泄漏）→ 追加 RUN_AUTHORIZED
4. 跑 → 双跑 → 判定书
5. 给 results/robust_v2/report/coral_baseline_report.md 与 coral_baseline_results.csv 加作废注记，
   指向本轮结果；原文保留
```

## 9. 解释边界

本协议只裁定 CORAL 与 SA 这两种**全局边缘分布对齐**方法在自采 94 维表示、三个 LORO 任务、
RF 单模型下的增益。它不构成：域适应族整体的结论（未测 TCA/JDA/对抗式等）；
类条件对齐方法的结论；strict59_ra 表示下的结论；UNSW 或 CIC 上的结论；
无标签部署方法的成立；层一/层二/层三任何一层的证据。

真正的对手是 **always-RF**（G1 路线 A 实测 regret 0.02025），不是 stacking。
任何"击败 stacking"的表述都不构成方法贡献。

---

## 10. 2026-09-03 修正条款（运行后追加，如实披露）

**性质**：§5 硬门 1 的**唯一键规格错误**修正。不是门槛放宽，不是因结果不利而改判据。

首跑 `acceptance.json` 的 `gate_1_no_leakage` 记为 `pass=false`。逐项诊断如下：

```text
轮次重叠（真正的泄漏风险）        [0, 0, 0]      本身已 PASS
(round, window_id)               重复 8992      ← §5 门 1 原文所用键
(round, source_file, window_id)  重复 0
(source_file, window_id)         重复 0
(round, label, window_id)        重复 0
```

根因：`window_id` 在缓存内**按 `source_file` 重新计数**（30 个 `source_file` = 6 轮次 × 5 标签，
`window_id` 取值 0–407），因此 `(round, window_id)` 从来就不是窗口的唯一键。
§5 门 1 把它写成唯一键是**我方规格错误**，与数据无关、与本轮任何结果无关。

**修正**：门 1 的窗口唯一键改为 `(round, source_file, window_id)`，
并要求同时**并报原键的计数**，使规格错误在产物内长期可见。

**可审计性保证**：修正只改这一道门的键，不触碰任何科学计算路径。
故修正后重跑的 `da_results.csv` / `passline.json` / `per_arm/*.json` 必须与首跑
**md5 逐字节一致**；若不一致则说明修正越界，整体 `INVALID_RUN_STOP`。
首跑的失败记录原样保存在 `RECOVERED_GATE1_KEY_FAILURE.json`，
诊断保存在 `GATE1_KEY_DIAGNOSIS.json`，两者都不删除。

**同时披露的两条结果解读约束**（运行后发现，写入此处以免事后合理化）：

1. `sa_raw` 的 `k` 被 §3 的确定性规则定成 **1**——原始特征上单一大方差列即占累计解释方差
   ≥0.95，故 SA 退化为一维投影，macro-F1 0.201–0.223。这是**规则与未标准化特征相遇的构造性退化**，
   不是 SA 方法的性能。有意义的 SA 臂是 `sa_zscore`（k = 22/22/23）。规则不因此改动，
   但报告必须写明 `sa_raw` 退化。
2. `zscore` 臂的 `delta_vs_none` 为 +0.000005 / −0.004773 / +0.005207，并非恒 0。
   RF 对逐特征仿射变换在**实数域**上不变，但标准化改变了浮点粒度与分裂点中值的取整，
   故存在 ±0.005 量级的数值抖动。双跑逐字节一致证明这不是随机性。
   `passline.json` 内 `invariance_check_zscore` 的注记（"非 0 说明管线有非预期随机性"）
   **表述过强**，正确解读以本条为准。
