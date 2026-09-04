# 条件预测协议：可分性 → 共识推翻率（运行前冻结）

**冻结日期**：2026-09-02
**状态**：`PROTOCOL_FROZEN_BEFORE_NUMERIC_RUN`
**性质**：把自采上测得的「簇内源域 OOF 可分性 → 共识推翻率」关系**冻结成预测函数**，
然后在 UNSW 上做出样检验。不修改任何既有冻结判定；不恢复 CPD 与 EC-MDM 的机制、预测或部署身份；
不重开 E1/E2/G1/RouteB/D12/UNSW-META-MISMATCH 任何判定。
**裁定方**：主线侧。执行方只实现与运行。

## 0. 为什么这个形式

前一轮 UNSW 机制检验（`UNSW-META-MISMATCH`）的判据是**二元签名**，判定 `MECHANISM_NOT_EXTERNALLY_REPLICATED`
（四条签名 2/3、3/3、0/3、2/3）。事后测得 UNSW 上共识推翻率**困难簇 12.9% vs 易类 0.07%（185×）**，
即定性形态在、定量严重度不在，而后者与 UNSW 的可分性（0.459–0.556，落在自采网格第 26–48 百分位）一致。

本协议改用**条件预测**而非二元复现，理由是关系两端都是**数据集内的无量纲量**：

```
可分性     = RF 在源域 OOF 上、只算簇内类的 macro-F1      （比例值，各自特征空间内部）
推翻率     = 三基模型一致且正确的窗口中 stacking 改判的比例（比例值）
```

因此它**不受跨数据集特征不可通约的限制**——自采与 UNSW 的 61 个同名列有 34/61 分布重叠 <0.20、
`interarrival_p10` 差 13,700 倍、窗口内包数中位 28 vs 4，那些差异阻断幅度比较，但不阻断这两个比例量。

**本协议不重开前一轮判定。** 它是一个新问题、新判据、新数据配置。

## 1. 唯一问题

在自采 G0 的 159 个任务上拟合并冻结的「可分性 → 推翻率」关系，
是否能预测 UNSW 上**从未测过**的任务配置的推翻率。

## 2. S1：拟合并冻结预测函数（自采侧，只读已落盘产物）

输入 = `results/g0_environment_grid/grid_override_diagnostic.csv`（159 行，已落盘）。
其中每行含 `severity_src_oof_rf_H`（可分性）与 `override_rate_H`（困难簇推翻率）。

**预测函数（非参数邻域法，避免函数形式假设）**：

```text
给定新任务的可分性 s：
  邻域 N(s) = { G0 任务 : |可分性 - s| <= delta }，delta 初值 0.05
  若 |N(s)| < 15，delta 每次 +0.01 直到 |N(s)| >= 15 或 delta = 0.20（超出则记 UNDEFINED）
  点预测   = median( N(s) 的 override_rate_H )
  区间预测 = [ P10, P90 ] of N(s) 的 override_rate_H
```

同时冻结**无条件基线**（用于 §4 技能门）：G0 全部 159 任务 `override_rate_H` 的中位数，
记为 `uncond_median`。该值与可分性无关。

S1 产物 `predictor_frozen.json` 必须包含：`delta` 规则、159 行的
(可分性, 推翻率) 原始对、`uncond_median`、以及一张 0.00–1.00 步长 0.01 的查表
（点预测 + P10 + P90），使预测**完全确定、可复算、不含运行时自由度**。

**S1 必须先提交（git commit）**，其 commit 早于任何 UNSW 新配置数字产生。

## 3. S2：UNSW 侧只测可分性并落盘预测（**不得计算推翻率**）

**面板（机械导出，不得手工调整）**：从 `results/unsw_features_full/` 的 20 个日文件中，
取「在 ≥15 个日子上有 ≥100 窗」的设备，得 15 台：

```text
AmazonEcho, BelkinWemoMotion, BelkinWemoSwitch, Dropcam, HPPrinter,
NetatmoWeather, NetatmoWelcome, SamsungSmartCam, SmartThings, TribySpeaker,
TPLinkPlug, WithingsBabyMonitor, WithingsAura, InsteonCam_wired, LiFXBulb
```

日集 = 这 15 台**同时**满足 ≥100 窗的日子（脚本实测并落盘，预期 15 天）。

**任务生成（机械 + 固定种子）**：枚举全部 `(2 个训练日, 1 个目标日)` 有序三元组（三日互不相同），
用 `random.Random(42)` 随机抽 **60 个**。抽样在**看到任何数字之前**完成并落盘 `task_list.json`。

**排除已污染配置**：前一轮 `UNSW-META-MISMATCH` 用过的三个日集组合
（`{09-23,09-30}→10-12`、`{09-23,10-12}→09-30`、`{09-30,10-12}→09-23`）
若被抽中则剔除并从剩余池按同一随机流补齐，剔除记录落盘。

**每任务的口径**（逐字沿用 `docs/PROTOCOL_UNSW_META_MISMATCH_20260902.md`）：

```text
标签 = 设备身份（非 category）
入选门槛 = 面板 15 台；模型 rf / xgboost / lightgbm / stacking
stacking OOF = 按 day 分组，n_splits = max(2, min(cv, 2)) = 2（脚本实测并落盘，非 2 即 INVALID_RUN_STOP）
平衡 = 主线 sample_balanced(max_rows=20000, random_state=42)
特征列 / build_model / clean_x / sample_balanced 全部 import，不重实现
簇 = 按 TAU=0.10 规则从该任务 RF 目标域混淆矩阵机械导出（同 UNSW 协议 §3）
可分性 = RF 在该任务源域 OOF 上、只算簇内类的 macro-F1
```

**S2 的输出严格限定为**：`task_list.json`、`cluster_per_task.csv`（逐任务簇成员与 f 值）、
`separability.csv`（逐任务可分性）、`prediction.csv`（逐任务点预测 + [P10,P90] + 邻域大小 + delta）、
`oof_folds.csv`、以及各模型的 `predictions/`（S3 需要，但 S2 **不读取其内容做任何统计**）。

**S2 脚本内不得出现推翻率的计算**（实现审阅逐行核对这一条）。
**S2 产物必须 git commit 之后才允许运行 S3**，使顺序外部可验证。

## 4. S3：揭示推翻率并判定

对每个任务计算共识推翻率（困难簇）：三个基模型预测一致**且正确**的窗口中 stacking 改判的比例。
同时并报易类推翻率。

**三条判据，运行前定死：**

| # | 判据 | 门槛 | 含义 |
|---|---|---|---|
| 1 | **技能门** | `MAE(条件预测) <= 0.80 × MAE(无条件基线)` | 用可分性做条件，误差必须比忽略可分性至少低 20% |
| 2 | **秩门** | UNSW 内部 `Spearman(可分性, 推翻率) <= -0.30` | 关系的方向与单调性在 UNSW 内部成立（自采是 −0.753，允许衰减） |
| 3 | **覆盖门** | 落在 `[P10, P90]` 内的任务 `>= 50%` | 区间预测的校准（名义覆盖 80%，允许打折） |

`MAE` 均在 60 个任务上按 `|实测 − 点预测|` 计算；无条件基线的点预测恒为 `uncond_median`。

**判据 1 是主判据**：它是唯一能排除"区间恰好重叠"这种偶然通过的检验——
无条件基线与条件预测用同一批实测值比较，只有可分性真的携带信息才会赢。

**三分支处置（不得事后新增）：**

| 分支 | 条件 | 状态码 | 处置 |
|---|---|---|---|
| A | 三门全过 | `RELATION_TRANSFERS_TO_UNSW` | 该条件关系升为跨测量系统结论，进正文；机制的主张形式改为「存在 + 条件」，条件量可预测 |
| B | 判据 1 过，2 或 3 不过 | `RELATION_DIRECTION_ONLY` | 只声称方向/单调性迁移，**不声称可定量预测**；区间预测的失败如实写入限制章节 |
| C | 判据 1 不过 | `RELATION_DOES_NOT_TRANSFER` | 该关系限定为自采内部结论；UNSW 仅承担现象成分（+0.0328）与事后形态观察（12.9% vs 0.07%） |

## 5. 硬门（任一失败即停，不写结论）

1. **顺序门**：`prediction.csv` 的 git commit 时间早于 S3 首次运行；S3 入口校验该文件存在且非空。
2. **OOF 折数**：逐任务实测 `n_splits = 2` 且分组字段为 `day`，落盘。非 2 即 `INVALID_RUN_STOP`。
3. **无泄漏**：训练日与目标日无交集；窗口级无重复 `(device, day, window_id)`。
4. **有限值**：61 维特征 0 NaN / 0 inf。
5. **簇非空**：任务的簇为空则该任务标 `NO_CLUSTER` 并从判据计算中剔除，剔除数落盘；
   若剔除后有效任务 < 40，整体 `INSUFFICIENT_TASKS_STOP`。
6. **双跑**：全部判定性产物 md5 逐字节一致。线程钉死
   `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` **且 `build_model` 的 `n_jobs=1`**
   （前一轮我用 `n_jobs=16` 导致双跑门失败，本轮不重犯）。不与其他作业并行。
7. 全程禁网、禁代理。解释器 `~/anaconda3/envs/iotcls/bin/python`。

## 6. 明文不做

- 不做幅度比较。**不得**把 UNSW 的落差、gain 或 macro-F1 与自采并列比大小
  （61 列不可通约：34/61 分布重叠 <0.20，`interarrival_p10` 差 13,700 倍，窗口包数中位 28 vs 4）。
- 不用 `category` 粒度（已实测掩盖现象：OOD−IID **−0.0131**，符号与设备身份粒度相反）。
- 不改窗口长度、不加特征列、不调参、不换元学习器、不做特征选择。
- 不计算或恢复 CPD；不恢复 CPD / EC-MDM 的机制、预测或部署身份。
- 不重开 `UNSW-META-MISMATCH` 的分支 C 判定；本协议无论结果如何都不改变它。
- 不得因结果不利而改 `TAU`、改 `delta` 规则、改任务抽样种子、改面板、改判据门槛。
- 不动 `dataset/`；不写入 `results/` 既有子目录；不改 `code/` 与 `docs/` 既有文件。

## 7. 交付物

`results/conditional_prediction_20260902/`：

```text
predictor_frozen.json      S1：delta 规则、159 个 (可分性,推翻率) 对、uncond_median、查表
task_list.json             S2：60 个任务 + 抽样种子 + 剔除记录
cluster_per_task.csv       S2：逐任务簇成员与逐对 f 值
separability.csv           S2：逐任务可分性、邻域大小、实际 delta
prediction.csv             S2：逐任务点预测与 [P10,P90]      ← 必须先 commit
oof_folds.csv              S2：逐任务 n_splits / 分组字段 / 每折日期
predictions/               S2：逐任务逐模型的窗口级预测
override.csv               S3：逐任务困难簇与易类推翻率
passline.json              S3：三条判据的实测与门槛并列
acceptance.json            §5 七道硬门逐项
provenance.json            git、argv、包版本、线程变量、耗时
VERDICT.md                 状态码 + §4 处置 + 解释边界
```

## 8. 登记与顺序

```text
1. 本协议提交（commit 早于任何 UNSW 新配置数字）
2. docs/EXPERIMENT_REGISTRY.md 登记 COND-PRED-UNSW 一行（运行前登记，含本协议 sha256）
3. 实现 → 主线侧实现审阅（重点核对 S2 内不含推翻率计算、n_jobs=1）→ 追加 RUN_AUTHORIZED
4. 跑 S1 → commit → 跑 S2 → commit prediction.csv → 跑 S3 → 判定书
5. 按 §4 分支同步 docs/CROSS_LINE_DISCUSSION_20260830.md；历史文档加注降格、保留原文
```

## 9. 解释边界

**这不是第三方外部验证。** 60 个 UNSW 配置共用同一批 15 台设备、同一个测试床、同一次采集，
所以它检验的是「该条件关系能否跨越**测量系统**（802.11 空口 → 以太网网关侧）」，
**不是**「能否跨机构复现」。任何引用本结果的场合必须同时说明这一点。

无论落在哪一分支，本协议都不构成：层一（非不变采集条件量）的外部检验
——公开数据集均无 radiotap，该族结构性不可复现；
现象幅度的外部复现；无标签部署方法的成立；同型号设备实例问题的回答
——UNSW 的近同型号候选 `NestDropcam`（1 天 209 窗）与 `InsteonCam_wifi`（1 天 45 窗）数据量不足。

## 10. 预期与风险（运行前记录，避免事后合理化）

**预期**：判据 2（秩门）最可能过——UNSW 三个已测配置的可分性 0.459–0.556 对应
网格邻域预期推翻率 ≈0.26，实测 0.129，方向一致但偏低一半。
**判据 1（技能门）风险最大**：如果 UNSW 的推翻率整体系统性偏低（0.129 vs 邻域 0.26），
那么条件预测会整体高估，而无条件基线（G0 中位数）也高估——两者都错，技能门可能过不了。

若出现「方向对但整体偏移」的形态，按分支 B 处置，**不得**事后引入校正项或缩放因子来救判据 1。
该形态本身是有价值的结论：关系的**序**迁移、**标度**不迁移。


