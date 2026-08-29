# EXECUTION_PLAN_20260829.md

**Status: FROZEN（执行层）v1.3**
**冻结日期**: 2026-08-29
**性质**: 执行层文档。本计划**不修改** `experiment_protocol_final.md`（FROZEN 2026-08-25）的任何条款；
凡涉及协议文本的地方均为"按既有文本执行"的口径澄清，同步登记于 `EXPERIMENT_REGISTRY.md`，不触发冻结例外。
**分工**: Fable 5 —— 决策与审阅（本文档作者）；Opus 5 子代理 —— 代码与基础工作实现。
**实现纪律**: 子代理不做设计决策、不执行任何 git add/commit/push；所有改动由审阅方核对 diff 后统一提交。
**环境铁律**: 服务器一切网络操作走校园网直连，**严禁配置或使用任何代理**。

---

## 0. 决策记录

### D1 GroupKFold 在 2 源轮次任务上退化为 2 折 —— 确认为期望口径（关闭登记表待决项）

- §9.1 的目标是 OOF 概率反映跨轮泛化且无相邻窗口泄漏；2 源轮次下按轮分组只有 2 折这一种无泄漏实现。
- 折数混淆由 A′ 臂（折数动态对齐 B）显式控制；E1-FULL 实测折数效应小而偏正（A→A′ 均值约 +0.008），
  崩溃加深全部来自分组效应（A′→B）。
- 备选方案（轮内时间块凑折数）会混合两种分组语义，偏离 §9.1 文本，不采用。
- E1-FULL（11 任务 × 5 种子）结果按此口径有效。

### D2 E1 向 G0 网格扩展（§12 任务范围的执行口径）

- **范围**：`|S|≥2` 的 120 任务为必做（§12 正文）；`|S|=1` 的 30 任务按 §12 括号条款一并执行，
  B 臂 = `window_start` 时间块，结果单独标注（`b_split_basis=time_block`）。
- **实现要点**：time_block 与 round_group 的判定依据必须是"训练轮次数 == 1"，不得沿用按任务名单
  （`SINGLE_ROUND` 常量）判断——G0 的 `|S|=1` 任务名不在名单内，按名单判断会把单轮任务错误送入轮次分组。
- **任务定义唯一来源**：grid 任务必须 import G0 生成器（`environment_grid_experiment.py`）取得，
  不得在 E1 脚本内重复实现任务生成（与 §11"唯一实现"同一纪律）。
- **种子**：先 seed=42 全量（与 G0 的 §14 单种子覆盖定位一致；§8.6 网格不用于显著性结论；
  11 个主线任务 × 5 种子的 E1-FULL 仍是唯一确证性 E1）。若单种子全量实测 ≤4 小时，
  报请审阅方决定是否追加 43–46；追加与否均登记。
- **特征集**：all_features（与 E1-FULL 一致）。**输出**：`results/e1_oof_arms_g0/`，不覆盖主线 E1。
- **一致性验收（对应 §22.1 P1 回归容差 1e-6）**：G0 的 stacking 已按 §9.1 grouped 口径训练
  （`robust_iot_research.py` L718），即 G0 自带 B 臂。全量前先在 ≥2 个 smoke 任务（1 个 |S|=2、
  1 个 |S|=1）上验证：E1 基模型 F1 ≡ G0 基模型 F1、E1 B 臂(seed 42) ≡ G0 stacking，容差 1e-6。
  验证通过后允许复用 G0 落盘结果省算力（复用与否在输出中记录）；验证不通过 → 停，报告，不得调参。
- **§19.2 合规**：先给 `e1_oof_arms.py` 补持久化（git hash、完整命令行、种子、关键包版本、
  逐任务折叠分配记录——轮级或时间块边界），再跑全量。

### D3 UNSW pilot（P2，9/10 截止）今日启动

- 全程直连下载（wget -c 断点续传，服务器上 nohup 后台），**禁代理**。
- 抽 3 天：`16-09-30`（§16.1 已核抽样日，最先下载）、`16-09-23`（首日）、`16-10-12`（末日）——
  跨度最大化以回答五问之 2（MAC 映射跨天一致性）。若末端日期文件过大危及 9/10 节点，
  可换其它远端日期，登记原因（选日为执行细节，非协议冻结项）。
- 数据落盘 `dataset/unsw/`（继承 `/dataset/` 的 gitignore）。
- 严格 §16.2：**pilot 以 pcap 为准**；官方 CSV 只可用于设备清点与窗口计数交叉核对，
  **不得用于验证任何时间特征**。
- `extract_features_generic.py`（§20.1 新建项）随 pilot 实现：仅 ~60 维通用特征
  （`len_*` / `interarrival_*` / `burst_*` / `subwin_*` / `up_down_*`），10 秒非重叠窗口，
  特征定义与命名对齐主线实现（对齐关系落成文档）；802.11 专属特征不做。
- 标注按 §16.1：`eth.src`/`eth.dst` 匹配设备 MAC（网关 `14:cc:20:51:33:ea`）。
- 产出：`results/unsw_pilot/PILOT_FIVE_QUESTIONS.md`（五问逐条作答、附证据）、许可证记录、
  MAC 映射表及跨天一致性核对、最小 RF LORO（RF 口径按 §7：n_estimators=500, class_weight=balanced）。
- 五问任一不通过 → 按 §16.3 立即上报换候选，不得自行降低标准。

### D4 同质环境×环境拓扑矩阵（§8.5 / §20.2）

- **语义**：6×6 矩阵，行 = 源环境 i，列 = 目标环境 j（i≠j，共 30 个有序对，对应 G0 `|S|=1`）。
  单元格 = `cpd_core.cpd_y(ref=CM_iid(i), tgt=CM_{i→j})`；模型 RF、all_features；
  混淆矩阵一律读 G0 落盘 CSV；计算只经 `cpd_core`（§11 唯一实现）。
- **参照系两个变体**（均为支撑材料，§8.6）：
  - primary：`ref = g0_iid_Ri_time_block`（诚实域内参照，§8.4）；
  - secondary：`ref = g0_iid_Ri_random`（与历史 IID 参照口径可比）。
- 同时输出由 CM 重算的 macro-F1 矩阵核对表，须与已有 `env_topology_matrix_rf.csv` 逐格一致（容差 1e-6）。
- `CPD_dir` 版本只做覆盖率评估（§4.3 n_err≥20 约束下预计大量缺失）：有多少报多少，缺失置 NaN，不强算。
- `six_env_confusion_similarity.py` 按 §20.2 改造后，全量跑 `test_cpd_core.py`，
  历史值 0.1521 的重建测试不得破坏。

### D5 文档加注批次一

- 执行 `REVIEW_DOCS_AUDIT_20260828.md` 第 1–5、8 项：根 README（CAUTION + 结构段改写 + 死链修复）、
  三份历史报告加注（`CPD_PAPER_LEVEL_ANALYSIS` / `COMPREHENSIVE_OOD_ANALYSIS` / `STACKING_COLLAPSE_ANALYSIS`）、
  `summary_report.md` 口径行（含生成器同步）、`results/README.md` 补 `p0_audit/`、`e1_oof_arms/`、
  `g0_environment_grid/` 三个目录。
- 纪律：历史报告**只加注不改原文**；README 中所有引用数字必须与落盘结果文件逐一核对并列出出处。
- 其余项（6、7、9–11）排 9/22 写作检查点前，不在本批。

### D6 S1 深度模型 5 种子（§22.2 立即执行项）

- 入口锁定 §14：`cnn_contrast_search_experiment.py --split-source
  results/robustness_scaling_20260706_v2/splits --random-state {42..46}`。
- **启动前硬性预检（§14 末条）**：确认 `--random-state` 只经 set_seed 影响初始化、不影响任何数据划分；
  确认 splits 为复制而非重建（引用行号留证）。预检不通过 → 停，报告，不得改脚本设计。
- RTX 4090 当前空闲；5 种子串行后台执行，日志落盘。

### D8 投票基线后处理（v1.1 新增；§7 基线 4–6，X2/X4/G1 的必备对照）

- **缺口**：§7 的 hard voting / 等权 soft voting / 校准 soft voting 是 X2 的必备对照、
  X4 的比较组、G1 的候选动作之一，但至今未在任何结果目录中计算过。
- **无需重训**：三者全部可由 G0 已落盘产物后处理得到——hard voting 读各基模型
  `predictions.csv` 做多数票；等权 soft voting 读各基模型 `pred_proba.csv` 平均后 argmax；
  校准 soft voting 用 stacking `oof_meta.csv` 中的基模型**源域 OOF 概率**拟合
  temperature / isotonic（§7：校准只在源域内做，此为无泄漏依据），再施加于测试
  `pred_proba.csv`。
- **实现**：`code/scripts/analysis/voting_baselines.py`，输出逐任务投票指标至
  `results/g0_environment_grid/voting_baselines.csv`（含 5-class / 4-class / 最差类 F1，§10）。
- **验收**：从 `predictions.csv` 重推各基模型 macro-F1 并与 `metrics.json` 核对（1e-6）作
  管道自检；票决/平局规则与校准拟合的数据边界写入脚本 docstring 与输出说明；
  校准仅触碰 OOF 概率的审计说明随产物落盘。
- **排期修正（v1.1 当日）**：输入全部为 G0 已落盘产物，与 D2 运行**无依赖**（唯一交互是 CPU 分时），即时可跑；须早于 G1 组装（目标 9/5 前）。
- 历史 110 条主线任务的投票基线依赖 §20.3 重打分补概率（D7 backlog），不阻塞 G1
  （G1 内层池用 G0 网格任务）。

### D9 E2 实现规格（§13 分层回归的执行口径；v1.2 新增，**先写后看**——本节提交先于任何 E2 数值产生）

- **因变量（DV）**：Stacking gain，**B 口径**（§9.1 分组 OOF；引用口径已由 E1_CONCLUSION 定）。
  主样本 = E1-G0-GRID 的 150 个网格任务（`results/e1_oof_arms_g0/e1_arms_raw_all.csv`
  B 臂 `gain_absolute`，seed 42；与 G0 有 ~1e-3 差异的 3 个任务用 E1 值——三臂同环境内部自洽）。
  敏感性：A 臂 DV 重跑同一回归（历史口径对照）。11 个主线任务只作描述性对照——
  n=11 支撑不了 7 维预测变量的回归（§13 统计纪律：不凑显著）。
- **统计单位与依赖（§15）**：任务为行，按**目标环境**（6 个）聚类；CI 用聚类 bootstrap
  （重抽环境，B=10000，seed 42）；另做 leave-one-target-env-out 敏感性（§15.5）。
  禁止点级 bootstrap，禁止把 n=150 当独立样本。
- **M0（逐字按 §13）**：对每类 c：`recall_src_IID(c) − recall_tgt(c)`（5 维）+ L2 范数 + 最大值 = 7 维。
  - 模型口径 = **RF**（§7 基线 1；用 stacking 自身 recall 会与 DV 机械纠缠）；敏感性 = 三基模型均值。
  - `recall_src_IID(c)` = 源集合 S 中各环境 G0 IID（**time_block**，与 D4 同一诚实参照）RF
    混淆矩阵逐类 recall 的均值；敏感性用 random IID。
  - `recall_tgt(c)` = 该任务 RF 测试混淆矩阵逐类 recall（G0 落盘 CM）。
- **M1 = M0 + `CPD_y`；M2 = M0 + `CPD_dir`**：ref = 源 S 的 IID CM（time_block），tgt = 任务 RF
  测试 CM；多源参照构造**沿用 0.8397 历史口径的同一构造**（CPD_DEFINITIONS §3，读该文档与
  test_cpd_core 的复现实现后复用，不新造）；计算只经 `cpd_core`。`CPD_dir` 按 §4.3
  n_err≥20：未定义任务从 M2 剔除并报告覆盖率；**M2 的 M0 对照在同一子集重新拟合**（否则增量不可比）。
- **估计与观测量（§13 双通道都做）**：① 标准化 OLS：M1/M2 相对 M0 的增量 R²、系数、
  效应量、环境聚类 bootstrap 95% CI；② 留一目标环境 CV：M0 vs M1/M2 的样本外 MSE 改善。
  另报每 |S| 分层的描述性均值（**不加回归项**）。预注册通道之外不加任何协变量；
  任何额外模型须标注 exploratory 且仅在审阅方要求时做。
- **解读纪律**：严格走 §13 预注册分支，判定由审阅方作出；实现只出数表不下结论。
  E1_CONCLUSION §5.2 的构造性代价口径警告约束全部解释文字。
- **产物**：`code/scripts/analysis/e2_conditional_explanation.py`（import cpd_core）；
  `results/e2_conditional/`：回归表、逐环境结果（§15.4）、M2 覆盖率、bootstrap 明细、
  §19.2 provenance、`E2_RESULTS_NOTE.md`（只记口径与数表）。
- **验收**：① M0 特征抽查 2 任务手算一致（1e-12）；② E2 的 `CPD_y` 在 |S|=1 的 30 任务上
  与 D4 拓扑矩阵（time_block 变体）**逐位一致**（同口径交叉验证）；③ bootstrap 固定种子可复现；
  ④ 本节与登记表行的 commit 先于任何 E2 数值。

### D9 追记（v1.3，两项裁定 + canonical 环境政策）

- **裁定 A（验收门 b 的 ULP 差异）**：canonical 分析环境 = `code/requirements-lock.txt`
  所锁的 iotcls。D4 拓扑 CSV 已在 iotcls 下重生成（初版产自 anaconda base；差异 9/30 格、
  max 2.22e-16，根因 numpy 1.23→2.4 的 fro-norm 归约路径变化；D4 自身 1e-6 硬门两版均过）。
  此后"逐位一致"类验收一律限定在 canonical 环境内执行；跨环境 ULP 差异按可复现性事实登记
  （与 E1-G0-GRID 的线程序发现同类）。
- **裁定 B（多源 `CPD_dir` 的 NaN 传播，D9 原文未定）**：primary = **lenient**
  （`nanmean`，在已定义参照上取均值，n=119）；strict（任一参照未定义即弃任务，n=26）
  **并列完整报告**，且必须带秩亏标记。裁定依据为**效度判据**：strict 子集上
  `d_recall_Socket` 零方差 → 设计阵在 10000/10000 次 bootstrap 中秩亏，其估计不具统计效度
  （lenient 为 0/10000）；lenient 亦与 §4.3 逐 CM 准入规则一致。
  **透明度声明**：本裁定作出时两变体的诊断数值已可见；因此两变体在所有产出中终身并列，
  正文引用任一变体时必须同时给出另一变体的数值与覆盖率。
- `d_recall_max` 维持字面读法（变化向量的最大元素，"受损最重的类"）；与最大绝对值读法
  在 16/150 任务上不同的事实记入 provenance。

### D7 工程收尾

- 随 D5：`pip freeze`（实际运行环境 anaconda3）出 `code/requirements-lock.txt`（§19.5）。
- 排 E1-G0 之后、不阻塞 9 月门槛：最小 CI（pytest workflow）；`robust_iot_research.py`
  划分索引持久化（审阅意见 #3）；历史 110 条 `pred_proba`/OOF 重打分脚本（§20.3——
  已核实 `results/robust_v2/**/stacking/` 无 `oof_meta.csv`/`pred_proba.csv`；
  G1/X3 所需网格任务的概率与 OOF 已由 G0 全量落盘，故此项不阻塞门槛）。

---

## 1. 时间表映射（协议 §22.1 不变）

| 节点 | 状态 |
|---|---|
| 8/31 P0 | 已完成（cpd_core 15/15、CPD_DEFINITIONS、r=-0.630 降格） |
| 9/7 P1 | G0 已跑完并于本日登记入库；余项 = D4 拓扑矩阵、D2 的 §19.2 持久化 |
| 9/7 E1 | 11 任务 × 5 种子已完成（§12 第一分支：现象基础成立）；余项 = D2 网格扩展 + E1 结论文档（审阅方撰写，含 position→R5 反转与 joint 域内口径错配两处专门解释） |
| 9/10 P2 | D3 今日启动 |
| 9/12 E2 / 9/14 G1 | 依赖 D2/D4 产物；D2 完成后立即排 E2 |

## 2. 验收标准

| 项 | 验收 |
|---|---|
| D2 | smoke 双一致性（1e-6）通过记录；全量行数 = 任务数 × 种子数 × 3 臂；§19.2 五要素齐 |
| D3 | 五问逐条有证据；MAC 映射表 + 跨天一致性输出；最小 LORO 的 metrics 落盘 |
| D4 | 两个 6×6 CPD_y CSV + F1 核对表逐格一致（1e-6）；test_cpd_core 全绿 |
| D5 | 历史报告 diff 只增不删（README 结构段除外）；引用数字有出处核对表 |
| D6 | 预检证据（行号）；5 个 run 目录 + 日志存在且首个 run 正常开训 |

## 3. Change Log

| 日期 | 事项 |
|---|---|
| 2026-08-29 | v1.0 冻结（Fable 5 审定） |
| 2026-08-29 | v1.1：新增 D8 投票基线后处理（复审发现 §7 基线 4–6 从未被计算，而 X2/X4/G1 都需要；全部可自 G0 落盘产物无重训推得）。同日协议 §23 增三条澄清/记录行（§14 种子语义边界、§12×§14 网格交集、§19.7 执行完毕），协议正文零改动 |
| 2026-08-29 | v1.2：新增 D9（E2 §13 分层回归执行口径，先写后看——D9 与登记行提交先于任何 E2 数值）；D8 排期修正（与 D2 无依赖） |
| 2026-08-29 | v1.3：D9 追记两项裁定（ULP/canonical 环境；CPD_dir NaN 传播 lenient-primary，效度依据 + 透明度声明）——见 D9 追记节 |
