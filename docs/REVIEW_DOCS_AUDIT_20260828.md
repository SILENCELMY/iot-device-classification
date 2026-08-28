# Markdown 文档盘点：更新 / 加注 / 删除建议（外部审阅）

**日期**：2026-08-28
**审阅者**：Claude（AI 辅助审阅）
**范围**：`research-results` 分支全部 26 个 md 文件 + `main` 分支 2 个 README。
**性质**：仅建议，未改动任何文档。

**总原则先说清楚**：协议冻结条款要求"不删除原始协议文本"，且历史报告的处置方式是
**加注降格、保留原文供溯源**（8/28 导师汇报也是这么执行的）。因此本盘点**没有任何
文件建议删除**——需要处理的都是"加注"或"更新"。

**核心发现一句话**：降格标注目前只落在了 2 个文件上（`CPD_FINDINGS.md`、
`CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md`），但携带旧结论的文件有 6 个——
根 README 和三份 6 月的报告还在无警示地陈述"r=-0.630 显著""传统观点已证伪""机制已验证"，
与协议 §2.3 的主张强度约束直接冲突。谁先读到 README 谁就先接收到已降格的结论。

---

## 一、需要尽快处理（旧结论无警示，与协议 §2.3 冲突）

### 1. `README.md`（根目录）——需要更新，优先级最高

这是任何人进入项目读的第一个文件，问题最多：

- **"当前结论"表**列出 `CPD vs Stacking gain Pearson r = -0.630` 和
  `CPD vs F1 下降 Pearson r = 0.9499`，均无降格/口径标注。前者已于 8/28 降为探索性
  结果；后者按协议 §4.3 属于"代数上内含误差幅度，相关性高有一部分是代数必然"，
  §2.3 规定只作描述性证据；
- **机制链条以"当前结论"陈述**（特征漂移 → CPD → 误差相关性坍缩 → 元学习器失配 →
  集成失效），但 E2（9/12）就是要检验这条链是不是机制——按协议现阶段它是**待检假设**；
- `loro_R2_R4_to_R3` 案例段的 Stacking 数值（0.5455 / -0.0693）是随机 OOF（E1 A 臂）
  口径，E1 预验证显示分组 OOF 下为 0.4907 / -0.1241，README 未提；
- "注意：项目根目录不是 git 仓库，`code/` 自身是 git 仓库"——已过时，现在两者是
  同一 GitHub 仓库的两个分支；
- 阅读路线中 `legacy/README.md`、`code/README.md` 两个链接在本分支/本仓库不存在
  （`legacy/` 未纳入版本控制，`code/` 在 main 分支）。

**建议**：顶部加一个与 `CPD_FINDINGS.md` 同款的 CAUTION 块；"当前结论"改为
"历史基线结论（部分已降格，见 p0_audit）"；补一段"仓库结构与检出方式"
（git worktree 用法见 `REVIEW_CODE_OPTIMIZATION_20260828.md` 第 4 条）；修链接。

### 2. `results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md` ——需要加注

三份未加注历史报告中问题最重的一份：

- 标题即"跨环境泛化失效的**核心机制**"、"**根本原因**"——协议冻结条款明确
  "相关性结论一律不得表述为因果关系"；
- "传统观点（**已证伪**）/ 本研究发现（**已验证**）"的表述超出证据强度；
- "CPD 可作为 OOD 泛化失效的**预测指标**"（r=0.9499）——§2.3 规定 `CPD_y` 与 F1
  下降的相关性只作描述性证据（两者共享目标标签与测试样本，且有代数耦合）。

**建议**：加与 `CONTROLLED_CPD_..._FINAL.md` 同款的 CAUTION 头（指向协议 §2.3、§4.3
与 `CPD_DEFINITIONS.md` 的口径归因），正文原样保留供溯源。README/报告 README 中
对它的定位从"CPD 主报告"改为"历史分析（2026-06 口径）"。

### 3. `results/robust_v2/report/COMPREHENSIVE_OOD_ANALYSIS.md` ——需要加注

同样问题："揭示了……灾难性失效的**根本原因**"、执行摘要以确定语气陈述四条机制结论。
另外其 CORAL 结论（"F1 平均下降 37.75%"）按协议 §2.2 已降为支撑实验。建议加注。

### 4. `results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md` ——需要加注

全文的 Stacking 数值（0.5455 / -0.0693 等）都是随机 OOF（现 E1 A 臂）口径。
E1 预验证已表明该口径整体偏乐观（分组 OOF 下旗舰任务崩溃加深至 -0.1241）。
建议加注说明"本报告 Stacking 数值为 A 臂口径，E1 正式结果出来后以 E1 为准"。

### 5. `results/robust_v2/report/summary_report.md` ——需要加注（轻量）

自动生成的 110 条汇总，其中 stacking 列同样是 A 臂口径。建议在生成器
（`generate_robustness_report.py`）里加一行口径说明，重新生成即可，
或手工在文件头加一行。

---

## 二、需要小幅更新（内容过时或缺项，无结论性风险）

### 6. `main` 分支 `README.md` + `scripts/README.md`

- "项目根目录不是 git 仓库"已过时；
- 所有命令引用 `code/scripts/...`，克隆件中不存在该路径（见代码审阅文档第 4 条）；
- 未提及 `oof_mode`（grouped/random）这一 §9.1 关键参数和 `e1_oof_arms.py` 入口。

建议改写为"本分支即代码目录，推荐用 git worktree 检出为 `<项目根>/code`"并给出命令。

### 7. `docs/WORKFLOW.md`

命令路径同上问题；"主要输出"清单缺 8 月新增的 `pred_proba.csv` / `oof_meta.csv`；
未收录 E1 入口。建议同步。

### 8. `results/README.md`

结构树只列了 5 个目录，缺 8 月新增的 `p0_audit/`（P0 审计结论，被协议和导师汇报
反复引用）和 `e1_oof_arms/`（E1 预验证）。这两个恰是当前最活跃的目录，建议补上。

### 9. `results/robust_v2/report/README.md`

- 引用的归档路径 `legacy/docs/archive/...` 在本分支不存在（legacy 未入库）；
- 对 `CONTROLLED_CPD_..._FINAL.md` 的定位仍是"CPD 对 Stacking 集成增益的机制验证"，
  该报告已降格，条目描述宜同步加"（已降格为探索性，见文件头标注）"。

### 10. `docs/CPD_FINDINGS.md`

已有 CAUTION 块（好），但仍有两处残留：

- "关键数值"表中 r=0.9499 一行无 §4.3 代数耦合说明（CAUTION 块只覆盖了 r=-0.630）；
- "定义"一节仍用泛称 "CPD"，与 §4.4 的命名纪律（`CPD_y` / `CPD_dir` / `UDS`
  唯一合法命名）不一致，建议改为指向 `CPD_DEFINITIONS.md` 的转引。

### 11. `docs/PAPER_MODEL_ORGANIZATION.md` + `docs/DEEP_MODEL_METHODS.md`

内容本身是论文写作素材，保留。两点建议：

- 表中深度模型数值均为**单种子**结果，S1（5 种子，§14）完成前建议在表头加
  "单种子，待 S1 补 mean ± std"的说明，防止直接誊进正文；
- 按 §2.2，深度架构比较已降为支撑实验，文档定位（正文 Table 1 vs 附录）建议
  在 9/22 写作检查点前和导师确认后在文头注明。

---

## 三、无需改动（当前有效或刻意保留）

| 文件 | 状态 |
|---|---|
| `docs/experiment_protocol_final.md` | 权威，FROZEN。只能按冻结条款经 Change Log 修改 |
| `docs/ADVISOR_BRIEF_20260828.md` | 当前有效（8/28） |
| `docs/CPD_DEFINITIONS.md` | P0 交付物，当前有效，口径对照齐全 |
| `docs/EXPERIMENT_REGISTRY.md` | 活跃维护中。唯一建议：HIST-gpu_capacity / HIST-cnn_contrast 两行 commit 列为"—"，能补则补 |
| `docs/论文冲刺计划.md`（v3） | 执行主干，当前有效 |
| `docs/论文冲刺计划_v2.md` | 刻意保留的审计依据（文头已写明定位），不动 |
| `results/p0_audit/R630_SENSITIVITY_CONCLUSION.md` | P0 交付物，当前有效 |
| `results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md` | 已正确加注，原文保留供溯源，不再动 |
| `results/robust_v2_rerun_.../RERUN_COMPARISON_SUMMARY.md` + `summary_report.md` | 复跑事实记录，有效 |
| `results/robust_v2/README.md` | 基本准确，可不动 |
| `results/robustness_scaling_.../ROBUSTNESS_SCALING_CONCLUSION.md` | 历史结论，保留。若引用注意：单种子 + 该脚本已被 §14 禁止用于多种子 |
| `results/gpu_capacity_.../DEEP_ROBUSTNESS_CONCLUSION.md` | 同上（其 CPD 列为旧口径，引用时对照 `CPD_DEFINITIONS.md`） |
| `results/cnn_architecture_.../CNN_CONTRAST_SEARCH_CONCLUSION.md` | 同上；它是 §14 认可的多种子正确入口的划分来源说明，保留 |
| `.claude/agents/worker.md` | 内部工具配置，与论文无关，不动 |

## 四、建议删除的文件

**没有。** 理由见文首总原则：协议的处置纪律是加注不删除；v1/v2 冲刺计划是刻意
并存（执行版 vs 审计依据）；各 conclusion 报告是实验登记表引用的原始产出。
唯一接近"可删"的是各报告间重复的旧数值表述，但它们都随"加注保留"策略一并处理即可。

---

## 附：处理优先级排序

1. 根 `README.md` 加 CAUTION + 修结构描述（10 分钟，读者面最大）；
2. `CPD_PAPER_LEVEL_ANALYSIS.md` / `COMPREHENSIVE_OOD_ANALYSIS.md` /
   `STACKING_COLLAPSE_ANALYSIS.md` 三份加注（各 5 分钟，消除与协议的表述冲突）；
3. `results/README.md` 补 `p0_audit/`、`e1_oof_arms/` 两条（2 分钟）;
4. 其余在 9/22 写作检查点前顺手做完即可。
