# 实验登记表

协议 §19.1 要求：每次实验一行，字段为实验 ID、commit/hash、配置文件、种子、输入路径、输出路径、结论、是否进入正文。
新建实验前先在此登记，跑完填「结论」列。历史实验（协议冻结前）以补录形式登记，只列进入正文候选的关键运行。

## 登记表

| ID | 日期 | commit / code hash | 配置 | 种子 | 输入 | 输出 | 结论 | 进正文 |
|---|---|---|---|---|---|---|---|---|
| HIST-robust_v2 | 2026-06-22 | `1f81219`（前后） | `configs/research_experiments.json`, test_size=0.3, w=10s | 42 | `features_raw_all_w10.csv` (11303×102) | `results/robust_v2/`（110 条） | 主线基线；含 r=-0.630 等已降格结论 | 是（作历史基线） |
| HIST-gpu_capacity | 2026-07-03 | — | 同上 | 42 | 同上 | `results/gpu_capacity_full_20260703/`（48 条） | 深度模型历史对比 | 是（作历史基线） |
| HIST-cnn_contrast | 2026-07-07 | — | — | 42 | 同上 | `results/cnn_architecture_contrast_20260707/`（40 条） | 架构对比 | 部分 |
| P0-R630 | 2026-08-26 | `code/` @ 见 `r630_sensitivity.py` mtime | 脚本内置：n_boot=10000, n_perm=10000 | 42 | `results/robust_v2/report/controlled_cpd_data.csv` (n=11) | `results/p0_audit/` | r=-0.630 降为探索性（LOTO 3/11 显著；Spearman 不显著） | 是（降格结论） |
| P0-CPD-CORE | 2026-08-26→27 | `code/` @ 见文件 | `test_cpd_core.py`，TOL=1e-10 | 42 | 各历史 confusion_matrix.csv | 15/15 PASS，exit=0 | 三历史值 0.8397/0.1521/0.801 逐位复现并归因 | 是（回归门） |
| E1-ARMS-VERIFY | 2026-08-28 | `code/` @ `6a3f678` | `max_rows=∞`, rf+stacking, all_features | 42 | `features_raw_all_w10.csv`，任务 `loro_R2_R4_to_R3` | `/tmp/regression_out`, `/tmp/arm_a_out`（临时，结论已录此处） | RF 与 A 臂（random OOF）逐位复现历史（Δ=0）；B 臂（grouped OOF）macro_f1 0.5455→**0.4907**，gain −0.0693→**−0.1241**，崩溃**加深约 1.8 倍** | 是（E1 预验证） |
| E1-FULL | 2026-08-29 | `code/` @ `1fa5bcc`（结果入库 `3952388`） | `e1_oof_arms.py --tasks all --seeds 42,43,44,45,46`，三臂 A/A′/B，A′ 折数动态对齐 B，all_features | 42–46 | `features_raw_all_w10.csv`，11 个主线任务 | `results/e1_oof_arms/`（e1_arms_raw.csv / e1_decomposition.csv / e1_arms.json） | B 臂崩溃在全部 OOD 任务持续并**加深**（gain 5 种子均值，A→B）：旗舰 `loro_R2_R4_to_R3` −0.066→−0.106；`loro_R2_R3_to_R4` −0.004→−0.223；`loro_R3_R4_to_R2` −0.024→−0.181；`joint` −0.001→−0.104；jitter 三任务 ≈−0.005→≈−0.08；IID 三任务两臂无差（≈−0.002）；唯一例外 `position→R5` 反转为 +0.024。折数效应（A→A′）小而偏正 → 加深源于分组效应。**按 §12 预注册规则走第一分支：现象基础成立，按 X2 继续机制检验**。结论文档待撰（含 R5 反转与 joint 域内口径错配两处专门解释） | 是 |
| G0-GRID | 2026-08-29 | `f740362` | `environment_grid_experiment.py` 全量：150 OOD（\|S\|=1/2/3 → 30/60/60 任务）+ 12 IID（6 环境 × random/time_block，§8.4 时间块对照）；模型 rf/xgboost/lightgbm/stacking；stacking 按 §9.1 grouped | 42（脚本常量 SEED，§14 单种子） | `features_raw_all_w10.csv`（缓存复用，无 tshark 重抽） | `results/g0_environment_grid/`：summary_metrics.csv（648 行 / 162 任务）、env_topology_matrix_rf.csv（RF macro-F1 6×6）、逐任务 CM / pred_proba / oof_meta（stacking） | 网格与时间块 IID 对照完成；CPD 口径同质拓扑矩阵待 §20.2 改造后生成（EXECUTION_PLAN D4）。§8.6：网格不用于显著性结论 | 支撑（§8.6） |

> **修订记录（2026-08-28）**：本行首次登记时误记 B 臂 macro_f1 为 0.5498（"崩溃略软"）。
> 该数字来自 `_splitter` 的一个缺陷：多轮分组分支缺 `return`，执行掉进单轮时间块分支并覆盖
> GroupKFold 结果，实际跑的是时间块 OOF 而非协议 §9.1 的轮次分组。修复后真实值为 0.4907，
> **方向与原记录相反**。修复见 commit `6a3f678`，严格化测试 `code/scripts/analysis/test_oof_modes.py`（11 项断言）。
>
> **注意（待你决定）**：`loro_R2_R4_to_R3` 只有 2 个源轮次（R2、R4），`GroupKFold` 的
> `n_splits` 因此退化为 **2**——每折用单轮训练、另一轮验证。这是协议 §9.1「group=round」的
> 直接后果，但元特征噪声显著大于历史的 5 折。E1 正式跑之前需确认这是否是期望口径。
>
> **决定（2026-08-29，Fable 5 审定，详见 `EXECUTION_PLAN_20260829.md` D1）——本待决项关闭**：
> 确认 2 折退化为协议 §9.1 的期望口径。理由：① 2 轮条件下按轮分组是唯一无泄漏实现；
> ② 折数混淆由 A′ 臂（折数动态对齐 B）显式控制，E1-FULL 实测折数效应 ≈ +0.008（小而偏正），
> 崩溃加深全部来自分组效应；③ 轮内时间块凑折数的替代方案混合两种分组语义，偏离 §9.1 文本。
> E1-FULL 结果按此口径有效。

## 待登记（已排期，执行口径见 `docs/EXECUTION_PLAN_20260829.md`）

- E1-G0 网格扩展（D2：`|S|≥2` 必做 + `|S|=1` 时间块 B 臂单独标注；seed 42 先行；
  输出 `results/e1_oof_arms_g0/`；全量前须过 1e-6 双一致性 smoke）
- UNSW pilot（D3，9/10 截止；含 `extract_features_generic.py` 新建，§16.2 pcap-only 约束）
- 同质环境拓扑矩阵（D4，§20.2 改造 `six_env_confusion_similarity.py`，读 G0 `|S|=1`）
- 深度模型 5 种子（D6，§14 正确入口，启动前过"种子只影响初始化"预检）

已完成并转入正式登记：G0 网格全量、时间块 IID 对照（并入 G0-GRID 行）、E1 全部 11 任务（E1-FULL 行，
含原"剩余 4 个任务"——其 B 臂为时间块口径，`b_split_basis` 字段已单独标注）。

## 维护规则

1. 每次实验**开工前**登记一行（ID 可先用日期前缀），跑完补「结论」。
2. commit 列填 `code/` 仓库的 `git rev-parse HEAD`；纯分析脚本改动也要先提交再跑。
3. 「进正文」= 该实验的数字可能出现在论文正文；改为「否」需注明理由。
4. 临时验证（如 `/tmp` 里的 smoke）若结论有价值，登记时在输出列注明"临时，结论已转录"。
