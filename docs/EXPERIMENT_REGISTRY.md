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
| E1-FULL | 2026-08-29 | `code/` @ `1fa5bcc`（结果入库 `3952388`） | `e1_oof_arms.py --tasks all --seeds 42,43,44,45,46`，三臂 A/A′/B，A′ 折数动态对齐 B，all_features | 42–46 | `features_raw_all_w10.csv`，11 个主线任务 | `results/e1_oof_arms/`（e1_arms_raw.csv / e1_decomposition.csv / e1_arms.json） | B 臂崩溃在全部 OOD 任务持续并**加深**（gain 5 种子均值，A→B）：旗舰 `loro_R2_R4_to_R3` −0.066→−0.106；`loro_R2_R3_to_R4` −0.004→−0.223；`loro_R3_R4_to_R2` −0.024→−0.181；`joint` −0.001→−0.104；jitter 三任务 ≈−0.005→≈−0.08；IID 三任务两臂无差（≈−0.002）；唯一例外 `position→R5` 反转为 +0.024。折数效应（A→A′）小而偏正 → 加深源于分组效应。**按 §12 预注册规则走第一分支：现象基础成立，按 X2 继续机制检验**。结论文档：`results/e1_oof_arms/E1_CONCLUSION.md`（含论文引用规则改为 B 口径、R5 反转与 joint 构造性代价两条专门记录） | 是 |
| G0-GRID | 2026-08-29 | `f740362` | `environment_grid_experiment.py` 全量：150 OOD（\|S\|=1/2/3 → 30/60/60 任务）+ 12 IID（6 环境 × random/time_block，§8.4 时间块对照）；模型 rf/xgboost/lightgbm/stacking；stacking 按 §9.1 grouped | 42（脚本常量 SEED，§14 单种子） | `features_raw_all_w10.csv`（缓存复用，无 tshark 重抽） | `results/g0_environment_grid/`：summary_metrics.csv（648 行 / 162 任务）、env_topology_matrix_rf.csv（RF macro-F1 6×6）、逐任务 CM / pred_proba / oof_meta（stacking） | 网格与时间块 IID 对照完成；CPD 口径同质拓扑矩阵待 §20.2 改造后生成（EXECUTION_PLAN D4）。§8.6：网格不用于显著性结论 | 支撑（§8.6） |
| D4-TOPOLOGY | 2026-08-29 | `1038f3b` | `six_env_confusion_similarity.py`（§20.2 重写）读 G0 \|S\|=1 + 双 IID 参照；口径 = `cpd_y(ref=CM_iid(i), tgt=CM_{i→j})`，见 EXECUTION_PLAN D4 | —（确定性分析） | `results/g0_environment_grid/raw_all/` 下 30 个有序对 + `g0_iid_R*_{time_block,random}` 的 rf/all_features 混淆矩阵 | `env_topology_cpd_y_ref_{time_block,random}_rf.csv`、`env_topology_macro_f1_from_cm_rf.csv`、`env_topology_cpd_dir_*`、`TOPOLOGY_MATRIX_NOTE.md` | 同质 6×6 替代废弃六环境矩阵。macro-F1 核对 30/30（最大偏差 1.1e-16）；审阅方独立重算抽查逐位一致；CPD_dir 在 n_err≥20 下仅 14/30 有定义（缺失主要来自 IID 参照侧误差过少，与 P0 §6 同向）；test_cpd_core 15/15、test_oof_modes 17/17 全绿。旧 `six_env_off_diag_frobenius_rf.csv` 原地保留作 0.1521 回归锁定源（legacy 迁移废止，见 CPD_DEFINITIONS §5.2） | 支撑（§8.6） |
| S1-DEEP-5SEED | 2026-08-29（进行中） | 起跑于工作区 @ `042fa03`（实验脚本零改动，仅 CLI 参数） | §14 入口 + 三项必要 CLI 偏离：`--output-root results/s1_deep_5seed_20260829/seed{N}`（默认路径会因 metrics.json 已存在而静默短路复用 seed 42 → std=0）、`--cnn-v5-source legacy/results/extreme_capacity_1p2m_20260706/raw_all`（默认源已迁 legacy）、解释器 `~/anaconda3/envs/iotcls/bin/python`（torch 2.5.1+cu121；base 无 CUDA 会静默跑 CPU）；`PYTHONPATH=code/scripts/analysis` | 42–46。预检通过：splits 逐字节复制（diff 无差异）、`--random-state` 经 set_seed 影响初始化 + DataLoader 批次顺序，**不影响划分**——语义记为"划分固定、训练随机性（初始化+批序）变"，符合 §14 意图（其补救条款仅针对影响划分的情形） | `results/robustness_scaling_20260706_v2/splits`（固定划分） | `results/s1_deep_5seed_20260829/seed{42..46}/`；日志 `results/s1_deep_5seed_logs/`；驱动命令全文见日志目录 `run_5seed.sh`（不入库，命令已录于此行） | 运行中（单种子约 85–90 分钟，ETA 8/29 晚）。锚点：seed 42 的 single_round_R2 三候选复现 7 月历史值至 4 位小数（0.8989/0.9092/0.9129）。**聚合警告：cnn_v5 行是复制的单种子参照，出表不得标 ±std** | 待跑完 |
| E1-G0-GRID | 2026-08-29（进行中） | 工作区（`e1_oof_arms.py` 网格扩展，代码入库待 D2 审阅） | `e1_oof_arms.py --grid all`：150 个 G0 OOD 任务 × 三臂 × all_features；\|S\|=1 的 B 臂 = `window_start` 时间块（判定按训练轮次数，非任务名单）；§19.2 provenance 落盘 | 42（§8.6 覆盖用途；43–46 视实测耗时由审阅方另定） | `features_raw_all_w10.csv` + G0 任务生成器（import，无重复实现） | `results/e1_oof_arms_g0/` | 运行中（并行会话观测到进程已启动）；启动前置条件为 smoke 双一致性（基模型 F1、B 臂 ≡ G0 stacking，容差 1e-6）通过，具体数字待 D2 报告归档后补录 | 支撑（§8.6） |
| P2-UNSW-PILOT | 2026-08-29 | 工作区 @ `0e5e791`（`extract_features_generic.py` 与 pilot 脚本随本行入库） | §16.3 五问 gate。4 天 pcap（09-23 / 09-30 / 10-11 / 10-12，约 11GB，直连 ~10MB/s 计约 12 分钟）；提取器 61 维通用特征（94 − 33 项 802.11 专属 = 61，对账闭合）；RF LORO import 主线 `build_model` / `sample_balanced`（§7/§16.4，不重实现）；脚本在 `code/scripts/analysis/unsw_pilot/` | 42 | `dataset/unsw/pcap/*.pcap` + `dataset/unsw/device_mac_map.csv`（入库副本见 `results/unsw_pilot/PROVENANCE_NOTE.md`） | `results/unsw_pilot/`：PILOT_FIVE_QUESTIONS.md、FEATURE_ALIGNMENT.md、INVENTORY.md、loro/、four_day/、BELKIN_PROBE.md、CSV_CROSSCHECK.md。特征缓存 `features_*.csv` 为可再生派生物且合并档 124/168MB 超 GitHub 单文件硬限，按 .gitignore 注释排除（各 run_meta.json 入库） | **五问全部通过（gate PASS，无需换候选）**：① MIT-0，论文使用与派生特征发布均可；② MAC 映射完整、四天零未知设备（唯一未列 MAC 为网关兄弟接口 `…:e9`，仅作 dst）——**限制进正文：23 个 IoT MAC 仅 11 个天天在线、10 个天天 ≥100 窗（测试床后期下线所致，10-11 全天对照排除短抓包与标签污染两种解释），§16.4 扩展必须按日交集取类并逐任务报实际类别数，不得笼统写 18 类**；③ 09-30 与 §16.1 预核数字逐项一致（20 活跃 / 18 ≥100 窗 / 18 ≥300 窗 / Dropcam 8640；CSV 清点交叉核对 673,414 / 86,398 / 84,291 逐位复现）；④ pcap 时间戳微秒级（唯一率 1.000000 vs CSV 0.125），61 特征 0 NaN / 0 inf；`side_/other_packet_ratio` 两列 Ethernet 上恒 0 仅作列对齐，**禁止进入跨数据集特征重要性比较**；⑤ 6 有序日对 LORO macro-F1 0.8164–0.8674（均值 0.8466，任务实际 10–14 类），Belkin Switch↔Motion 混淆经判别检验排除标签互换（域内即最难两类、双向混合形态，证据 BELKIN_PROBE.md）。执行决定：10-12 实为 14.48h 抓包，保留（真实末日、最大跨度）并加录 10-11 全天对照。相邻日对分数最高（0.8901/0.8778）、远隔 19 天对最低的梯度已记录不解读（属 §16.4 检验 1）。提取成本 ~11 min/天（窗口级 pandas 开销），20 天全量前需按天并行化 | 是（P2 gate 交付物） |

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

- D8 投票基线后处理（进行中，回报后登记）

2026-08-29 转入正式登记：D4-TOPOLOGY（已完成）、S1-DEEP-5SEED（运行中）、E1-G0-GRID（运行中）、
P2-UNSW-PILOT（已完成，gate PASS）。

已完成并转入正式登记：G0 网格全量、时间块 IID 对照（并入 G0-GRID 行）、E1 全部 11 任务（E1-FULL 行，
含原"剩余 4 个任务"——其 B 臂为时间块口径，`b_split_basis` 字段已单独标注）。

## 维护规则

1. 每次实验**开工前**登记一行（ID 可先用日期前缀），跑完补「结论」。
2. commit 列填 `code/` 仓库的 `git rev-parse HEAD`；纯分析脚本改动也要先提交再跑。
3. 「进正文」= 该实验的数字可能出现在论文正文；改为「否」需注明理由。
4. 临时验证（如 `/tmp` 里的 smoke）若结论有价值，登记时在输出列注明"临时，结论已转录"。
