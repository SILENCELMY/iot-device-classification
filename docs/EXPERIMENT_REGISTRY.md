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
| C3-CONFIRM-UNSW16 | 2026-09-05 | `4291f23` | `docs/PROTOCOL_C3_CONFIRMATORY_20260905.md`（预登记，§3 主判据 / §4 配置 / §5 成功判据 冻结） | 42 与 43 | `results/unsw_features_full/features_day_16-09-23.csv`（训练）、`…16-09-30.csv`（选参）、`results/unsw_reserve_16day/features_unsw_w10_reserve16.csv` + `lenhist_unsw_w10_reserve16.csv`（**保留 16 天，读取前 0 次访问**）、`dataset/unsw/pcap/`（臂 A 需原始包） | `results/c3_confirmatory_20260905/` | **待跑** —— 判 C3a 粒度饱和曲线在留出天复现且饱和 ≥0.99；**C3b′**（协议 §8 改写，仍在读保留集前）闸门在选参天的判决符号在留出天复现（区上符号翻转 0 对）且端到端 Δ>0；C3c Δ ≥ −0.002。§8 记录闸门记账缺陷的修正：整区记账 + 区∈{R2,R3∩R1,R3} 逐对选 | 是（确证测试，三种结果全部照原样写） |

> **修订记录（2026-08-28）**：本行首次登记时误记 B 臂 macro_f1 为 0.5498（"崩溃略软"）。
> 该数字来自 `_splitter` 的一个缺陷：多轮分组分支缺 `return`，执行掉进单轮时间块分支并覆盖
> GroupKFold 结果，实际跑的是时间块 OOF 而非协议 §9.1 的轮次分组。修复后真实值为 0.4907，
> **方向与原记录相反**。修复见 commit `6a3f678`，严格化测试 `code/scripts/analysis/test_oof_modes.py`（11 项断言）。
>
> **注意（待你决定）**：`loro_R2_R4_to_R3` 只有 2 个源轮次（R2、R4），`GroupKFold` 的
> `n_splits` 因此退化为 **2**——每折用单轮训练、另一轮验证。这是协议 §9.1「group=round」的
> 直接后果，但元特征噪声显著大于历史的 5 折。E1 正式跑之前需确认这是否是期望口径。

## 待登记（已排期）

- C3 确证测试的前置：保留 16 天 lenhist 抽取 **已完成**（`lenhist_unsw_w10_reserve16.csv`，1139981 行 × 72 列，目标 top32 与`16-09-23` 原始导出逐位相同）

- G0 网格全量（约 156 任务，种子 1 个，协议 §14）；实测单任务 4 模型约 11–16s，全量约 36 分钟
- 时间块 IID 对照
- 深度模型 5 种子（§14 正确入口）
- UNSW pilot
- E1 剩余 4 个任务（3 个 `single_round` + `joint_R2_R3_R4`）：均为单轮/IID，无法按轮次分组，
  按 §12 用时间块并单独标注，A′ 的「分组效应」在这些任务上含义不同（时间块 vs 随机，非轮次分组）

## 维护规则

1. 每次实验**开工前**登记一行（ID 可先用日期前缀），跑完补「结论」。
2. commit 列填 `code/` 仓库的 `git rev-parse HEAD`；纯分析脚本改动也要先提交再跑。
3. 「进正文」= 该实验的数字可能出现在论文正文；改为「否」需注明理由。
4. 临时验证（如 `/tmp` 里的 smoke）若结论有价值，登记时在输出列注明"临时，结论已转录"。
