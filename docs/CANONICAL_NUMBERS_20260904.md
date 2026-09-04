# 规范数字表（CANONICAL NUMBERS）

**日期**：2026-09-04　**目的**：正文里每一个承重数字只允许一个值、一个出处、一个估计量口径。
同一个量在项目里往往存在多个估计值（不同种子数、不同测试块、不同任务集），审稿人会把它们当成矛盾。
本表规定**正文引用哪个**、**其它版本为何不同**、**该数当前的证据等级**。

**证据等级**：
- **F（冻结）**：有登记行 + 冻结协议/规格 + 双跑产物在盘。可直接进正文。
- **P（产物在盘，无登记行）**：数字可从盘上产物逐位复算，但产生它的网格/脚本无登记行。进正文前须补登记或随协议重算。
- **X（探索性，无产物）**：只存在于文档文字中，产生它的 scratch 已丢失。**不得进正文**，须重算。
- **待追**：出处尚未定位。

**本表本身的规则**：改动任一承重数字须同时改本表；新增承重数字须先在本表登记出处。

---

## A. 干预（TWO-CHANNEL-SELF）—— 等级 F

| 量 | 正文值 | 产物 | 口径 | 已知其它版本 |
|---|---:|---|---|---|
| ⓪ 源域 CV 选模型 | **0.5191** | `results/two_channel_20260903/eval_five_points.csv` 列 `P0_src_cv` 9 任务均值 | seed 42 单次；目标域 macro-F1 | — |
| ⓪′ RF 重要性 + CV | **0.5258** | 同上 `P0p_importance` | 保留 57 列（含 7 列 rssi） | — |
| ① 全特征 + stacking | **0.5249** | `C1_all_stack` | | — |
| ② 全特征不堆叠 | **0.5581** | `C2_all_nostack` | | — |
| ③ 删 rssi + stacking | **0.7544** | `C3_drop_stack` | | — |
| ④ 流程终点 | **0.7604** | `C4_endpoint_1then2` = `C4_endpoint_2then1` | `PATH_DEPENDENT = False` | — |
| ⑤ 逐任务 oracle | **0.7648** | `C5_oracle` | regret 0.0045 | — |
| ④−⓪ | **+0.2413** | 上两行之差 | **样本内**（协议 §3.3） | — |
| 按 `\|S\|` 拆分（⓪‴ 多源基线） | 见 `MAINLINE` §7 表 | `results/two_channel_20260903/multisource_baseline_posthoc.csv`（2026-09-04 事后聚合，等级 **P**） | `\|S\|=1` 6 任务 / `\|S\|=2` 3 任务 | — |

## B. 坐标 1 诊断量 —— 等级 F / P

| 量 | 正文值 | 产物 | 口径 | 已知其它版本 |
|---|---:|---|---|---|
| `S(F)` rssi | **+5.2440** | `results/two_channel_20260903/diag_coord1.csv` 求和 | ΔAUC 代理，9 任务 | — |
| `S(F)` 次名 subwin | **+0.0156** | 同上 | | 登记表曾写 0.016 |
| 头名/次名比 | **336.2** | 5.2440/0.0156 | | **328**（用四舍五入的 0.016 算出；登记表 `UNSW-TWO-CHANNEL-VALIDATE` 行仍写 328，门槛 <10 不受影响） |
| `S(F)` 按 `\|S\|` 分层 rssi | `\|S\|=2` +1.1076（唯一正值族）/ `\|S\|=1` +4.1364 | 同上按任务分层 | 登记表 `HISTINT` 行已记 | — |
| `ΔLORO` rssi | **+0.2920** | `results/histint_objective_signal_20260903/s_obj.csv` `max_base` 口径 `d_loro` | 3 个 `\|S\|=2` 内层任务合计，seed 42 | `best_base_src` 口径 +0.2963 |
| `ΔJOINT` rssi | **−0.0406** | 同上 `d_joint` | 天花板代价 | `best_base_src` −0.0289 |
| `ΔLORO` len | **−0.1610** | 同上 | 反向对照 | `best_base_src` −0.1633 |
| AUC 代理 vs 目标函数 Spearman | **0.8182** | `passline.json` `max_base.spearman_vs_frozen_S_F` | | `best_base_src` 0.6848 |
| `JOINT`（H 内联合时间块） | **0.9101** | `joint_reference.json` `__full__` | seed 42 单次，测试 = 每轮 5 块中第 5 块 | **0.8936**（`MAINLINE` §2，5 种子均值，**出处待追**，见 E） |
| rssi 的 RF 重要性份额 | **25.29%** | `results/robust_v2/raw_all/joint_R2_R3_R4/all_features/rf/feature_importance.csv` | 单任务、指示性 | **27.4%**（登记表 ⓪′ 语境，**出处待追**）；两者口径不同，正文只引一个并注明 |

## C. 机制：四格分解与推翻率 —— 等级 P（网格无登记行）

| 量 | 正文值 | 产物 | 口径 | 已知其它版本 |
|---|---:|---|---|---|
| 共识推翻率（三旗舰） | **78.06% = 1149/1472** | `results/g0_environment_grid/fourcell_posthoc_self.csv` full94 hard `\|S\|=2` 合计 | 三基模型一致且正确 → stacking 改判 | — |
| 自采四格 full94（9 任务，困难簇） | **3379 / 1152 / 4653 / 204 / 594，净 −948** | 同上 | 固定簇 {Light_T1, Light_XM, Sensor} | **4065 / 1152 / 4884 / 205 / 594，净 −947**（`MAINLINE` §7 旧表，scratch 已失，分母**复现不出**，已撤回） |
| 自采四格 strict59_ra | **7240 / 59 / 1127 / 1 / 9，净 −58** | 同上（网格 `_r5`；`_r3` 逐位相同） | | **7821 / 58 / 2170 / 58 / 10，净 0**（旧表，同上撤回；「精确 0」不成立） |
| 破坏率降幅 full94→strict59 | **42 倍**（0.3409→0.0081） | 同上 | | 旧表 38 倍 |
| UNSW 四格（Belkin 对，3 任务） | **4900 / 633 / 4358 / 656 / 23，净 +23** | `results/unsw_meta_mismatch_20260902/fourcell_posthoc.csv` | 同定义；`broken` 与 `n_overridden` 逐位对账 | — |
| UNSW 破坏率 / 救回率 | **0.1292 / 0.1505**（任务 0：0.2965 / 0.4191） | 同上 | | — |
| 自采 vs UNSW 破坏:救回 | **5.8:1 vs 1.0:1** | 两文件 `\|S\|=2` | **这是解离的承重量**；「同推翻率」修辞已撤回 | — |
| `meta_soft_cls` gap / stacking gap | **−0.0058 / −0.1741** | `results/g0_environment_grid/weight_vs_map_diagnostic.csv` 三旗舰均值 | 权重 vs 映射消融 | — |

## D. 外部数据集 —— 等级 F

| 量 | 正文值 | 产物 | 口径 |
|---|---:|---|---|
| UNSW 跨天落差 | **+0.032778**，6/6 为正 | `results/unsw_iid_reference_20260902/paired_comparison.csv` | 设备身份粒度，rf，日内时间块 IID 配对跨天 |
| UNSW 同日 IID 区间 | 0.8684–0.8907 | `iid_summary.csv` | **跨越 10/13/14 类**，不是同口径单一量，不得作单一参照数 |
| UNSW `gap_H` 三任务 | +0.0117 / −0.0075 / −0.0105；`mean_abs` **0.0099** | `unsw_meta_mismatch_20260902/signature_table.csv`, `passline.json` | 预注册区间 [0.03, 0.20] |
| Belkin 互混 | `f_ab` 0.361 / `f_ba` 0.446 | `cluster_definition.json` | **跨天均值**，逐天稳定性未测 |
| UNSW 摄像头域内 F1 组均值 | 0.9364–0.9585 | `unsw_iid_reference_20260902/per_class_iid_*_selfgate.csv` | 5–6 台过门槛 |
| 同型号对窗口数 | NestDropcam 209/1 天；InsteonCam_wifi 45/1 天 | `results/unsw_features_full/device_window_counts_by_day.csv` | 存活者偏差依据 |

## E. 阶梯与 `\|S\|` 分层 —— 等级 P / X / 待追

| 量 | 文档值 | 状态 | 处置 |
|---|---:|---|---|
| 同轮次随机 / 时间块 / 联合内部 / 联合时间块 / LORO | 0.9622 / 0.9296 / 0.9200 / **0.8936** / 0.6979 | `MAINLINE` §2，5 种子；0.9622 与 0.6979 另见 `PROJECT_STATUS…0902` 与 UNSW 协议，**中间三档出处待追** | 正文引用前须定位产物或重算落盘 |
| `\|S\|` 分层 best_base full94 | **0.5771 / 0.6907 / 0.7730**（gap −0.0700 / −0.0931 / −0.0541；IID 时间块 best_base 0.8896） | `results/g0_environment_grid/nsource_strata_posthoc.csv`（2026-09-04 重算落盘，逐位复现文档值） | 等级 **P**（网格无登记行） |
| `\|S\|` 分层 strict59_ra | **0.7337 / 0.7987 / 0.8243**（gap −0.0028 / −0.0103 / −0.0039；IID 0.8616） | 同上，网格 `_r5` | 等级 **P** |
| 迁移效率 `TE` 自采 | 0.7879 / 0.9237（**9 任务口径**，脚本已失）；按 `\|S\|` 全网格口径为 full94 0.649/0.776/0.869、strict59 0.852/0.927/0.957（`nsource_strata_posthoc.csv` 列 `TE`） | 9 任务版等级 X、全网格版等级 P | **`TE` 依赖任务集**，正文引用须注明口径 |
| rssi 反转率 | 0.2628（transferable-59 的 3.6 倍） | `MAINLINE` §4，**出处待追** | 同上 |
| 25 维 RSSI 消融 | 0.6745→0.6531（→R5）；0.7550→0.7310（→R6+R7） | `docs/RSSI特征分析.md`——**该文件仍 untracked** | 入库；`R567` 在 94 维重测 |

## F. E1 / E2 —— 等级 F

| 量 | 正文值 | 产物 |
|---|---:|---|
| E1 B 臂旗舰 `loro_R2_R4_to_R3` gain | **−0.106**（5 种子） | `results/e1_oof_arms/`；**−0.1241 是 seed 42 单种子值**，8/28 简报用的是它 |
| E1 B 臂最大崩溃 `loro_R2_R3_to_R4` | **−0.223** | 同上 |
| E1 A→A′（纯折数） | 全部 ±0.012 内 | `e1_decomposition.csv` |
| E2 `CPD_y` 增量解释力 | 消失 → §13 降级分支 | `results/e2_conditional/E2_CONCLUSION.md`, `e2_acceptance.json` |

---

## 使用规则

1. 正文中的每个数字必须能在本表找到一行；找不到就先加行。
2. 等级 **X** 的数字进正文前必须先变成 **P** 或 **F**（重算并落盘，或补登记行）。
3. 「已知其它版本」列存在的数字，正文**只引正文值**，并在首次出现处脚注为何与其它版本不同。
4. 本表 2026-09-04 建立时的三个已知不一致：`JOINT` 0.9101/0.8936、rssi 份额 25.29%/27.4%、头名次名比 336.2/328。
   前两个各自出处待追，第三个已解决（328 是舍入产物，正文用 336.2）。
