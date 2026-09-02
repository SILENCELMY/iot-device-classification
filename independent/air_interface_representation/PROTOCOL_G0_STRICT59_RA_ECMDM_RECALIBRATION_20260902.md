# G0 strict59_ra 下 EC-MDM oracle 重裁定协议（运行前冻结）

**冻结日期**：2026-09-02（Asia/Shanghai）  
**状态**：`PROTOCOL_FROZEN_BEFORE_ANY_STRICT59_RA_G0_NUMERIC_RUN`  
**归属**：独立空口表示接口审计线；不并入主线，不修改任何既有冻结文件或失败目录。  
**替代关系**：本协议取代但不修改
`docs/PROTOCOL_G0_STRICT59_ECMDM_20260902.md`。旧协议基于方向提取有缺陷的 `strict59`，禁止执行。

## 0. 已知事实、协议动因与最大允许主张

本协议冻结前只允许知道以下既有事实：

1. `strict59_ra` 方向修复审计状态为 `STRICT59_RA_DIRECTION_REPAIR_ACCEPTED`。在 R2--R4 上，旧/RA
   上行零窗口由 79.88% 降至 0.53%，`up_len_std` 零窗口由 100% 降至 3.96%；相对 `strict45`
   的五模型等权 `Delta LORO=+0.0498`、`Delta 最差 LORO=+0.0754`。
2. 旧 `strict59` 的下行 `TA==BSSID` 正确，但上行误用 `DA==BSSID`；正确空口接收方语义是
   `RA==BSSID`。旧 `strict59` 只能作有缺陷历史诊断，不得作为科学门或输入。
3. full94 下既有 M1/M1-R/M2 表明：有目标标签的完整 target-CF 元层有可恢复上界，C 级是首个充分
   阶段；这些量从来不是无标签估计或部署收益。
4. full94 下既有 M4 仅得到 `INTERMEDIATE_RECOVERY_FROM_FIXED_OBSERVABLES`。该结果不能代替
   `strict59_ra` 下的 observable estimability 裁定，也不能由本协议改判。

本协议最多把 EC-MDM 裁定为：**在语义修复的 59 维表示下得到支持的、有目标标签 oracle 证据的
CPD 候选上位构念**。这里“候选”不可删除。本协议不计算 CPD，因而不证明 EC-MDM 是 CPD 的机制、
因果来源、可观测替代量或可部署上位构念。

## 1. 唯一问题与三层证据隔离

唯一问题是：在 R2--R7 全部使用 `strict59_ra` 后，固定基模型目标概率不变时，源元层相对 target-CF
元层的 OOD-IID 超额损失是否仍有实质量级，以及 `I -> G -> C -> F` 阶梯中 C 是否仍是首个充分阶段。

三层证据必须分开命名、分开裁定：

| 层 | 本协议是否检验 | 标签条件 | 允许术语 |
|---|---|---|---|
| oracle recoverability / structure | 是 | 目标标签用于 target-CF 与 I/G/C/F 交叉拟合 | 离线上界、结构诊断、候选构念 |
| observable estimability | 否 | 目标标签不得进入部署时估计量 | 未裁定；须另冻协议 |
| deployability | 否 | 未知目标环境的可用信息与时序约束 | 未建立；不得由 oracle 推出 |

`MMG`、`excess_F` 和 `ER` 均属于第一层。即使全部通过，也不得写成“无标签可估计”“上线可用”或
“部署收益”。

## 2. 冻结输入与哈希锚

### 2.1 已接受 strict59_ra 接口

| 文件 | SHA-256 |
|---|---|
| `results/air_interface_representation_audit/strict59_ra_run_20260902_r2/VERDICT.md` | `279eefa6358a7cf132fb623a82283f3cd0b7e7b8a49f3e1c4fbe1a638865b53a` |
| `acceptance.json` | `67a510f732a726752e5840c2d865fd176dd0a715dc4907e62b9c52777758941e` |
| `feature_arms.json` | `5a44b90e9a7cf30588c348addb10ad3d0292abd3adb9ef5c7227ee9247910005` |
| `corrected_direction_features.csv` | `732751066390e615b3f3dbf20d3be808e1f90982ae73773ca3186e94cba7f552` |
| `split_audit.csv` | `6b3cd823cb74d2e847dc629eefb31cecc293509ecaab16e9f20b7ccd9195dafb` |
| `manifest.json` | `f57975a459eabb3ccd1c1a6a12db5042d3ecfb3c4f49e095f9c4bdd5b0c26bd0` |

`acceptance.json::status` 必须精确为 `STRICT59_RA_DIRECTION_REPAIR_ACCEPTED`，且 manifest 对其覆盖的
每个文件逐项精确通过。`strict59_ra` 列名单必须从上述 `feature_arms.json` 的同名项读取，恰为 59 个
唯一字段；不得从旧 bridge 的 `strict59` 项读取。

### 2.2 缓存、任务与冻结实现

| 输入/实现 | SHA-256 |
|---|---|
| `results/robust_v2/raw_all/features_raw_all_w10.csv` | `9bf191f0fb74d66463c829bbc39de73d752265163bf6dc1729a668e3d1c6ca41` |
| `code/scripts/core/environment_grid_experiment.py` | `eeb7cda6c71b14f5ee6b725faef5c0bc056a703f956f9acb93b09243d87499e9` |
| `code/scripts/core/robust_iot_research.py` | `1d29434570d35422ce2b7cd9d485259f5520761aa11999342ca9b0cc2f36a5e3` |
| `independent/meta_mismatch/PROTOCOL_M1.md` | `b055dca8c430ef2e42d817d5e16f9bedf963cdd4104a8cb0e34618c10cb386c8` |
| `independent/meta_mismatch/PROTOCOL_M1R.md` | `0c6195bc17ea972f7cc18ad343cf7939e1b35814f4bfdd4ac0771409dc0ee2b6` |
| `independent/meta_mismatch/PROTOCOL_M2.md` | `4b23bfa1509ef1fa26b6afe945f4191a4099a6bc7c0446ffbc8bd2cdee53cfb5` |
| `m1_meta_counterfactual.py` | `96f5419608b66e77617420f381e149da4738b8780205a6cc994e36fcab330dea` |
| `m1r_matched_control.py` | `07110482c458a256c4582762d8217a52a835b0bb239ab88a913291ce4ac17884` |
| `m2_meta_mechanism.py` | `2ca7e4a7e9a516910b038e8d16529611f75e443182385414c8794205609d2e51` |

缓存必须为 11303 行、8 个精确 meta 字段与 94 个数值字段；轮次数固定为
`R2=1816, R3=1837, R4=1853, R5=1816, R6=1988, R7=1993`。数值必须全部有限，行键
`(label, round, source_file, window_id)` 唯一。

### 2.3 R2--R7 的 30 个 pcap 锁

| pcap | SHA-256 |
|---|---|
| `camera_r2` | `9e8dab77b439fa97865a2705a3d91b99c3d567fe0282f8e85b7c081b3080cad4` |
| `light_T1_r2` | `9c62775af38207f128941da2de98ae7b91ef12b798ed9f68b79cfecf49275ce6` |
| `light_xm_r2` | `88b46b2b2c68b6725e70a4820173f580bdaf96042253e34e90b62c8af822d5c2` |
| `sensor_r2` | `068d027f5d1a06324f84d38aadddbdde7bba70453b12a4654e9c58a7a2668e57` |
| `socket_r2` | `133fb099ccb33a3b1bc4e817c3ebe1c408c593681248584c00ebab5690769760` |
| `camera_r3` | `b86bc442465ab53ae1d0850f0096f6d5d3e4a92f2f7067a968b82669e77c3202` |
| `light_T1_r3` | `ebfa62cbc5d25855a4b47c80bd48644665a5a604e1f78ad658e35c1eea6cfa9c` |
| `light_xm_r3` | `410d415615646d322a78f72c0112c8d1db1ac99949831fe9eb5b38fbac95026d` |
| `sensor_r3` | `4665629a665ce062ed4682d476b72e8e896ec9e1e7bc19287579d37e759deedc` |
| `socket_r3` | `15ca818ea3a927f46fb057667c8fcce9a76fb1709985af00dbdba84719859990` |
| `camera_r4` | `ad68d87c6c705b38bdac901e9fb5038824a84f9e2ea43a87866486c04af52974` |
| `light_T1_r4` | `50b7968226ccb6ea254b704795befe89bb6d4e648b9f3bd0ff22c3de528bdb99` |
| `light_xm_r4` | `5cff4efe8862525051f5a97362dd8b6c18c04fec4108d82186a4c5df45315c97` |
| `sensor_r4` | `4ce712b7ab9c05d181cd0819bf31f94e82b80e311e0389f44d1a7ad2a5caf7a6` |
| `socket_r4` | `901b25b1af1f4c3baef527f88a3f602b8606a3f3f7aaac378f261bd716267ea6` |
| `camera_r5` | `fb5280ae19283f38abbf0ea0ac8f1295f06b7f83d5ecef47e0efb5b2fbe1fbf5` |
| `light_T1_r5` | `fb70d9ef4ec7c277e8287ed6f213b5cfa9bf7949b69cc7e3400fcfe1fb5a8ad2` |
| `light_xm_r5` | `e93120f0707bd741dfaff22cd26add7b00ba4c132d2871254b9a0c3fbf6d9fb8` |
| `sensor_r5` | `70a78607d215ba4af46f605ab9630860c34a7421bace60b9b6193abe61d02b57` |
| `socket_r5` | `de622b7be00643a1ef97f244eee803b4cd353ba66b7ae6ecc76629f79ff02fd5` |
| `camera_r6` | `55df60693bd517ffbfc63c89df00c6a134f9cedca911cb70f95b3f1d664d45e6` |
| `light_T1_r6` | `9fab293b83490fe8e58b3de1007cadb30511475aef7143dd2204844d2fd5f5c0` |
| `light_xm_r6` | `17fe8d7f0fa0e17ffcb00f9fdbb13109f2b4d313b9618ebb19baa40259571c8e` |
| `sensor_r6` | `c2ab988b677f5cd436aa25f8f05055041a6b9cd89e2cfe872cebcb7d866e38d7` |
| `socket_r6` | `62ee7a337a5adf43d852922a0c5e20b9993e4f481db3e6f83b3d9e6b416a890b` |
| `camera_r7` | `a3e9a0282268ad3fca1a3c25f13b03dc6c0ed5c374d07e96cbf8f19c16a44ca3` |
| `light_T1_r7` | `36af506bd89947e7e9c693b8fbfc6bc70fe46cefd58120e19f113c7a14f2bdc3` |
| `light_xm_r7` | `e752598f7fd7ad472a0c58dea9232f71341601451cdaf1301187f6dbb39427dd` |
| `sensor_r7` | `8645dfd930733d104035d294ff76f9ad3d1510b2687baf43d6ed52c638c86d8b` |
| `socket_r7` | `d2dec476114dc99845affd5332fa6c78a52605788230e2c876e38f32104fc3ed` |

任一锚不符即 `INVALID_RUN_STOP`，不得用相近文件或旧 `strict59` 代替。

## 3. R2--R7 表示物化与接口硬门

已接受审计只为原 bridge 的 R2--R4 重提取了 5506 个窗口；G0 使用 R2--R7。因此本协议必须先把
**同一个已接受的 TA/RA 语义**机械扩展到全部 30 个 pcap，而不是让 R5--R7 沿用旧 DA 错误。

物化规则逐字继承方向修复审计：使用冻结 tshark 字段、10 秒非重叠窗口和最少 2 包门；先走旧实现，
随后只在内存副本中令 `da=ra` 再调用同一个 `summarize_window()`，且只覆盖以下 14 列：

```text
up_packet_ratio, down_packet_ratio, up_down_ratio,
up_len_mean, up_len_std, up_len_p50,
down_len_mean, down_len_std, down_len_p50,
up_ia_mean, up_ia_std, down_ia_mean, down_ia_std,
len_up_down_diff
```

工程硬门全部满足才可训练：

1. pcap 重建旧 full94 与缓存逐值最大绝对误差 `<=1e-9`，11303/11303 窗口完整；
2. 8 个 meta 字段、缓存原行序、原 pandas 行位置和 45 个公共字段逐位不变；
3. 四方向比例闭合最大误差 `<=1e-12`；所有 59 个模型字段有限；
4. R2--R4 的 5506 个键与已接受 `corrected_direction_features.csv` 完全一致，14 个 `ra_*` 字段
   最大绝对误差 `<=1e-9`；
5. 全部 R2--R7 与各轮分别报告旧/RA 上行零比例、`up_len_std` 零比例和变化比例；总体 RA 上行零比例
   `<=0.50`、总体至少 50% 窗口的上行比例发生变化、总体 RA `up_len_std` 不得 100% 为零；
6. 物化缓存列序必须精确为原缓存中的 8 个 meta 字段原顺序，随后为 `strict59_ra` 的 59 列冻结顺序；
   共 67 列，不保留其它 35 个数值字段。

该步骤是表示接口补全，不是新的科学臂；不得用 R5--R7 的提取统计改门或选择字段。

## 4. G0、M1、M1-R、M2 的冻结运行口径

### 4.1 G0(strict59_ra)

- 任务唯一由 `environment_grid_experiment.build_task_grid()` 生成：150 OOD + 6 IID time-block +
  6 IID random，共 162 次运行；环境 R2--R7，`MAX_SOURCES=3`。
- 模型固定为 `[rf, xgboost, lightgbm, stacking]`，种子 42，`FILTER_MODE=raw_all`，窗口 10 秒，
  `max_rows=10^9`，`feature_mode=all`，禁用特征选择，`n_jobs=4`。
- 总模型单元必须为 `162 x 4 = 648`。每个模型的 `feature_columns.json` 必须为同一 59 列且顺序精确。
- 不改 `code/`。只允许 wrapper 在进程内覆盖 G0 的 `CACHE_SRC` 与 `OUT_ROOT` 后调用其原入口；
  任务、模型和训练函数不得复制或重写。

### 4.2 M1/M1-R/M2(strict59_ra)

M1、M1-R、M2 的数学定义、目标时间五折、类别轴、基模型块顺序、匹配支撑、I/G/C/F 参数化、优化器、
正则、充分性阈值和 4-class 敏感性全部逐字继承各自冻结协议。本协议只允许 wrapper 改输入/输出 root：

- M1 `INPUT_ROOT` -> 新 G0 的 `raw_all/`；
- M1-R `M1_RESULT_ROOT` -> 同一重复的 M1 输出；
- M2 `M1R_ROOT` 与 `G0_ROOT` -> 同一重复的新输出。

不得修改三个既有脚本。本协议的实现须以静态测试证明只覆盖路径常量，没有覆盖任务、标签、模型、阈值、
聚合、优化或判定函数。

## 5. 运行前 full94 判否门

在生成任何正式 `strict59_ra` 模型数值前，须先完成以下两项：

1. 对 §2.2 三个脚本/协议及 full94 冻结 M1/M1-R 输入与 gate 做 SHA-256 完整性审计；
2. 在两个全新临时目录上，用冻结 M2 代码对既有 full94 G0 + M1-R 输入做 full 双跑并调用其原 verifier。

双跑通过后必须精确复现（容差 `1e-12`）：

| 量 | full94 冻结值 |
|---|---:|
| M1 `MMG_ood_equal`（冻结 gate 只读审计） | 0.20292761623438427 |
| M1 OOD 正环境数（冻结 gate 只读审计） | 6/6 |
| M1-R `matched OOD MMG`（冻结 gate 只读审计） | 0.1919156647423589 |
| M1-R `excess_F_equal` | 0.18336639758796638 |
| M2 `ER_C_equal` | 0.8909627931962164 |
| M2 `ER_C(e)>=0.50` 环境数 | 6/6 |

任一失败即 `FULL94_REPRODUCTION_INVALID_STOP`，不得启动 G0(strict59_ra)。full94 只作管道判否与历史量级
参照；由于它保留旧 DA 方向字段，而新臂同时修复方向并删除 35 列，两者差值不是纯表示消融，不得声称
“只删除无线字段”的因果效应或表示无关性。

## 6. 双跑、资源与执行顺序

固定解释器：`/home/lmy/anaconda3/envs/iotcls/bin/python`。启动前六个大小写代理变量必须全空；全程禁网、
禁代理、不安装依赖、不用 GPU。`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、
`NUMEXPR_NUM_THREADS` 均固定为 1；G0 的模型 `n_jobs` 固定为 4。不与其它计算作业并行。

严格顺序如下，任一步未过门即停：

```text
F0  冻结本协议并记录 SHA；确认三个正式新输出根均不存在
F1  实现 wrapper + 独立测试；另写 implementation freeze，锁协议/runner/tests SHA
F2  full94 完整性审计 + M2 full 双跑复现门
S1  R2--R7 strict59_ra 物化 + 11303 行接口审计
S2A G0(strict59_ra) 正式重复 A：162 x 4
S2B G0(strict59_ra) 临时重复 B：162 x 4
S3A/S3B M1：各 162 次；比较除 provenance 外全部判定性产物
S4A/S4B M1-R：各 156 次；调用原 verifier
S5A/S5B M2：各 156 次；调用原 verifier
S6  只有双跑与全部工程门通过后，读取 strict59_ra 科学量并按 §7 裁定
```

G0 两次须逐字节比较全部 `metrics.json`、`predictions.csv`、`pred_proba.csv`、
`feature_columns.json`、Stacking `oof_meta.csv` 与汇总文件；M2 A/B 分别读取各自 G0 的 model 文件，
从而由 M2 双跑同时约束元层模型语义。任一判定性文件或 M1/M1-R/M2 输出不一致即
`NONDETERMINISTIC_RUN_STOP`，不得挑选某次结果。

## 7. 预冻结科学门与唯一状态树

全部主量先在每个目标环境内聚合，再对六环境等权。

### 7.1 oracle recoverability 门

四项全部满足：

1. matched OOD `MMG_equal >= 0.010`；
2. matched OOD `MMG(e)>0` 至少 4/6 环境；
3. `excess_F_equal = matched OOD MMG - IID time-block MMG >= 0.005`；
4. `excess_F(e)>0` 至少 4/6 环境。

这些是 M1-R 原冻结支持门，不因 full94 结果重设。

### 7.2 实质量级门

`excess_F_equal >= 0.020`。该门只区分可忽略上界与实质上界；必须同时报告六个逐环境值，不得只报均值。

### 7.3 C 结构门

C 必须是 I -> G -> C 顺序中的首个充分阶段：

- `ER_C_equal >= 0.80`，且至少 4/6 环境 `ER_C(e)>=0.50`；
- I 与 G 均不得在相同两条件下先达标。

`ER` 分母为 `excess_F`；不得裁剪比值。若存在门通过但个别环境分母非正，按 M2 原定义完整报告，且该环境
不能计入 `ER>=0.50`。

### 7.4 状态树

按以下顺序唯一裁定：

| 条件 | 状态 | 后续 |
|---|---|---|
| 任一输入、复现、完整性、确定性或工程门失败 | `INVALID_RUN_STOP`（保留更具体失败码） | 不读科学结果 |
| oracle recoverability 门失败 | `EC_MDM_ORACLE_RECOVERABILITY_NOT_SUPPORTED_STRICT59_RA_STOP` | 停止；不进入估计性 |
| recoverability 过、实质量级不过 | `EC_MDM_ORACLE_SIGNAL_BELOW_MATERIALITY_STRICT59_RA_STOP` | 可描述，不作为承重结构；不进入估计性 |
| 前两门过、C 不是首个充分阶段 | `EC_MDM_ORACLE_STRUCTURE_NOT_REPLICATED_STRICT59_RA_STOP` | 报告实际首个阶段；不进入估计性 |
| 三门全过 | `EC_MDM_ORACLE_CANDIDATE_SUPPORTED_STRICT59_RA` | 仅授权另冻 observable estimability 协议 |

通过状态的最强表述固定为：

> 在当前自采 R2--R7、语义修复的 59 维表示和目标标签离线交叉拟合条件下，存在实质的元决策可恢复上界，
> 且类别阈值加类别条件基模型块重加权是首个达到预注册充分性门的参数化；这使 EC-MDM 获得作为 CPD
> 候选上位构念的 oracle 结构证据。

不得把 `candidate`、`oracle` 或“当前自采”删去。

## 8. 后续 observable -> commissioning 决策纪律

本协议不运行 M4、Route A/B/C 或 commissioning，但冻结后续分流，防止把三层证据混称：

1. 只有 §7 三门全过，才可另立 `strict59_ra` observable estimability 协议。该协议必须只用 source 标签与
   target 无标签概率/预测/时序构造估计量；目标标签只能在估计与动作完全冻结、哈希落盘后用于最终评分。
2. 若该无目标标签估计协议通过其运行前门，才可另立部署评估；oracle 结果本身不能触发部署主张。
3. 若 oracle 三门全过而无目标标签估计未过，允许进入**少量目标标签/入网确认 commissioning**，但必须
   另冻预算、采样、时序与方法；主比较必须是使用完全相同目标标签预算的基线，而不是零标签旧模型。
4. commissioning 至少并列：同预算 target-only 元层重拟合、同预算类别阈值校准、同预算全局块权重、
   同预算模型/策略选择，以及不适配 source 模型；只有相对最强同预算基线通过预冻结改善与跨环境一致性门，
   才能声称 commissioning 增益。
5. 若 commissioning 未战胜相同标签预算基线，结论仍只停留在 oracle recoverability/structure；不得把
   “少量标签有效”归因于 EC-MDM 特有结构。

具体标签数、取样方式和改善门不得在本协议中凭空授权；须在打开该阶段任何目标标签结果前由独立协议冻结。

## 9. 输出合同与不可写边界

唯一允许的新正式 roots：

```text
results/g0_environment_grid_strict59_ra/
results/meta_mismatch_exploratory/strict59_ra_ecmdm/
results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902/
```

第三个 root 至少包含：

```text
input_audit.json
pcap_manifest.json
extraction_audit.json
strict59_ra_features_raw_all_w10.csv
full94_reproduction_gate.json
g0_double_run_verification.json
pipeline_double_run_verification.json
oracle_passline.json
per_environment.csv
provenance.json
acceptance.json
VERDICT.md
manifest.json
```

M1/M1-R/M2 的正式 A 产物放第二个 root 的 `m1/`、`m1r/`、`m2/`；重复 B 与 full94 复现只放 `/tmp`
全新目录。任何正式 root 在启动时已存在即拒绝覆盖。失败只在第三个新 root 写 `FAILED.json`；已经产生的
新 staging 不删除、不覆盖，另冻修复协议后换 root。

禁止修改：`docs/`、`code/`、`dataset/`、旧 `results/` 子目录、旧失败目录、M1--M6 既有产物与
本协议。不得执行旧 G0 strict59 协议，不登记或同步主线 registry/cross-line 文档。

## 10. 实现冻结与判定书边界

本协议 SHA-256 必须在任何实现 smoke/full 或科学数值产生前写入独立 freeze record。实现完成后须另写
implementation freeze，至少锁：本协议、wrapper、测试、G0/M1/M1-R/M2 源码、正式输出 roots 与
`numeric_outputs_absent_before_authorization=true`。只有静态审计、纯合成单测和 `--preflight-no-fit`
允许在 implementation freeze 前运行；它们不得读取或拟合正式任务的模型结果。

`VERDICT.md` 只转录 §7 状态、三门原始量、逐环境值、工程门和本节解释边界。任何论文位置、scope 合并、
CPD 身份或主线状态变化均不属于本协议裁定范围。
