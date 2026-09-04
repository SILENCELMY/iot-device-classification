# 协议：交集特征扩池（自采侧第一段，运行前冻结）

**日期**：2026-09-04
**性质**：表示容量扩充 + 诊断。**两个嵌套配置，主判据在未筛选的那个上。只测不选** —— 不让流程在新池上做任何删除决策。
**不改动任何已冻结协议、不重开任何已冻结判定、不改动 `full94` 一列。**
**上游**：`docs/METHOD_SPEC_20260904.md` §1.1（可采性门）、§1.2（位置分类学与「先扩交集」规则）、
§5 限制 6（候选接近时的选择行为已测且失败）；
`docs/PROTOCOL_TWO_CHANNEL_20260903.md` sha256
`dc298198e70957dcd4d1b445900ce85e6ae7bbfac60cac4c8c546515594c0e86`（`pair_auc` / `coord1_diag` /
`accept_family` / `joint_eval` 实现来源）。

---

## 1. 为什么现在做，以及为什么只测不选

**已测的三条把靶收窄到「未被计算过的交集特征」**：

1. `UNSW-FAMILY-REVERSAL`：UNSW 池 7 族 `d_auc_sum` **全部为负、正值族 0/7** →
   该池无反转族，干预效应量的外部复现在 UNSW 上已确定不可能。
2. 自采侧两池共有的那 7 族**也全都很弱**（`diag_coord1.csv`：`subwin` +0.0156 为唯一正值，
   且只在 4/9 个任务上为正；其余六族总和为负）。**唯一强反转族 `rssi` 落在空口独有那一块。**
3. `UNSW-BELKIN-CAPACITY`：Belkin 对 61 列最强单列强度仅 **0.1168**，且**方向族
   （`down` 0.1168 / `up` 0.1123）比其余五族强 3–5 倍、是唯一超过 0.10 的**。

**故扩池方向由数据指定，不由想象指定**：沿方向这一维往深处抽，并补上从未计算过的
时间结构与序列结构。**三者只需 (时间戳, 包长, 方向)，全在交集内**（§1.2 规则：
交集是新特征能否被外部验证的唯一区域）。

**为什么只测不选**：`HIER-CLASSPAIR` 实测选择规则在 9 个接近候选间选错
（inner 前四名相差 0.005，outer 上 Camera 代价 0.42）。**扩池会把候选数放大，
故本协议不产生任何删除动作**；删不删仍由冻结的 `accept_family`（多条件）在
选择规则修好后另立协议决定。

**已披露：作者已知现有 94 列的构成与未覆盖清单**（见 §1.2），本协议的候选清单据此拟定。
**候选清单在本协议冻结时定死，运行中不得增删** —— 否则「抽 200 个报 3 个」不可证伪。

## 2. 候选清单（运行前定死，逐列公式）

新增 **89 列 / 5 族**（其中 33 列构成子集 `N33`，见 §2.4）（族名由 `derive_families` 的 `c.split("_")[0]` 机械导出）。
窗口定义、`min_packets_per_window`、轮次集合一律沿用冻结提取器。

### N1 `dir_*`（14 列）—— 方向的精细刻画

**方向判定采用修正规则 `wlan.ra == BSSID`（上行），而非冻结提取器的 `DA == BSSID`。**
【偏离披露】冻结提取器第 357 行自陈 `tshark 3.2.3 does not expose ToDS/FromDS, so we use TA/DA`；
该规则已知使非 Socket 四类的 `up_packet_ratio` 在 **100%** 窗口为 0
（`independent/air_interface_representation/` 独立线已量化，`RA==BSSID` 为 15.7%–43.6%）。
**本协议不改动 `full94` 中原有的 `up_*`/`down_*`（保持逐字节不变）**，
新族与之并存 —— 由此 `dir` 族与 `up`/`down` 族的诊断值对比，**直接量出方向 bug 的代价**。

| 列 | 定义 |
|---|---|
| `dir_up_ia_p50` / `dir_up_ia_p90` | 相邻上行包时间差的 50/90 分位（现有仅 mean/std） |
| `dir_down_ia_p50` / `dir_down_ia_p90` | 相邻下行包时间差的 50/90 分位 |
| `dir_resp_lat_mean` / `dir_resp_lat_p50` / `dir_resp_lat_p90` | 每个上行包到**下一个**下行包的时延，取均值/50/90 分位（无后继下行者跳过） |
| `dir_alt_count` | 方向序列中方向改变次数 ÷ 包数 |
| `dir_up_run_mean` / `dir_up_run_max` | 连续上行包 run-length 的均值 / 最大值 |
| `dir_down_run_mean` / `dir_down_run_max` | 连续下行包 run-length 的均值 / 最大值 |
| `dir_up_len_p90` / `dir_down_len_p90` | 逐方向包长的 90 分位（现有仅 mean/std/p50） |

### N2 `time_*`（10 列）—— 时间结构（只需时间戳）

| 列 | 定义 |
|---|---|
| `time_ac_lag1` / `time_ac_lag2` / `time_ac_lag5` | 到达间隔序列在 lag 1/2/5 的自相关（Pearson，样本不足返回 0） |
| `time_ac_absmax` | lag 1..10 上 \|自相关\| 的最大值 |
| `time_fft_peak_ratio` | 把窗口按 0.1 s 分 100 桶得到到达计数序列，实 FFT 后**主峰功率 ÷ 总功率**（剔除 DC） |
| `time_fft_peak_freq` | 上述主峰对应频率（Hz） |
| `time_ia_n_modes` | `log10(间隔)` 直方图（固定 20 桶，范围 [-6, 2]）中局部极大值个数 |
| `time_ia_mode_gap` | 前两个峰桶中心之差（不足 2 峰记 0） |
| `time_ia_entropy` | 上述直方图的 Shannon 熵 ÷ log(20)（归一化到 [0,1]） |
| `time_burstiness` | `(σ−μ)/(σ+μ)`，间隔序列的 Goh–Barabási 突发度（μ=0 记 0） |

### N3 `seq_*`（9 列）—— 序列结构（需包长与方向）

包长量化：16 个对数等距桶，范围 `[min(length), max(length)]` 用**全局固定边界**
`[0, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1518, 2048, 4096, 8192, ∞]`（15 个边界 → 16 桶），
**不随窗口自适应**（否则同一序列在不同窗口的编码不同，破坏可比性）。

| 列 | 定义 |
|---|---|
| `seq_len_entropy` | 量化包长直方图的 Shannon 熵 ÷ log(16) |
| `seq_len_bigram_entropy` | 量化包长 bigram 分布的熵 ÷ log(256) |
| `seq_len_bigram_top1` | 最高频包长 bigram 的占比 |
| `seq_len_distinct_ratio` | 出现过的量化桶数 ÷ 包数 |
| `seq_dir_entropy` | 方向 bigram（uu/ud/du/dd）分布的熵 ÷ log(4) |
| `seq_dir_top1` | 最高频方向 bigram 的占比 |
| `seq_dir_trans_ud` / `seq_dir_trans_du` | 上→下 / 下→上 转移在全部 bigram 中的占比 |
| `seq_lz_ratio` | 量化包长字符串经 `zlib.compress(level=6)` 后长度 ÷ 原始长度（复杂度代理） |

### N4 `timeup_*` / `timedn_*`（各 10 列，共 20 列）—— 时间结构的逐方向版本

现有 `time_*` 把上下行混在一起算。本族对**只取上行包**与**只取下行包**的到达序列，
各自重算 N2 的十个量（列名与 N2 逐一对应，前缀改为 `timeup_` / `timedn_`）。
族名由 `split("_")[0]` 导出为 `timeup` / `timedn`，**故为两个独立可诊断的族**。
若该方向包数 < 3，该方向的十列一律记 0（并计入 §5.6 的恒零比例）。

### N5 补全项（36 列）—— 同族内的参数补全

| 列 | 定义 | 归属族 |
|---|---|---|
| `time_ac_lag3` `time_ac_lag4` `time_ac_lag6` … `time_ac_lag10` | 自相关补全至 lag 1..10（7 列） | `time` |
| `time_fft_peak2_ratio` `time_fft_peak3_ratio` | FFT 第 2、3 主峰功率占比（2 列） | `time` |
| `dir_up_ia_p10/p25/p75`、`dir_down_ia_p10/p25/p75` | 逐方向间隔分位补全（6 列） | `dir` |
| `dir_up_len_p10/p25/p75`、`dir_down_len_p10/p25/p75` | 逐方向包长分位补全（6 列） | `dir` |
| `dir_up_burst_count` `dir_up_burst_size_mean` `dir_up_burst_size_max` | 上行突发（定义沿用冻结 `burst_*` 的间隔阈值）（3 列） | `dir` |
| `dir_down_burst_count` `dir_down_burst_size_mean` `dir_down_burst_size_max` | 下行突发（同上）（3 列） | `dir` |
| `seq_len8_entropy` `seq_len8_bigram_entropy` `seq_len8_bigram_top1` `seq_len8_distinct_ratio` | 包长量化改 **8 桶**（取 §2 N3 固定边界的每隔一个）后重算 4 列 | `seq` |
| `seq_len32_entropy` `seq_len32_bigram_entropy` `seq_len32_bigram_top1` `seq_len32_distinct_ratio` | 包长量化改 **32 桶**（在 §2 N3 固定边界间各插一个几何中点）后重算 4 列 | `seq` |
| `seq_len_trigram_entropy` | 16 桶量化包长三元组分布的熵 ÷ log(4096) | `seq` |

合计 7+2+6+6+3+3+4+4+1 = **36 列**。加 N1 14 + N2 10 + N3 9 + N4 20 = **89 列 / 5 族**
（`dir` `time` `seq` `timeup` `timedn`）。

### 2.4 子集 `N33` 的定义（运行前定死）

`N33` = **N1 的 14 列 + N2 的 10 列 + N3 的 9 列**，即本协议 2026-09-04 早先冻结的那份
按证据筛选的靶单。**`N33 ⊂ N89`，两配置嵌套。**

**为什么 N3 是最直接的可证伪假设**：`BelkinWemoMotion` 与 `BelkinWemoSwitch` 跑同一套固件、
同一个云，故其**边缘统计**（包长分布、间隔分布 —— 即现有 94/61 列全部在测的东西）必然相似；
可能不同的是**对话形状**（谁先说、几轮、间隔多久），而那正是 N3 抓的。

## 3. 配置与口径

| 配置 | 列数 | 族数 | 说明 |
|---|---:|---:|---|
| `full94` | 94 | 10 | **逐字节不变**，复用 `results/robust_v2/raw_all/features_raw_all_w10.csv`（md5 `703984b6ad2fde2f45e0cce1c6df31be`）；本协议**在同一 runner 下重跑一次**以保证与新配置同估计量 |
| `full94+N33` | 127 | 13 | `full94` 左连接 `N33`（§2.4） |
| `full94+N89` | **183** | **15** | `full94` 左连接全部 89 新列 —— **主判据在此配置上** |

新表另存，**不覆盖任何既有缓存**。轮次 R2–R7 全抽（成本低，为后续 outer 单元备好）。
模型 `rf` / `xgboost` / `lightgbm` / `stacking`，种子 **42–46**。

## 4. 主判据（运行前定死，门槛不因结果调整）

**主判据只在一个配置上，运行前定死为 `full94+N89`**（未经作者筛选的那一个）。
理由：手工筛选是一个自由度；把**未筛选**的配置定为主判据，即把该自由度从判定中移除。
`full94+N33` 为并报项，**其数值在任何情况下都不得替代主判据** —— 否则「两个配置对同一门槛开两枪、
报过的那个」正是本协议要防的选择性报告。两配置嵌套（`N33 ⊂ N89`）故 Δ 高度相关，
多重比较的严重性低于两个独立检验，但**不因此放宽上述规定**。

主量 = `Δceiling = joint_tb(full94+N89) − joint_tb(full94)`，
`joint_tb` 沿用 `run_histint.joint_eval`（H = R2+R3+R4，前 4 时间块训练、第 5 块测试），
取 `best_base`（三非 stacking 模型逐种子取 max），五种子均值。
**`joint_tb(full94)` 在本协议内重跑取得**（同 runner、同估计量）；
外部参照值 0.9084（`r567_94dim_retest_20260904/deltas.csv`，sd 0.0011），
若本协议重跑值与之相差 > 0.005 须在偏离段说明。

| 分支 | 条件 | 处置 |
|---|---|---|
| `EXPANSION_RAISES_CEILING` | `Δceiling ≥ +0.02` | 新族含现有 94 列没有的信息 → 表示容量确实可扩 → 移植到 UNSW（另立协议）测 Belkin |
| `EXPANSION_MARGINAL` | `0 ≤ Δceiling < +0.02` | 新族信息量在噪声量级 → 如实报，不移植；并报逐族诊断以定位是否某一族单独有用 |
| `EXPANSION_USELESS` | `Δceiling < 0` | 新族纯噪声（更多列反而稀释）→ 如实报；「信息在未计算的交集特征里」这一假设被否 |

**门槛 +0.02 的依据（定于任何本协议数字之前）**：`joint_tb` 五种子 sd = 0.0011，
故 0.02 ≈ **18 sd**，远超噪声；且表示代价的可用余量为 0.092（0.9084 → 1.0），
0.02 占其 **22%** —— 既有意义又非不可达。

**并报的第三个量（无门槛）**：`Δ(N89 − N33) = joint_tb(full94+N89) − joint_tb(full94+N33)`。
**它测的是作者的证据筛选帮了还是害了**：为正说明筛选丢掉了信息，为负说明多余 56 列在稀释。
该量**不参与分支判定**。

## 5. 并报（无论方向，不得省略）

1. **逐族 `d_auc_sum`（13 族全列）**：在 `full94+N` 上跑 `coord1_diag` 的定义
   （9 个 g0 跨轮次任务、全部类对求和），**含 `dir` / `time` / `seq` 三个新族与原 10 族**；
2. **逐任务符号一致性（闸门 A，并报项、无否决权）**：每族 `d_auc` 在 9 个任务中为正的个数。
   校准依据（来自本协议冻结前已存在的冻结产物）：`rssi` **8/9**、`subwin` **4/9**；
3. **占优度**：`top1 / max(top2, 0.05)`；参照 `rssi/subwin = 336×`；
4. **`dir` 族 vs 原 `up`/`down` 族的诊断值对比** → 方向 bug 的代价；
5. `joint_tb` 逐类 F1（Camera / Light_T1 / Light_XM / Sensor / Socket）：
   现有 `full94` 域内逐类为 0.980 / 0.807 / 0.778 / 0.818 / 0.999，
   **天花板损失集中在困难簇三类** —— 新族是否打在那里；
6. 逐新列的恒零窗口比例（防再出现 `side_packet_ratio` 那种构造性死列）；
7. 新表与 `full94` 的连接完整性：未匹配行数必须为 0。

## 6. 硬门

1. **`full94` 缓存只读**，md5 前后核对一致；新列另存新文件，不覆盖。
2. **候选清单锁定**：runner 中新列名与 §2 逐字一致，`N89` 数量恰为 **89**、`N33` 恰为 **33** 且为 `N89` 的子集，均以 `assert` 保证。
3. **只测不选**：本协议不调用 `accept_family`、不产生任何 `removed` 列表、不改任何配置选择。
4. 复用冻结实现：`run_tshark` / 窗口切分 / `pair_auc` / `coord1_diag` 定义 / `joint_eval`，一律不重实现。
5. 线程 `OMP=MKL=OPENBLAS=1`、`n_jobs=1`；**双跑 md5 逐字节一致**（除 `_volatile`）；
   **双跑之间不得提交任何 commit**。
6. 解释器 `~/anaconda3/envs/iotcls/bin/python`；禁网禁代理、不安装依赖。
7. 特征提取的 tshark 版本与命令行记入 `provenance.json`（冻结提取器用 tshark 3.2.3）。

## 7. 产物与成本

`results/feature_expansion_20260904/`：`features_new33_w10.csv`（新 33 列 + 连接键）、
`join_audit.json`（连接完整性、恒零比例）、`family_dauc.csv`、`pair_family_dauc.csv`、
`ceiling.csv`（`joint_tb` 逐配置逐模型逐种子 + 逐类 F1）、`passline.json`、`provenance.json`、`VERDICT.md`。

**预估成本**：自采 pcap 共 **169 MB**（camera 36M / light_T1 27M / light_xm 29M / sensor 32M / socket 45M）；
tshark 重解 + 89 新列计算约 **25–30 min**（89 列抽一次，`N33` 是其子集，**不重复抽**）；
`coord1_diag` 在 `N89`（15 族）与 `N33`（13 族）上各一次，共约 **4 min**
（`pair_auc` 单次实测 0.023 s）；`joint_tb` **3 个配置** × 4 模型 × 5 种子约 **37 min**。
合计约 **70 min/单跑、140 min/双跑**。

## 8. 本协议不做什么

不删任何族、不做配置选择、不碰 UNSW/CIC、不改 `full94`、不动窗长（10 s 不变）、
不抽空口独有或网关独有的特征（只抽交集）；
不改动 `TWO-CHANNEL-SELF`、`HISTINT-OBJ-SIGNAL`、`R567-94DIM-RETEST`、
`PIPELINE-DROPRSSI-E2E`、`UNSW-BELKIN-CAPACITY`、`HIER-CLASSPAIR`、
`UNSW-FAMILY-REVERSAL` 的任何判定。
