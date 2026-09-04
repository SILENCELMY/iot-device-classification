# 协议：流水线在 `drop_rssi` 口径上的端到端重做与源域侧决策检验（运行前冻结）

**日期**：2026-09-04
**性质**：口径统一 + 决策可判定性检验。**不改动任何已冻结协议，不重开任何已冻结判定。**
**上游**：`docs/PROTOCOL_TWO_CHANNEL_20260903.md` sha256
`dc298198e70957dcd4d1b445900ce85e6ae7bbfac60cac4c8c546515594c0e86`（判定 `PROCEDURE_BEATS_PRACTICE`）；
`docs/PROTOCOL_R567_94DIM_RETEST_20260904.md` sha256
`6ab54aea29288e7ec157b75f8d7fb6c450b424c2de659008653e3f3ae15db64c`（判定 `REGIME_DEPENDENT`，本协议复用其 `eval_unit`）。

---

## 1. 为什么必须做这一个

2026-09-04 的探索性测量给出了一条完整流水线的端到端数字（R5/R6/R7，设备级 5 分钟
0.947–0.957，标准做法 0.8014），**但那一臂用的是 `strict59_ra`（删 35 列 + 方向修复），
不是流程登记的坐标 1 动作 `drop_rssi`（删 8 列）**。`strict59_ra` 列清单属
`independent/air_interface_representation/` 独立线，**无登记行**，跨线引用需另立协议。

同时，那一臂里「是否堆叠」的选择是**看过目标之后做的**。已用冻结 `metrics.json` 事后核对，
源域侧内层 LORO 证据在两个特征集上都给出了与目标侧一致的方向（`full94` 内层 +0.1741 → 拒绝堆叠，
外层真实 +0.0529 窗口级 / −0.1181 设备级；`strict59` 内层 −0.0012 → 不拒绝，外层 −0.0000 / +0.0104）。
但那是**两个网格的既有产物**，两者由不同 runner、不同时间产生，且 `drop_rssi` 上从未测过。

**本协议同时闭合这两个洞**：在 `drop_rssi` 口径上、用同一 runner、同一实现，把①→⑤跑一遍，
并预注册检验「源域侧决策是否与目标侧最优一致」。

## 2. 任务与配置（运行前定死）

| 单元 | 类型 | 源域 | 目标 | 用途 |
|---|---|---|---|---|
| `inner_to_R2` | inner | R3+R4 | R2 | **源域侧证据**：目标轮次在源域内，标签对流程合法可见 |
| `inner_to_R3` | inner | R2+R4 | R3 | 同上 |
| `inner_to_R4` | inner | R2+R3 | R4 | 同上 |
| `pos_R5` | outer | R2+R3+R4 | R5 | **主判据单元**：目标从未参与任何决策 |
| `jit_R6` | outer | R2+R3+R4 | R6 | 主判据单元 |
| `jit_R7` | outer | R2+R3+R4 | R7 | 主判据单元 |

配置两个：`full94`（94 列）与 `drop_rssi`（86 列，`TC.derive_families` 导出的 `rssi` 族 8 列）。
模型四个：`rf` / `xgboost` / `lightgbm` / `stacking`。种子 **42–46**，报均值与样本标准差。

`strict59_ra` **不进本协议**（独立线，无登记行）；其既有数字只作**只读对照**列出，不参与任何判据。

## 3. 主判据（运行前定死，门槛不因结果调整）

`best_base` = 逐单元逐种子取三个非 stacking 模型 macro-F1（窗口级）的最大值。
`g = best_base − stacking`（窗口级 macro-F1，五种子均值）。

- **源域侧决策**（每个配置一个，只读 inner 三单元）：
  `g_in` = 三个 inner 单元 `g` 的均值。`g_in > 0` → `REFUSE_STACKING`，否则 `ALLOW_STACKING`。
- **目标侧真相**（每个配置 × 每个 outer 单元）：`g_out > 0` → `REFUSE`，否则 `ALLOW`。
- **一致格** = 决策与真相同向。共 2 配置 × 3 outer = **6 格**。

| 分支 | 条件 | 处置 |
|---|---|---|
| `SOURCE_SIGNAL_AGREES` | 一致 ≥ 5/6 | 步骤③可由源域独立判定 → 流水线终点是**流程输出**而非事后选择，可如此陈述 |
| `SOURCE_SIGNAL_PARTIAL` | 一致 3–4/6 | 步骤③只在部分条件下可判定，须逐格如实报，终点陈述必须降级为「配置上界」 |
| `SOURCE_SIGNAL_FAILS` | 一致 ≤ 2/6 | 源域侧无法定步骤③ → 流水线缺一环，端到端数字只能作上界报，不得称流程输出 |

**打平带**：`|g_out| < 0.01` 的格记为 `TIE`，**仍计入上述 6 格分母**（严格口径），
同时**并报**剔除 TIE 后的一致率作为参考。分支由严格口径决定。

## 4. 并报（无论方向，不得省略）

- 逐单元、逐配置、逐模型、逐种子窗口级 macro-F1（均值 ± 样本标准差）；
- **准确率-延迟曲线**：N=1（10 s）/ N=6（1 min）/ N=30（5 min）设备级 macro-F1，逐单元逐配置逐模型；
- **端到端终点**：配置 `drop_rssi`、模型由 §3 源域侧规则选出，在三个延迟档上的值；
  与标准做法（`full94` + 源域 CV 选出的模型）并列；
- 逐设备众数裕度（真类份额 − 次名份额）与众数错设备数；
- 逐类 F1；
- `strict59_ra` 既有数字**只读对照**（不重跑、不参与判据）。

## 5. 硬门

1. 复用冻结实现：`import run_two_channel`（`Data` / `derive_families` / `make_model` /
   `time_blocks`）；`eval_unit` **逐字复用 `results/r567_94dim_retest_20260904/run_r567.py`**，
   仅扩展为同时返回逐窗口预测；R567 的 `GATE7`（`eval_unit` 与 `TC.fit_eval` 逐位一致，容差 1e-12）
   **保留并必须通过**，否则 `INVALID_RUN_STOP`。
2. **新增代码只有两处，且必须声明**：(a) 逐窗口预测转储；(b) 块投票（定义与
   `results/g0_environment_grid/diag_window_aggregation.py` 完全一致：按 `true_label` 分组、
   按 `window_start` 排序、连续 N 窗、不足 `max(1, N//2)` 的尾块丢弃）。
3. 种子切换经 `run_two_channel.SEED` 重绑定，不改冻结文件。
4. 线程钉死 `OMP=MKL=OPENBLAS=1`，模型 `n_jobs=1`。
5. **双跑 md5 逐字节一致**（除 `provenance.json` 的 `_volatile` 段）。
   **双跑之间不得产生任何 commit**（`git_head` 必须一致——`HISTINT-OBJ-SIGNAL` 的教训）。
6. 解释器 `~/anaconda3/envs/iotcls/bin/python`；禁网禁代理。
7. **决策隔离断言**：计算源域侧决策的函数只接收 inner 三单元的结果，签名上不出现 outer 单元；
   代码中以 `assert` 保证。R5/R6/R7 只用于打分。

## 6. UNSW 外部臂（并报，不设门槛，零新拟合）

用冻结产物 `results/unsw_meta_mismatch_20260902/subset_f1.csv`（含 `srcOOF_H` 与 `target_H`
逐模型）计算同样的「源域侧 vs 目标侧」方向一致性，3 个任务。**结构与自采不同**
（源域侧是 OOF，不是内层 LORO），故只作旁证，不进 §3 计数。同时并报 UNSW 设备级曲线
（已有产物 `aggregation_audit_devices_posthoc.csv`）。

## 7. 产物与成本

`results/pipeline_droprssi_20260904/`：`per_unit.csv`（逐单元逐配置逐模型逐种子，窗口级 + 三档设备级）、
`decision.json`（`g_in`、决策、逐格一致性、分支）、`latency_curve.csv`、`device_margin.csv`、
`unsw_arm.json`、`passline.json`、`provenance.json`、`VERDICT.md`。

**预估成本**：2 配置 × 6 单元 × 4 模型 × 5 种子 = **240 次拟合**。按 R567 实测
（320 次 / 3484 s ≈ 10.9 s/次）折算约 **44 min/单跑、88 min/双跑**。

## 8. 本协议不做什么

- 不测 `strict59_ra`（独立线）；不改任何特征定义；不重抽 pcap；
- 不做类对级选择性使用 `rssi`（另立协议）；不做窗长消融（另立协议）；
- 不改动 `TWO-CHANNEL-SELF` 与 `R567-94DIM-RETEST` 的任何判定。
