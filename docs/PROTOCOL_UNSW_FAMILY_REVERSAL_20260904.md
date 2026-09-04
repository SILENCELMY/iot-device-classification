# 协议：UNSW 池上的族级反转诊断（运行前冻结）

**日期**：2026-09-04
**性质**：诊断性测量，**第一段**。不产生流程判定、不删任何族、不改任何已冻结判定。
**上游**：`docs/PROTOCOL_TWO_CHANNEL_20260903.md` sha256
`dc298198e70957dcd4d1b445900ce85e6ae7bbfac60cac4c8c546515594c0e86`（`coord1_diag` / `pair_auc` 实现来源）；
`docs/PROTOCOL_UNSW_BELKIN_CAPACITY_20260904.md` sha256
`dc58f07725f988c2dca962f3b33e33cd675d84563f6380d097877300b80db630`（H/T 切分与门槛口径来源）；
`docs/METHOD_SPEC_20260904.md` §1.1（特征池为声明的输入）。

---

## 1. 唯一要回答的问题

`docs/METHOD_SPEC_20260904.md` §1.1 已定：**特征池是声明的输入，方法声称的是
「在任何位置所能提供的池上都适用的配置推导程序」。** 该主张目前 **n=1** ——
坐标 1 的族级反转诊断**只在自采（位置 A 的池，含 `rssi`）上跑过**。

UNSW 侧从未跑过：`code/scripts/analysis/unsw_two_channel/check_unsw_families.py`
文件头自陈 *"Produces no judgment-bearing numbers -- only column lists and group sizes"*，
只做了结构检查（`len`14 `interarrival`12 `subwin`9 `burst`7 `up`7 `down`6 `singletons`6 = 61）。

**本协议只答一句**：**UNSW 的 61 列池里，有没有「强且反转」的族？**

**它不答的**：该池能不能把 Belkin 对分开（`UNSW-BELKIN-CAPACITY` 已答：
`PARTIALLY_SEPARABLE`，二类 0.6504、最强单列强度 0.1168）。
**容量与稳定性是两个独立问题**：同一 61 列对易类信息充足（设备级 0.999），
对 Belkin 几乎为零 —— 故「信息不够」不蕴含「没有反转族」。

## 2. 数据与切分（运行前定死）

- 缓存：`results/unsw_features_full/features_day_*.csv`（20 天），**复用不重抽**。
- **只读 H = 前 14 天中的前 10 天（`H_inner`）**，按文件名日期升序。
  **第 11–14 天（`H_outer`）本协议内不被读取；T（后 6 天）同样不被读取。**
  两者均以路径白名单硬性拦截，违反即中止。
  理由：`H_outer` 保留给第二段（oracle 距离）作打分用，本段若读了它，第二段就不再是「从未参与决策」。
- 设备集：取 `H_inner` 十天**全部**通过 `≥100 窗/天` 门槛的设备**交集**
  （实测 14 台：`AmazonEcho` `BelkinWemoMotion` `BelkinWemoSwitch` `Dropcam` `HPPrinter`
  `NetatmoWeather` `NetatmoWelcome` `PIX-STAR` `SamsungSmartCam` `SmartThings`
  `TPLinkCam` `TPLinkPlug` `TribySpeaker` `WithingsBabyMonitor`）→ **91 个无序类对**。
- 平衡口径逐字对齐 `UNSW-IID-REF`：`sample_balanced(max_rows=20000)`。
- 族划分：`derive_families` 机械导出（`c.split("_")[0]`，规模 <2 并入 `singletons`），**7 族**。

## 3. 任务集（运行前定死）

`H_inner` 内留一天出（LORO），共 **10 个任务**：第 k 个任务源域 = 除第 k 天外的 9 天，目标 = 第 k 天。
**所有任务的目标都在 `H_inner` 内**，故其标签对流程合法可见（与自采 9 任务同构）。

## 4. 观测量（复用冻结实现，不重实现）

逐 (任务, 无序类对, 族) 计算 `pair_auc`（`run_two_channel.py:198`，源域拟合标准化逻辑判别、
目标域 AUC，`max_iter=2000, C=1.0`；样本门槛 40/20 逐字沿用），再取
`d_auc = auc(剔除族 f) − auc(全部 61 列)`，在**全部类对上求和**得逐族 `d_auc_sum`
（即 `coord1_diag` 的定义，`run_two_channel.py:214`）。

**单种子 42**：`pair_auc` 用 `LogisticRegression(lbfgs)`，对给定数据确定；
冻结实现中 `coord1_diag` 亦只调用一次、不逐种子重复。

## 5. 主判据（运行前定死，门槛不因结果调整）

主量 = 逐族 `d_auc_sum`（10 任务求和）。记最大者为 `top1`、次大者为 `top2`。

| 分支 | 条件 | 处置 |
|---|---|---|
| `REVERSING_FAMILY_FOUND` | 存在族 `d_auc_sum > 0` **且** `top1 / max(top2, ε) ≥ 3.0`（`ε=0.05`） | 该池含强反转族 → **坐标 1 的动作在位置 B 的池上有候选** → 进入第二段（oracle 距离），并作干预外部复现的前置 |
| `WEAK_CANDIDATES_ONLY` | 存在族 `> 0` 但 `top1/top2 < 3.0` | 候选存在但不占优 → **正是规格 §5 限制 6 所警告的「接近候选」情形**；不进第二段，如实记「该池上选择风险高，需先修选择规则」 |
| `NO_REVERSING_FAMILY` | 无任何族 `d_auc_sum > 0` | 该池无反转族 → 坐标 1 自动不开火（`cand` 为空，`run_two_channel.py:321` 的行为，非外加分支）→ **预注册的「不开火」在族级独立确认**；不进第二段 |

**门槛依据（选定于任何本协议数字之前）**：
- **3.0 倍**：自采上 `rssi` 对次名的比是 **22.8×**（`HISTINT-OBJ-SIGNAL` `max_base` 口径
  `S_obj` 0.4140 vs `singletons` 0.0182）。取 3.0 是「明显占优」的宽松下界 ——
  远低于自采实测值，故不是为迁就任何预期而设；同时高于 `HIER-CLASSPAIR` 实测失败时的
  候选比（前四名相差 0.005，比值 ≈1.006），故能把该失败形态挡在 `WEAK_CANDIDATES_ONLY` 内。
- **ε=0.05**：防 `top2 ≤ 0` 时比值发散；取值与自采次名量级（0.018）同数量级。

**运行前预期（只作记录，不改判据）**：作者预期落 `NO_REVERSING_FAMILY`。
理由：自采上这 7 个共有族全部很弱（最高 `singletons` 0.0182，`len` 为负 −0.1610），
而 `rssi` 是 0.4140；且 `UNSW-META-MISMATCH` 四格分解在 UNSW 上净反转基本为零
（破坏 633 vs 救回 656、换错法 23 vs 自采 594）。**但该证据是元层聚合层面的间接证据，
从未在族级测过 —— 故本协议的结果不由该预期决定。**

## 6. 并报（无论方向，不得省略）

1. 逐族 `d_auc_sum`（7 族全列）与 `top1/top2` 比值；
2. 逐 (任务, 类对, 族) 的 `d_auc` 明细落盘（与自采 `diag_coord1.csv` 同形）；
3. **逐类对聚合**的 `d_auc`：哪些类对上哪一族反转最强 —— 用于检验反转是否如自采一样是**类对级**的；
4. 因样本门槛（40/20）被跳过的 (任务, 类对) 计数；
5. 全 61 列基线 AUC 的逐类对分布（`auc_base`），用于区分「族有害」与「该类对本来就不可分」；
6. 与自采逐族 `d_auc_sum` 的并列表，**明写这是跨池比较、不构成受控对比**。

## 7. 硬门

1. **只读 `H_inner`（前 10 天）**；`H_outer`（第 11–14 天）与 T（后 6 天）路径白名单硬性拦截。
2. **`window_start_epoch` 与 8 个元数据列不入特征**，逐项核对并**先于任何 AUC 数字落盘**。
3. **缓存 md5 先于任何 AUC 数字落盘**（`UNSW-BELKIN-CAPACITY` 首跑因漏此项作废，本协议照此执行）。
4. 复用冻结实现：`pair_auc` / `coord1_diag` 的定义与参数逐字沿用，不重实现；
   族划分、门槛、平衡口径分别沿用 `derive_families` / `UNSW-IID-REF`。
5. **双跑 md5 逐字节一致**（除 `provenance.json` 的 `_volatile` 段）；
   线程 `OMP=MKL=OPENBLAS=1`、`n_jobs=1`；**双跑之间不得提交任何 commit**。
6. 解释器 `~/anaconda3/envs/iotcls/bin/python`；全程禁网禁代理、不安装依赖。

## 8. 产物与成本

`results/unsw_family_reversal_20260904/`：`split.json`（先落盘：白名单、缓存 md5、元数据排除核对）、
`family_dauc.csv`、`pair_family_dauc.csv`、`pair_baseline_auc.csv`、`skipped_pairs.csv`、
`passline.json`、`provenance.json`、`VERDICT.md`。

**预估成本（实测折算）**：`pair_auc` 单次 **0.023 s**（实测）；
每任务 91 类对 × (1 基线 + 7 族) = 728 次 → **约 17 s/任务**；
10 任务 → **约 3 min/单跑、6 min/双跑**（另加 10 个日文件读取）。

## 9. 第二段的前置条件（本协议内预注册，不在本协议内执行）

仅当落 `REVERSING_FAMILY_FOUND` 时，另立协议执行第二段：在 `H_outer`（第 11–14 天，
本协议未读取）上测「程序距该池 oracle 的距离」，与自采的 0.0045 并列。
**第二段成本已实测折算**：四模型一次 280 s（rf 10.0 / xgboost 16.5 / lightgbm 20.1 / stacking 233.4），
含 stacking 约 4–5 h 单跑、不含约 50 min 单跑 —— **该取舍在第二段协议内定，不在本协议内定。**

## 10. 本协议不做什么

不删任何族；不评估任何配置；不做 oracle 距离；不碰 `H_outer` 与 T；
不改动 `TWO-CHANNEL-SELF`、`UNSW-META-MISMATCH`、`UNSW-BELKIN-CAPACITY`、
`PIPELINE-DROPRSSI-E2E`、`HIER-CLASSPAIR` 的任何判定。
