# UNSW 元层失配机制外部检验协议（运行前冻结）

**冻结日期**：2026-09-02
**状态**：`PROTOCOL_FROZEN_BEFORE_NUMERIC_RUN`
**性质**：把自采上已量化的元层失配机制拿到 UNSW 上做外部检验。设计思路与自采逐项对应，
不修改任何既有冻结判定，不恢复 CPD 的机制/预测/部署身份，不改 D12 判定，
不把本协议结果与 E1 的 −0.07~−0.22 并列为「同一机制跨数据集复现」（D12 禁令仍然生效）。
**裁定方**：主线侧。执行方只实现与运行。

## 0. 触发事实（全部产自本协议之前）

自采侧的机制已被量到，四个环节都有落盘数字（`results/g0_environment_grid/`，
分组 OOF、seed 42）：

| 环节 | 实测 |
|---|---|
| 失效定位 | 困难簇 3 类上 stacking 相对最佳基模型 **−0.3740 / −0.1892 / −0.3002**；易类 2 类 **−0.0037 / −0.0266 / +0.0131** |
| 簇内源域 OOF 难度 | RF macro-F1 **0.2838 / 0.5148 / 0.2433**（自类信号近乎无用） |
| 簇内目标域难度 | RF **0.4577 / 0.3692 / 0.7008**（与 OOF regime 不一致） |
| 系数签名 | 簇内自类项接近零或为负（`xgb.Sensor→Sensor −2.35`），最大项为跨类借用（`rf.Light_T1→Sensor +3.07`）；易类为 RF 直通（`rf.Camera→Camera +9.28`） |

UNSW 侧已确认存在可混淆簇，且证据全部来自 **RF-only 产物**（`results/unsw_pilot/loro/`
与 `results/unsw_iid_reference_20260902/`），未涉及任何 UNSW stacking 拟合：
`BelkinWemoMotion` 在 `16-09-30→16-10-12` 上 recall **610/1538 = 0.397**、719 判成
`BelkinWemoSwitch`；逐设备 IID−LORO 差 `BelkinWemoSwitch` **+0.1143**、`LiFXBulb` **+0.1506**、
`WithingsAura` **+0.1507**，而 `Dropcam` **+0.0005**。

## 1. 唯一问题

在 UNSW 上，用与自采逐项同构的构造，元层失配是否呈现与自采相同的四条签名；
其簇内损失幅度是否与簇内源域 OOF 难度同向。

本实验不改窗口、不加特征、不调参、不做特征选择、不计算 CPD、不实现任何适配器。

## 2. 冻结构造（与自采逐项对应）

| 维度 | 自采 | UNSW |
|---|---|---|
| 标签 | 5 个设备身份 | **设备身份**（10–14 类，非 `category`） |
| 源 | 2 个轮次 | **2 个采集日** |
| 目标 | 1 个未见轮次 | 1 个未见采集日 |
| Stacking OOF | 按轮次分组，`n_splits` 退化为 **2** | 按**日**分组，`n_splits` 退化为 **2** |
| 模型 | rf / xgboost / lightgbm / stacking | 同 |
| 元学习器 | `LogisticRegression(max_iter=2000, class_weight=balanced)`，cv=5 | 同（主线 `build_model("stacking")`，不改） |

三个任务，与自采三个 LORO 一一对应：

```text
09-23 + 09-30 -> 10-12
09-23 + 10-12 -> 09-30
09-30 + 10-12 -> 09-23
```

其余口径逐字沿用 `docs/SPEC_UNSW_IID_REFERENCE_20260902.md`：特征 =
`results/unsw_pilot/four_day/features_unsw_w10_4day.csv`；`feature_columns()` /
`build_model` / `clean_x` / `sample_balanced` 全部 import，不重实现；入选门槛 =
三天各自 ≥100 窗的设备交集；`max_rows=20000`；`random_state=42`。

## 3. 簇定义：规则冻结在此，成员由规则机械导出

**簇成员不得由 stacking 结果决定。** 规则如下，只读 **RF** 的目标域混淆矩阵：

```text
对每个有序对 (i,j)：f_ij = CM_rf[i][j] / rowsum(i)，按任务取均值
若 max(f_ij, f_ji) >= TAU = 0.10 则 i—j 连边
簇 = 该图中节点数 >= 2 的连通分量；其余设备为「易类对照」
```

**该规则已在自采上验证**，逐位复算 `results/g0_environment_grid/` 的 RF 混淆矩阵：

| 任务 | 簇成员（f 值） | 被排除（f 值） |
|---|---|---|
| `g0_R2_R3_to_R4` | Light_XM→Light_T1 **0.897**、Sensor→Light_T1 **0.149**、Light_T1→Sensor **0.178** | Camera **0.095**、Socket **0.000** |
| `g0_R3_R4_to_R2` | Light_T1→Light_XM **0.218**、Sensor→Light_T1 **0.231**、Light_XM→Sensor **0.156** | Camera **0.069**、Socket **0.000** |

即 `TAU=0.10` 在自采上恰好给出 `{Light_T1, Light_XM, Sensor}`，成员值 0.149–0.897、
非成员值 0.069–0.095，无边界争议。**该阈值不得为 UNSW 调整。**

**S0（先于任何 stacking 拟合）**：在本协议的三个 2→1 任务上只跑 RF，按上式导出 UNSW 簇成员，
写入 `cluster_definition.json` 并**在该文件落盘后才允许拟合其余模型**。
同时用 `results/unsw_pilot/loro/` 的 6 个 1→1 RF 混淆矩阵独立导出一次作为一致性对照：
两次成员不一致时**以 S0 的 2→1 结果为准**，差异如实记录，不得据此挑选。

## 4. 四条签名判据（运行前定死）

设簇 = `H`、易类对照 = `E`，`gap_S = macro_f1(stacking, S) − max_m macro_f1(m, S)`。

1. **定位**：`gap_H < 0` 且 `|gap_H| >= 3 × |gap_E|`，在 **3/3** 任务成立。
2. **免疫**：`gap_E >= −0.03`，在 **3/3** 任务成立（自采实测 −0.0037 / −0.0266 / +0.0131）。
3. **系数签名**：对簇内每个类 `c`，元学习器系数满足二者之一——
   自类项 `coef[c, oof_m_c]` 至少一个 `m` 为 **≤ 0**；或 `|coef|` 最大的输入**不是**任一自类项。
   要求在簇内 **≥2/3 的类** 上成立，且在 **≥2/3 的任务** 上成立。
   易类对照必须相反：最大项为自类项，在 ≥2/3 易类上成立。
4. **regime 错配**：簇内 RF macro-F1 在源域 OOF 与目标域之间的绝对差 `≥ 0.05`，在 3/3 任务成立
   （自采实测 0.174 / 0.146 / 0.458）。

**四条全过 = 机制外部复现成立。** 逐条实测值无论通过与否一律并报。

## 5. 有风险的定量预测（运行前写死，可被打错）

自采簇内源域 OOF 难度为 macro-F1 **0.2433–0.5148**（自类信号近乎废）；
UNSW 的 Belkin 对 RF recall 约 **0.397**，坏但明显好于自采下限。因此预测：

> **UNSW 的 `|gap_H|` 三任务均值应显著小于自采的 0.2878**（自采三任务
> −0.3740 / −0.1892 / −0.3002 的绝对值均值），**预声明区间 0.03 ~ 0.20**。

- 落在区间内 → 支持「失配幅度随簇内 OOF 难度走」。
- `|gap_H| < 0.03` → 幅度预测失败（机制在 UNSW 上不足以产生可测失配），
  即使签名 1–4 通过也必须记为 `MECHANISM_SIGNATURES_ONLY`。
- `|gap_H| > 0.20`（达到或超过自采水平）→ 幅度预测失败，
  说明幅度不由簇内 OOF 难度决定，机制描述需要修订，**不得**当作「更强的复现」宣传。

## 6. 三分支处置（不得事后新增分支）

| 分支 | 条件 | 状态码 | 处置 |
|---|---|---|---|
| A | 四条签名全过 **且** 幅度落在 0.03–0.20 | `META_MISMATCH_MECHANISM_EXTERNALLY_REPLICATED` | 机制升为两测试床结论，进正文承重结构；自采承担部署事件成分，UNSW 承担机制成分 |
| B | 四条签名全过，幅度预测失败 | `MECHANISM_SIGNATURES_ONLY` | 机制结构外部成立、幅度关系不成立；正文只声称结构复现，幅度差异写入限制章节并如实给出两侧数字 |
| C | 任一签名失败 | `MECHANISM_NOT_EXTERNALLY_REPLICATED` | 机制只作自采单测试床刻画；UNSW 仅承担现象成分（`+0.0328` 参照测量）；失败的具体签名必须逐条写入正文 |

## 7. 硬门（任一失败即停，不写结论）

1. **簇定义先落盘**：`cluster_definition.json` 的 mtime 必须早于任何 stacking 产物；
   脚本在拟合非 RF 模型前检查该文件存在，否则退出。
2. **OOF 折数**：每任务 stacking 的 `n_splits` 实测必须为 **2** 且分组字段为 `day`，逐任务落盘。
   若实现回退到时间块或随机折，即 `INVALID_RUN_STOP`（自采曾有 `_splitter` 缺 `return` 的先例）。
3. **划分不重叠**：训练日与测试日无交集；窗口级无重复 `(device, day, window_id)`。
4. **有限值**：61 维特征 0 NaN / 0 inf。
5. **类集一致**：三任务实际类集与 §2 门槛机械导出的结果一致并落盘。
6. **双跑**：全部判定性产物 md5 逐字节一致。线程钉死
   `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`，不与其他作业并行
   （E1-G0-GRID 记 stacking 跨线程拓扑复现界约 `2e-3`）。
7. 全程禁网、禁代理、不安装依赖。解释器 `~/anaconda3/envs/iotcls/bin/python`。

## 8. 明文不做

- 不改窗口长度、不加特征列、不调参、不做特征选择、不换元学习器。
- 不做 20 天全量、不做随机 1 小时段构造、不做 `strict59` 对照——三者各自另立协议。
- 不计算或恢复 CPD；不恢复 CPD 与 EC-MDM 的机制、预测或部署身份。
- 不重开 E1 / E2 / G1 / Route B / D12 / M1–M6 任何判定；不修改 `results/unsw_test1/`。
- 不用 `category` 粒度做任何主判据（类别粒度已实测掩盖现象：OOD−IID **−0.0131**）。
- 不得因结果不利而改 `TAU`、换簇成员、换天、换门槛或换粒度。
- 不动 `dataset/`；不写入 `results/` 既有子目录；不改 `code/` 与 `docs/` 既有文件。

## 9. 交付物

`results/unsw_meta_mismatch_20260902/`：

```text
cluster_definition.json    S0 导出的簇成员 + 逐对 f 值 + 与 pilot 1->1 的一致性对照
signature_table.csv        四条签名的逐任务实测值与通过与否
subset_f1.csv              逐任务逐模型的 all / H / E macro-F1（源域 OOF 与目标域）
meta_coefficients.csv      元学习器逐 (输出类, 输入通道) 系数
confusion/                 逐任务逐模型混淆矩阵
oof_folds.csv              逐任务 n_splits、分组字段、每折日期
passline.json              §4 四条 + §5 幅度预测的实测与门槛并列
acceptance.json            §7 七道硬门逐项
provenance.json            git、argv、包版本、线程变量、耗时
VERDICT.md                 状态码 + §6 处置 + 解释边界
```

## 10. 登记与顺序

1. 本协议先提交（commit 必须先于任何 UNSW stacking 数值）。
2. `docs/EXPERIMENT_REGISTRY.md` 登记 `UNSW-META-MISMATCH` 一行（运行前登记，含本协议 sha256）。
3. 实现完成后主线侧作实现审阅，追加 `RUN_AUTHORIZED` 方可运行。
4. 判定书写入 `VERDICT.md`，并按 §6 分支同步 `docs/CROSS_LINE_DISCUSSION_20260830.md`；
   历史文档一律加注降格、保留原文。

## 11. 解释边界

无论落在哪一分支，本协议都不构成：现象幅度的外部复现（UNSW 缺部署事件变更，
其跨天 gap `+0.0328` 仅为自采 `strict59` `+0.1201` 的约四分之一）；
CPD 或 EC-MDM 任何身份的恢复；无标签部署方法的成立；D12 判定的改变。
任何引用本协议结果的场合，必须同时报告 D12 在类别粒度上的零结果（OOD−IID `−0.0131`），
并说明粒度差异，不得只报有利的一次。


