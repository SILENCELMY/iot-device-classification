# G0 strict59_ra EC-MDM R5 有界稳定性原地续跑协议（续跑前冻结）

**日期**：2026-09-03（Asia/Shanghai）

**状态**：`R5_BOUNDED_STABILITY_CONTINUATION_FROZEN_BEFORE_S3`

**适用范围**：独立的空口表示接口审计线；不并入主线。

## 1. 触发事实与边界

R5 已独立完成 G0-A 与 G0-B：每轮 162 个任务、648 个模型单元。原生模型进程隔离期间无
`SIGSEGV`、`SIGBUS`、`SIGABRT` 或重试。R5 在 `S2_G0_VERIFY` 的首个逐字节不一致处按冻结协议
停止：

```text
raw_all/g0_R2_R3_R4_to_R5/all_features/lightgbm/pred_proba.csv
```

停止后、续跑协议冻结前进行的只读取证表明：RF/XGBoost 概率、预测和指标逐字节一致；LightGBM
标签与指标一致但部分概率不逐字节一致；这些差异在 4 个 Stacking 单元中造成总计 6 个标签差异，
最大单元标签分歧率约 0.001089，四项汇总指标最大差值约 0.001130。该事实必须永久披露，不能把
R5 描述为逐字节双跑复现。

本协议是失败后的工程恢复协议，不追认原始逐字节门已经通过，也不改变父协议的科学阈值。用户已
明确授权在同一 R5 audit root 内恢复并继续；原停止记录必须改名保存，禁止静默删除。

## 2. 冻结输入

只允许读取下列已经完成的 R5 staging：

```text
G0-A: results/g0_environment_grid_strict59_ra_r5/
G0-B: /tmp/strict59_ra_ecmdm_ge7nrsdf/g0_b/
audit: results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r5/
```

进入 S3 前必须验证：

1. `FAILED.json`、R5 runner、R5 implementation freeze 及本协议/冻结记录哈希匹配；
2. A/B 各有 5835 个文件、648 个 `metrics.json`、648 个 `predictions.csv`、648 个
   `pred_proba.csv`、648 个 `feature_columns.json` 和 162 个 `oof_meta.csv`；
3. A/B 相对文件集合完全相同；59 特征列记录完全相同；模型隔离审计显示两轮各 648/648 成功；
4. 不读取 R2/R3/R4 失败 staging 作为续跑输入；不修改 G0-A、G0-B 或旧协议/实现文件。

任一条件失败立即保留/写回 `FAILED.json` 并停止。

## 3. G0 有界重复稳定性门

该门一次性扫描全部 A/B 单元，禁止只检查已知差异文件。

### 3.1 仍要求精确一致

- 所有样本键、真实标签、顺序、形状与特征列；
- RF、XGBoost 的 `predictions.csv`、`pred_proba.csv`、`metrics.json`；
- LightGBM 的全部预测标签和全部 `metrics.json`；
- 每轮各自的预测标签必须等于相应概率的 `argmax`；
- 概率必须有限、位于 `[0,1]`，每个概率块行和误差不超过 `1e-10`。

### 3.2 预先固定的有界门

- LightGBM、Stacking 的目标概率最大绝对差：`<= 0.05`；
- OOF 三个基模型概率块最大绝对差：`<= 0.05`；
- Stacking 全部 289362 个目标预测的全局标签分歧率：`<= 1e-4`；
- 任一 Stacking 单元标签分歧率：`<= 0.002`；
- 任一 Stacking 单元的 accuracy、precision、recall、macro-F1 绝对差：`<= 0.002`；
- Stacking 之外的模型标签分歧数和指标差必须为零。

通过只能称为 `G0_BOUNDED_REPLICATE_STABILITY_PASS`，不得写成 byte-identical。完整遥测与所有非
逐字节文件清单写入 `g0_double_run_verification.json`。

## 4. 原地恢复和下游双跑

有界门通过后：

1. 将原 `FAILED.json` 原子改名为 `RECOVERED_S2_G0_VERIFY_FAILURE.json`；
2. G0-A 是原协议已指定的正式 root，作为唯一 canonical science input；G0-B 只承担独立拟合稳定性
   审计，不作为第二份扰动输入；
3. M1、M1-R、M2 仍各运行两次，但两次均从 canonical G0-A 出发，以分别验证固定输入下的计算
   确定性；冻结科学模块、阈值和 `adjudicate()` 逻辑一律不改；
4. M1 gate 必须完全相同；M1-R 与 M2 继续使用各自冻结的逐字节双跑 verifier；
5. 正式 A 结果写入原 R5 science root，B 结果仍写入原 R5 `/tmp`；不得覆盖已有 G0 文件。

若续跑失败，必须生成新的 `FAILED.json`，同时保留原停止记录和所有新 staging。若成功，audit root
中必须没有 `FAILED.json`，必须生成 `VERDICT.md`、`acceptance.json`、`provenance.json`、manifest
以及恢复审计；provenance 必须记录本协议、恢复实现和原停止记录的哈希。

## 5. 裁定措辞不变

即使恢复成功，也只裁定带目标标签的 **oracle recoverability/structure**。observable estimability
仍未检验，deployability 仍未建立；EC-MDM 只能称为 CPD 的候选上位构念。本恢复不证明因果机制、
无标签可估计性或部署收益。若后续无标签估计失败，commissioning 仍须另冻协议，并战胜相同目标
标签预算基线。

## 6. 运行约束

- 不联网、不安装依赖、六个代理变量必须为空；
- 使用全新 user-systemd continuation unit，`Restart=no`、`KillMode=control-group`、`Linger=yes`；
- 先通过合成测试和 `--preflight-no-fit`，再允许 `--continue-after-g0`；
- 续跑不重复训练 G0-A/G0-B，不删除 `/tmp/strict59_ra_ecmdm_ge7nrsdf`。
