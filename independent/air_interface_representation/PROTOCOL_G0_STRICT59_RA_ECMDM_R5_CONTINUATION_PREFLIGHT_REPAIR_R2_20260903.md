# R5 有界稳定性续跑 preflight 概率精度修正 R2（续跑前冻结）

**日期**：2026-09-03（Asia/Shanghai）

**状态**：`R5_CONTINUATION_FLOAT32_SIMPLEX_TOLERANCE_FROZEN_BEFORE_S3`

## 1. 触发与范围

父续跑协议及实现完成后，第一次只读 `--preflight-no-fit` 在检查 XGBoost `pred_proba.csv` 时停止：

```text
probability row-sum error 8.585266209060194e-08
```

随后对冻结 A/B 的全部目标概率与 OOF 概率只读扫描确认：两轮 XGBoost/OOF 最大行和误差均为
`8.900133252609521e-08`；RF 最大为 `2.220446049250313e-16`，LightGBM 与 Stacking 最大为
`7.771561172376096e-16`。该误差来自 XGBoost float32 概率表示，不是 A/B 差异。

第一次 preflight 未移动 `FAILED.json`、未创建 science root、未运行 M1/M1-R/M2，也未修改 G0-A/G0-B。
本 R2 只修正单轮概率单纯形合法性容差；父续跑协议的全部 A/B 稳定性阈值、canonical G0-A 规则、
科学模块和裁定逻辑保持不变。

## 2. 唯一修正

父续跑协议第 3.1 节的：

```text
每个概率块行和误差不超过 1e-10
```

替换为：

```text
每个概率块行和误差不超过 1e-6
```

`1e-6` 是针对 float32 概率输出的格式合法性上界，不是 A/B 相似度阈值。以下门槛不变：

- LightGBM、Stacking 目标概率最大绝对差 `<= 0.05`；
- OOF 概率最大绝对差 `<= 0.05`；
- Stacking 全局标签分歧率 `<= 1e-4`；
- 任一 Stacking 单元标签分歧率 `<= 0.002`；
- 任一 Stacking 单元四项汇总指标绝对差 `<= 0.002`；
- RF/XGBoost 概率仍须 A/B 逐字节一致；三个基模型标签和指标仍须逐字节一致。

## 3. 其余约束

父续跑协议所有未被本文件明确替换的条款继续有效。仍须先更新并冻结续跑实现、通过 16 项合成测试
和完整只读 preflight，才允许原地移动停止记录并进入 S3。即使最终通过，oracle recoverability、
observable estimability、deployability 与 CPD 候选上位构念的证据边界不变。
