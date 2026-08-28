# CPD 核心发现

> [!CAUTION] **部分结论已降格（2026-08-28，协议 §11）**
> 下表「CPD vs Stacking Gain Pearson r = -0.630」已按协议完成敏感性审计，**降为探索性结果**
> （LOTO 3/11 次显著；Spearman 不显著；口径依赖）。详见
> `results/p0_audit/R630_SENSITIVITY_CONCLUSION.md`。本文其余数值截至 2026-07-07。

CPD 是本项目的主线概念，全称 **Confusion Pattern Drift**。它衡量的不是准确率变化，而是模型“错法”的结构变化。

## 定义

对两个混淆矩阵 `C_i`、`C_j`：

```text
CPD(C_i, C_j) = ||Off(C_i) - Off(C_j)||_F
```

其中：

- `Off(C)`：去掉对角线，只保留误分类结构
- `||.||_F`：Frobenius 范数
- 实现中通常先对混淆矩阵按行归一化

## 核心论点

传统解释通常停在：

```text
特征分布漂移 -> 性能下降
```

本项目进一步把机制拆成：

```text
特征漂移
  -> 混淆模式漂移
  -> 误差相关性坍缩
  -> 元学习器失配
  -> Stacking 失效
```

也就是说，跨场景后模型不仅变差，而且错误拓扑变了；Stacking 在训练场景学到的基模型互补关系，在 OOD 场景失效。

## 关键数值

| 指标 | 数值 |
|---|---:|
| CPD vs F1 下降 Pearson r | 0.9499 |
| CPD vs F1 下降 Spearman rho | 0.9429 |
| IID 平均 CPD | 0.1466 |
| OOD 平均 CPD | 0.5563 |
| OOD - IID CPD 差值 | +0.4097 |
| CPD vs Stacking Gain Pearson r | -0.630 |

## 典型失效案例

`loro_R2_R4_to_R3` 是最严重的 Stacking 崩溃场景：

- CPD 最高
- RF macro-F1 约 0.615
- Stacking macro-F1 约 0.546
- Stacking 比最佳基础模型更差
- Sensor 召回率从近乎稳定变成大幅误判为 Light_T1

## 推荐报告

按这个顺序读：

1. [CPD_PAPER_LEVEL_ANALYSIS.md](../results/robust_v2/report/CPD_PAPER_LEVEL_ANALYSIS.md)
2. [CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md](../results/robust_v2/report/CONTROLLED_CPD_MECHANISM_VALIDATION_FINAL.md)
3. [STACKING_COLLAPSE_ANALYSIS.md](../results/robust_v2/report/STACKING_COLLAPSE_ANALYSIS.md)

次要分析和旧版报告已经归档到 `legacy/docs/archive/result-reports-secondary/`。
