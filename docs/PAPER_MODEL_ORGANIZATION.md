# 深度学习实验论文组织方案

本文档用于把深度学习探索阶段的版本号命名整理为论文正文可用的正式模型体系。后续论文正文不再使用 `v1/v2/v3` 等开发版本号，而使用能够反映结构特征的学术化名称。

四个深度模型的具体 Architecture、Trainer 和 Data Split 口径见 [DEEP_MODEL_METHODS.md](DEEP_MODEL_METHODS.md)。

## 1. 最终命名方案

| 正文名称 | 原实验名称 | 模型系列 | 命名理由 |
|---|---|---|---|
| Random Forest | `rf` | Classical ML | 传统机器学习基线，不涉及深度表示学习，用于建立鲁棒性参考下界和强传统模型基线。 |
| Multi-Scale Residual CNN | `cnn1d_v3` | CNN | 该模型由多尺度卷积分支、残差连接、attention pooling 和 raw feature skip 组成；名称突出其主要归纳偏置，而不保留开发版本号。 |
| Temporal Dilated Residual CNN | `cnn1d_tcn` | CNN | 该模型使用 dilated residual convolution 建模跨特征位置的长程关系，接近 TCN 思路；名称强调 temporal/dilated relation modeling。 |
| Lightweight Feature Transformer | `transformer` | Transformer | 该模型为轻量 Transformer encoder，使用 feature token embedding 与 mean pooling，作为 Transformer 系列基线。 |
| Attention-Pooled Feature Transformer | `transformer_v2` | Transformer | 该模型增加 encoder depth、attention pooling 和 raw feature skip，体现增强型全局关系建模。 |

推荐论文中首次出现时写作：

- **Multi-Scale Residual CNN (MSR-CNN)**
- **Temporal Dilated Residual CNN (TDR-CNN)**
- **Lightweight Feature Transformer (LF-Transformer)**
- **Attention-Pooled Feature Transformer (APF-Transformer)**

正文中可使用缩写，但表格建议保留完整名称或“完整名称 + 缩写”。

## 2. Table 1: Representative Models

| Model | Architecture Type | Core Design | Parameter Size |
|---|---|---|---:|
| Random Forest | Classical ensemble | Bagged decision trees with feature-subspace sampling | N/A |
| Multi-Scale Residual CNN | Convolutional neural network | Multi-scale 1D convolution, residual blocks, attention pooling, raw feature skip | 1.01M |
| Temporal Dilated Residual CNN | Convolutional neural network | Dilated residual convolution blocks, lightweight channel attention, attention/statistical pooling | 0.81M |
| Lightweight Feature Transformer | Transformer encoder | Feature-token embedding, 2-layer encoder, mean pooling | 73.5K |
| Attention-Pooled Feature Transformer | Transformer encoder | Deeper feature-token encoder, attention pooling, raw feature skip, MLP head | 605.6K |

## 3. Table 2: Performance Comparison

| Model | Parameters | IID Macro-F1 | OOD Macro-F1 | OOD Drop | CPD |
|---|---:|---:|---:|---:|---:|
| Random Forest | N/A | 0.9537 | 0.7142 | 0.2396 | 0.5973 |
| Multi-Scale Residual CNN | 1.01M | 0.9310 | 0.5859 | 0.3451 | 0.8497 |
| Temporal Dilated Residual CNN | 0.81M | 0.9297 | 0.6020 | 0.3277 | 0.8177 |
| Lightweight Feature Transformer | 73.5K | 0.8993 | 0.6223 | 0.2770 | 0.7199 |
| Attention-Pooled Feature Transformer | 605.6K | 0.9292 | 0.5942 | 0.3350 | 0.8236 |

指标定义：

- `IID Macro-F1`: R2/R3/R4 internal split 平均 Macro-F1。
- `OOD Macro-F1`: LORO、Position、Jitter 场景平均 Macro-F1。
- `OOD Drop = IID Macro-F1 - OOD Macro-F1`。
- `CPD`: OOD confusion matrix 相对 IID confusion topology 的平均漂移程度。

## 4. 最终实验组织结构

### 4.1 Classical Machine Learning

正文保留：

- Random Forest

定位：

Random Forest 是传统机器学习强基线。它在 IID 和 OOD 下均表现稳定，CPD 也最低，说明传统集成树模型虽然表达能力有限，但对跨场景 topology drift 更不敏感。论文中可将其作为“鲁棒性参照模型”，而不是简单 accuracy baseline。

### 4.2 CNN Family

正文保留：

- Multi-Scale Residual CNN
- Temporal Dilated Residual CNN

定位：

Multi-Scale Residual CNN 代表高容量多尺度卷积关系建模。它在 IID 场景表现较强，但 OOD drop 和 CPD 明显增大，体现了过强 topology fitting 的风险。

Temporal Dilated Residual CNN 代表另一种 CNN 归纳偏置：通过 dilated residual convolution 捕获更平滑的跨特征关系。相较 Multi-Scale Residual CNN，它几乎不损失 IID，但 OOD Macro-F1 更高、CPD 更低，因此可作为 CNN 系列中的 topology robustness sweet spot。

建议正文叙事：

```text
Multi-Scale Residual CNN
  -> high IID performance, high CPD
Temporal Dilated Residual CNN
  -> similar IID, improved OOD, reduced CPD
```

这支持论文主线：模型表达能力本身并不等价于 topology robustness，关键还包括 relation modeling 的结构偏置。

### 4.3 Transformer Family

正文保留：

- Lightweight Feature Transformer
- Attention-Pooled Feature Transformer

定位：

Lightweight Feature Transformer 是低容量全局关系建模基线。它 IID 低于增强型 Transformer，但 OOD 和 CPD 更稳定。

Attention-Pooled Feature Transformer 增强了 encoder depth 和 pooling/head 表达能力。它 IID 明显提升，但 OOD 下降、CPD 增大，说明增强全局关系建模可能学习到更强的 environment-specific topology。

建议正文叙事：

```text
Lightweight Feature Transformer
  -> moderate IID, better OOD stability
Attention-Pooled Feature Transformer
  -> higher IID, larger OOD drop, larger CPD
```

这与 CNN 系列共同支撑核心论点：更强表示能力不必然带来更强 OOD/topology robustness。

## 5. 正文、附录和归档划分

### 5.1 正文保留

| 内容 | 用途 |
|---|---|
| Random Forest | 传统模型强基线和鲁棒性参照 |
| Multi-Scale Residual CNN | CNN 高容量多尺度关系建模代表 |
| Temporal Dilated Residual CNN | CNN topology robustness sweet spot |
| Lightweight Feature Transformer | Transformer 轻量全局关系建模基线 |
| Attention-Pooled Feature Transformer | Transformer 增强全局关系建模代表 |
| Table 1 Representative Models | 说明最终模型体系 |
| Table 2 Performance Comparison | 展示 IID/OOD/CPD 核心结果 |
| CPD definition and analysis | 支撑论文核心机制 |
| IID vs OOD degradation figure | 展示泛化差异 |
| Confusion topology comparison | 展示 confusion topology drift |

### 5.2 附录保留

| 内容 | 建议位置 |
|---|---|
| CNN-Inception | Supplementary ablation: alternative CNN inductive bias |
| CNN-ConvNeXt | Supplementary ablation: depthwise separable CNN variant |
| CNN-v5 | Supplementary capacity scaling result |
| CNN-v3-smooth / CNN-v3-sharp / CNN-v3-hybrid | Supplementary local CNN variants |
| CNN-v2 / CNN-v4 | Supplementary capacity scaling trajectory |
| Scenario-wise performance | Supplementary tables |
| Scenario-wise CPD | Supplementary tables |

### 5.3 归档保存

| 内容 | 归档理由 |
|---|---|
| CNN-v1 | 初始轻量 CNN，性能不足，仅反映开发起点 |
| CNN-v2 | 中间容量版本，正文叙事不再需要 |
| CNN-v4 | 后续容量扩展版本，未形成更优正文代表 |
| CNN-v5 | 极端/高容量验证版本，作为 scaling archive |
| CNN-v3-smooth / sharp / hybrid | 围绕 CNN-v3 的局部变体，属于开发搜索过程 |
| CNN-Inception | 架构探索模型，性能有参考价值但不如 TDR-CNN 适合作为正文代表 |
| CNN-ConvNeXt | 架构探索模型，CPD 低但 OOD 提升有限 |
| 其它探索版本 | 保留复现路径，不进入正文主线 |

建议归档目录口径：

```text
results/cnn_architecture_contrast_20260707/   # 架构级 CNN 探索，正文引用 TDR-CNN
results/robustness_scaling_20260706_v2/       # CNN capacity scaling / fixed splits
legacy/results/cnn_contrast_search_20260707/  # CNN-v3 局部变体探索
legacy/results/extreme_capacity_1p2m_20260706/ # high-capacity archive
legacy/results/topology_sweet_spot_20260703/  # CNN-v2.5 过渡实验
```

## 6. 论文主线建议

建议正文按以下逻辑组织：

```text
1. Classical baseline
   Random Forest establishes a robust non-deep reference.

2. CNN family
   Multi-Scale Residual CNN shows strong IID but large CPD.
   Temporal Dilated Residual CNN improves OOD and reduces CPD.

3. Transformer family
   Lightweight Feature Transformer provides a low-capacity global-relation baseline.
   Attention-Pooled Feature Transformer improves IID but increases CPD/OOD drop.

4. Mechanism analysis
   Higher relation modeling capability can improve IID,
   but may also increase environment-specific topology fitting.
   CPD explains why IID performance and OOD robustness diverge.
```

可用于正文的核心表述：

> The results indicate that stronger representation learning does not necessarily imply stronger topology robustness. The Temporal Dilated Residual CNN achieves a more favorable balance between IID accuracy, OOD generalization, and CPD, while enhanced global relation modeling in the transformer family increases IID performance at the cost of larger topology drift.

## 7. 最终推荐表述

正文中不建议写：

- CNN-v1 / CNN-v2 / CNN-v3
- Transformer-v1 / Transformer-v2
- extreme capacity model
- best tuned model

正文中建议写：

- Random Forest
- Multi-Scale Residual CNN
- Temporal Dilated Residual CNN
- Lightweight Feature Transformer
- Attention-Pooled Feature Transformer

这样可以把实验从“开发版本比较”整理为“代表性架构比较”，更符合 SCI 论文中方法和实验章节的组织方式。
