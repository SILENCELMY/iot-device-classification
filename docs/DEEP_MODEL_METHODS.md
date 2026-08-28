# 深度模型方法说明：Architecture, Trainer, and Data Split

本文档整理论文正文最终保留的四个深度模型，重点说明：

- **Architecture (Arch)**：网络结构和设计动机；
- **Trainer**：训练策略、优化器、损失函数和复现设置；
- **Data Split**：IID/OOD 划分、标准化和评估口径。

本文档中的正式模型名称沿用 [PAPER_MODEL_ORGANIZATION.md](PAPER_MODEL_ORGANIZATION.md)。

## 1. 统一输入与任务设置

四个深度模型均不直接读取原始 pcapng，而是使用统一特征缓存：

```text
results/robust_v2/raw_all/features_raw_all_w10.csv
```

统一数据口径：

| 项目 | 设置 |
|---|---|
| Windowing | 10s non-overlap windows |
| Feature mode | `raw_all` / `all_features` |
| Input | 每个窗口对应一个 tabular feature vector |
| Numeric feature count | 94 |
| Classes | Camera, Light_T1, Light_XM, Sensor, Socket |
| Metric | Macro-F1 |
| OOD Drop | IID Macro-F1 - OOD Macro-F1 |
| Topology metric | CPD, based on normalized off-diagonal confusion topology |

模型输入张量可记为：

```text
x in R^d, d = 94
```

CNN 模型将 `x` reshape 为：

```text
(batch, 1, d)
```

即把 feature vector 当作一维 feature sequence 处理。

Transformer 模型将每个 scalar feature 视作一个 feature token：

```text
(batch, d, 1) -> feature embedding -> (batch, d, hidden_dim)
```

这保证所有深度模型使用完全相同的特征、标签、窗口和评估指标。

## 2. Data Split Protocol

### 2.1 IID split

IID 结果来自 R2/R3/R4 三个 normal round 的 internal split：

| Task | Train/Test source |
|---|---|
| `single_round_R2` | R2 internal stratified split |
| `single_round_R3` | R3 internal stratified split |
| `single_round_R4` | R4 internal stratified split |

实现口径：

- 对单轮数据按 label 做 stratified train/test split；
- `test_size = 0.3`；
- `random_state = 42`；
- 最终 IID Macro-F1 为 R2、R3、R4 三个任务的平均值。

### 2.2 OOD split

OOD 包含三类场景：

| OOD Type | Task | Meaning |
|---|---|---|
| LORO | R2+R3 -> R4 | leave-one-round-out across normal rounds |
| LORO | R2+R4 -> R3 | leave-one-round-out across normal rounds |
| LORO | R3+R4 -> R2 | leave-one-round-out across normal rounds |
| Position | R2+R3+R4 -> R5 | device position shift |
| Jitter | R2+R3+R4 -> R6+R7 | operation jitter / unstable traffic condition |

最终 OOD Macro-F1 为上述 OOD 任务平均值。

### 2.3 Fixed split artifacts

深度模型最终对比使用固定 split artifacts：

```text
results/robustness_scaling_20260706_v2/splits/<task>/
├── train_idx.npy
├── val_idx.npy
├── test_idx.npy
└── split_metadata.json
```

所有保留模型共享相同：

- train indices；
- test indices；
- label order；
- feature set；
- evaluation tasks；
- Macro-F1 calculation；
- confusion matrix and CPD calculation。

本轮实验不使用 validation-based model selection：

```text
val_idx.npy = empty array
validation_policy = empty_val_no_model_selection
```

因此结果反映固定训练配置下的结构差异，而不是 early stopping 或调参搜索差异。

### 2.4 Standardization rule

所有深度模型严格使用 train-only standardization：

```text
scaler = StandardScaler()
x_train = scaler.fit_transform(train_features)
x_test  = scaler.transform(test_features)
```

禁止：

- 在 full dataset 上 fit scaler；
- 在 test set 上重新 fit；
- 使用跨任务共享 scaler；
- 使用 OOD test distribution 信息做 normalization。

该规则避免 normalization leakage。

## 3. Unified Trainer

四个深度模型共用同一训练流程。

### 3.1 Loss function

使用 class-weighted cross entropy：

```text
CrossEntropyLoss(weight = normalized inverse class frequency)
```

class weight 由当前 task 的 training labels 计算：

```text
class_weight_c = total_train_samples / count_c
class_weight = class_weight / mean(class_weight)
```

目的：

- 缓解不同设备类别窗口数量不均衡；
- 保持 Macro-F1 目标下的类别均衡性；
- 不引入额外采样策略。

### 3.2 Optimizer

使用 AdamW：

| Item | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.0009 |
| Weight decay | 0.0001 |
| Batch size | 128 |
| Epochs | 70 |
| Gradient clipping | max norm = 5.0 |
| Device | CUDA when available |

### 3.3 Reproducibility

固定随机种子：

```text
random_state = 42
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
DataLoader generator seed = 42
```

同时设置：

```text
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 3.4 Training loop

训练流程：

```text
for epoch in epochs:
    for batch in shuffled_train_loader:
        logits = model(x_batch)
        loss = weighted_cross_entropy(logits, y_batch)
        backpropagation
        clip_grad_norm_(max_norm=5.0)
        AdamW update
```

推理流程：

```text
model.eval()
argmax(logits) -> predicted label
```

没有使用：

- data augmentation；
- external pretraining；
- AutoML；
- multi-stage training；
- early stopping；
- validation-based checkpoint selection。

这使模型差异主要来自 architecture，而不是复杂训练技巧。

## 4. Model 1: Multi-Scale Residual CNN (MSR-CNN)

原实验名称：

```text
cnn1d_v3
```

参数量：

```text
1.01M
```

### 4.1 Architecture

MSR-CNN 是 CNN 系列中的高容量多尺度卷积基线。它将 tabular feature vector 视作一维 feature sequence。

结构流程：

```text
Input feature vector
-> LayerNorm
-> Conv1D stem
   Conv1D(1 -> 64, k=3)
   GELU
   Conv1D(64 -> 128, k=5)
   GELU
   Conv1D(128 -> 160, k=3)
   GELU
-> Multi-Scale Residual Block x 3
-> Attention Pooling
-> Mean Pooling
-> Max Pooling
-> Raw Feature Skip
-> MLP Head
-> Classifier
```

Multi-scale residual block:

```text
Input channels = 160
Branches:
  Conv1D(k=3)
  Conv1D(k=5)
  Dilated Conv1D(k=3, dilation=2)
Concat
-> Conv1D(k=1) mixing
-> Dropout
-> Conv1D(k=3)
-> SE channel attention
-> residual add
-> GELU
```

Pooling representation：

```text
z = concat(
    attention_pool(conv_features),
    mean_pool(conv_features),
    max_pool(conv_features),
    raw_skip(input_features)
)
```

Head：

```text
LayerNorm
Dropout(0.25)
Linear(640 -> 256)
GELU
Dropout(0.15)
Linear(256 -> 128)
GELU
Linear(128 -> num_classes)
```

### 4.2 Design intention

MSR-CNN 的目标是增强局部和多尺度 feature relation modeling：

- `k=3/k=5/dilated k=3` 捕获不同尺度的 feature interaction；
- residual blocks 提升深层训练稳定性；
- SE attention 建模 channel importance；
- attention/mean/max pooling 融合不同 summary statistics；
- raw feature skip 保留原始 tabular signal。

### 4.3 Paper role

论文中建议定位为：

```text
high-capacity CNN relation modeling baseline
```

它的意义不是作为最鲁棒模型，而是展示：

- IID 表现很强；
- OOD Drop 较大；
- CPD 较高；
- 强关系建模可能带来 environment-specific topology fitting。

## 5. Model 2: Temporal Dilated Residual CNN (TDR-CNN)

原实验名称：

```text
cnn1d_tcn
```

参数量：

```text
0.81M
```

### 5.1 Architecture

TDR-CNN 是最终 CNN sweet spot 模型。它同样把 feature vector 当作一维 feature sequence，但使用 TCN-style dilated residual blocks，而不是多分支 Inception-style block。

结构流程：

```text
Input feature vector
-> LayerNorm
-> Conv1D stem
   Conv1D(1 -> 80, k=3)
   GELU
   Conv1D(80 -> 144, k=3)
   GELU
-> Dilated Residual Block(dilation=1)
-> Dilated Residual Block(dilation=2)
-> Dilated Residual Block(dilation=4)
-> Dilated Residual Block(dilation=8)
-> Attention Pooling
-> Mean Pooling
-> Std Pooling
-> Raw Feature Skip
-> MLP Head
-> Classifier
```

Dilated residual block：

```text
Input channels = 144
Conv1D(k=3, dilation=d)
GELU
Dropout(0.14)
Conv1D(k=1)
GELU
Dropout(0.14)
Conv1D(k=3, dilation=d)
SE channel attention
Residual add
GELU
```

其中 dilation 依次为：

```text
1, 2, 4, 8
```

Pooling representation：

```text
z = concat(
    attention_pool(conv_features),
    mean_pool(conv_features),
    std_pool(conv_features),
    raw_skip(input_features)
)
```

Head：

```text
LayerNorm
Dropout(0.28)
Linear(576 -> 224)
GELU
Dropout(0.18)
Linear(224 -> 112)
GELU
Linear(112 -> num_classes)
```

### 5.2 Design intention

TDR-CNN 的设计核心是用 dilated convolution 建模更平滑的跨特征关系：

- dilation=1/2/4/8 逐步扩大 receptive field；
- residual connection 保持稳定优化；
- SE attention 提供轻量 channel recalibration；
- std pooling 补充 feature activation dispersion 信息；
- 相比 MSR-CNN，TDR-CNN 避免过强多分支局部模式拟合。

### 5.3 Paper role

论文中建议定位为：

```text
topology robustness sweet spot in the CNN family
```

相对 MSR-CNN：

| Model | IID | OOD | OOD Drop | CPD |
|---|---:|---:|---:|---:|
| MSR-CNN | 0.9310 | 0.5859 | 0.3451 | 0.8497 |
| TDR-CNN | 0.9297 | 0.6020 | 0.3277 | 0.8177 |

关键解释：

- IID 几乎持平；
- OOD 明显提升；
- OOD Drop 下降；
- CPD 下降。

因此 TDR-CNN 支持论文中的重要判断：

```text
appropriate relation modeling bias can improve OOD robustness
without simply increasing capacity.
```

## 6. Model 3: Lightweight Feature Transformer (LF-Transformer)

原实验名称：

```text
transformer
```

参数量：

```text
73.5K
```

### 6.1 Architecture

LF-Transformer 是 Transformer 系列的轻量基线。它把每个 scalar feature 看作一个 token。

结构流程：

```text
Input feature vector
-> Unsqueeze each feature as scalar token
-> Linear projection: 1 -> d_model
-> Learnable positional encoding
-> Transformer Encoder x 2
-> Mean Pooling over feature tokens
-> LayerNorm
-> Dropout
-> Linear classifier
```

主要超参数：

| Item | Value |
|---|---:|
| d_model | 64 |
| Heads | 4 |
| Encoder layers | 2 |
| FFN hidden dim | 128 |
| Dropout | 0.10 |
| Pooling | mean pooling |

Classifier：

```text
LayerNorm(64)
Dropout(0.15)
Linear(64 -> num_classes)
```

### 6.2 Design intention

LF-Transformer 用最小 Transformer 结构测试 feature-token global relation modeling：

- self-attention 可直接建模任意 feature-feature relation；
- 2-layer encoder 控制容量；
- mean pooling 避免额外复杂 attention pooling；
- 作为 Transformer family 的 baseline。

### 6.3 Paper role

论文中建议定位为：

```text
low-capacity global relation modeling baseline
```

它的作用是给 APF-Transformer 提供对照：

| Model | IID | OOD | OOD Drop | CPD |
|---|---:|---:|---:|---:|
| LF-Transformer | 0.8993 | 0.6223 | 0.2770 | 0.7199 |
| APF-Transformer | 0.9292 | 0.5942 | 0.3350 | 0.8236 |

该对照说明：增强 Transformer 表达能力后，IID 可提升，但 OOD robustness 和 topology stability 可能下降。

## 7. Model 4: Attention-Pooled Feature Transformer (APF-Transformer)

原实验名称：

```text
transformer_v2
```

参数量：

```text
605.6K
```

### 7.1 Architecture

APF-Transformer 是增强型 Transformer。它比 LF-Transformer 更深、更宽，并引入 attention pooling 和 raw feature skip。

结构流程：

```text
Input feature vector
-> Unsqueeze each feature as scalar token
-> Feature embedding: Linear(1 -> 128)
-> Learnable positional encoding
-> Transformer Encoder x 4
-> Attention Pooling
-> Mean Pooling
-> Raw Feature Skip
-> MLP Head
-> Classifier
```

主要超参数：

| Item | Value |
|---|---:|
| d_model | 128 |
| Heads | 4 |
| Encoder layers | 4 |
| FFN hidden dim | 256 |
| Dropout | 0.10 |
| Pooling | attention pooling + mean pooling |

Attention pooling：

```text
score_i = Linear(LayerNorm(token_i))
weight_i = softmax(score_i)
attn_pool = sum(weight_i * token_i)
```

Raw feature skip：

```text
LayerNorm(input)
Linear(feature_count -> 128)
GELU
```

Head：

```text
concat(attention_pool, mean_pool, raw_skip)
-> LayerNorm(384)
-> Dropout(0.20)
-> Linear(384 -> 128)
-> GELU
-> Dropout(0.10)
-> Linear(128 -> num_classes)
```

### 7.2 Design intention

APF-Transformer 的目标是增强全局 relation modeling：

- 更深 encoder 提高 token interaction capacity；
- 更大 hidden dimension 提高表示能力；
- attention pooling 让模型学习 feature-token importance；
- raw feature skip 保留原始 tabular signal；
- MLP head 提升非线性分类能力。

### 7.3 Paper role

论文中建议定位为：

```text
enhanced global relation modeling model
```

它用于验证：

```text
higher representation capacity
does not necessarily imply stronger topology robustness.
```

相对 LF-Transformer：

- IID 从 0.8993 提升到 0.9292；
- OOD 从 0.6223 降到 0.5942；
- OOD Drop 从 0.2770 增加到 0.3350；
- CPD 从 0.7199 增加到 0.8236。

这说明增强 attention-based relation modeling 后，模型可能更容易拟合 environment-specific confusion topology。

## 8. Cross-Model Interpretation

四个深度模型形成两条对照线：

```text
CNN family:
MSR-CNN -> TDR-CNN

Transformer family:
LF-Transformer -> APF-Transformer
```

### 8.1 CNN family conclusion

MSR-CNN 的 IID 略高，但 OOD 和 CPD 更差。TDR-CNN 几乎保持 IID，同时提升 OOD 并降低 CPD。

解释：

```text
multi-scale high-capacity local interaction
may learn environment-specific topology;

dilated residual relation modeling
provides smoother cross-feature relation bias,
leading to better OOD topology stability.
```

### 8.2 Transformer family conclusion

APF-Transformer 相比 LF-Transformer 提升 IID，但 OOD 下降且 CPD 增大。

解释：

```text
enhanced global feature-token attention
can improve IID fitting,
but may amplify environment-specific topology fitting.
```

### 8.3 Main paper message

最终论文主线可表述为：

> In complex IoT device identification, stronger representation learning does not necessarily lead to stronger topology robustness. OOD generalization depends not only on capacity, but also on the inductive bias of relation modeling. CPD captures this divergence by measuring how confusion topology changes across environments.

## 9. Suggested Method Section Structure

建议论文方法部分这样组织：

```text
3.1 Feature Windowing and Dataset Protocol
3.2 Confusion Pattern Drift
3.3 Representative Models
    3.3.1 Multi-Scale Residual CNN
    3.3.2 Temporal Dilated Residual CNN
    3.3.3 Lightweight Feature Transformer
    3.3.4 Attention-Pooled Feature Transformer
3.4 Training Protocol
3.5 IID and OOD Evaluation Protocol
```

实验部分可这样组织：

```text
4.1 IID Performance
4.2 OOD Generalization
4.3 Confusion Topology Drift
4.4 Capacity and Topology Robustness
4.5 CNN vs Transformer Relation Modeling
```

## 10. Reproducibility Pointers

关键代码：

| Purpose | Path |
|---|---|
| Model definitions | `code/scripts/analysis/deep_robustness_validation.py` |
| Fixed split evaluation | `code/scripts/analysis/topology_sweet_spot_experiment.py` |
| Architecture contrast experiment | `code/scripts/analysis/cnn_contrast_search_experiment.py` |

关键结果：

| Purpose | Path |
|---|---|
| Final CNN architecture contrast | `results/cnn_architecture_contrast_20260707/report/cnn_contrast_table.csv` |
| Transformer and CNN baseline capacity table | `results/gpu_capacity_full_20260703/report/capacity_performance_table.csv` |
| Fixed split artifacts | `results/robustness_scaling_20260706_v2/splits/` |

复现实验时应优先复用 fixed split artifacts，而不是重新随机划分。
