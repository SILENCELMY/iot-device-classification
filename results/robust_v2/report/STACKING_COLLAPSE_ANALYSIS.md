# Stacking 在漂移场景下的崩盘原因分析

> [!CAUTION] **口径说明（2026-08-29，协议 §9.1 / §12）**
> 本报告全部 Stacking 数值（含 `0.5455` / `-0.0693`）为**随机折叠 OOF** 口径，
> 即 E1 的 **A 臂**。该口径按协议 §9.1 存在跨轮次泄漏、整体偏乐观。
> E1 正式结果（11 任务 × 5 种子 × 三臂）显示分组 OOF 下旗舰任务 `loro_R2_R4_to_R3`
> 的 gain 由 `-0.0662` 深化至 `-0.1057`。**以 `results/e1_oof_arms/` 的 E1 结果为准。**
> 正文原样保留供溯源。

**日期**: 2026-06-23  
**分析对象**: Robust V2 实验 (110 runs)  
**崩盘定义**: Stacking 相比 Best Base Learner 下降 > 0.05 (macro-F1)

---

## 一、核心发现

### 1.1 崩盘现象

| 任务 | 训练样本 | Stacking F1 | Best Base (RF) | Delta | 状态 |
|---|---|---|---|---|---|
| **LORO R2+R4→R3** | 3669 | 0.5455 | 0.6148 | **-0.0693** | ⚠️ 严重崩盘 |
| Position R2-R4→R5 | 5506 | 0.6678 | 0.7012 | -0.0334 | 负效应 |
| LORO R3+R4→R2 | 3690 | 0.7905 | 0.8098 | -0.0193 | 轻微负效应 |
| LORO R2+R3→R4 | 3653 | 0.6634 | 0.6662 | -0.0028 | 接近持平 |

**关键观察**:
- **LORO R2+R4→R3 是最严重的崩盘场景** (delta -0.0693)
- 训练集 3500-5500 样本的 drift 任务普遍负效应
- Jitter 任务 (5506 样本) 接近持平或略好 (delta -0.0001 到 -0.0053)

---

## 二、混淆矩阵分析: 为何 LORO R2+R4→R3 崩盘？

### 2.1 RF (Best Base) vs Stacking 混淆矩阵对比

#### RF 表现 (F1 = 0.6148)
```
            Camera  Light_T1  Light_XM  Sensor  Socket
Camera       362         2         1       0       0     (99.2% 召回)
Light_T1       1        88       179      95       0     (24.2% 召回)  → 主要混淆到 Light_XM, Sensor
Light_XM       3        95       270       1       0     (73.2% 召回)
Sensor        17       232        46      68       1     (18.7% 召回)  → 主要混淆到 Light_T1
Socket         0         0         0       0     376     (100% 召回)
```

#### Stacking 表现 (F1 = 0.5455)
```
            Camera  Light_T1  Light_XM  Sensor  Socket
Camera       363         2         0       0       0     (99.5% 召回)
Light_T1       2        52       224      85       0     (14.3% 召回)  → 召回下降 10%!
Light_XM      17        69       282       1       0     (76.4% 召回)
Sensor        21       271        61      10       1     (2.7% 召回)   → 召回暴跌到 2.7%!
Socket         0         0         0       0     376     (100% 召回)
```

### 2.2 差异分析 (Stacking - RF)

| 真实类别 | 预测为 Camera | 预测为 Light_T1 | 预测为 Light_XM | 预测为 Sensor | 预测为 Socket |
|---|---|---|---|---|---|
| Camera | +1 | 0 | -1 | 0 | 0 |
| **Light_T1** | +1 | **-36** ⚠️ | **+45** ⚠️ | -10 | 0 |
| Light_XM | +14 | -26 | +12 | 0 | 0 |
| **Sensor** | +4 | **+39** ⚠️ | +15 | **-58** ⚠️ | 0 |
| Socket | 0 | 0 | 0 | 0 | 0 |

**关键问题**:
1. **Sensor 召回暴跌**: 68 → 10 (**-85%**)，大量样本被误判为 Light_T1 (+39 个样本)
2. **Light_T1 召回下降**: 88 → 52 (**-41%**)，更多样本被误判为 Light_XM (+45 个样本)
3. **Stacking 过度拟合了 "Light_T1 ↔ Sensor" 的错误混淆模式**

---

## 三、根本原因推断 (深入版)

### 3.0 Meta-learner 内部机制剖析

直接加载 Stacking 模型，查看 Meta-learner (Logistic Regression) 学到的系数：

#### 3.0.1 LORO R2+R4→R3 (崩盘场景) 的 Meta-learner 系数

```
对角线权重 (base 预测该类 → Meta 输出该类):
Target       RF       XGB      LGBM
Camera       2.258    1.726    2.222
Light_T1     2.474    1.408    0.982
Light_XM     1.657    2.386    0.725
Sensor       2.034    2.498    0.465
Socket       2.126    2.141    2.145
```

**关键发现**: LGBM 对 Sensor 的对角线权重仅 0.465 (其他都是 2+)，说明 LGBM 在 OOF 时 Sensor 预测很差，Meta 几乎不信任 LGBM 的 Sensor 预测。

#### 3.0.2 跨场景对比: Light_T1 ↔ Sensor 错误关联强度

| 场景 | 输出 Sensor 时, base 预测 Light_T1 的权重 (RF, XGB, LGBM) | 一致性 |
|---|---|---|
| **LORO R2+R4→R3 (崩盘)** | -0.969, -0.715, **+0.349** | ❌ LGBM 符号相反 |
| Single R3 (IID) | -0.326, -0.547, -0.644 | ✅ 全部为负 |
| Joint R2R3R4 | -0.615, -0.491, -0.186 | ✅ 全部为负 |
| Jitter R6+R7 | -0.398, -0.611, -0.265 | ✅ 全部为负 |

**关键发现**: 在 LORO 崩盘场景下，**只有 LGBM 权重是正的** (+0.349)，RF/XGB 都是负的。这意味着:
- Meta-learner 在训练时发现 LGBM 的预测"反着来"有时能纠错
- 但在 R3 测试时，这种"反向纠正"完全失效
- RF/XGB 的负权重主导，导致 Meta 强烈反对输出 Sensor

#### 3.0.3 模拟: 当 base models 都预测 Light_T1 (但真实是 Sensor) 时

输入概率: Camera=0.05, Light_T1=0.70, Light_XM=0.15, Sensor=0.05, Socket=0.05

| 类别 | LORO (Drift) 得分 | Single R3 (IID) 得分 |
|---|---|---|
| Light_T1 | **3.531** | **3.110** |
| Light_XM | 0.852 | -0.381 |
| Sensor | -0.528 | -0.526 |
| Camera | -1.550 | -0.923 |
| Socket | -2.305 | -1.280 |

**两个场景的 Meta 都输出 Light_T1**，因为 base models 的一致错误 (Sensor → Light_T1) 强烈主导。但 LORO 场景下 Light_XM 得分也高 (0.852)，因为 Meta 学到 "LGBM 的 Sensor 预测反着听" 反而把 Light_XM 推高了。

### 3.1 Meta-learner 的过拟合陷阱

Stacking 架构:
```
Base Learners (RF, XGBoost, LightGBM)
    ↓ 输出 3×5 = 15 维概率特征
Meta-learner (Logistic Regression with 5-fold CV)
    ↓ 在 base learners 的 OOF 预测上训练
Final Prediction
```

**问题**:
- **Base learners 在 drift 场景下都不准** (LORO R2+R4→R3 全员 F1 < 0.62)
- Base learners 的**错误模式高度相关**: 都倾向于把 Sensor 误判为 Light_T1
- Meta-learner 在小训练集 (3669 样本) 上学习这些**共同的错误**，反而强化了错误模式
- 5-fold CV 的 OOF 预测基于 R2+R4 的内部分布，**完全没见过 R3 的模式**

### 3.1.1 训练集规模 vs Stacking 表现

| 训练样本数 | 任务类型 | Delta | 说明 |
|---|---|---|---|
| 1271-1297 | Single-round | -0.0017 到 -0.0019 | 小样本但 **IID 分布**，Stacking 接近持平 |
| 3653-3690 | **LORO** | **-0.0028 到 -0.0693** | 小样本 + **严重 drift**，Stacking 崩盘 |
| 3854 | Joint | -0.0030 | 中等样本，**联合分布**，Stacking 接近持平 |
| 5506 | Jitter | -0.0001 到 -0.0053 | 较大样本 + **轻度 drift**，Stacking 接近持平 |
| 5506 | Position | -0.0334 | 较大样本 + **中等 drift**，Stacking 负效应 |

**结论**:
- **样本量不是唯一因素** (Single-round 1200 样本也能持平)
- **Drift 严重程度 + 小样本** = Stacking 噩梦
- **LORO 任务是最严重的 session drift** (跨轮次泛化失败)

### 3.3 为何 RF 在 LORO R2+R4→R3 上表现最好？

#### 各模型在 Sensor 类上的召回率:
```
RF:           68/364 = 18.7%  (最佳)
ExtraTrees:   74/364 = 20.3%  (次佳，但 Light_T1 更差)
LightGBM:     17/364 = 4.7%
XGBoost:      6/364  = 1.6%
Stacking:     10/364 = 2.7%   (崩盘)
```

#### 各模型在 Light_T1 类上的召回率:
```
RF:           88/363 = 24.2%  (最佳)
ExtraTrees:   59/363 = 16.3%
LightGBM:     51/363 = 14.0%
Stacking:     52/363 = 14.3%
XGBoost:      42/363 = 11.6%
```

**RF 的优势**:
1. **随机特征子集** 减少了对特定特征的依赖 → 更鲁棒
2. **Bagging** 集成策略在 drift 场景下比 Boosting 更稳定
3. **不追求完美拟合训练集** → 在 R2+R4 上没有过拟合错误模式

**LightGBM/XGBoost 的问题**:
- Boosting 在训练集上强化了 "Sensor → Light_T1" 的错误关联
- 测试集 R3 的分布偏移导致这种关联完全失效

**Stacking 的问题**:
- Meta-learner 学习了 **3 个 base learners 的共同错误**
- 错误加权后反而比最差的 base learner 还差
- **更严重**: Meta-learner 对 LGBM 赋予了"反向纠正"的权重 (LGBM 预测 Light_T1 → Meta 反而倾向 Sensor)，但这种"纠正"在 R3 上完全失效

### 3.4 跨轮次混淆模式漂移 (核心发现!)

**直接对比 R2, R3, R4 单独训练时的混淆模式**:

| 类别 | R2 召回率 | R3 召回率 | R4 召回率 | 主要混淆方向 |
|---|---|---|---|---|
| Camera | 100% | 98.2% | 99.2% | 无混淆 |
| Light_T1 | 89.0% | **100%** | 92.7% | R2→Light_XM, R4→Sensor |
| Light_XM | 89.9% | 90.1% | 92.7% | R2→Light_T1, R4→Sensor |
| **Sensor** | **100%** | **91.7%** | **87.2%** | **R3→Light_XM, R4→Light_T1** |
| Socket | 100% | 100% | 100% | 无混淆 |

**关键观察**:
- **R2 的 Sensor 完美分类** (100%)，无混淆
- **R3 的 Sensor 主要和 Light_XM 混淆** (9 个 → Light_XM)
- **R4 的 Sensor 同时和 Light_T1, Light_XM 混淆** (7+7 个)

**当用 R2+R4 训练时**:
- Meta-learner 看到的混淆模式是: Sensor ↔ Light_T1 (主要从 R4 学到)
- Meta-learner 从未见过 R3 特有的 "Sensor ↔ Light_XM" 模式

**在 R3 测试时**:
- R3 的 Sensor 样本，在 R2+R4 的特征空间下，变成了"像 R4 的 Sensor"
- 即: R3 的 Sensor 被 base models 分类为 Light_T1 (因为 R4 的 Sensor 经常被错认为 Light_T1)
- 232/364 = 63.7% 的 Sensor 被错认为 Light_T1
- Meta-learner 强化了这种错误关联 (因为它学到的就是 Sensor → Light_T1 的模式)

**核心洞察**: **Confusion Pattern Drift (混淆模式漂移)** 是 Stacking 崩盘的直接原因。
- R2+R4 的混淆模式: Sensor → Light_T1
- R3 的真实混淆模式: Sensor → Light_XM (在单轮训练时) 或 Sensor → Light_T1 (在 LORO 时)
- Meta-learner 永远无法适应测试集的混淆模式变化

---

## 四、其他场景分析

### 4.1 为何 Jitter 任务 Stacking 接近持平？

| 任务 | 训练样本 | Stacking F1 | Best Base | Delta |
|---|---|---|---|---|
| Jitter R2-R4→R6+R7 | 5506 | 0.7969 | 0.7970 | **-0.0001** (几乎持平) |
| Jitter R2-R4→R7 | 5506 | 0.8204 | 0.8220 | -0.0015 |
| Jitter R2-R4→R6 | 5506 | 0.7731 | 0.7784 | -0.0053 |

**原因**:
1. **Jitter 是操作抖动，不是 session drift** → base learners 的错误模式更随机，不高度相关
2. **训练集更大** (5506 vs 3669) → Meta-learner 有更多样本学习正确的加权
3. **R7 比 R6 更接近训练分布** (F1 0.82 vs 0.77) → Stacking 在 R7 上表现更好

### 4.2 为何 Position 任务 Stacking 负效应 (delta -0.0334)？

| 模型 | Macro-F1 | 说明 |
|---|---|---|
| RF | 0.7012 | 最佳 |
| LightGBM | 0.6584 | |
| Stacking | 0.6678 | **比 RF 差 3.3%** |
| XGBoost | 0.6603 | |
| ExtraTrees | 0.6423 | |

**推测**:
- Position drift (R5 vs R2-R4) 是**位置变化导致的信道特征漂移**
- Base learners 的错误模式**部分相关** (都受信道影响)
- Meta-learner 学习到的加权策略**在 R5 上不泛化**

---

## 五、总结与建议

### 5.1 Stacking 失效的四要素 (基于深度分析)

1. **Drift 严重** (跨 session/位置/时间的分布偏移)
2. **训练集小** (< 5000 样本，Meta-learner 容易过拟合)
3. **Base learners 错误高度相关** (共同的混淆模式被强化)
4. **混淆模式漂移 (Confusion Pattern Drift)**: 训练集的混淆模式与测试集完全不同

#### 5.1.1 混淆模式漂移的具体表现

**R2+R4 训练时的混淆模式**:
- Sensor → Light_T1 (R4 特有)
- Light_T1 → Light_XM (R2 特有)

**R3 测试时的混淆模式 (LORO 场景)**:
- Sensor → Light_T1 (R4 模式被"传染"给 R3)
- Light_T1 → 完全不混淆 (R3 单独训练时 100% 召回)

**Meta-learner 的灾难**:
- 它在 R2+R4 上学到 "base 预测 Light_T1, 输出 Light_T1"
- 在 R3 上, Sensor 被错误地预测为 Light_T1, Meta 强化了这个错误
- Meta-learner 完全没有见过 R3 的真实混淆模式

#### 5.1.2 论文关键论据

这个发现支撑了论文的核心论点:
- **跨场景的不仅是特征分布漂移，更是混淆模式漂移**
- **Stacking 的 Meta-learner 假设训练集和测试集的混淆模式一致，这在 drift 场景下完全不成立**
- **简单模型 (RF) 的"直接决策"反而比复杂集成 (Stacking) 更鲁棒**

### 5.2 论文结论

**不应使用 Stacking 的场景**:
- ❌ LORO 跨轮次泛化 (session drift)
- ❌ Position 跨位置部署 (信道漂移)
- ❌ 训练集 < 4000 样本 + drift 场景

**可以使用 Stacking 的场景**:
- ✅ IID 分布 (Single-round, Joint training)
- ✅ 轻度 drift + 大训练集 (Jitter R7, 5506 样本)

### 5.3 改进方向 (Future Work)

1. **Domain-adversarial Stacking**: Meta-learner 添加 domain confusion loss
2. **Uncertainty-aware Stacking**: 只在 base learners 一致时才信任 stacking
3. **Adaptive Ensemble**: 根据测试样本的 uncertainty 动态选择 base learner 或 stacking
4. **Use RF directly**: 在 drift 场景下，RF 的鲁棒性可能比复杂集成更好

---

## 附录: 可视化图表

![Stacking 崩溃分析](stacking_collapse_analysis.png)

**图表说明 (基础分析)**:
1. **左上**: 训练集规模 vs Stacking Delta (LORO 任务全线负效应)
2. **右上**: LORO 三个任务的模型对比 (Stacking 在 R2R4→R3 崩盘)
3. **左下**: LORO R2R4→R3 混淆矩阵差异热图 (Sensor 和 Light_T1 混淆加剧)
4. **右下**: 各场景 Stacking Delta 分布 (LORO 和 Position 最差)

![Stacking 元学习器深度分析](stacking_meta_learner_analysis.png)

**图表说明 (深入分析)**:
1. **左上**: LORO 崩盘场景的 Meta-learner 15 维系数矩阵
2. **右上**: 跨场景 Light_T1 ↔ Sensor 错误关联强度 (崩盘场景 LGBM 权重符号异常)
3. **左下**: 跨轮次混淆模式漂移 (R2/R3/R4 的 Sensor 召回率完全不同)
4. **右下**: 真实 Sensor 样本在 LORO R2R4→R3 上的预测分布 (Stacking 最差)
