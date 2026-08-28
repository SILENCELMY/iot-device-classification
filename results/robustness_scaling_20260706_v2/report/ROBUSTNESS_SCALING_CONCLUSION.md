# Robustness Scaling Experiment - Stage 4

## 实验协议

- 特征缓存：`results/robust_v2/raw_all/features_raw_all_w10.csv`
- 数据窗口：10s non-overlap，沿用主线 feature cache
- Split artifact：`splits/<task>/train_idx.npy`, `val_idx.npy`, `test_idx.npy`
- Validation policy：当前训练框架不做模型选择/early stopping，`val_idx.npy` 保留为空数组
- 标准化：CNN-v4 使用 `StandardScaler.fit(train)`，再 transform test；复制 baseline 保留原实验产物
- 深度训练：epochs=70, batch_size=128, lr=0.0009, device=cuda

## 最终结果

| Model          | Params  | IID    | OOD    | OOD_Drop | CPD    |
| -------------- | ------- | ------ | ------ | -------- | ------ |
| RF             | N/A     | 0.9537 | 0.7142 | 0.2396   | 0.5973 |
| CNN-v2         | 260049  | 0.7801 | 0.5310 | 0.2491   | 0.7238 |
| CNN-v3         | 1012154 | 0.9310 | 0.5859 | 0.3451   | 0.8497 |
| CNN-v4         | 2206080 | 0.9177 | 0.5700 | 0.3477   | 0.8762 |
| Transformer-v1 | 73541   | 0.8993 | 0.6223 | 0.2770   | 0.7199 |
| Transformer-v2 | 605634  | 0.9292 | 0.5942 | 0.3350   | 0.8236 |

## OOD 场景拆解

| Model          | IID    | OOD_Jitter | OOD_LORO | OOD_Position |
| -------------- | ------ | ---------- | -------- | ------------ |
| RF             | 0.9537 | 0.7858     | 0.6946   | 0.7012       |
| CNN-v2         | 0.7801 | 0.6383     | 0.4882   | 0.5519       |
| CNN-v3         | 0.9310 | 0.6481     | 0.6155   | 0.4347       |
| CNN-v4         | 0.9177 | 0.6461     | 0.6072   | 0.3825       |
| Transformer-v1 | 0.8993 | 0.6930     | 0.6181   | 0.5643       |
| Transformer-v2 | 0.9292 | 0.6659     | 0.6230   | 0.4361       |

## CPD 场景拆解

| Model          | OOD_Jitter | OOD_LORO | OOD_Position |
| -------------- | ---------- | -------- | ------------ |
| RF             | 0.3404     | 0.6930   | 0.5670       |
| CNN-v2         | 0.5450     | 0.7325   | 0.8764       |
| CNN-v3         | 0.7914     | 0.7306   | 1.2654       |
| CNN-v4         | 0.7212     | 0.7501   | 1.4094       |
| Transformer-v1 | 0.5620     | 0.7366   | 0.8276       |
| Transformer-v2 | 0.6807     | 0.7113   | 1.3037       |

## 核心判断

1. **CNN-v4 是否继续 CPD ↑ / OOD Drop ↑？** CNN-v3 CPD=0.8497, OOD Drop=0.3451；CNN-v4 CPD=0.8762, OOD Drop=0.3477。

2. **是否出现 OOD robustness recovery？** CNN-v3 OOD=0.5859；CNN-v4 OOD=0.5700。

3. **CNN-v4 未完全满足本轮 scaling 前提。** 虽然参数量继续增加，但 IID 没有超过 CNN-v3，说明更大的结构没有自动转化为更有效的 IID representation。在这个前提下，CNN-v4 的 OOD 低于 CNN-v3，CPD/OOD Drop 高于 CNN-v3，因此没有出现 invariant representation emergence。

4. **Phase transition 判断：** 本轮没有观察到 OOD robustness recovery，也没有观察到 CPD 下降/稳定。结果更接近 Hypothesis A 的方向，但由于 CNN-v4 的 IID 没有超过 CNN-v3，应表述为“继续增大 CNN capacity 没有带来 phase transition”，而不是“更强有效表示已经形成后仍然失败”。

## 理论解释

本阶段把 CNN capacity 继续推过 CNN-v3，用于检验 topology sensitivity 是否持续单调增加。最终结论不以最高 IID 为依据，而以 OOD recovery 与 CPD 是否下降/稳定作为判断 invariant representation emergence 的核心证据。
