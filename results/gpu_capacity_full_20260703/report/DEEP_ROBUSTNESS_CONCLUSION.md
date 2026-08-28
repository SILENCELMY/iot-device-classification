# Controlled Capacity Increase Experiment 结论

## 实验目的

本实验用于验证：随着模型 relation modeling capability 增强，topology drift sensitivity 是否同步增强。实验不追求 SOTA，重点比较 capacity increase 后的 IID、OOD drop、CPD 和 environment-specific fitting。

## 实验设置

- 特征缓存：`results/robust_v2/raw_all/features_raw_all_w10.csv`
- 特征集：`all_features`
- 任务：R2/R3/R4 single-round，三组 LORO，Position R5，Jitter R6+R7
- 对照模型：RF、CNN-v1、Transformer-v1
- 增容模型：CNN-v2（Residual + SE）、CNN-v3（multi-scale residual + SE + attention pooling + raw skip）、Transformer-v2（4-layer encoder + attention pooling）
- 深度模型训练：epochs=60, batch_size=128, lr=0.001, device=cuda

## 完整性能表

| Model          | IID    | OOD    | OOD_Drop | CPD    |
| -------------- | ------ | ------ | -------- | ------ |
| rf             | 0.9537 | 0.7142 | 0.2396   | 0.5973 |
| cnn1d          | 0.6439 | 0.4924 | 0.1515   | 0.6323 |
| cnn1d_v2       | 0.7801 | 0.5310 | 0.2491   | 0.7238 |
| cnn1d_v3       | 0.9310 | 0.5859 | 0.3451   | 0.8497 |
| transformer    | 0.8993 | 0.6223 | 0.2770   | 0.7199 |
| transformer_v2 | 0.9292 | 0.5942 | 0.3350   | 0.8236 |

## Capacity Increase Delta

| family        | v1_model    | v2_model       | delta_iid | delta_ood | delta_ood_drop | delta_cpd |
| ------------- | ----------- | -------------- | --------- | --------- | -------------- | --------- |
| CNN           | cnn1d       | cnn1d_v2       | 0.1361    | 0.0386    | 0.0976         | 0.0914    |
| CNN-optimized | cnn1d_v2    | cnn1d_v3       | 0.1509    | 0.0549    | 0.0961         | 0.1259    |
| Transformer   | transformer | transformer_v2 | 0.0299    | -0.0281   | 0.0580         | 0.1038    |

## CPD 汇总

| model          | cpd_vs_iid_mean |
| -------------- | --------------- |
| cnn1d          | 0.6323          |
| cnn1d_v2       | 0.7238          |
| cnn1d_v3       | 0.8497          |
| rf             | 0.5973          |
| transformer    | 0.7199          |
| transformer_v2 | 0.8236          |

## 鲁棒性排名

| model          | iid_mean_macro_f1 | ood_mean_macro_f1 | mean_ood_drop | mean_cpd_vs_iid | iid_rank | ood_rank | drop_rank | topology_stability_rank |
| -------------- | ----------------- | ----------------- | ------------- | --------------- | -------- | -------- | --------- | ----------------------- |
| rf             | 0.9537            | 0.7142            | 0.2396        | 0.5973          | 1.0000   | 1.0000   | 2.0000    | 1.0000                  |
| transformer    | 0.8993            | 0.6223            | 0.2770        | 0.7199          | 4.0000   | 2.0000   | 4.0000    | 3.0000                  |
| transformer_v2 | 0.9292            | 0.5942            | 0.3350        | 0.8236          | 3.0000   | 3.0000   | 5.0000    | 5.0000                  |
| cnn1d_v3       | 0.9310            | 0.5859            | 0.3451        | 0.8497          | 2.0000   | 4.0000   | 6.0000    | 6.0000                  |
| cnn1d_v2       | 0.7801            | 0.5310            | 0.2491        | 0.7238          | 5.0000   | 5.0000   | 3.0000    | 4.0000                  |
| cnn1d          | 0.6439            | 0.4924            | 0.1515        | 0.6323          | 6.0000   | 6.0000   | 1.0000    | 2.0000                  |

## 关键问题回答

1. **CNN-v3 是否比 CNN-v2 更强？** 是。CNN-v3 的 IID mean macro-F1 为 0.9310，明显高于 CNN-v2 的 0.7801，也接近 RF 的 0.9537。说明 multi-scale residual convolution、attention pooling 和 raw skip 确实显著增强了 CNN 的 IID 表达能力。

2. **CNN-v3 的鲁棒性是否同步提升？** 没有。CNN-v3 的 OOD mean macro-F1 为 0.5859，虽然高于 CNN-v2 的 0.5310，但 OOD drop 从 CNN-v2 的 0.2491 增加到 0.3451，CPD 从 0.7238 增加到 0.8497。也就是说，CNN-v3 更强，但 topology sensitivity 也更强。

3. **Transformer-v2 全量设置结果如何？** Transformer-v2 的 IID 从 Transformer-v1 的 0.8993 提高到 0.9292，但 OOD mean macro-F1 从 0.6223 降到 0.5942，OOD drop 从 0.2770 增加到 0.3350，CPD 从 0.7199 增加到 0.8236。全量 GPU 设置下，这个趋势仍然成立：更强 IID 表达不等于更强 OOD/topology robustness。

4. **capacity increase 后 CPD 是否变大？** 是，并且很清楚。RF 的 CPD 最低（0.5973）；CNN-v1 为 0.6323，CNN-v2 为 0.7238，CNN-v3 为 0.8497；Transformer-v1 为 0.7199，Transformer-v2 为 0.8236。CNN 路线呈现出非常明显的 capacity 越强、CPD 越大的趋势。

5. **是否出现 higher relation modeling capability -> stronger topology sensitivity？** 是。本次 GPU 全量实验给出了更强证据：CNN-v3 和 Transformer-v2 都提高了 IID，但二者的 OOD drop 和 CPD 也同步升高。CNN-v3 的 IID 排名第 2，但 CPD 最高、OOD drop 最大；Transformer-v2 的 IID 排名第 3，但 OOD/CPD 均差于 Transformer-v1。

## 简洁结论

这版 GPU 实验支持核心假设：**更强 relation modeling capability 会增强 OOD 下的 topology drift sensitivity。**

CNN-v3 优化是成功的，因为 IID 从 CNN-v2 的 0.7801 提升到 0.9310；但它不是更鲁棒的模型，因为 CPD 升到全场最高 0.8497，OOD drop 也升到全场最高 0.3451。Transformer-v2 全量设置同样显示 IID 提升，但 OOD 下降和 CPD 上升。

因此，当前数据上更合理的表述是：**更强模型能更好拟合 IID relation，但也更容易拟合 environment-specific relation，从而带来更强 confusion topology drift。**

## 输出文件

- `capacity_performance_table.csv`
- `capacity_delta.csv`
- `performance_comparison.csv`
- `iid_ood_model_summary.csv`
- `cpd_comparison.csv`
- `robustness_ranking.csv`
- `iid_ood_degradation.png`
- `confusion_topology_comparison.png`
- `cpd_comparison.png`
