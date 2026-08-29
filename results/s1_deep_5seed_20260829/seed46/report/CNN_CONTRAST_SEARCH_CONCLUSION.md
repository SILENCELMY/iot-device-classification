# CNN Architecture IID/OOD Contrast Search

## Protocol

- Features: `results/robust_v2/raw_all/features_raw_all_w10.csv`
- Split source: `results/robustness_scaling_20260706_v2/splits`
- All candidates reuse the same train/test indices and train-only StandardScaler.
- Training: epochs=70, batch_size=128, lr=0.0009, weight_decay=0.0001, device=cuda

## Main Table

| Display       | Params       | IID    | OOD    | OOD_Drop | CPD    | delta_iid_vs_v3 | delta_ood_vs_v3 | relation_vs_v3        |
| ------------- | ------------ | ------ | ------ | -------- | ------ | --------------- | --------------- | --------------------- |
| CNN-v3        | 1012154.0000 | 0.9310 | 0.5859 | 0.3451   | 0.8497 | 0.0000          | 0.0000          | same_direction_or_tie |
| CNN-v5        | 1225296.0000 | 0.9199 | 0.5877 | 0.3322   | 0.8362 | -0.0111         | 0.0018          | IID_lower_OOD_higher  |
| CNN-Inception | 763092.0000  | 0.9186 | 0.6062 | 0.3124   | 0.7672 | -0.0124         | 0.0203          | IID_lower_OOD_higher  |
| CNN-TCN       | 814854.0000  | 0.9314 | 0.5927 | 0.3387   | 0.8417 | 0.0004          | 0.0068          | same_direction_or_tie |
| CNN-ConvNeXt  | 1079570.0000 | 0.9268 | 0.5801 | 0.3467   | 0.8473 | -0.0042         | -0.0057         | same_direction_or_tie |

## Scenario Performance

| IID    | OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ------ | ---------- | -------- | ------------ | ------------- |
| 0.9310 | 0.6481     | 0.6155   | 0.4347       | CNN-v3        |
| 0.9199 | 0.6487     | 0.6162   | 0.4412       | CNN-v5        |
| 0.9186 | 0.6492     | 0.6214   | 0.5176       | CNN-Inception |
| 0.9314 | 0.6185     | 0.6268   | 0.4646       | CNN-TCN       |
| 0.9268 | 0.6258     | 0.6083   | 0.4501       | CNN-ConvNeXt  |

## Scenario CPD

| OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ---------- | -------- | ------------ | ------------- |
| 0.7914     | 0.7306   | 1.2654       | CNN-v3        |
| 0.7946     | 0.6980   | 1.2922       | CNN-v5        |
| 0.6227     | 0.7209   | 1.0506       | CNN-Inception |
| 0.8839     | 0.6950   | 1.2397       | CNN-TCN       |
| 0.7589     | 0.7520   | 1.2217       | CNN-ConvNeXt  |

## Contrast Hits

| Display       | Params       | IID    | OOD    | OOD_Drop | CPD    | delta_iid_vs_v3 | delta_ood_vs_v3 | relation_vs_v3       |
| ------------- | ------------ | ------ | ------ | -------- | ------ | --------------- | --------------- | -------------------- |
| CNN-v5        | 1225296.0000 | 0.9199 | 0.5877 | 0.3322   | 0.8362 | -0.0111         | 0.0018          | IID_lower_OOD_higher |
| CNN-Inception | 763092.0000  | 0.9186 | 0.6062 | 0.3124   | 0.7672 | -0.0124         | 0.0203          | IID_lower_OOD_higher |
