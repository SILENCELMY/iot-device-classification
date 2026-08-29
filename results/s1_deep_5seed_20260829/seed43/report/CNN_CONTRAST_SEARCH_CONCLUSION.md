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
| CNN-Inception | 763092.0000  | 0.9213 | 0.6071 | 0.3143   | 0.7752 | -0.0097         | 0.0212          | IID_lower_OOD_higher  |
| CNN-TCN       | 814854.0000  | 0.9243 | 0.6013 | 0.3231   | 0.8073 | -0.0067         | 0.0154          | IID_lower_OOD_higher  |
| CNN-ConvNeXt  | 1079570.0000 | 0.9192 | 0.5689 | 0.3503   | 0.9027 | -0.0118         | -0.0170         | same_direction_or_tie |

## Scenario Performance

| IID    | OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ------ | ---------- | -------- | ------------ | ------------- |
| 0.9310 | 0.6481     | 0.6155   | 0.4347       | CNN-v3        |
| 0.9199 | 0.6487     | 0.6162   | 0.4412       | CNN-v5        |
| 0.9213 | 0.6908     | 0.6090   | 0.5175       | CNN-Inception |
| 0.9243 | 0.6655     | 0.6153   | 0.4950       | CNN-TCN       |
| 0.9192 | 0.5615     | 0.6086   | 0.4570       | CNN-ConvNeXt  |

## Scenario CPD

| OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ---------- | -------- | ------------ | ------------- |
| 0.7914     | 0.7306   | 1.2654       | CNN-v3        |
| 0.7946     | 0.6980   | 1.2922       | CNN-v5        |
| 0.6303     | 0.7147   | 1.1014       | CNN-Inception |
| 0.7467     | 0.7287   | 1.1038       | CNN-TCN       |
| 0.9414     | 0.7828   | 1.2235       | CNN-ConvNeXt  |

## Contrast Hits

| Display       | Params       | IID    | OOD    | OOD_Drop | CPD    | delta_iid_vs_v3 | delta_ood_vs_v3 | relation_vs_v3       |
| ------------- | ------------ | ------ | ------ | -------- | ------ | --------------- | --------------- | -------------------- |
| CNN-v5        | 1225296.0000 | 0.9199 | 0.5877 | 0.3322   | 0.8362 | -0.0111         | 0.0018          | IID_lower_OOD_higher |
| CNN-Inception | 763092.0000  | 0.9213 | 0.6071 | 0.3143   | 0.7752 | -0.0097         | 0.0212          | IID_lower_OOD_higher |
| CNN-TCN       | 814854.0000  | 0.9243 | 0.6013 | 0.3231   | 0.8073 | -0.0067         | 0.0154          | IID_lower_OOD_higher |
