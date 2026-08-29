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
| CNN-Inception | 763092.0000  | 0.9177 | 0.5997 | 0.3180   | 0.8226 | -0.0133         | 0.0138          | IID_lower_OOD_higher  |
| CNN-TCN       | 814854.0000  | 0.9297 | 0.6020 | 0.3277   | 0.8177 | -0.0013         | 0.0161          | IID_lower_OOD_higher  |
| CNN-ConvNeXt  | 1079570.0000 | 0.9208 | 0.5869 | 0.3339   | 0.8152 | -0.0102         | 0.0010          | IID_lower_OOD_higher  |

## Scenario Performance

| IID    | OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ------ | ---------- | -------- | ------------ | ------------- |
| 0.9310 | 0.6481     | 0.6155   | 0.4347       | CNN-v3        |
| 0.9199 | 0.6487     | 0.6162   | 0.4412       | CNN-v5        |
| 0.9177 | 0.6589     | 0.6132   | 0.5000       | CNN-Inception |
| 0.9297 | 0.6531     | 0.6291   | 0.4694       | CNN-TCN       |
| 0.9208 | 0.6179     | 0.5966   | 0.5266       | CNN-ConvNeXt  |

## Scenario CPD

| OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ---------- | -------- | ------------ | ------------- |
| 0.7914     | 0.7306   | 1.2654       | CNN-v3        |
| 0.7946     | 0.6980   | 1.2922       | CNN-v5        |
| 0.7235     | 0.7564   | 1.1203       | CNN-Inception |
| 0.6538     | 0.7247   | 1.2607       | CNN-TCN       |
| 0.7920     | 0.7353   | 1.0782       | CNN-ConvNeXt  |

## Contrast Hits

| Display       | Params       | IID    | OOD    | OOD_Drop | CPD    | delta_iid_vs_v3 | delta_ood_vs_v3 | relation_vs_v3       |
| ------------- | ------------ | ------ | ------ | -------- | ------ | --------------- | --------------- | -------------------- |
| CNN-v5        | 1225296.0000 | 0.9199 | 0.5877 | 0.3322   | 0.8362 | -0.0111         | 0.0018          | IID_lower_OOD_higher |
| CNN-Inception | 763092.0000  | 0.9177 | 0.5997 | 0.3180   | 0.8226 | -0.0133         | 0.0138          | IID_lower_OOD_higher |
| CNN-TCN       | 814854.0000  | 0.9297 | 0.6020 | 0.3277   | 0.8177 | -0.0013         | 0.0161          | IID_lower_OOD_higher |
| CNN-ConvNeXt  | 1079570.0000 | 0.9208 | 0.5869 | 0.3339   | 0.8152 | -0.0102         | 0.0010          | IID_lower_OOD_higher |
