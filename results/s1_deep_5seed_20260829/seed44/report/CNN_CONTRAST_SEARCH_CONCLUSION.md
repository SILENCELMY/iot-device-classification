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
| CNN-Inception | 763092.0000  | 0.9115 | 0.5937 | 0.3177   | 0.8072 | -0.0196         | 0.0079          | IID_lower_OOD_higher  |
| CNN-TCN       | 814854.0000  | 0.9295 | 0.6056 | 0.3240   | 0.8191 | -0.0015         | 0.0197          | IID_lower_OOD_higher  |
| CNN-ConvNeXt  | 1079570.0000 | 0.9202 | 0.5911 | 0.3290   | 0.8313 | -0.0108         | 0.0053          | IID_lower_OOD_higher  |

## Scenario Performance

| IID    | OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ------ | ---------- | -------- | ------------ | ------------- |
| 0.9310 | 0.6481     | 0.6155   | 0.4347       | CNN-v3        |
| 0.9199 | 0.6487     | 0.6162   | 0.4412       | CNN-v5        |
| 0.9115 | 0.6686     | 0.6102   | 0.4696       | CNN-Inception |
| 0.9295 | 0.6525     | 0.6262   | 0.4966       | CNN-TCN       |
| 0.9202 | 0.6145     | 0.6160   | 0.4932       | CNN-ConvNeXt  |

## Scenario CPD

| OOD_Jitter | OOD_LORO | OOD_Position | Display       |
| ---------- | -------- | ------------ | ------------- |
| 0.7914     | 0.7306   | 1.2654       | CNN-v3        |
| 0.7946     | 0.6980   | 1.2922       | CNN-v5        |
| 0.6311     | 0.7206   | 1.2432       | CNN-Inception |
| 0.7082     | 0.7378   | 1.1738       | CNN-TCN       |
| 0.7819     | 0.7336   | 1.1737       | CNN-ConvNeXt  |

## Contrast Hits

| Display       | Params       | IID    | OOD    | OOD_Drop | CPD    | delta_iid_vs_v3 | delta_ood_vs_v3 | relation_vs_v3       |
| ------------- | ------------ | ------ | ------ | -------- | ------ | --------------- | --------------- | -------------------- |
| CNN-v5        | 1225296.0000 | 0.9199 | 0.5877 | 0.3322   | 0.8362 | -0.0111         | 0.0018          | IID_lower_OOD_higher |
| CNN-Inception | 763092.0000  | 0.9115 | 0.5937 | 0.3177   | 0.8072 | -0.0196         | 0.0079          | IID_lower_OOD_higher |
| CNN-TCN       | 814854.0000  | 0.9295 | 0.6056 | 0.3240   | 0.8191 | -0.0015         | 0.0197          | IID_lower_OOD_higher |
| CNN-ConvNeXt  | 1079570.0000 | 0.9202 | 0.5911 | 0.3290   | 0.8313 | -0.0108         | 0.0053          | IID_lower_OOD_higher |
