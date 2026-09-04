# UNSW-BELKIN-CAPACITY 判定书

**判定**：`PARTIALLY_SEPARABLE`

协议 sha256 `dc58f07725f988c2dca962f3b33e33cd675d84563f6380d097877300b80db630`　种子 [42, 43, 44, 45, 46]　H = 前 14 天　T（6 天）白名单拦截

## 主判据与正控

| 单元 | 类对 | 特征集 | F1（种子→天→模型 max） | 门槛 |
|---|---|---|---:|---|
| **A** 主判据 | BelkinWemoMotion vs BelkinWemoSwitch | UNSW 61 | **0.6504** | ≥0.80 / 0.60–0.80 / <0.60 |
| **B** 正控 | Dropcam vs SamsungSmartCam | UNSW 61 | 0.9996 | ≥0.90 否则 INVALID |
| **C** 参照 | Light_T1 vs Light_XM | 自采 full94 | 0.9283 | 只报不设门槛 |

逐模型（单元 A）：`lightgbm` 0.6483　`rf` 0.6498　`xgboost` 0.6504

单元 A 跨天对照（§6.3，单种子）最大 0.5785；域内 − 跨天 = +0.0719　→ 预期两者接近（可分性问题而非漂移问题）

单元 A − 单元 C = -0.2779　**跨数据集比较，不构成受控对比（§6.5）**


## 逐列单变量 AUC（协议 §6，只报不设门槛）

最强单列 `down_len_mean`　AUC 0.6168　强度 |AUC−0.5| = 0.1168

| 族 | 族内最强 \|AUC−0.5\| |
|---|---:|
| `down` | 0.1168 |
| `up` | 0.1123 |
| `len` | 0.0337 |
| `subwin` | 0.0310 |
| `singletons` | 0.0310 |
| `burst` | 0.0282 |
| `interarrival` | 0.0220 |

top-10 单列：

| 列 | AUC | \|AUC−0.5\| |
|---|---:|---:|
| `down_len_mean` | 0.6168 | 0.1168 |
| `up_len_p50` | 0.3877 | 0.1123 |
| `up_down_ratio` | 0.4027 | 0.0973 |
| `down_packet_ratio` | 0.5972 | 0.0972 |
| `up_packet_ratio` | 0.4028 | 0.0972 |
| `down_len_p50` | 0.5828 | 0.0828 |
| `up_len_mean` | 0.4268 | 0.0732 |
| `down_len_std` | 0.5428 | 0.0428 |
| `len_cv` | 0.5337 | 0.0337 |
| `byte_count` | 0.5310 | 0.0310 |