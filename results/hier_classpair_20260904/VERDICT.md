# HIER-CLASSPAIR 判定书

**判定**：`HIER_FAILS`　条件1 3/6　条件2 1/3

协议 sha256 `ebb2eb642a322337012e5a5a6a6ac62df7fb77054b7a7f01936cf76191738297`　种子 [42, 43, 44, 45, 46]

## §3 簇导出（源域程序性，τ=0.1）

导出簇 **[np.str_('Light_T1'), np.str_('Light_XM'), np.str_('Sensor')]**；等于预期 `['Light_T1', 'Light_XM', 'Sensor']`？**True**；跨模型一致？True；跨种子一致？True

## §4 源域侧选出（只看 inner）

`hier` = **rf + xgboost**；`flat_full94` = rf；`flat_drop_rssi` = lightgbm

## 主判据（outer 三单元，逐类，五种子均值）

| 单元 | 条件 | 类 | hier | flat_best | 通过 |
|---|:-:|---|---:|---:|:-:|
| `pos_R5` | 1 | Camera | 0.8369 | 0.8709 | ✗ |
| `pos_R5` | 1 | Socket | 1.0000 | 1.0000 | ✓ |
| `pos_R5` | 2 | Light_T1|Light_XM|Sensor | 0.7162 | 0.6808 | ✓ |
| `jit_R6` | 1 | Camera | 0.4713 | 0.8131 | ✗ |
| `jit_R6` | 1 | Socket | 1.0000 | 1.0000 | ✓ |
| `jit_R6` | 2 | Light_T1|Light_XM|Sensor | 0.6647 | 0.6820 | ✗ |
| `jit_R7` | 1 | Camera | 0.8394 | 0.9402 | ✗ |
| `jit_R7` | 1 | Socket | 1.0000 | 1.0000 | ✓ |
| `jit_R7` | 2 | Light_T1|Light_XM|Sensor | 0.7551 | 0.7697 | ✗ |

## 并报：macro 与设备级

| 单元 | hier macro | V1 oracle macro | hier 设备级 5min |
|---|---:|---:|---:|
| `pos_R5` | 0.7971 | 0.6959 | 0.9294 |
| `jit_R6` | 0.6931 | 0.7593 | 0.8574 |
| `jit_R7` | 0.8210 | 0.8260 | 1.0000 |

## 偏离
- §3 要求用 §4 选出的第一层模型导出簇，§4 要求按 §5 分层实现选模型 —— 循环依赖。本实现用三 inner 单元 flat full94 预测导出簇（与 §4 解耦），三基模型要求一致、不一致取交集；其余 §3 条款照执行。见 runner 文件头。