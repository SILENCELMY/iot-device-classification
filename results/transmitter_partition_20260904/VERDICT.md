# TRANSMITTER-PARTITION-VALIDATE 判定书

**判定对**：A = `PARTITION_PARTIAL`（9/10）　B = `DEVICE_SIGNAL_SAME_DAY`（`F_gw` = 0.6820）

协议 sha256 `d0ad95186383e295f91d34787f4c95b4a999139fbfdef986b8df576bdc3cfd43`　种子 [42, 43, 44, 45, 46]　模型 ['rf', 'xgboost', 'lightgbm']（`stacking` 不进本协议）

单元：源域恒 R2+R3+R4，目标域 R5 / R6 / R7　τ = 0.01　饱和阈值 = 0.999

**烧毁隔离**：A 的预测所依据的符号模式先于本协议被看到，观测来自 9 个 g0 任务（目标域全在 R2/R3/R4 内），本协议不使用；R5/R6/R7 从未被该诊断读取。

## A：符号能否还原发射机划分

| 类对 | 同发射机 | `full94` 基线 AUC 均值 | 饱和 | `D`（三单元求和） | 观测符号 | 预测符号 | 依据规则 | 命中 |
|---|---|---:|---|---:|---|---|---|---|
| `Light_T1|Light_XM` | 是 | 0.4902 | 否 | +0.6460 | `+` | `+` | rule1_same_transmitter | ✓ |
| `Light_T1|Sensor` | 是 | 0.7087 | 否 | +0.4283 | `+` | `+` | rule1_same_transmitter | ✓ |
| `Light_XM|Sensor` | 是 | 0.9054 | 否 | +0.0038 | `0` | `+` | rule1_same_transmitter | ✗ |
| `Sensor|Socket` | 否 | 1.0000 | 是 | +0.0000 | `0` | `0` | rule2_saturated | ✓ |
| `Light_XM|Socket` | 否 | 1.0000 | 是 | +0.0000 | `0` | `0` | rule2_saturated | ✓ |
| `Light_T1|Socket` | 否 | 1.0000 | 是 | +0.0000 | `0` | `0` | rule2_saturated | ✓ |
| `Camera|Socket` | 否 | 0.9991 | 是 | -0.0000 | `0` | `0` | rule2_saturated | ✓ |
| `Camera|Light_T1` | 否 | 0.9428 | 否 | -0.1597 | `-` | `-` | rule3_diff_transmitter | ✓ |
| `Camera|Sensor` | 否 | 0.9898 | 否 | -0.2055 | `-` | `-` | rule3_diff_transmitter | ✓ |
| `Camera|Light_XM` | 否 | 0.9817 | 否 | -0.3142 | `-` | `-` | rule3_diff_transmitter | ✓ |

**`hits` = 9/10** → `PARTITION_PARTIAL`（门槛：10/10 还原、8–9 部分、≤7 否证；随机命中 1/3，`P(10/10 | 随机) ≈ 1.7e-05`）

A 逐五种子完全相同：**True**（`pair_auc` 用 lbfgs，确定性由此被检验而非假定）

## B：同日单元 R6（05-26，三个网关设备同日）的逐类 F1

| 单元 | 配置 | Camera | Light_T1 | Light_XM | Sensor | Socket | macro | `F_gw` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `jit_R6` | `full94` | 0.9022 | 0.6231 | 0.6528 | 0.7153 | 1.0000 | 0.7787 | 0.6637 |
| `jit_R6` | `full94_minus_rssi` | 0.6711 | 0.6214 | 0.6421 | 0.7826 | 1.0000 | 0.7434 | 0.6820 |
| `jit_R7` | `full94` | 0.9402 | 0.6913 | 0.7181 | 0.7677 | 1.0000 | 0.8235 | 0.7257 |
| `jit_R7` | `full94_minus_rssi` | 0.8708 | 0.7299 | 0.7773 | 0.8017 | 1.0000 | 0.8359 | 0.7696 |
| `pos_R5` | `full94` | 0.8709 | 0.2833 | 0.6827 | 0.6430 | 1.0000 | 0.6960 | 0.5363 |
| `pos_R5` | `full94_minus_rssi` | 0.6260 | 0.5491 | 0.7869 | 0.7059 | 1.0000 | 0.7336 | 0.6806 |

**主判据 B** = 单元 `jit_R6`、配置 `full94_minus_rssi` 下 ['Light_T1', 'Light_XM', 'Sensor'] 三类逐类 F1 均值 = **0.6820** → `DEVICE_SIGNAL_SAME_DAY`（门槛 0.50 = 1.5×(1/3)、0.40 = 1.2×(1/3)；1/3 为三类不可区分时的退化值）

**`Camera` 的已知代价**（`jit_R6`）：`full94` 0.9022 → `full94_minus_rssi` 0.6711（Δ -0.2311）

## 并报

- 拓扑真值：30 个 pcap 逐文件按 `wlan.ta` 计数落入 `topology.json`；**自有发射机恒为最大非 AP 站点 = True**
- 跨标签污染（三个网关文件）：来自 Camera MAC 5437 帧、来自 Socket MAC 3279 帧，占网关帧 4.17%
- 逐 (类对, 单元) 明细 30 行落入 `pair_task_dauc.csv`；逐类 F1 全表 120 行（5 类 × 3 单元 × 2 配置 × 5 种子 × 3 模型 + best_base）
- 未饱和的 `Socket` 类对：0 个（无，规则 ② 全部命中）

## 残余未控项（协议 §1.4，如实记）

- R6 三场的小时仍不同（`light_T1` 17h / `light_xm` 16h / `sensor` 14h）；彻底解开「设备身份 vs 会话身份」需会话交织式重采，本协议不做。
- 缓解证据：三个网关类各自恰好 **3 场上午 / 3 场下午**，故时段不构成跨轮次仍存活的逐标签关联。

## 硬门自检

- 6.1 `full94` 缓存只读，md5 `703984b6ad2fde2f45e0cce1c6df31be` 与协议 §2 声明值一致，先于任何数字核对
- 6.2 预注册（拓扑真值、10 条预测、τ、全部门槛）先于任何 AUC/F1 数字落入 `prereg.json`
- 6.3 烧毁隔离：A 的单元目标域 ⊆ {R5,R6,R7}，以 `assert` 保证；9 个 g0 任务未调用
- 6.4 `Data`/`pair_auc`/`derive_families`/`make_model`/`clean_x` 逐字复用，未重实现
- 6.7 只测不选：未调用 `accept_family`、未产生 `removed` 列表、未做配置选择