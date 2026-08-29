# FEATURE_ALIGNMENT.md — UNSW 通用特征提取器 ↔ 主线实现 对齐表

**生成日期**: 2026-08-29
**本脚本**: `code/scripts/core/extract_features_generic.py`（协议 §20.1 新建项，Ethernet / UNSW）
**主线实现**: `code/scripts/core/robust_iot_research.py` → `summarize_window()`（L241-439，802.11 / 自采数据）
**协议依据**: §16.4「特征不必与自采数据对齐 …… 只用约 60 维通用特征
（`len_*` / `interarrival_*` / `burst_*` / `subwin_*` / `up/down_*`）」

---

## 0. 结论摘要

| 项 | 值 |
|---|---|
| 主线 `summarize_window` 数值特征总数 | **94**（对应 §16.4 所称「那 94 个 802.11 特征」） |
| 本脚本数值特征总数 | **61** |
| 其中**同名同定义**（逐字复制主线代码路径） | **56** |
| 其中**同名、定义语义等价但机制改写**（链路层不同） | **3**（`up_packet_ratio` / `down_packet_ratio` / `up_down_ratio`） |
| 其中**同名但在 Ethernet 上退化为常量 0** | **2**（`side_packet_ratio` / `other_packet_ratio`） |
| 主线有、本脚本**故意不做**（802.11 专属） | **33** |
| 本脚本新增的主线没有的特征 | **0** |

分方向的 11 个 len/ia 统计（`up_len_*` / `down_len_*` / `up_ia_*` / `down_ia_*` /
`len_up_down_diff`）计算公式与主线逐字一致，但其输入依赖上面那 3 个方向判定，
故归入「同名同公式、方向来源改写」一栏（见 §3）。

**数值一致性保证**：`quantile` / `safe_mean` / `safe_std` / `coefficient_of_variation`
四个统计辅助函数在本脚本中**逐字复制**主线 L223-238，包括 `std(ddof=0)`、
`len(values) > 1` 才算 std、空序列 quantile 返回 0.0、cv 分母 `abs(mean) > 1e-12` 的
全部边界行为。窗口切分（`floor(relative_time / window_seconds)`）、
`min_packets_per_window` 默认 2、`window_seconds` 默认 10.0 亦与主线 L77-78、L460 一致。

---

## 1. 同名同定义（56 项，逐字复制主线实现）

### 1.1 基础量（2）

| 本脚本 | 主线 | 同定义 | 主线行号 |
|---|---|---|---|
| `packet_count` | `packet_count` | 是 | L275 |
| `byte_count` | `byte_count` | 是 | L276 |

### 1.2 `len_*`（13）

| 本脚本 | 主线 | 同定义 | 主线行号 |
|---|---|---|---|
| `len_mean` | `len_mean` | 是 | L277 |
| `len_std` | `len_std` | 是（`ddof=0`） | L278 |
| `len_min` | `len_min` | 是 | L279 |
| `len_max` | `len_max` | 是 | L280 |
| `len_range` | `len_range` | 是 | L281 |
| `len_cv` | `len_cv` | 是 | L282 |
| `len_p10` | `len_p10` | 是 | L283 |
| `len_p25` | `len_p25` | 是 | L284 |
| `len_p50` | `len_p50` | 是 | L285 |
| `len_p75` | `len_p75` | 是 | L286 |
| `len_p90` | `len_p90` | 是 | L287 |
| `len_p95` | `len_p95` | 是 | L288 |
| `len_iqr` | `len_iqr` | 是 | L289 |

**口径注**：主线 `lengths = group["length"]` 来自 `frame.len`；本脚本同样取 `frame.len`。
802.11 的 `frame.len` 含 radiotap + 802.11 头，Ethernet 的 `frame.len` 含 Ethernet 头 ——
**同一字段名、同一含义（链路层整帧字节数），但绝对数值不可跨数据集直接比较**。
这不影响 pilot 目的（每个数据集内部各自建模）。

### 1.3 `interarrival_*`（12）

| 本脚本 | 主线 | 同定义 | 主线行号 |
|---|---|---|---|
| `interarrival_mean` | 同 | 是 | L290 |
| `interarrival_std` | 同 | 是 | L291 |
| `interarrival_min` | 同 | 是（空序列 → 0.0） | L292 |
| `interarrival_max` | 同 | 是（空序列 → 0.0） | L293 |
| `interarrival_cv` | 同 | 是 | L294 |
| `interarrival_p10` | 同 | 是 | L295 |
| `interarrival_p25` | 同 | 是 | L296 |
| `interarrival_p50` | 同 | 是 | L297 |
| `interarrival_p75` | 同 | 是 | L298 |
| `interarrival_p90` | 同 | 是 | L299 |
| `interarrival_p95` | 同 | 是 | L300 |
| `interarrival_iqr` | 同 | 是 | L301 |

**口径注**：主线 L252-253 先 `times.sort_values()` 再 `diff().dropna()`；本脚本
在 `summarize_window` 入口 `group.sort_values("time_epoch")`，等价。

### 1.4 `burst_*`（8）

| 本脚本 | 主线 | 同定义 | 主线行号 |
|---|---|---|---|
| `burst_packet_ratio` | 同 | 是（阈值 `ia <= 0.10`） | L323-324 |
| `long_gap_ratio` | 同 | 是（阈值 `ia >= 1.0`） | L325 |
| `burst_count` | 同 | 是 | L343 |
| `burst_size_mean` | 同 | 是 | L344 |
| `burst_size_std` | 同 | 是 | L345 |
| `burst_size_max` | 同 | 是 | L346 |
| `burst_size_min` | 同 | 是 | L347 |
| `burst_packet_fraction` | 同 | 是 | L348 |

burst 分段逻辑（`burst_starts = [True] + ~burst_ia[:-1]` 的累加）逐行复制主线 L328-342，
包括 `long_gap_ratio` 归在 burst 族这一分组习惯。

### 1.5 `subwin_*`（10）

| 本脚本 | 主线 | 同定义 | 主线行号 |
|---|---|---|---|
| `subwin_packet_mean` | 同 | 是 | L427 |
| `subwin_packet_std` | 同 | 是 | L428 |
| `subwin_packet_min` | 同 | 是 | L429 |
| `subwin_packet_max` | 同 | 是 | L430 |
| `subwin_packet_cv` | 同 | 是 | L431 |
| `subwin_byte_mean` | 同 | 是 | L432 |
| `subwin_byte_std` | 同 | 是 | L433 |
| `subwin_byte_min` | 同 | 是 | L434 |
| `subwin_byte_max` | 同 | 是 | L435 |
| `active_subwin_count` | 同 | 是 | L436 |

子窗口数 `sub_count = 5`（主线 L410）、bin 边界 `np.linspace(0, window_seconds, 6)`、
最后一格用闭区间 `<=`（主线 L417）—— 全部一致。

### 1.6 分方向 len/ia 统计（11）—— 公式同定义，方向来源改写

| 本脚本 | 主线 | 同定义 | 主线行号 |
|---|---|---|---|
| `up_len_mean` | 同 | 公式同；`up` 判定见 §3 | L398 |
| `up_len_std` | 同 | 同上 | L399 |
| `up_len_p50` | 同 | 同上 | L400 |
| `down_len_mean` | 同 | 同上 | L401 |
| `down_len_std` | 同 | 同上 | L402 |
| `down_len_p50` | 同 | 同上 | L403 |
| `up_ia_mean` | 同 | 同上 | L404 |
| `up_ia_std` | 同 | 同上 | L405 |
| `down_ia_mean` | 同 | 同上 | L406 |
| `down_ia_std` | 同 | 同上 | L407 |
| `len_up_down_diff` | 同 | 同上 | L408 |

`interarrival` 长度比 `group` 少 1，主线用 `reindex(ia_index, fill_value=False)` 对齐
（L391-393），本脚本逐字复制该对齐方式。

---

## 2. 同名、语义等价、机制改写（3 项）

| 本脚本 | 主线 | 同定义？ | 说明 |
|---|---|---|---|
| `up_packet_ratio` | `up_packet_ratio` | **语义同，机制不同** | 主线（802.11，L364-370）：`DA == BSSID` → uplink。本脚本（Ethernet）：`eth.src == 设备 MAC` → up |
| `down_packet_ratio` | `down_packet_ratio` | **语义同，机制不同** | 主线：`TA == BSSID` → downlink。本脚本：`eth.dst == 设备 MAC` → down |
| `up_down_ratio` | `up_down_ratio` | 公式同（L381-383），方向来源如上 | `up_cnt / max(down_cnt, 1)` |

**为什么必须改写**：主线的方向判定是 802.11 空口概念（TA/DA 相对 BSSID），
Ethernet 抓包里根本没有 BSSID/TA。UNSW 是**网关侧**抓包，`eth.src`/`eth.dst`
直接给出设备是发送方还是接收方 —— 这是比 802.11 启发式**更直接**的方向判定，
语义完全一致（up = 设备发出，down = 设备收到），且无歧义。

**副作用（诚实记录）**：本脚本的方向判定更干净，主线在 802.11 上有一个
"other" 兜底类（TA/DA 都不等于 BSSID 的帧）。这意味着两个数据集的
`up_packet_ratio` 分布不是同一测量过程的产物，**不可跨数据集直接比大小**。
pilot 的用途是「同一特征族在另一链路层上能否支撑同类现象」，不做跨集数值比较，
因此该差异不构成问题。

---

## 3. 同名但在 Ethernet 上退化（2 项）

| 本脚本 | 主线 | 同定义？ | 说明 |
|---|---|---|---|
| `side_packet_ratio` | `side_packet_ratio` | **否，恒为 0** | 主线 side = 802.11 管理/控制帧（`fc.type ∈ {0,1}`，L371-375）。Ethernet 无管理/控制帧 |
| `other_packet_ratio` | `other_packet_ratio` | **否，恒为 0** | 主线 other = TA/DA 都不匹配 BSSID 的帧。本脚本按设备 MAC 归流后每个包必为 up 或 down |

**保留理由**：保住列名与列序，使 UNSW 特征表能直接喂给主线以列名索引的下游代码
（`feature_columns()` / `clean_x()` / `cpd_core`）。两列方差为 0，RF 的
`feature_importances_` 会自动给出 0，不影响建模，也不构成信息泄漏。
**在任何跨数据集特征重要性对比中必须剔除这两列**，否则会被误读为"该特征在 UNSW 上不重要"。

---

## 4. 主线有、本脚本故意不做（33 项，802.11 专属）

协议 §16.4 明令：「802.11 专属特征（`subtype_*_ratio` / `retry_ratio` / `rssi_*` /
`bssid_known` / `unique_sa/da`）在 Ethernet 抓包上不存在」。

| 主线特征 | 数量 | 主线行号 | 不做的原因 |
|---|---|---|---|
| `data_ratio` / `mgmt_ratio` / `ctrl_ratio` | 3 | L302-304 | `wlan.fc.type` 不存在于 Ethernet |
| `retry_ratio` | 1 | L305 | `wlan.fc.retry` 不存在 |
| `unique_sa_count` / `unique_da_count` | 2 | L306-307 | `wlan.sa/da` 是空口地址；Ethernet 已按 MAC 归流，该计数恒为 1 |
| `rssi_mean/std/min/max/p10/p50/p90` + `rssi_missing_ratio` | 8 | L308-315 | `radiotap.dbm_antsignal` 需 monitor 模式，网关有线抓包无此字段 |
| `subtype_0_ratio` … `subtype_15_ratio` | 16 | L318-319 | `wlan.fc.subtype` 不存在 |
| `null_data_ratio` / `qos_data_ratio` | 2 | L320-321 | 同上 |
| `bssid_known` | 1 | L384 | `wlan.bssid` 不存在 |
| **合计** | **33** | | |

94 − 33 = **61** = 本脚本特征数。**核对通过。**

---

## 5. 元数据列对照

| 本脚本 | 主线 | 说明 |
|---|---|---|
| `device` | `label` | UNSW 的类名 = device_id。本脚本**同时输出** `label`（值相同）以便直接复用主线以 `label` 为类名的下游代码（`sample_balanced` L951-959 按 `label` 分组） |
| `label` | `label` | 同上，冗余列 |
| `day` | `round` | UNSW 的环境轴 = 采集日（协议 §16.1：20 个连续逐日采集）；主线的环境轴 = 采集轮次 |
| `source_file` | `source_file` | 同 |
| `window_id` | `window_id` | 同 |
| `window_start` | `window_start` | 同（`group["relative_time"].min()`，相对本文件首包）。主线 `time_block_split()`（L962-983）按 `source_file` 分组再按 `window_start` 排序，本脚本输出可直接喂入 |
| `window_end` | `window_end` | 同 |
| `window_start_epoch` | *（无）* | **新增**：窗口起点的绝对 Unix 时间。跨天推理需要绝对时间轴，主线单轮次内不需要。列入 `META_COLUMNS`，不参与建模 |
| *（无）* | `traffic` | UNSW 全部是「稳态运行」单一流量类型（§16.1），无对应轴 |
| *（无）* | `filter_mode` | 主线的 `raw_all` / `data_only` / `data_non_null` 三档是 802.11 帧类型过滤，Ethernet 无对应概念 |

---

## 6. 窗口栅格的一处刻意差异（重要）

| | 主线 | 本脚本 |
|---|---|---|
| 一个 pcap 文件含 | **1 台设备** | **全部设备**（网关侧全网抓包） |
| `relative_time` 原点 | 该文件首包（= 该设备首包） | 该 pcap 首包（= **全天首包，所有设备共享**） |

若照搬主线"每设备各自从 0 起算"，则设备 A 的 `window_id=0` 与设备 B 的 `window_id=0`
落在不同墙钟时刻，跨设备窗口不可比、跨天对齐也无意义。因此本脚本用**全天共享栅格**：
所有设备的 `window_id = floor((t − 全天首包时刻) / 10)`。

**验证判据**：全天饱和的设备应得到约 `86400 / 10 = 8640` 个窗口 ——
与协议 §16.1 记录的「抽样日 Dropcam 8640 窗口（全天饱和）」一致。
实测值见 `results/unsw_pilot/device_window_counts_16-09-30.csv`。

---

## 7. 复现命令

```bash
~/anaconda3/bin/python3 code/scripts/core/extract_features_generic.py \
    --pcap-dir dataset/unsw/pcap \
    --mac-map  dataset/unsw/device_mac_map.csv \
    --output   results/unsw_pilot/features_unsw_w10.csv \
    --days     16-09-30,16-09-23,16-10-12
```

运行元信息（命令行、git hash、包版本、特征名全表）自动落盘到
`results/unsw_pilot/features_unsw_w10.run_meta.json`（协议 §19.2）。
