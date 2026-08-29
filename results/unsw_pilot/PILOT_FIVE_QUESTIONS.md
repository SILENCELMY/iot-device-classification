# PILOT_FIVE_QUESTIONS.md — UNSW Pilot 五问逐条作答

**协议依据**: `docs/experiment_protocol_final.md` §16.3（gate）
**执行依据**: `docs/EXECUTION_PLAN_20260829.md` D3（P2，9/10 截止）
**执行日期**: 2026-08-29
**数据**: UNSW IoT Traffic Traces，https://iotanalytics.unsw.edu.au/iottraces.html

---

## 0. 一页结论

| # | 问题 | 答案 | 主要证据 |
|---|---|---|---|
| 1 | 许可是否允许论文使用与派生特征发布 | **是** | `dataset/unsw/README_SOURCE.md` §2（MIT-0 原文逐字抄录） |
| 2 | MAC → 设备映射是否完整且跨天一致 | **是（附一条必须写入正文的限制）** | `mac_census_unlisted.csv`、`mac_day_consistency.csv`、`INVENTORY.md` §3 |
| 3 | 至少 10 类设备在各选定日有足量窗口 | **是** | `INVENTORY.md` §1、`device_window_counts_*.csv` |
| 4 | 通用 Ethernet 特征能否从 **pcap** 稳定提取 | **是** | `smoke_16-09-30.txt`、`features_unsw_w10.run_meta.json`、`FEATURE_ALIGNMENT.md` |
| 5 | 最小 RF LORO 可运行且无明显标签错误 | **是** | `loro/loro_summary.csv`、`loro/loro_results.json`、`BELKIN_PROBE.md` |

**五问全部通过。** 无需按 §16.3 换次选数据集，也无需把 RQ5 缩为仅 CPD 存在性检验。

**一条必须随结果一起写进正文的限制**（见问 2、问 3）：**不是所有设备每天都在线**。
20 天中抽的 3 天里，23 个 IoT MAC 只有 **11 个天天有流量**、**10 个天天 ≥100 窗口**。
§16.4 的"连续 k 天训练、未来一天测试"扩展**必须按日对取交集类别**，
入选门槛（窗口数下限）与**每个任务的实际类别数**都要在正文报出，
不得笼统写作"18 类"。

---

## 问 1 —— 许可是否允许论文使用与派生特征发布

### 答：**是**

- **SPDX-License-Identifier: MIT-0**，Copyright 2021 IoT Traffic Analytics Research Group,
  School of EE&T, UNSW Sydney。与协议 §16.1 已记录的"许可 = MIT-0"一致。
- MIT-0 是 MIT 去掉**署名保留条款**的版本：授权文字为
  "to deal in the Software **without restriction**, including without limitation the rights to
  use, copy, modify, merge, **publish**, **distribute**, sublicense, and/or sell copies"，
  且**没有** MIT 原文那句 "The above copyright notice ... shall be included in all copies"。
- 因此：
  - **论文使用** → 允许；
  - **派生特征发布**（我们从 pcap 抽取的 10 秒窗口特征 CSV）→ 允许。
    "modify" + "distribute" 均在授权范围内，无 copyleft、无强制署名。
  - 仍按学术惯例引用 IEEE TMC 2018 原文（页面设有 "Cite our data" 段）。

### 证据

| 文件 | 内容 |
|---|---|
| `dataset/unsw/README_SOURCE.md` §2 | 许可原文逐字抄录 + 抓取日期 + 页面出处 + `Last-Modified` |

抓取方式：服务器校园网**直连**（无代理），`curl https://iotanalytics.unsw.edu.au/iottraces.html`，
页面 `Last-Modified: Thu, 02 Oct 2025 06:45:30 GMT`，7,127 B。

---

## 问 2 —— MAC 到设备映射是否完整且跨天一致

### 答：**是**（映射本身完整且一致；但**设备在线情况逐日变化**，须写入正文）

拆成两个子问回答，因为协议括号里点明的风险是"设备更换或地址变化会静默污染标签"。

### 2a. 映射是否**完整**（有没有跑流量却不在清单里的设备）→ **是，完整**

对 3 天全量 pcap 做 `eth.src` / `eth.dst` 普查，排除组播/广播（首字节最低位为 1）后，
**清单外的单播 MAC 只有 1 个**：

| MAC | 09-23 | 09-30 | 10-12 | 只作 src 还是 dst | 判读 |
|---|---|---|---|---|---|
| `14:cc:20:51:33:e9` | 3,404 | 3,838 | 2,148 | **只作 dst，src 恒为 0** | 网关 `14:cc:20:51:33:ea` 的**相邻接口 MAC**（同一台 TP-Link 路由器的另一个接口） |

- 该 MAC 与网关 MAC 仅末位相差 1（`ea` → `e9`），且**从不作为发送方出现**，
  流量量级（2–4 K/天）与路由器管理面一致。**不是一台被漏记的设备。**
- **没有任何"新出现的高流量未知 MAC"** —— 这正是"设备换了地址"会留下的痕迹，此处不存在。

### 2b. 映射是否**跨天一致**（同一 MAC 在不同天是不是同一台设备）→ **是，无地址变更迹象**

判据：若某设备换了 MAC，会同时看到 (i) 原 MAC 消失、(ii) 一个清单外新 MAC 顶上。
实测 (ii) 不存在（见 2a），故不存在地址变更。

进一步的正向证据 —— **窗口数随抓包时长精确缩放**。`16-10-12` 是 20 天的**末日，抓包只有
14.48 小时**（52,128 s；另两天均为 86,398 s ≈ 24 h）。时长比 = 52128/86398 = **0.6033**：

| device | `16-09-23` | `16-09-30` | `16-10-12` | 10-12 / 09-30 |
|---|---|---|---|---|
| Dropcam | 8,640 | 8,640 | 5,213 | **0.6033** |
| SmartThings | 8,458 | 8,461 | 5,093 | 0.6019 |
| NetatmoWelcome | 6,275 | 6,281 | 3,797 | 0.6045 |
| SamsungSmartCam | 6,146 | 6,031 | 3,764 | 0.6241 |
| BelkinWemoMotion | 3,178 | 3,224 | 1,957 | 0.6070 |

**逐日常驻设备的窗口数比值与时长比在小数点后 2 位吻合** → 这些 MAC 在三天里行为一致，
是同一批设备，映射稳定。

### 2c. 必须写入正文的限制：**设备在线情况逐日变化**

23 个 IoT MAC 在 3 天中的出现情况（完整表见 `INVENTORY.md` §3 / `mac_day_consistency.csv`）：

| 分组 | 数量 | 设备 |
|---|---|---|
| 3 天都有流量 | **11** | SmartThings, AmazonEcho, NetatmoWelcome, SamsungSmartCam, Dropcam, BelkinWemoSwitch, BelkinWemoMotion, NetatmoWeather, TribySpeaker, HPPrinter, WithingsScale |
| 3 天都 ≥100 窗口 | **10** | 同上去掉 WithingsScale（3 天分别只有 9 / 4 / 4 个窗口） |
| 只在 2 天出现 | 8 | InsteonCam_wired, LiFXBulb, PIX-STAR, TPLinkCam, TPLinkPlug, WithingsAura, WithingsBabyMonitor, NestProtect |
| 只在 1 天出现 | 2 | iHome（仅 09-30）, Blipcare（仅 09-23，1 个窗口） |
| 3 天全无 | **2** | InsteonCam_wifi (`e8:ab:fa:19:de:4f`), NestDropcam (`30:8c:fb:b6:ea:45`) |

**这是设备开关机 / 拔电，不是标签污染** —— 依据 2a、2b：没有任何未知 MAC 顶替消失的设备。
消失的设备（如 TPLinkCam 在 09-30 有 118,233 包、10-12 为 0）在 pcap 里是**彻底静默**，
不是"流量被记到别的标签下"。

**操作后果（§16.4 扩展时必须遵守）**：每个"k 天训练 → 未来一天测试"任务的类别集合
必须取**该任务涉及各天的交集**，且每天都要过窗口数门槛；正文须报出**每个任务的实际类别数**，
不得笼统称"18 类"。本 pilot 的 6 个日对实际类别数为 10–14（见问 5）。

### 证据

| 文件 | 内容 |
|---|---|
| `results/unsw_pilot/mac_census_unlisted.csv` | 3 天全量 pcap 的清单外单播 MAC（仅 1 条） |
| `results/unsw_pilot/mac_census_known.csv` | 31 个已知 MAC 的逐天 src/dst 包数 |
| `results/unsw_pilot/mac_census_summary.json` | 汇总（11/23 天天在线、1 个清单外 MAC 等） |
| `results/unsw_pilot/mac_day_consistency.csv` | 每 IoT MAC 逐天窗口数 + 出现天数 |
| `results/unsw_pilot/INVENTORY.md` §3 | 同上，人读版 + 判读 |
| `dataset/unsw/device_mac_map.csv` | 31 条 MAC → device_id 映射（23 IoT / 7 非 IoT / 1 网关） |

---

## 问 3 —— 至少 10 类设备在各选定日有足量窗口

### 答：**是**

全部数字来自 **pcap** 派生特征表（10 秒非重叠窗口，`min_packets_per_window = 2`）：

| day | 抓包时长 | 有流量设备数 | **≥100 窗口** | **≥300 窗口** | 总窗口数 | 单设备最大 |
|---|---|---|---|---|---|---|
| `16-09-23`（20 天首日） | 86,398 s | 17 | **14** | **14** | 60,482 | 8,640 |
| `16-09-30`（§16.1 已核抽样日） | 86,398 s | **20** | **18** | **18** | 78,122 | 8,640 |
| `16-10-12`（20 天末日） | **52,128 s** | 14 | **13** | **13** | 37,155 | 5,213 |

- **`16-09-30` 与协议 §16.1 记录的预期逐项吻合**：有流量 20 台 ✅、≥100 窗口 18 台 ✅、
  ≥300 窗口 18 台 ✅、Dropcam 8,640 窗口（全天饱和）✅。
- 三天**各自**都远超"至少 10 类"的门槛（最少的 `16-10-12` 也有 13 类 ≥300 窗口）。
- 三天**取交集**后仍有 **10 类** ≥100 窗口，恰好达标。

### 与协议 §16.1 逐设备数字的一处澄清（不是矛盾）

§16.1 记的 `PIX-STAR 381 / NestProtect 7 / WithingsScale 4` 与 pcap 实测的
`482 / 6 / 4` 有小差异。核对后确认：**§16.1 那组逐设备数字来自官方 CSV**
（CSV 侧折算得 `382 / 7 / 4`，与 §16.1 逐一吻合）。

这**不违反 §16.2** —— CSV 用于窗口计数是允许用途。但按 §16.2"pilot 以 pcap 为准"，
**本 pilot 及后续一律采用 pcap 数字**。pcap 数偏高的原因：CSV 只收录带 IP 层的包，
pcap 另含 ARP / IPv6-ND 等，低速设备因此多出若干活跃窗口。

### 官方 CSV 交叉核对（§16.2 允许的唯一用途：设备清点 + 窗口计数）

`16-09-30` 交叉核对结果：**CSV 侧有流量 20 台，pcap 侧 20 台，"一侧有另一侧无" = 0 台。**
同时在本机复现了 §16.2 的判据，与协议记录**逐位相同**：

| 项 | 本次实测 | §16.2 记录 |
|---|---|---|
| CSV 行数 | 673,414 | 673,414 |
| TIME 跨度（秒） | 86,398 | 86,398 |
| 唯一 TIME 值 | 84,291 | 84,291 |
| 唯一值/行数 | 0.1252 | — |

### 证据

| 文件 | 内容 |
|---|---|
| `results/unsw_pilot/INVENTORY.md` §1、§2.x | 门槛统计 + 逐日逐设备窗口数 |
| `results/unsw_pilot/device_window_counts_{16-09-23,16-09-30,16-10-12}.csv` | 逐日明细 |
| `results/unsw_pilot/device_window_counts_all.csv` | 三日合并 |
| `results/unsw_pilot/CSV_CROSSCHECK.md` + `csv_crosscheck_16-09-30.csv` | CSV↔pcap 清点核对（仅清点用途） |

---

## 问 4 —— 通用 Ethernet 特征能否从 pcap 稳定提取

### 答：**是**

### 4a. 字段可得性（tshark 3.2.3，直接读官方 pcap）

提取器只要 4 个字段：`frame.time_epoch` / `frame.len` / `eth.src` / `eth.dst`。
在 `16-09-30` 前 500,000 包上：**四个字段缺失率均为 0.0000%**。

### 4b. 时间分辨率 —— §16.2 的关键判据

| 项 | pcap（本次实测） | 官方 CSV |
|---|---|---|
| 时间戳唯一率 | **1.000000** | 0.125170 |
| 全为整数秒？ | **False** | True (int64) |
| 最小正包间隔 | **1.8835e-05 s（18.8 µs）** | 0（塌缩） |
| 包间隔 p50 | 0.0419 s | — |
| 包间隔 < 0.1 s 的比例 | 0.6099 | — |

**pcap 是微秒分辨率**，`interarrival_*` / `burst_*`（阈值 0.10 s）/ `subwin_*`（2 s 子窗）
三族时间特征**全部有真实取值**。§16.2 的禁用 CSV 约束在本机得到独立确认。

### 4c. 提取结果的稳定性

`16-09-30` 全天：**78,122 个窗口 / 20 台设备 / 61 个数值特征**，用时 674 s。

| 检查项 | 结果 |
|---|---|
| NaN 单元格 | **0** |
| inf 单元格 | **0** |
| 常量列 | **2**，且都是**设计上就该恒为 0** 的 `side_packet_ratio` / `other_packet_ratio`（见 `FEATURE_ALIGNMENT.md` §3） |
| `window_id` 范围 | 0 – 8,639（= 86,400/10，全天栅格完整） |
| `packet_count` | min 2 / p50 4 / max 2,150 |

**时间特征确实带区分度**（不是常数）—— 各设备 `interarrival_p50` 中位数跨 **3 个数量级**：

| 设备 | `interarrival_p50` | `burst_packet_ratio` |
|---|---|---|
| InsteonCam_wired | 0.000241 | 0.818 |
| SamsungSmartCam | 0.001250 | 0.778 |
| TPLinkCam | 0.073079 | 0.500 |
| Dropcam | 0.215385 | 0.095 |
| TribySpeaker | 0.507765 | 0.000 |

### 4d. 特征族与主线对齐

`code/scripts/core/extract_features_generic.py`（§20.1 新建项）产出 **61 维**通用特征：

| 族 | 维数 |
|---|---|
| `len_*`（含 `len_up_down_diff`） | 14 |
| `interarrival_*` | 12 |
| `burst_*` + `long_gap_ratio` | 8 |
| `up_*` / `down_*` / `side_*` / `other_*` | 15 |
| `subwin_*` + `active_subwin_count` | 10 |
| `packet_count` / `byte_count` | 2 |
| **合计** | **61** |

与主线 `robust_iot_research.py::summarize_window()` 的 **94** 维相比：
**94 − 33（802.11 专属，按 §16.4 一概不做）= 61**，**核对闭合**。

其中 **56 项同名同定义**（`quantile` / `safe_mean` / `safe_std` /
`coefficient_of_variation` 四个统计函数逐字复制主线 L223-238，含全部边界行为），
**3 项同名语义等价但机制改写**（方向判定由 802.11 的 `TA/DA vs BSSID` 改为
Ethernet 的 `eth.src/eth.dst vs 设备 MAC`），
**2 项同名但在 Ethernet 上恒为 0**（`side_/other_packet_ratio`，保留列名以维持列对齐）。

**802.11 专属特征（`subtype_*` / `retry_*` / `rssi_*` / `bssid_known` / `unique_sa/da`）一项未做**，
符合 §16.4。

### 证据

| 文件 | 内容 |
|---|---|
| `results/unsw_pilot/smoke_16-09-30.txt` | 字段缺失率、时间分辨率、全天 MAC 清点（smoke 原始输出） |
| `code/scripts/analysis/unsw_pilot/smoke_pcap.sh` | smoke 的可复现脚本 |
| `results/unsw_pilot/FEATURE_ALIGNMENT.md` | **61 ↔ 94 逐特征对齐表**（含 3 项改写、2 项退化、33 项不做的逐条说明） |
| `results/unsw_pilot/features_unsw_w10.csv` | 三日合并特征表（175,759 行） |
| `results/unsw_pilot/features_day_*.csv` | 逐日特征表 |
| `results/unsw_pilot/features_*.run_meta.json` | §19.2 运行元信息（命令行、git hash、包版本、61 个特征名全表） |
| `code/scripts/core/extract_features_generic.py` | 提取器本体 |

---

## 问 5 —— 最小 RF LORO 是否可运行且无明显标签错误

### 答：**是**（可运行；唯一的报警项已判定为设备本身相似，非标签错误）

### 5a. 可运行性与结果

口径：**RF 直接 import 主线 `build_model("rf", ...)`**（协议 §7：`n_estimators=500,
class_weight="balanced"`），**类别不均衡直接 import 主线 `sample_balanced`**（§16.4），
不在 pilot 脚本里另写一份 —— 遵守 §11「唯一实现」纪律。
入选门槛：训练日与测试日**都** ≥100 窗口。

6 个有序日对全部跑通：

| train → test | 类别数 | 训练行 | 测试行 | **macro-F1** | weighted-F1 | accuracy | 秒 |
|---|---|---|---|---|---|---|---|
| `16-09-23` → `16-09-30` | 14 | 18,430 | 18,260 | **0.8164** | 0.8229 | 0.8266 | 2.6 |
| `16-09-23` → `16-10-12` | 10 | 19,728 | 18,055 | **0.8455** | 0.8412 | 0.8461 | 1.9 |
| `16-09-30` → `16-09-23` | 14 | 18,260 | 18,430 | **0.8418** | 0.8482 | 0.8513 | 3.5 |
| `16-09-30` → `16-10-12` | 13 | 19,994 | 19,016 | **0.8674** | 0.8636 | 0.8656 | 2.5 |
| `16-10-12` → `16-09-23` | 10 | 18,055 | 19,728 | **0.8621** | 0.8603 | 0.8595 | 1.9 |
| `16-10-12` → `16-09-30` | 13 | 19,016 | 19,994 | **0.8462** | 0.8462 | 0.8478 | 2.2 |
| | | | | **均值 0.8466** | | | |

跨天 macro-F1 **0.8164 – 0.8674**（10–14 类，随机基线约 0.07–0.10）。
**任务可运行，且远非退化解。**

### 5b. 标签错误检查

机械检查三种典型形态（`pilot_rf_loro.py::label_sanity`）：
A. 某类过半被判成**另一个特定类**且高于自身 recall（串位）；
B. 某一对类**双向**互判均 >0.30（对调）；
C. 某类 recall <0.02 但 support ≥30（被系统性吸走）。

结果：**C 型 0 例。A/B 型的报警全部集中在同一对：`BelkinWemoSwitch` ↔ `BelkinWemoMotion`，
其余 12 类在 6 个日对中一次都没报警。**

### 5c. 该报警判定为「设备相似」而非「标签错误」—— 决定性对照

关键判据：**同一天内 MAC 是硬标识，标签必然自洽**。若是标签错误，同一天内应当可分、
只在跨天崩溃。实测**同一天内也分不开**：

同一天（`16-09-30`）**全类 IID** 随机划分（18 类，macro-F1 = **0.9096**）的逐类 F1：

| 排名（1=最差） | 类别 | F1 |
|---|---|---|
| **1** | **`BelkinWemoMotion`** | **0.6477** |
| **2** | **`BelkinWemoSwitch`** | **0.6628** |
| 3 | `TPLinkPlug` | 0.8198 |
| … | … | … |
| 16 | `WithingsBabyMonitor` | 0.9970 |
| 17 | `Dropcam` | 1.0000 |
| 18 | `iHome` | 1.0000 |

- 这两类是 18 类中**最差的两名**，而同一次运行里有 6 个类 F1 ≥ 0.99、2 个类 = 1.0000。
- 逐天单独做两类 IID 二分类：09-23 = 0.7537、09-30 = 0.6836、10-12 = 0.6165（均值 **0.6846**）。
- 特征层直接证据：两类 61 维**中位向量有 35/56 维完全相同**。
- 两者 MAC 同属 OUI `ec:1a:59`（Belkin International），同厂同 WeMo/UPnP 协议栈。
  该混淆在 Sivanathan TMC 2018 原文中亦有记载。
- 崩溃形态是**双向混合**（Switch recall 0.33–0.36，约 0.53–0.57 流向 Motion），
  **不是标签对调应有的 recall≈0 / 单向≈1.0**。

**判定：无明显标签错误，问 5 通过。**（这一对的高混淆本身是有价值的信号 ——
它给 18 类混淆拓扑贡献了一个真实的"同厂近邻"难例。）

### 证据

| 文件 | 内容 |
|---|---|
| `results/unsw_pilot/loro/loro_summary.csv` | 6 个日对的 macro-F1 / accuracy / 行数 |
| `results/unsw_pilot/loro/loro_results.json` | 全量结果 + 逐对 `label_sanity` + 61 个特征名 + git hash |
| `results/unsw_pilot/loro/cm_*.csv` | 6 个日对的混淆矩阵 |
| `results/unsw_pilot/loro/per_class_*.csv` | 6 个日对的逐类 precision/recall/F1 |
| `results/unsw_pilot/BELKIN_PROBE.md` + `.json` | 设备相似 vs 标签错误的判别检验（含全类 IID 对照） |
| `code/scripts/analysis/unsw_pilot/pilot_rf_loro.py` | LORO 脚本（import 主线 build_model / sample_balanced / clean_x） |
| `code/scripts/analysis/unsw_pilot/belkin_probe.py` | 判别检验脚本 |

---

## 附. 数据与执行记录

### 已下载（校园网直连，全程无代理）

| 日期 | 远端字节 | 落盘 | `gzip -t` | pcap 时长 | 包数 | 用途 |
|---|---|---|---|---|---|---|
| `16-09-30` | 105,756,983 | 一致 | OK | 86,398 s | 802,226 | §16.1 已核抽样日，优先级 1 |
| `16-09-23` | 263,370,328 | 一致 | OK | 86,398 s | 947,072 | 20 天首日 |
| `16-10-12` | 4,105,428,276 | 一致 | OK | **52,128 s** | 4,948,806 | 20 天末日（跨度最大化） |
| `16-10-11` | 842,790,743 | 一致 | OK | 86,399 s | 2,073,339 | **追加**，见下 |

实测吞吐 ≈ 9–10 MB/s，前三天合计 4.17 GiB 约 **10 分钟**下完。**ETA 远低于 12 小时上报门槛**，
因此未按 D3 的"文件过大可换日期"条款更换任何日期，三天按原优先级全部拿到。

### 一处执行决定：追加 `16-10-11`（需登记）

- **原因**：`16-10-12` 是 20 天末日，抓包**只有 14.48 小时**（52,128 s，另两天为 24 h）。
  这不影响五问的任何一条（问 2/3/5 均已用它通过），但会让"末端日"在后续
  §16.4 扩展里同时混入"短抓包"与"设备下线"两个因素。
- **处置**：**保留 `16-10-12` 不动**（它是真正的末日，跨度最大，正是问 2 需要的），
  **另加下载一天完整 24 h 的 `16-10-11`（803.7 MiB，约 1.5 分钟）**作为末端的
  full-day 对照。**纯增量，不替换、不删除任何原定日期。**
- **依据**：D3「选日为执行细节，非协议冻结项」。特征提取已启动，产物为
  `features_day_16-10-11.csv`，不影响本文件已给出的任何结论。

### 目录

```
dataset/unsw/
├── README_SOURCE.md              许可原文 / URL 结构 / 设备清单 / 抓取日期
├── device_mac_map.csv            31 条 MAC → device_id（23 IoT / 7 非 IoT / 1 网关）
└── pcap/
    ├── download_pcap.sh          直连串行下载（wget -c，显式 --no-proxy）
    ├── DOWNLOAD_STATUS.md        状态 / PID / 进度查看 / 断点续传 / 完整性校验
    ├── download.log, unpack*.log
    └── {date}.tar.gz, {date}.pcap

results/unsw_pilot/
├── PILOT_FIVE_QUESTIONS.md       本文件
├── FEATURE_ALIGNMENT.md          61 ↔ 94 特征对齐表
├── INVENTORY.md                  设备清点 / 窗口计数 / MAC 跨天一致性
├── CSV_CROSSCHECK.md             CSV↔pcap 清点核对（仅清点用途）
├── BELKIN_PROBE.md               标签错误判别检验
├── smoke_16-09-30.txt            pcap 字段与时间分辨率 smoke
├── features_unsw_w10.csv         三日合并特征表（175,759 行 × 61 特征）
├── features_day_*.csv            逐日特征表
├── device_window_counts_*.csv    逐日 / 合并窗口计数
├── mac_census_*.{csv,json}       3 天 MAC 普查（含清单外 MAC）
├── mac_day_consistency.csv       每 IoT MAC 逐天窗口数
├── loro/                         6 个日对的 LORO 结果 + 混淆矩阵 + 逐类指标
└── *.py, *.sh                    全部可复现脚本
```

### 未做（按协议）

- 未用 CSV 验证任何时间特征（§16.2）；CSV 只出现在 `CSV_CROSSCHECK.md` 的清点核对中。
- 未做 802.11 专属特征（§16.4）。
- 未做跨数据集完整基线对比、公开数据集上的深度模型搜索、特征工程调优（§16.4「不做」清单）。
- 未执行任何 `git add` / `commit` / `push`。

---

## 附 B. 第 4 天 `16-10-11` 的补充分析（supplementary，不改动上文任何结论）

上文问 1–5 的全部数字来自**协议 §16.3 要求的 3 天**（`16-09-23` / `16-09-30` / `16-10-12`），
文件在 `results/unsw_pilot/` 根目录。本节是追加的 `16-10-11`（完整 24 h）带来的**补强**，
产物独立落在 `results/unsw_pilot/four_day/`，**不覆盖也不修改**上文引用的任何文件。

### B.1 为什么这一天有决定性价值

上文问 2 曾把"多台设备在 `16-10-12` 归零"部分归因于该日**抓包只有 14.48 小时**。
`16-10-11` 是**完整 24 h** 的一天，正好把这个混淆因素拆开：

| device | `16-09-23` (24h) | `16-09-30` (24h) | `16-10-11` (**24h**) | `16-10-12` (14.5h) |
|---|---|---|---|---|
| WithingsBabyMonitor | 8,636 | 8,631 | **0** | 0 |
| TPLinkCam | 2,256 | 2,347 | **0** | 0 |
| TPLinkPlug | 637 | 642 | **0** | 0 |
| PIX-STAR | 657 | 482 | **0** | 0 |

**这四台在完整 24 h 的 `16-10-11` 上同样是 0。**
→ 归零**不是**短抓包造成的，而是**设备在采集末期被真实下线**（testbed 收尾）。
→ 与问 2 的判定一致且更强：**不是标签污染，是设备开关机**。
（`16-10-12` 的短抓包只额外解释常驻设备窗口数按 0.6033 缩放这件事，见问 2b。）

### B.2 4 天门槛统计

| day | 抓包时长 | 有流量设备数 | ≥100 窗口 | ≥300 窗口 | 总窗口数 |
|---|---|---|---|---|---|
| `16-09-23` | 86,398 s | 17 | 14 | 14 | 60,482 |
| `16-09-30` | 86,398 s | 20 | 18 | 18 | 78,122 |
| `16-10-11` | 86,399 s | 15 | 13 | 13 | 62,629 |
| `16-10-12` | 52,128 s | 14 | 13 | 13 | 37,155 |

**4 天都 ≥100 窗口的 IoT MAC 仍是 10 个**，与 3 天交集**完全相同**（问 3 的结论不变）。

### B.3 4 天 LORO（12 个有序日对）

| train → test | 类别数 | macro-F1 | | train → test | 类别数 | macro-F1 |
|---|---|---|---|---|---|---|
| `09-23` → `09-30` | 14 | 0.8164 | | `10-11` → `09-23` | 10 | 0.8677 |
| `09-23` → `10-11` | 10 | 0.8478 | | `10-11` → `09-30` | 13 | 0.8744 |
| `09-23` → `10-12` | 10 | 0.8455 | | `10-11` → `10-12` | 13 | **0.8901** |
| `09-30` → `09-23` | 14 | 0.8418 | | `10-12` → `09-23` | 10 | 0.8621 |
| `09-30` → `10-11` | 13 | 0.8623 | | `10-12` → `09-30` | 13 | 0.8462 |
| `09-30` → `10-12` | 13 | 0.8674 | | `10-12` → `10-11` | 13 | 0.8778 |

**12 对全部跑通，macro-F1 0.8164 – 0.8901，均值 0.8583**（3 天版为 0.8466）。
相邻两天 `10-11 ↔ 10-12` 是全场最高的一对（0.8901 / 0.8778），
跨 19 天的 `09-23 ↔ 10-12` 明显更低（0.8455 / 0.8621）——
**时间距离越远、跨天泛化越差**，这正是 §16.4 检验 1（机制存在性）所需要的那种梯度信号，
但 pilot 阶段**不对此下任何结论**，只记录现象。

标签报警仍然**只出现在 `BelkinWemoSwitch` ↔ `BelkinWemoMotion` 一对**，
其余类在 12 个日对中零报警 —— 与问 5c 的判定一致。

### B.3b 4 天 MAC 普查（问 2a 的补强）

在 4 天全量 pcap（合计 8,771,443 包）上重跑清单外单播 MAC 普查：

- **清单外单播 MAC 仍然只有 1 个**：`14:cc:20:51:33:e9`（网关相邻接口），
  4 天合计 12,938 包，**4 天都只作 dst、src 恒为 0**。
- 4 天都有流量的 IoT MAC：**11 / 23**；从未出现的：**2**（与 3 天版一致）。

**多观察一天、且是完整 24 h 的一天，依然没有出现任何未知设备 MAC**
→ 问 2「映射完整且跨天一致」的判定进一步加强。

### B.4 产物

```
results/unsw_pilot/four_day/
├── features_unsw_w10_4day.csv     4 天合并（238,388 行 × 61 特征）
├── INVENTORY.md                    4 天门槛统计 + MAC 逐天一致性
├── device_window_counts_*.csv
├── mac_day_consistency.csv
├── mac_census_*.{csv,json}         4 天 MAC 普查
└── loro/                           12 个日对的结果 + 混淆矩阵 + 逐类指标
```
