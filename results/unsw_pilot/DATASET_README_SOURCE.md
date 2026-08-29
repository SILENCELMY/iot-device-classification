# UNSW IoT Traffic Traces — 数据来源记录

**抓取日期**: 2026-08-29（服务器 `galaxy`，校园网直连，**未使用任何代理**）
**用途**: 协议 §16 UNSW pilot（执行计划 D3，P2 节点 9/10）

---

## 1. 页面出处

| 项 | 值 |
|---|---|
| 数据集主页 | https://iotanalytics.unsw.edu.au/iottraces.html |
| 页面 `Last-Modified` | Thu, 02 Oct 2025 06:45:30 GMT |
| 页面字节数 | 7,127 B |
| 对应论文 | IEEE TMC 2018（Sivanathan et al.），"Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics" |
| 设备清单原文件 | https://iotanalytics.unsw.edu.au/resources/List_Of_Devices.txt （33 行，含 2 行表头/空行） |
| 分发方式 | AWS S3 + CloudFront（响应头 `server: AmazonS3` / `via: ... cloudfront.net`） |

---

## 2. 许可（五问之 1 的证据）

页面 License 段原文（逐字抄录）：

```
SPDX-License-Identifier: MIT-0
Copyright 2021 IoT Traffic Analytics Research Group, School of EE&T, UNSW Sydney.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

**结论**：**MIT-0**，与协议 §16.1 记录一致。MIT-0 = MIT 去掉署名保留条款（"without restriction"，且**没有** MIT 原文的
"The above copyright notice ... shall be included in all copies" 一句）。允许 use / copy / modify / merge /
**publish** / **distribute** / sublicense / sell，不附加署名义务。

因此：
- **论文使用** —— 允许；
- **派生特征发布**（我们从 pcap 抽取的 10 秒窗口特征 CSV） —— 允许（"modify" + "distribute" 且无 copyleft、无署名强制）；
- 仍按学术惯例引用 IEEE TMC 2018 原文（页面设 "Cite our data" 段）。

---

## 3. 下载 URL 结构

从页面 HTML 的 JS 模板中提取（`${pcapDates[i]}` 为形如 `16-09-30` 的日期）：

```
pcap : https://iotanalytics.unsw.edu.au/iottestbed/pcap/{YY-MM-DD}.tar.gz
csv  : https://iotanalytics.unsw.edu.au/iottestbed/csv/{YY-MM-DD}.csv.zip
```

- 日期覆盖 `16-09-23` → `16-10-12`，共 20 个连续日；
- `.tar.gz` 内为**单个** `{YY-MM-DD}.pcap`（已核：`tar -tzf 16-09-30.tar.gz` 输出 1 行）；
- 页面另有一个「Download all files using following shell script」的 GitHub gist 链接
  （`gist.github.com/arunmir/dd428d8de787bcc07ef4b513b97d0da3`），实测其 raw 端点返回 **404**，
  因此本项目自写下载脚本 `pcap/download_pcap.sh`（wget -c 直连串行）。

### HEAD 探测结果（2026-08-29，服务器直连）

| 日期 | Content-Length (B) | ≈ | Content-Type |
|---|---|---|---|
| `16-09-30` | 105,756,983 | 100.9 MiB | application/x-tar |
| `16-09-23` | 263,370,328 | 251.2 MiB | application/x-tar |
| `16-10-12` | 4,105,428,276 | 3.82 GiB | binary/octet-stream |

三者 `Last-Modified` 均为 2025-10-02（S3 重新上传时间，非采集时间）。

---

## 4. 设备 / MAC 清单

原始清单 `List_Of_Devices.txt` 共 31 条 MAC 记录，本项目整理为 `device_mac_map.csv`（列：
`device_id, device_name, mac, connection, category, is_iot`）。

| 类别 | 条数 | 说明 |
|---|---|---|
| IoT MAC | **23** | 与协议 §16.1「23 个 IoT MAC」一致 |
| 其中不同设备名 | 22 | `Insteon Camera` 占 2 个 MAC（`00:62:6e:51:27:2e` 有线 / `e8:ab:fa:19:de:4f` 无线） |
| 非 IoT（手机 / 平板 / 笔记本） | 7 | Samsung Galaxy Tab、Android Phone ×2、Laptop、MacBook ×1、IPhone、MacBook/Iphone |
| 网关（基础设施） | 1 | TPLink Router Bridge LAN `14:cc:20:51:33:ea` |
| 合计 | 31 | |

IoT 设备按可混淆分组（对应 §16.1「含 5 台摄像头 / 3 台插座开关 / 多个传感器」）：

- `camera`（7 个 MAC）：NetatmoWelcome、TPLinkCam、SamsungSmartCam、Dropcam、InsteonCam_wired、
  InsteonCam_wifi、WithingsBabyMonitor、NestDropcam
  —— 注：§16.1 点名的「5 台摄像头」指 Dropcam / TP-Link / Samsung SmartCam / Insteon / Netatmo Welcome，
  本表另把 Withings Baby Monitor 与 Nest Dropcam 也归为 camera（同为视频类），分组更宽。
- `switch`（3）：BelkinWemoSwitch、TPLinkPlug、iHome
- `sensor`（4）：BelkinWemoMotion、NestProtect、NetatmoWeather、WithingsAura
- `health`（2）：WithingsScale、Blipcare
- `speaker`（2）：AmazonEcho、TribySpeaker
- `appliance`（2）：PIX-STAR、HPPrinter
- `light`（1）：LiFXBulb
- `hub`（1）：SmartThings

**`device_id` 命名**与协议 §16.1 中已出现的短名保持一致：`Dropcam` / `PIX-STAR` / `NestProtect` /
`WithingsScale` / `InsteonCam_wifi` / `Blipcare` / `NestDropcam`。

---

## 5. 标注方式（协议 §16.1）

网关侧抓包，**Ethernet 帧**。按 `eth.src` / `eth.dst` 精确匹配设备 MAC 归流：

- `eth.src == <device MAC>` → 该设备的**上行**（up）包；
- `eth.dst == <device MAC>` → 该设备的**下行**（down）包；
- 网关 MAC `14:cc:20:51:33:ea` 作为对端，不作为一个待分类设备。

一个包若两端都是已知 IoT MAC（局域网内设备间流量），会同时进入两台设备的流（对 A 记 up、对 B 记 down）。

---

## 6. pcap / CSV 硬约束（协议 §16.2，不可放宽）

- pilot **结论一律以 pcap 为准**；
- 官方 CSV 版（`{date}.csv.zip`）的 `TIME` 字段是**秒级整数**，
  `interarrival_*` / `burst_*` / `subwin_*` 三族时间特征在其上全部失效；
- **本目录下不得出现"用 CSV 验证时间特征"的产物**；CSV 仅允许用于
  **设备清点、窗口计数、类别可用性初筛**的交叉核对。

---

## 7. 本地落盘

```
dataset/unsw/
├── README_SOURCE.md          # 本文件
├── device_mac_map.csv        # 31 条 MAC → device_id 映射
└── pcap/
    ├── download_pcap.sh      # 直连串行下载脚本（wget -c）
    ├── DOWNLOAD_STATUS.md    # 下载状态 / 断点续传方法
    ├── download.log          # wget 输出
    └── {date}.tar.gz         # 每日 pcap 压缩包（内含单个 {date}.pcap）
```

`dataset/` 已在仓库根 `.gitignore` 中（`/dataset/`），原始数据不入库。
