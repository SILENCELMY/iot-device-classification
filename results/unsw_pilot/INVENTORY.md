# INVENTORY.md — UNSW pilot 设备清点与窗口计数

**全部数字来源：pcap 派生特征表** `features_unsw_w10.csv`（协议 §16.2：pilot 以 pcap 为准）。

- 特征表行数: **175,759**
- 覆盖天数: **3** ['16-09-23', '16-09-30', '16-10-12']
- 窗口长度: 10 秒非重叠；`min_packets_per_window = 2`（与主线默认一致）

## 1. 每日门槛统计（五问之 3）

| day | 有流量设备数 | ≥100 窗口设备数 | ≥300 窗口设备数 | 总窗口数 | 最大单设备窗口数 |
|---|---|---|---|---|
| `16-09-23` | 17 | 14 | 14 | 60,482 | 8,640 |
| `16-09-30` | 20 | 18 | 18 | 78,122 | 8,640 |
| `16-10-12` | 14 | 13 | 13 | 37,155 | 5,213 |

协议 §16.1 记录的抽样日 `16-09-30` 预期：**有流量 20 台，≥100 窗口 18 台，≥300 窗口 18 台**。上表为本次 pcap 实测值。

## 2.1 `16-09-23` 每设备窗口数

| device_id | device_name | category | MAC | 窗口数 | 包数 |
|---|---|---|---|---|---|
| `Dropcam` | Dropcam | camera | `30:8c:fb:2f:e4:b2` | 8,640 | 242,703 |
| `WithingsBabyMonitor` | Withings Smart Baby Monitor | camera | `00:24:e4:11:18:a8` | 8,636 | 46,901 |
| `SmartThings` | Smart Things | hub | `d0:52:a8:00:67:5e` | 8,458 | 29,442 |
| `NetatmoWelcome` | Netatmo Welcome | camera | `70:ee:50:18:34:43` | 6,275 | 38,397 |
| `SamsungSmartCam` | Samsung SmartCam | camera | `00:16:6c:ab:6b:88` | 6,146 | 84,991 |
| `AmazonEcho` | Amazon Echo | speaker | `44:65:0d:56:cc:d3` | 6,120 | 52,014 |
| `BelkinWemoMotion` | Belkin wemo motion sensor | sensor | `ec:1a:59:83:28:11` | 3,178 | 89,491 |
| `BelkinWemoSwitch` | Belkin Wemo switch | switch | `ec:1a:59:79:f4:89` | 3,115 | 68,299 |
| `HPPrinter` | HP Printer | appliance | `70:5a:0f:e4:9b:c0` | 2,383 | 13,662 |
| `TPLinkCam` | TP-Link Day Night Cloud camera | camera | `f4:f2:6d:93:51:f1` | 2,256 | 10,328 |
| `TribySpeaker` | Triby Speaker | speaker | `18:b7:9e:02:20:44` | 2,239 | 8,817 |
| `NetatmoWeather` | Netatmo weather station | sensor | `70:ee:50:03:b8:ac` | 1,728 | 13,466 |
| `PIX-STAR` | PIX-STAR Photo-frame | appliance | `e0:76:d0:33:bb:85` | 657 | 6,823 |
| `TPLinkPlug` | TP-Link Smart plug | switch | `50:c7:bf:00:56:39` | 637 | 2,596 |
| `WithingsScale` | Withings Smart scale | health | `00:24:e4:1b:6f:96` | 9 | 376 |
| `NestProtect` | NEST Protect smoke alarm | sensor | `18:b4:30:25:be:e4` | 4 | 204 |
| `Blipcare` | Blipcare Blood Pressure meter | health | `74:6a:89:00:2e:25` | 1 | 59 |
| `InsteonCam_wifi` | Insteon Camera | camera | `e8:ab:fa:19:de:4f` | 0 | 0 |
| `InsteonCam_wired` | Insteon Camera | camera | `00:62:6e:51:27:2e` | 0 | 0 |
| `LiFXBulb` | Light Bulbs LiFX Smart Bulb | light | `d0:73:d5:01:83:08` | 0 | 0 |
| `NestDropcam` | Nest Dropcam | camera | `30:8c:fb:b6:ea:45` | 0 | 0 |
| `WithingsAura` | Withings Aura smart sleep sensor | sensor | `00:24:e4:20:28:c6` | 0 | 0 |
| `iHome` | iHome | switch | `74:c6:3b:29:d7:1d` | 0 | 0 |

## 2.2 `16-09-30` 每设备窗口数

| device_id | device_name | category | MAC | 窗口数 | 包数 |
|---|---|---|---|---|---|
| `Dropcam` | Dropcam | camera | `30:8c:fb:2f:e4:b2` | 8,640 | 222,575 |
| `WithingsBabyMonitor` | Withings Smart Baby Monitor | camera | `00:24:e4:11:18:a8` | 8,631 | 47,184 |
| `SmartThings` | Smart Things | hub | `d0:52:a8:00:67:5e` | 8,461 | 29,184 |
| `WithingsAura` | Withings Aura smart sleep sensor | sensor | `00:24:e4:20:28:c6` | 8,134 | 33,557 |
| `NetatmoWelcome` | Netatmo Welcome | camera | `70:ee:50:18:34:43` | 6,281 | 33,702 |
| `SamsungSmartCam` | Samsung SmartCam | camera | `00:16:6c:ab:6b:88` | 6,031 | 56,570 |
| `AmazonEcho` | Amazon Echo | speaker | `44:65:0d:56:cc:d3` | 5,303 | 53,996 |
| `InsteonCam_wired` | Insteon Camera | camera | `00:62:6e:51:27:2e` | 4,378 | 48,386 |
| `LiFXBulb` | Light Bulbs LiFX Smart Bulb | light | `d0:73:d5:01:83:08` | 3,262 | 12,352 |
| `BelkinWemoMotion` | Belkin wemo motion sensor | sensor | `ec:1a:59:83:28:11` | 3,224 | 83,374 |
| `BelkinWemoSwitch` | Belkin Wemo switch | switch | `ec:1a:59:79:f4:89` | 3,101 | 62,940 |
| `iHome` | iHome | switch | `74:c6:3b:29:d7:1d` | 2,881 | 8,262 |
| `HPPrinter` | HP Printer | appliance | `70:5a:0f:e4:9b:c0` | 2,360 | 8,448 |
| `TPLinkCam` | TP-Link Day Night Cloud camera | camera | `f4:f2:6d:93:51:f1` | 2,347 | 117,633 |
| `TribySpeaker` | Triby Speaker | speaker | `18:b7:9e:02:20:44` | 2,288 | 10,585 |
| `NetatmoWeather` | Netatmo weather station | sensor | `70:ee:50:03:b8:ac` | 1,666 | 13,998 |
| `TPLinkPlug` | TP-Link Smart plug | switch | `50:c7:bf:00:56:39` | 642 | 2,605 |
| `PIX-STAR` | PIX-STAR Photo-frame | appliance | `e0:76:d0:33:bb:85` | 482 | 2,988 |
| `NestProtect` | NEST Protect smoke alarm | sensor | `18:b4:30:25:be:e4` | 6 | 274 |
| `WithingsScale` | Withings Smart scale | health | `00:24:e4:1b:6f:96` | 4 | 161 |
| `Blipcare` | Blipcare Blood Pressure meter | health | `74:6a:89:00:2e:25` | 0 | 0 |
| `InsteonCam_wifi` | Insteon Camera | camera | `e8:ab:fa:19:de:4f` | 0 | 0 |
| `NestDropcam` | Nest Dropcam | camera | `30:8c:fb:b6:ea:45` | 0 | 0 |

## 2.3 `16-10-12` 每设备窗口数

| device_id | device_name | category | MAC | 窗口数 | 包数 |
|---|---|---|---|---|---|
| `Dropcam` | Dropcam | camera | `30:8c:fb:2f:e4:b2` | 5,213 | 117,656 |
| `SmartThings` | Smart Things | hub | `d0:52:a8:00:67:5e` | 5,093 | 17,836 |
| `WithingsAura` | Withings Aura smart sleep sensor | sensor | `00:24:e4:20:28:c6` | 3,798 | 12,287 |
| `NetatmoWelcome` | Netatmo Welcome | camera | `70:ee:50:18:34:43` | 3,797 | 13,831 |
| `SamsungSmartCam` | Samsung SmartCam | camera | `00:16:6c:ab:6b:88` | 3,764 | 31,712 |
| `AmazonEcho` | Amazon Echo | speaker | `44:65:0d:56:cc:d3` | 3,758 | 32,817 |
| `InsteonCam_wired` | Insteon Camera | camera | `00:62:6e:51:27:2e` | 2,538 | 27,685 |
| `BelkinWemoSwitch` | Belkin Wemo switch | switch | `ec:1a:59:79:f4:89` | 1,981 | 36,400 |
| `BelkinWemoMotion` | Belkin wemo motion sensor | sensor | `ec:1a:59:83:28:11` | 1,957 | 53,086 |
| `HPPrinter` | HP Printer | appliance | `70:5a:0f:e4:9b:c0` | 1,616 | 6,791 |
| `TribySpeaker` | Triby Speaker | speaker | `18:b7:9e:02:20:44` | 1,269 | 6,067 |
| `NetatmoWeather` | Netatmo weather station | sensor | `70:ee:50:03:b8:ac` | 1,232 | 9,701 |
| `LiFXBulb` | Light Bulbs LiFX Smart Bulb | light | `d0:73:d5:01:83:08` | 1,135 | 7,230 |
| `WithingsScale` | Withings Smart scale | health | `00:24:e4:1b:6f:96` | 4 | 159 |
| `Blipcare` | Blipcare Blood Pressure meter | health | `74:6a:89:00:2e:25` | 0 | 0 |
| `InsteonCam_wifi` | Insteon Camera | camera | `e8:ab:fa:19:de:4f` | 0 | 0 |
| `NestDropcam` | Nest Dropcam | camera | `30:8c:fb:b6:ea:45` | 0 | 0 |
| `NestProtect` | NEST Protect smoke alarm | sensor | `18:b4:30:25:be:e4` | 0 | 0 |
| `PIX-STAR` | PIX-STAR Photo-frame | appliance | `e0:76:d0:33:bb:85` | 0 | 0 |
| `TPLinkCam` | TP-Link Day Night Cloud camera | camera | `f4:f2:6d:93:51:f1` | 0 | 0 |
| `TPLinkPlug` | TP-Link Smart plug | switch | `50:c7:bf:00:56:39` | 0 | 0 |
| `WithingsBabyMonitor` | Withings Smart Baby Monitor | camera | `00:24:e4:11:18:a8` | 0 | 0 |
| `iHome` | iHome | switch | `74:c6:3b:29:d7:1d` | 0 | 0 |

## 3. MAC 映射跨天一致性（五问之 2）

每个 IoT MAC 在 3 天中的窗口数（0 = 该天完全无流量）：

| device_id | MAC | `16-09-23` | `16-09-30` | `16-10-12` | 出现天数 | ≥100窗口天数 |
|---|---|---|---|---|---|---|
| `AmazonEcho` | `44:65:0d:56:cc:d3` | 6,120 | 5,303 | 3,758 | 3 | 3 |
| `BelkinWemoMotion` | `ec:1a:59:83:28:11` | 3,178 | 3,224 | 1,957 | 3 | 3 |
| `BelkinWemoSwitch` | `ec:1a:59:79:f4:89` | 3,115 | 3,101 | 1,981 | 3 | 3 |
| `Dropcam` | `30:8c:fb:2f:e4:b2` | 8,640 | 8,640 | 5,213 | 3 | 3 |
| `HPPrinter` | `70:5a:0f:e4:9b:c0` | 2,383 | 2,360 | 1,616 | 3 | 3 |
| `NetatmoWeather` | `70:ee:50:03:b8:ac` | 1,728 | 1,666 | 1,232 | 3 | 3 |
| `NetatmoWelcome` | `70:ee:50:18:34:43` | 6,275 | 6,281 | 3,797 | 3 | 3 |
| `SamsungSmartCam` | `00:16:6c:ab:6b:88` | 6,146 | 6,031 | 3,764 | 3 | 3 |
| `SmartThings` | `d0:52:a8:00:67:5e` | 8,458 | 8,461 | 5,093 | 3 | 3 |
| `TribySpeaker` | `18:b7:9e:02:20:44` | 2,239 | 2,288 | 1,269 | 3 | 3 |
| `InsteonCam_wired` | `00:62:6e:51:27:2e` | 0 | 4,378 | 2,538 | 2 | 2 |
| `LiFXBulb` | `d0:73:d5:01:83:08` | 0 | 3,262 | 1,135 | 2 | 2 |
| `PIX-STAR` | `e0:76:d0:33:bb:85` | 657 | 482 | 0 | 2 | 2 |
| `TPLinkCam` | `f4:f2:6d:93:51:f1` | 2,256 | 2,347 | 0 | 2 | 2 |
| `TPLinkPlug` | `50:c7:bf:00:56:39` | 637 | 642 | 0 | 2 | 2 |
| `WithingsAura` | `00:24:e4:20:28:c6` | 0 | 8,134 | 3,798 | 2 | 2 |
| `WithingsBabyMonitor` | `00:24:e4:11:18:a8` | 8,636 | 8,631 | 0 | 2 | 2 |
| `iHome` | `74:c6:3b:29:d7:1d` | 0 | 2,881 | 0 | 1 | 1 |
| `WithingsScale` | `00:24:e4:1b:6f:96` | 9 | 4 | 4 | 3 | 0 |
| `NestProtect` | `18:b4:30:25:be:e4` | 4 | 6 | 0 | 2 | 0 |
| `Blipcare` | `74:6a:89:00:2e:25` | 1 | 0 | 0 | 1 | 0 |
| `InsteonCam_wifi` | `e8:ab:fa:19:de:4f` | 0 | 0 | 0 | 0 | 0 |
| `NestDropcam` | `30:8c:fb:b6:ea:45` | 0 | 0 | 0 | 0 | 0 |

- 所有 3 天都有流量的 IoT MAC: **11 / 23**
- 所有 3 天都 ≥100 窗口的 IoT MAC: **10 / 23**
- 任何一天都没出现过的 IoT MAC: **2**

**判读**：MAC 映射本身是静态清单，跨天一致性的风险是「同一 MAC 在不同天对应了不同物理设备」或「设备换了 MAC」。上表能检出的是**出现/消失模式**；若某 MAC 在部分天完全消失、而同期出现一个不在清单里的高流量 MAC，即为可疑信号。见下一节的「清单外 MAC」检查。

