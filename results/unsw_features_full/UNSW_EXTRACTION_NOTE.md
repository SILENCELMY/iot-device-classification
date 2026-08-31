# UNSW 20 天特征提取与设备清点（主线，供 §16.4 检验 1 规格使用）

**性质**：机械性提取与清点，不含任何科学判定。生成于 2026-08-31，未修改
`results/unsw_pilot/`、`independent/` 或 `results/meta_mismatch_exploratory/`。

## 1. 提取

- 脚本：`code/scripts/core/extract_features_generic.py`（与 pilot 同一版本，逐日 `run_meta.json`）
- 驱动：`code/scripts/core/unsw_extract_all.sh`，8 路并行，user systemd transient unit
  `unsw-extract-full.service`（`Result=success`, `ExecMainStatus=0`）
- 输入：`dataset/unsw/pcap/`，20 天（16-09-23 ~ 16-10-12）
- 输出：`results/unsw_features_full/features_day_<day>.csv` + `.run_meta.json`（已 gitignore，可确定性再生）
- 墙钟：17:13:21 → 17:30:43（17.4 分钟），20/20 天 `rc=0`
- 规模：**1,317,887** 窗口，61 个数值特征，窗口 10 s，`min_packets_per_window=2`，网关 MAC 已排除

## 2. 确定性核验（免费获得）

`PILOT_CROSSCHECK.txt`：与 pilot 重叠的 4 天 **md5 逐位相同**（16-09-23 / 16-09-30 / 16-10-11 /
16-10-12）。跨 2 天、跨编排方式（pilot 串行 vs 本次 8 路并行）复现，提取管线确定性得到确认。

## 3. 设备清点（`device_window_counts_by_day.csv`）

- IoT 设备 MAC 共 **23** 个；`label` 列是**设备身份**，不是设备类型
- **每日 ≥100 窗的设备数逐日为 13–18**（16-09-28 ~ 16-10-05 为 18；最后五天降至 13）
- **在全部 20 天都 ≥100 窗的设备只有 10 个**，且该集合对阈值不敏感
  （≥50 / ≥100 / ≥200 得到同一 10 个）——是结构边界，不是阈值假象：

  `AmazonEcho, BelkinWemoMotion, BelkinWemoSwitch, Dropcam, HPPrinter,
  NetatmoWeather, NetatmoWelcome, SamsungSmartCam, SmartThings, TribySpeaker`

## 4. 类型级构成（`device_mac_map.csv` 的 `category` 列）

| category | 全部 23 设备 | 全 20 天稳定的 10 设备 |
|---|---:|---:|
| camera | 8 | **3**（Dropcam, NetatmoWelcome, SamsungSmartCam） |
| sensor | 4 | **2**（BelkinWemoMotion, NetatmoWeather） |
| switch | 3 | 1（BelkinWemoSwitch） |
| speaker | 2 | **2**（AmazonEcho, TribySpeaker） |
| health | 2 | 0 |
| appliance | 2 | 1（HPPrinter） |
| light | 1 | 0 |
| hub | 1 | 1（SmartThings） |
| **合计** | **8 类 / 23 设备** | **6 类 / 10 设备** |

六个类型在全部 20 天均有 ≥100 窗（按类型汇总）。

## 5. 规格必须先声明的三件事（清点结论，非判定）

1. **标签层级**：设备身份（23 / 稳定 10 类）还是设备类型（8 / 稳定 6 类）。二者是不同任务。
   注意自采数据的标签集 `Camera / Light_T1 / Light_XM / Sensor / Socket` 本身是**类型与实例的混合**
   （light 被拆成两个实例类），跨数据集可比性须显式处理。
2. **设备面板**：固定面板（全 20 天稳定的 10 设备 / 6 类型，保守）还是**逐任务按 k+1 天取交集**
   （§16.4 的连续 k 天训练 / 次日测试只要求任务内稳定，可纳入更多设备）。须逐任务报实际类别数。
3. **leave-device-out 的可执行范围**：在 UNSW 上**首次可执行**，但稳定面板下只有 camera 有 3 台、
   speaker/sensor 各 2 台（留一后仅剩 1 台）。因此它把"未见设备实例"从自采数据上的
   **不可能**推进到**可做但很薄**，不足以单凭此消除"设备类型分类 vs 设备实例指纹"的质疑。
