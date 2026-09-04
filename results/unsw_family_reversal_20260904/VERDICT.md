# UNSW-FAMILY-REVERSAL 判定书

**判定**：`NO_REVERSING_FAMILY`

协议 sha256 `33b7f5be5638c0be007ff8a9f1adbc03e933cc7d0a65dc8b158fe08a0b3dc5bb`　种子 42　H_inner = features_day_16-09-23.csv .. features_day_16-10-02.csv（10 天）；H_outer（4 天）与 T（6 天）白名单拦截

14 台设备（10 天交集）　91 个无序类对　7 族　61 列　10 个留一天出任务

## 逐族 `d_auc_sum`（10 任务、全部类对求和）

| 族 | 列数 | `d_auc_sum` | 自采参照（跨池，不构成受控对比） |
|---|---:|---:|---:|
| `subwin` | 9 | -0.7831 | — |
| `singletons` | 6 | -1.1896 | +0.0182 |
| `down` | 6 | -3.2795 | — |
| `interarrival` | 12 | -3.5239 | +0.0045 |
| `up` | 7 | -3.7289 | — |
| `burst` | 7 | -4.1189 | — |
| `len` | 14 | -11.8153 | -0.1610 |

`top1` = `subwin` -0.7831；`top2` = `singletons` -1.1896；**比值 -15.66**（门槛 3.0）；正值族 0/7

自采参照：`rssi` 对次名 **22.8×**（`HISTINT-OBJ-SIGNAL`）

## 并报

- 全 61 列基线 AUC（逐类对均值）：均值 0.9725、最小 0.6653（`BelkinWemoMotion|BelkinWemoSwitch`）、最大 0.9998
- 明细行数 6370；因样本门槛 40/20 被跳过的 (任务,类对) 计 0
- 运行前预期（协议 §5，只记录不改判据）：`NO_REVERSING_FAMILY`

## 硬门自检

- 7.1 只读 H_inner 10 天；H_outer 与 T 白名单拦截（`guarded_read` 断言）
- 7.2 元数据排除逐项核对，先于任何 AUC 数字落入 `split.json`
- 7.3 缓存 md5（10 项）先于任何 AUC 数字落盘
- 7.4 `pair_auc` / `derive_families` 逐字复用冻结实现，未重实现