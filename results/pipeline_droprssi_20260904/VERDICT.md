# PIPELINE-DROPRSSI-E2E 判定书

**判定**：`SOURCE_SIGNAL_PARTIAL`　一致 **4/6** 格（TIE 2 格；剔除 TIE 后 3/4）

协议 sha256 `6e85b2fb283667b20dfa9e718311dbfb686868afe448d5c6381857a003b6abac`　种子 [42, 43, 44, 45, 46]　打平带 0.01

## 源域侧决策（只读 inner 三单元，硬门 7 assert 保证）

| 配置 | inner g 均值 | 决策 | inner_to_R2 | inner_to_R3 | inner_to_R4 |
|---|---:|---|---:|---:|---:|
| `full94` | +0.1697 | `REFUSE_STACKING` | +0.1807 | +0.1057 | +0.2225 |
| `drop_rssi` | +0.0043 | `REFUSE_STACKING` | -0.0064 | +0.0117 | +0.0076 |

## 逐格一致性（目标从未参与决策）

| 配置 | 单元 | g_out | 真相 | 决策 | 一致 | TIE |
|---|---|---:|---|---|:-:|:-:|
| `full94` | `pos_R5` | -0.0244 | ALLOW | REFUSE | ✗ | · |
| `full94` | `jit_R6` | +0.0806 | REFUSE | REFUSE | ✓ | · |
| `full94` | `jit_R7` | +0.0878 | REFUSE | REFUSE | ✓ | · |
| `drop_rssi` | `pos_R5` | +0.0125 | REFUSE | REFUSE | ✓ | · |
| `drop_rssi` | `jit_R6` | -0.0039 | ALLOW | REFUSE | ✗ | TIE |
| `drop_rssi` | `jit_R7` | +0.0078 | REFUSE | REFUSE | ✓ | TIE |

## 准确率-延迟曲线：端到端终点 vs 标准做法夹逼

| 单元 | 流程 10s | 流程 1min | 流程 5min | full94+stk 5min | full94+bb 5min |
|---|---:|---:|---:|---:|---:|
| `pos_R5` | 0.7336 | 0.8629 | 0.9327 | 0.7692 | 0.7299 |
| `jit_R6` | 0.7434 | 0.8999 | 1.0000 | 0.8690 | 1.0000 |
| `jit_R7` | 0.8359 | 0.9855 | 1.0000 | 0.8175 | 1.0000 |

流程终点用 best_base（由源域侧规则在 `drop_rssi` 上选出，未看目标）。

## 偏离与自检

- GATE7：`eval_unit` == 冻结 `fit_eval` 逐位一致（容差 1e-12）
- 硬门 7：`source_side_decision` 以 assert 保证只接收 inner 单元
- §4 偏离：`full94 + 源域 CV 选出的模型`未重算（需未登记的 P0 机制），改以 full94 的 stacking / best_base 两臂夹逼，见 `passline.json`
