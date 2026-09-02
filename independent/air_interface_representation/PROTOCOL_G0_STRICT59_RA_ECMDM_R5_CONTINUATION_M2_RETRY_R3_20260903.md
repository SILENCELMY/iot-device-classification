# R5 原地续跑 M2 瞬时优化失败恢复 R3（重试前冻结）

**日期**：2026-09-03（Asia/Shanghai）

**状态**：`R5_CONTINUATION_M2_TRANSIENT_RETRY_FROZEN_BEFORE_RETRY`

## 1. 触发事实

R5 有界稳定性续跑已完成并双跑验证 M1 与 M1-R，随后 M2-A 在打印第 90/156 个任务后停止：

```text
I: optimizer failed: ABNORMAL:
```

失败来自冻结 M2 的 SciPy L-BFGS-B `result.success == False` 硬门。M2 只在全部 156 个任务成功后才写
正式结果，因此停止后的正式 M2-A 目录为空，M2-B 目录不存在。

停止后进行的只读/无输出诊断，以相同 canonical G0-A、相同 M1-R、相同目标函数、初值、bounds 和
L-BFGS-B options 重新计算 `g0_R6_R7_to_R5` 的 5 个 fold；所有默认调用均成功收敛。诊断说明该
返回是可重试的瞬时优化器状态，但诊断结果不作为正式科学输入。

## 2. 冻结恢复点

恢复前必须核验：

- 当前 `FAILED.json` 是上述 `R5C_S5_M2` 错误且哈希匹配冻结记录；
- 最初 S2 停止记录仍以 `RECOVERED_S2_G0_VERIFY_FAILURE.json` 原字节保存；
- canonical M1 gate、M1-R gate 和 M1-R double-run verification 哈希匹配；
- G0 bounded stability verification 为 PASS，M1/M1-R 已完成；
- 正式 M2-A 目录存在且严格为空，M2-B 不存在；
- R5 continuation runner、协议 R1/R2 和实现冻结哈希匹配。

任一条件失败不得重试。

## 3. 唯一允许的重试

1. 原当前 `FAILED.json` 原子改名保存为 `RECOVERED_M2_OPTIMIZER_FAILURE.json`；
2. 只可用 `rmdir` 删除已核验为空的 M2 attempt 目录；禁止删除任何非空目录或旧 G0/M1/M1-R；
3. M2-A 和 M2-B 各自最多 3 次**整轮尝试**；每次仍从任务 1 开始；
4. 只有异常类型为冻结 M2 的 `M2Error`，且消息匹配
   `^(I|G|C): optimizer failed: ABNORMAL:` 时才允许重试；其他异常立即停止；
5. 允许重试时，失败 M2 目录必须仍为空；若已有任何文件或子目录，立即停止并保留现场；
6. 不改变 optimizer、初值、目标函数、梯度、bounds、`maxiter`、`ftol`、`gtol` 或科学阈值；
7. A/B 成功后仍必须通过冻结 M2 的逐字节 `verify()`，随后才可执行原 adjudication。

重试次数、错误消息、attempt 起止时间必须写入恢复审计和 provenance。若 3 次仍失败，保留新的
`FAILED.json`；不得把未成功结果包装为通过。

## 4. 成功终态与证据边界

成功后原 R5 audit root 不得存在 `FAILED.json`，必须存在两份 recovered failure 记录、最终
`VERDICT.md`、acceptance、provenance 和 manifest。VERDICT 必须继续披露 G0-A/G0-B 是有界稳定而非
逐字节一致。

科学边界不变：只裁定 oracle recoverability/structure；observable estimability 未检验，
deployability 未建立，EC-MDM 只能称为 CPD 的候选上位构念。
