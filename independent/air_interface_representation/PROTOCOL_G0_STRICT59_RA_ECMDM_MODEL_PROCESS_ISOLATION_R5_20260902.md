# G0 strict59_ra EC-MDM R5 原生模型进程隔离协议（运行前冻结）

**冻结日期**：2026-09-02（Asia/Shanghai）  
**状态**：`NATIVE_MODEL_PROCESS_ISOLATION_FROZEN_BEFORE_R5_RUN`  
**性质**：只把每个 G0 模型单元的训练、预测和序列化放入独立子进程，并以事务式目录发布完整产物；
不改变特征、数据划分、模型、超参数、种子、任务顺序、指标、重复、科学门、状态树或解释边界。

## 1. 继承链

本协议完整继承且不得修改：

| 冻结对象 | SHA-256 |
|---|---|
| `PROTOCOL_G0_STRICT59_RA_ECMDM_RECALIBRATION_20260902.md` | `d3c0c19821effee9ce1c9370fe6dc0046742c0f2038a887c34b8952445bfee5a` |
| `PROTOCOL_G0_STRICT59_RA_ECMDM_PATH_REPAIR_R2_20260902.md` | `ad0644d3f396fcbe44caac8e4c4398fb6eda23dd872c18cddcd50b1cb6047269` |
| `PROTOCOL_G0_STRICT59_RA_ECMDM_TERMINAL_RECOVERY_R3_20260902.md` | `e98aef970698827e531a685f6d5f87a30ce68a1f6360138556a4621fc710ea01` |
| `PROTOCOL_G0_STRICT59_RA_ECMDM_TMP_DISPLAY_REPAIR_R4_20260902.md` | `607686217f0d18a71b8df6d5ff63e03d8599f7b714df8c10035e12baa29adcbf` |
| `G0_STRICT59_RA_ECMDM_R4_IMPLEMENTATION_FREEZE.json` | `62de14924bfeff8b0597f7c685846374d1ff9ccb028ec6c3f30bbe48831d5c5b` |

父协议的 oracle recoverability、`excess_F >= 0.020` 实质量级、C-first 结构门，以及 oracle
recoverability / observable estimability / deployability 三层证据边界全部不变。

## 2. R4 原生崩溃事实

R4 user-systemd unit 于 2026-09-02 22:40:25（Asia/Shanghai）结束。journal 的确定性记录为：

```text
Main process exited, code=dumped, status=11/SEGV
Failed with result 'core-dump'
```

崩溃发生在正式 G0-A 的 `g0_iid_R3_time_block / xgboost`：

1. RF 已完整完成；XGBoost 的 `predictions.csv`、`pred_proba.csv`、分类报告、混淆矩阵、特征列和特征
   重要性均已写完。
2. `model.joblib` 已打开但为 0 字节；`metrics.json` 尚未写入。冻结源码的固定顺序表明崩溃发生在
   `joblib.dump(model, model.joblib)` 内。
3. XGBoost 3.2.0 的 pickle 路径在 `Booster.__getstate__()` 调用原生
   `XGBoosterSerializeToBuffer()`；纯 Python 异常不能产生该 `SIGSEGV`。
4. R3 曾对完全相同任务成功生成 3,087,938 字节的 `model.joblib`。R3/R4 崩溃前的六项预测与诊断文件
   SHA-256 逐项相等，因此不是数据、划分、拟合或预测的确定性错误。
5. 崩溃时未形成 Python `FAILED.json`，因为主 runner 本身被原生信号杀死。服务器没有 `coredumpctl`，
   Apport 未留下可用 core，故不能宣称已定位具体 C++ 栈帧。
6. 检查时磁盘尚余约 1.6 TiB、inode 使用约 1%、可用内存约 121 GiB；journal 记录为 SEGV 而非 OOM。

R4 的 `/tmp` 显示路径修复与本次崩溃无关：故障发生在仓库内 G0-A，该调用保持原显示锚点。

## 3. R4 冻结快照

| root | 状态 | 文件数 | 总字节 | `metrics.json` 数 | 树摘要 |
|---|---:|---:|---:|---:|---|
| `results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r4/` | 存在 | 5 | 10,431,614 | 0 | `3d07f65f332ca7b9a782a2c5ea6546b66f00801ef1dd49edf4998d1747261eda` |
| `results/g0_environment_grid_strict59_ra_r4/` | 存在 | 5,526 | 20,395,864,950 | 613 | `0dbccd96bb57e16d5f988ffed7097b20035277a4197bf5ba0c3ac7d1b12744a6` |
| `results/meta_mismatch_exploratory/strict59_ra_ecmdm_r4/` | 不存在 | 0 | 0 | 0 | — |

R4 G0 树摘要沿用 R4 协议定义：对全部 GNU `sha256sum` 行按完整行排序后再计算 SHA-256。R4 audit
树摘要沿用 R3 协议的路径排序定义。R4 systemd 日志为 59,688 字节，SHA-256
`d2698cac565da1bf60947b85c36d0cf19ef0d453e60456d71d176a336825d905`。

R4 audit 五个文件的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `extraction_audit.json` | `f937d7b49c8e8d2fe6aa9abbed56457d82fe845c0c0add88d03a2247f7456c62` |
| `full94_reproduction_gate.json` | `b1b92665e19566f73959f4c5798382808c31aa800b1ade93b77620d521796e83` |
| `input_audit.json` | `6675ae12c0c9544cf4887127a4f63b54c60539a8e040e2d0727017f5ccc5dd7c` |
| `pcap_manifest.json` | `0d6d1822a57c2e7f5e2edf3a37cd1fe8193edde084d04ae3c8f26b787239dead` |
| `strict59_ra_features_raw_all_w10.csv` | `f7430bc32d50b66d1e4975f1bf2678d55fd692c441bec0093965a77969e10ca5` |

R2、R3、R4 正式 roots、日志和临时 roots 均只保留取证，不得删除、改名、补写、覆盖、读取为 R5 输入
或复用。

## 4. R5 原生进程隔离与事务发布

R5 只允许 wrapper 对冻结 `robust_iot_research.evaluate_model()` 增加下列调用隔离；冻结 G0 与科学模块源码
不得修改：

1. `evaluate_task()` 仍按冻结顺序构造同一模型对象、传入同一数据和参数；wrapper 在调用
   `evaluate_model()` 时 `fork()` 一个全新子进程。训练、预测、指标计算、全部文件写入和 `joblib.dump()`
   都只在子进程中发生；父 runner 不持有拟合后的原生 Booster。
2. 每个子进程写入与目标同一文件系统上的唯一 flat attempt 目录。正式模型目录在子进程运行期间必须
   不存在。只有子进程以 0 退出，且固定产物集合完整、`metrics.json` 可解析、`model.joblib` 非空时，父
   runner 才以同文件系统 `rename()` 将整个 attempt 目录原子发布为正式模型目录。
3. 固定产物集合：RF/XGBoost/LightGBM 各 8 个文件（报告、混淆矩阵、特征列、特征重要性、metrics、
   model、概率、预测）；Stacking 另须 `oof_meta.csv`，共 9 个。禁止发布部分目录。
4. 子进程若因 `SIGSEGV`、`SIGBUS` 或 `SIGABRT` 终止，只允许使用同一未拟合父模型和同一输入再开一个
   全新 attempt；每个模型最多 3 次总尝试（首次 + 2 次恢复）。不得重试 Python 异常、非零正常退出、
   其他信号或文件完整性失败。
5. 所有失败 attempt 永久保留，不删除、不覆盖；其路径、信号、尝试序号写入隔离审计。子进程启用
   `faulthandler`，若系统允许则在失败 attempt 留下 Python fatal-signal 栈。
6. 正式 G0-A 的 attempt 根为
   `results/.g0_environment_grid_strict59_ra_r5_model_attempts/`；G0-B 的 attempt 根为同一全新 `/tmp`
   pipeline root 下的 `.g0_b_model_attempts/`。成功 attempt 通过 rename 离开 attempt root；失败 attempt
   留在原位。
7. G0-A 和 G0-B 分别返回隔离审计；wrapper 在双跑校验前写
   `model_process_isolation_audit.json`。最终仍必须通过原 648 单元计数、逐文件双跑 SHA 和全部后续门。

`fork()` 只改变故障域和原生对象生命周期；子进程执行的仍是哈希锁定的原函数。随机种子、线程参数、
文件内容生成逻辑和任务顺序不变。父进程能观察的模型结果只来自原函数最后写出的 `metrics.json`。

## 5. R5 roots、测试与运行

唯一新正式 roots：

```text
results/g0_environment_grid_strict59_ra_r5/
results/meta_mismatch_exploratory/strict59_ra_ecmdm_r5/
results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r5/
```

上述三处、正式 A attempt root 和 R5 systemd 日志在本协议冻结时均不存在。实现后另写 R5 implementation
freeze。正式运行前只允许：

- 原状态树合成测试；
- 子进程成功、原子发布、允许信号重试、Python 异常不重试、上下文恢复的合成测试；
- `--preflight-no-fit` 静态检查。

正式 R5 必须从 full94 双跑判否门开始，不得续用旧 staging。仍由
`strict59-ra-ecmdm-r5-20260902.service` 持久承载，`Linger=yes`、`Restart=no`、代理变量清空，不联网、
不安装依赖。日志为
`results/air_interface_representation_audit/strict59_ra_ecmdm_r5_20260902.systemd.log`。

启动后只做一次 active/MainPID/audit-root/立即失败确认。若某子进程原生崩溃，父 runner 按本协议自行
隔离恢复，无需 systemd 重启；若 3 次均失败则全链 fail-stop，不得降低门槛或改用旧结果。

## 6. 解释边界

R5 仍只能按父协议 oracle 三门裁定。即使状态通过，EC-MDM 也只能称为 CPD 候选上位构念的 oracle
结构证据；observable estimability 未检验，deployability 未建立。若无标签估计失败，commissioning 必须
另冻并战胜最强相同目标标签预算基线。独立线仍不并入主线。
