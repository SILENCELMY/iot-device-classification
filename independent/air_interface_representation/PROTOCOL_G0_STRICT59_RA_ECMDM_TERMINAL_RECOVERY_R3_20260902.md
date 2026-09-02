# G0 strict59_ra EC-MDM R3 终端承载恢复协议（重跑前冻结）

**冻结日期**：2026-09-02（Asia/Shanghai）  
**状态**：`TERMINAL_CARRIER_RECOVERY_FROZEN_BEFORE_R3_RERUN`  
**性质**：只把正式运行从交互 PTY 改为服务器 user-systemd 持久任务，并换用全新 R3 roots；不改变
特征、数值实现、任务、模型、种子、重复、指标、科学门、状态树或解释边界。

## 1. 继承链

本协议完整继承且不得修改：

| 冻结对象 | SHA-256 |
|---|---|
| `PROTOCOL_G0_STRICT59_RA_ECMDM_RECALIBRATION_20260902.md` | `d3c0c19821effee9ce1c9370fe6dc0046742c0f2038a887c34b8952445bfee5a` |
| `PROTOCOL_G0_STRICT59_RA_ECMDM_PATH_REPAIR_R2_20260902.md` | `ad0644d3f396fcbe44caac8e4c4398fb6eda23dd872c18cddcd50b1cb6047269` |
| `G0_STRICT59_RA_ECMDM_R2_IMPLEMENTATION_FREEZE.json` | `c1f4990c19c4e3d1447ad7207308fe42bdba32b50c61b396198d39b1566a698c` |

父协议关于 oracle recoverability、`excess_F >= 0.020` 实质量级、C-first 结构门以及
oracle recoverability / observable estimability / deployability 三层证据边界全部不变。

## 2. R2 终止事实与冻结快照

R2 正式运行于 2026-09-02 16:29:37（Asia/Shanghai）由交互 `pts/0` 启动。确认其 Python runner
PID `1295060` 仍从属于该交互会话后，于 17:25 发送一次 `SIGINT`；runner 自行在 `S2A_G0` 阶段写入
`FAILED.json`，错误类型为 `KeyboardInterrupt`。这只是运行承载终止，不是科学门、输入完整性、数值一致性
或模型训练错误；R2 没有形成可裁定结果。

R2 staging 在进程完全退出后的冻结快照如下：

| root | 状态 | 文件数 | 总字节 | `metrics.json` 数 | 树摘要 |
|---|---:|---:|---:|---:|---|
| `results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r2/` | 存在 | 6 | 10,433,922 | 0 | `42afdf34243fe27f678c3cad2ef24591818b808239273fc08ac16d27925dc415` |
| `results/g0_environment_grid_strict59_ra_r2/` | 存在 | 2,835 | 10,562,160,576 | 315 | `5bd46658601e536da925ea3aecbd73f8813b68c122f2b61f971f31d4fc523fa9` |
| `results/meta_mismatch_exploratory/strict59_ra_ecmdm_r2/` | 不存在 | 0 | 0 | 0 | — |

“树摘要”定义为：按仓库相对路径字节序排列该 root 下所有普通文件，对每个文件生成 GNU
`sha256sum` 标准行，再对完整行流计算 SHA-256。它同时锁定路径集合与每个文件内容哈希。

R2 audit 六个文件的逐文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `FAILED.json` | `3e4ab6fb70bfdc66e72f76e47de8c58209d1c2f690cfa2636500a69f45bf42ee` |
| `extraction_audit.json` | `f937d7b49c8e8d2fe6aa9abbed56457d82fe845c0c0add88d03a2247f7456c62` |
| `full94_reproduction_gate.json` | `7ea6fae9ea7397522aea8fd28b11f14538d53fbef1ef3f03a557f53b5ccf5fa6` |
| `input_audit.json` | `6675ae12c0c9544cf4887127a4f63b54c60539a8e040e2d0727017f5ccc5dd7c` |
| `pcap_manifest.json` | `0d6d1822a57c2e7f5e2edf3a37cd1fe8193edde084d04ae3c8f26b787239dead` |
| `strict59_ra_features_raw_all_w10.csv` | `f7430bc32d50b66d1e4975f1bf2678d55fd692c441bec0093965a77969e10ca5` |

临时 root `/tmp/strict59_ra_ecmdm_thmq2qta` 在终止时存在（342 个普通文件，78,127,840 字节），只保留
为失败取证，不得读取、删除、补写或复用。以上 R2 roots 同样不得删除、改名、补写、覆盖或复用。

## 3. R3 唯一允许的实现变化

1. 正式输出常量只改为以下全新 R3 roots：

   ```text
   results/g0_environment_grid_strict59_ra_r3/
   results/meta_mismatch_exploratory/strict59_ra_ecmdm_r3/
   results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r3/
   ```

2. runner 新增本恢复协议及其 freeze 的 SHA 校验，并要求 CLI 显式提供本协议 SHA-256。
3. 测试只新增 R3 继承链、R3 roots 与旧 R2 roots 不相等的断言。
4. 除上述常量、静态校验、provenance 字段与 CLI 参数外，R2 runner 和测试不得改变。
5. 实现后必须另写 R3 implementation freeze；正式运行前只允许合成测试和 `--preflight-no-fit`。

三处 R3 roots 在本协议冻结时均不存在。R3 必须从 full94 双跑判否门开始完整重跑，不得从 R2 staging
或临时 root 续跑。

## 4. 持久运行承载

正式 R3 只能以 user-systemd transient service 启动：

- unit：`strict59-ra-ecmdm-r3-20260902.service`
- 工作目录：`/home/lmy/iot-device-classification`
- Python：`/home/lmy/anaconda3/envs/iotcls/bin/python`
- 合并日志：
  `results/air_interface_representation_audit/strict59_ra_ecmdm_r3_20260902.systemd.log`
- 所有父协议禁止的代理变量在 service 环境中显式清空；不联网、不安装依赖。
- unit 必须带 `Restart=no`，科学或工程失败不得自动重试。

启动后只允许一次性确认 unit 为 `active (running)`、MainPID 非零、R3 audit root 已创建且日志没有立即
fail-stop；此后不轮询、不自动重启、不自动裁定。VS Code/SSH 断开不得终止该 user service。

## 5. 解释边界

R3 仍只能按父协议的 oracle 三门裁定。即使状态为
`EC_MDM_ORACLE_CANDIDATE_SUPPORTED_STRICT59_RA`，也只能称为 CPD 候选上位构念的 oracle 结构证据；
observable estimability 未检验，deployability 未建立。若后续无标签估计失败，commissioning 必须另冻，
并战胜最强相同目标标签预算基线。独立线仍不并入主线。
