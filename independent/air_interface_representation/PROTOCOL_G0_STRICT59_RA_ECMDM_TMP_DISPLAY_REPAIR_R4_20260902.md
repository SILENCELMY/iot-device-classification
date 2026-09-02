# G0 strict59_ra EC-MDM R4 `/tmp` 显示路径修复协议（重跑前冻结）

**冻结日期**：2026-09-02（Asia/Shanghai）  
**状态**：`TMP_DISPLAY_PATH_REPAIR_FROZEN_BEFORE_R4_RERUN`  
**性质**：只修复冻结 G0 模块对仓库外 `/tmp` 重复 B 输出路径的显示假设，并换用全新 R4 roots；
不改变实际输入/输出位置、特征、数值实现、任务、模型、种子、重复、指标、科学门、状态树或解释边界。

## 1. 继承链

本协议完整继承且不得修改：

| 冻结对象 | SHA-256 |
|---|---|
| `PROTOCOL_G0_STRICT59_RA_ECMDM_RECALIBRATION_20260902.md` | `d3c0c19821effee9ce1c9370fe6dc0046742c0f2038a887c34b8952445bfee5a` |
| `PROTOCOL_G0_STRICT59_RA_ECMDM_PATH_REPAIR_R2_20260902.md` | `ad0644d3f396fcbe44caac8e4c4398fb6eda23dd872c18cddcd50b1cb6047269` |
| `PROTOCOL_G0_STRICT59_RA_ECMDM_TERMINAL_RECOVERY_R3_20260902.md` | `e98aef970698827e531a685f6d5f87a30ce68a1f6360138556a4621fc710ea01` |
| `G0_STRICT59_RA_ECMDM_R3_IMPLEMENTATION_FREEZE.json` | `77584dc701f10f68759e960d559e77616fc8c281a0247fcee29d3d3ad219192f` |

父协议要求重复 B 与 full94 复现只放 `/tmp`，本协议继续严格遵守。oracle recoverability、
`excess_F >= 0.020` 实质量级、C-first 结构门，以及 oracle recoverability / observable estimability /
deployability 三层证据边界全部不变。

## 2. R3 失败事实与冻结快照

R3 由 user-systemd unit `strict59-ra-ecmdm-r3-20260902.service` 持久承载。full94 双跑判否门、R2--R7
RA 重提取和正式 G0 重复 A 的 648/648 模型单元均完成；进入 `S2B_G0` 后，在任何重复 B 模型拟合前
fail-stop。`FAILED.json` 错误类型为 `ValueError`：

```text
'/tmp/strict59_ra_ecmdm_blgigwiw/g0_b' is not in the subpath of
'/home/lmy/iot-device-classification'
```

触发点是冻结 `environment_grid_experiment.py` 第 163 行打印
`out_root.relative_to(REPO_ROOT)`。冻结源码中 `REPO_ROOT` 的三个运行期引用（第 131、163、223 行）均只
用于打印相对显示路径；实际缓存和输出分别由 wrapper 显式设置的绝对 `CACHE_SRC`、`OUT_ROOT` 决定。
因此这是包装层未适配 `/tmp` 显示锚点的工程失败，不是输入、数值、可重复性或科学门失败。

R3 staging 在 unit 退出后的冻结快照：

| root | 状态 | 文件数 | 总字节 | `metrics.json` 数 | 树摘要 |
|---|---:|---:|---:|---:|---|
| `results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r3/` | 存在 | 6 | 10,433,311 | 0 | `a8f45b1eb216cf3e263356507358dbb99ea4eca882313fff031ecb9d918c4a3b` |
| `results/g0_environment_grid_strict59_ra_r3/` | 存在 | 5,835 | 20,886,330,456 | 648 | `0b628f1164322daa31cc251c1e23e1f4730a921e5a813212dc3a7b85353294a0` |
| `results/meta_mismatch_exploratory/strict59_ra_ecmdm_r3/` | 不存在 | 0 | 0 | 0 | — |

R3 G0“树摘要”定义为：为 root 下所有普通文件生成 GNU `sha256sum` 标准行，对这些行按完整行字节序排序，
再对完整行流计算 SHA-256；它同时锁定路径集合与每个文件内容哈希。R3 audit root 的摘要沿用 R3 协议定义。

R3 audit 六个文件的逐文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `FAILED.json` | `973c77c10d391466564f24a47678e7f69be78a088dab5867bdddbc6c35df906d` |
| `extraction_audit.json` | `f937d7b49c8e8d2fe6aa9abbed56457d82fe845c0c0add88d03a2247f7456c62` |
| `full94_reproduction_gate.json` | `266a4e32bc2397adadaba253e43825cbab223472fcdd235e8011f40298640d3f` |
| `input_audit.json` | `6675ae12c0c9544cf4887127a4f63b54c60539a8e040e2d0727017f5ccc5dd7c` |
| `pcap_manifest.json` | `0d6d1822a57c2e7f5e2edf3a37cd1fe8193edde084d04ae3c8f26b787239dead` |
| `strict59_ra_features_raw_all_w10.csv` | `f7430bc32d50b66d1e4975f1bf2678d55fd692c441bec0093965a77969e10ca5` |

承载日志 `strict59_ra_ecmdm_r3_20260902.systemd.log` 为 64,419 字节，SHA-256
`bf5d3ed9631c7ff432da01f04e1f7c8858515ef1e6a5607c008cb3207467d0e1`。临时 root
`/tmp/strict59_ra_ecmdm_blgigwiw` 存在（342 个普通文件，78,127,840 字节）。R3 正式 roots、日志与临时
root 均只保留取证，不得删除、改名、补写、覆盖或复用。

## 3. R4 唯一允许的实现变化

1. 正式输出常量只改为以下全新 R4 roots：

   ```text
   results/g0_environment_grid_strict59_ra_r4/
   results/meta_mismatch_exploratory/strict59_ra_ecmdm_r4/
   results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r4/
   ```

2. `run_g0()` 保存并在 `finally` 恢复冻结模块的 `g0.REPO_ROOT`。若实际 `output_root` 位于原仓库 root
   之外，只在该调用期间把 `g0.REPO_ROOT` 设为二者的词法公共祖先；对 `/home/...` 与 `/tmp/...` 即 `/`。
   该变量在冻结 G0 源码中的运行期用途仅为三处 `relative_to()` 打印，不参与路径选择或数值计算。
3. `g0.CACHE_SRC` 和 `g0.OUT_ROOT` 仍分别指向同一 R4 物化缓存与真实 `/tmp/.../g0_b`；禁止把 B 写入
   仓库、禁止复用 R3 G0-A、禁止修改冻结 G0 源码。
4. 新增合成测试：仓库内输出保持原显示锚点；仓库外临时输出得到可用公共显示锚点；`run_g0()` 无论
   成功或异常都恢复 `g0.REPO_ROOT`。
5. runner 新增本修复协议/freeze 的 SHA 校验及 CLI 显式 SHA；provenance 增加 R4 继承字段。
6. 除上述显示锚点、R4 roots、冻结链校验、测试与 provenance 外，R3 runner 和测试不得改变。

三处 R4 roots 在本协议冻结时均不存在。实现后另写 R4 implementation freeze；正式运行前只允许合成测试
和 `--preflight-no-fit`。R4 必须从 full94 双跑判否门开始完整重跑，不得读取或续跑任何 R2/R3 staging。

## 4. 持久承载

R4 继续使用 user-systemd，且已确认 `Linger=yes`：

- unit：`strict59-ra-ecmdm-r4-20260902.service`
- 日志：`results/air_interface_representation_audit/strict59_ra_ecmdm_r4_20260902.systemd.log`
- `Restart=no`，代理变量显式清空，不联网、不安装依赖。

启动后只做一次 `active (running)`、MainPID、R4 audit root 与立即失败检查，之后停止轮询。

## 5. 解释边界

R4 仍只能按父协议 oracle 三门裁定。即使状态通过，EC-MDM 也只能称为 CPD 候选上位构念的 oracle
结构证据；observable estimability 未检验，deployability 未建立。若后续无标签估计失败，commissioning
必须另冻并战胜最强相同目标标签预算基线。独立线仍不并入主线。
