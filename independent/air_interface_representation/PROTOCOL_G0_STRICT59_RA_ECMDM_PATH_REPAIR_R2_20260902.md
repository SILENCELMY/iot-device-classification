# G0 strict59_ra EC-MDM R2 source_file 路径修复协议（重跑前冻结）

**冻结日期**：2026-09-02（Asia/Shanghai）  
**状态**：`IMPLEMENTATION_PATH_REPAIR_FROZEN_BEFORE_RERUN`  
**性质**：只修复 pcap 重建表与缓存表的 `source_file` 绝对/相对路径表示；不改变特征值、任务、模型、
种子、M1/M1-R/M2 定义、科学门、状态树或解释边界。

## 1. 继承协议与首次失败

完整继承 `PROTOCOL_G0_STRICT59_RA_ECMDM_RECALIBRATION_20260902.md`，SHA-256
`d3c0c19821effee9ce1c9370fe6dc0046742c0f2038a887c34b8952445bfee5a`。原协议已由 commit `e8adaad`
先于任何本轮数值冻结；原实现由 commit `39b0fde` 冻结。

首次正式 root
`results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902/` 只保留：

| 文件 | SHA-256 |
|---|---|
| `FAILED.json` | `0496449ae0b372e3f8f40960a063fe8e02b3d1849a4dda6825ff23db7fd7a3f9` |
| `full94_reproduction_gate.json` | `8d5ec8a09badbd804f667dcf5fa1078722700ead64d396de6a13e07fb7388064` |

失败阶段为 `S1_MATERIALIZATION`，签名为 `pcap reconstructed keys differ from source cache`。失败发生在
30 个 pcap 重提取后的内存对齐门，任何 `strict59_ra` G0 模型拟合前；G0 与 M1/M1-R/M2 新正式 roots
均未创建。full94 双跑判否门已通过，但 R2 为保持完整执行链，允许按原代码不变地再跑一次。

失败 root 与 `/tmp/strict59_ra_ecmdm_lvhfo416` 不得删除、补写、改名、覆盖或复用。

## 2. 根因

冻结缓存的 `source_file` 是仓库内 pcap 的**绝对路径**；首次 runner 向 `summarize_window()` 传入相对路径，
所以内存重建表的 `source_file` 是相对路径。二者指向同一哈希锁定 pcap，但首次 runner 把
`source_file` 纳入四列键后直接逐字符串比较，因表示不同而 fail-stop。

既有、已接受的 R2--R4 方向修复 runner 本来只用 `(label, round, window_id)` 对齐 5506 个窗口，并把
诊断表的 `source_file` 记录为相对路径；因此该失败不是窗口或方向特征值不一致，而是新 wrapper 扩展出的
路径规范化遗漏。

## 3. 允许的唯一实现变化

1. tshark 仍读取同一 30 个 SHA-256 锁定的 pcap；传给 `preprocess_packets()` 与
   `summarize_window()` 的路径改为 `(REPO_ROOT / relative_path).resolve()`，使重建 meta 与缓存中的绝对
   `source_file` 逐字符串一致。
2. 全 11303 行重建表、方向表与缓存表仍用
   `(label, round, source_file, window_id)` 四列唯一键对齐，不放松缓存完整性门。
3. 对已接受的 R2--R4 诊断表，先要求其相对 `source_file` 逐行 `resolve()` 后等于当前绝对
   `source_file`，再仅以其既有唯一键 `(label, round, window_id)` 对齐 5506 行并比较 14 个 `ra_*`
   字段至 `1e-9`。不得忽略、模糊匹配或重写路径。
4. 新增合成测试覆盖：相对/绝对路径必须解析到同一文件；不同文件即失败；四列缓存键仍保持严格。
5. 除上述路径规范化、测试与 R2 root 常量外，runner 与测试不得改变。原科学协议 §3--§10 全部不变。

## 4. R2 输出与执行

唯一新正式 roots：

```text
results/g0_environment_grid_strict59_ra_r2/
results/meta_mismatch_exploratory/strict59_ra_ecmdm_r2/
results/air_interface_representation_audit/strict59_ra_ecmdm_recalibration_20260902_r2/
```

三处在本协议冻结时必须不存在，存在即停止。实现修复后另写 R2 implementation freeze，锁父协议、本修复
协议、runner、tests 与全部原冻结依赖 SHA；只允许合成测试和 `--preflight-no-fit`，通过后才可正式重跑。

R2 仍从 full94 双跑判否门开始，随后完整运行 R2--R7 物化、G0 A/B、M1 A/B、M1-R A/B、M2 A/B 与
原状态树。不得从首次失败内存续跑，不得读取首次运行未落盘的方向值。

## 5. 解释边界

R2 只能按父协议原 oracle recoverability、实质量级与 C 结构三门裁定。无论结果为何，observable
estimability 仍未裁定，deployability 仍未建立，EC-MDM 最多仍是 CPD 的候选上位构念。若 oracle 三门
全过，才允许另冻无标签估计；若无标签估计失败，commissioning 仍必须另冻并战胜相同标签预算基线。
