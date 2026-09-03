# HISTINT-OBJ-SIGNAL 判定书

**判定**：`OBJECTIVE_SIGNAL_AGREES`（分支 2）

规格 sha256 `71fabf3a717a4f13e8f42705c4a68a31a46f59ae504c85e5db680e13ce173e06`
主判据：`max_base` 口径下 `rssi` 排第一（二值，运行前定死）。

## S_obj 排序（主口径 max_base）

| 族 | 列数 | `S_obj` | `ΔLORO` | `ΔJOINT` | 冻结 `S(F)` |
|---|---:|---:|---:|---:|---:|
| `rssi` | 8 | +0.4140 | +0.2920 | -0.0406 | +5.2440 |
| `singletons` | 13 | +0.0182 | -0.0014 | -0.0065 | -0.2281 |
| `subtype` | 16 | +0.0076 | -0.0038 | -0.0038 | -0.0575 |
| `interarrival` | 12 | +0.0045 | -0.0142 | -0.0062 | -0.2262 |
| `unique` | 2 | +0.0041 | +0.0012 | -0.0010 | -0.3346 |
| `subwin` | 9 | -0.0104 | -0.0127 | -0.0008 | +0.0156 |
| `down` | 6 | -0.0115 | -0.0154 | -0.0013 | -0.6186 |
| `up` | 7 | -0.0190 | -0.0377 | -0.0062 | -0.3667 |
| `burst` | 7 | -0.0210 | -0.0396 | -0.0062 | -0.8892 |
| `len` | 14 | -0.0439 | -0.1610 | -0.0390 | -0.7237 |

头名/次名比值：22.774322222311234；正值族数：5（rssi, singletons, subtype, interarrival, unique）
对冻结 `S(F)` 的 Spearman：0.8182（只报不设门槛）

`best_base_src` 口径排序：rssi > interarrival > unique > subtype > down > singletons > subwin > up > burst > len

`JOINT`(c_full, max_base) = 0.9101，MAINLINE §2 参照 0.8936，差 +0.0165

恒等式 `S_obj = ΔLORO − 3×ΔJOINT` 逐族断言通过（容差 1e-12）。

本规格未执行任何删除动作、未改变任何配置、未产生新的流程终点。
