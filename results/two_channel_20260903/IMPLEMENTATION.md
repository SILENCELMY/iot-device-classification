# 实现登记

协议 `docs/PROTOCOL_TWO_CHANNEL_20260903.md`（含 §12 修正条款）
sha256 `21da959212135497d85c1fc3ea503597013d859c4c3b1e1cfd220a62c79dc4c6`

## runner

```text
路径    results/two_channel_20260903/run_two_channel.py
行数    589
sha256  
```

**协议 §19.7 的白名单使 `results/**/*.py` 不入库**，故此处登记 sha256 以固定实现版本。
（协议卫生待办：后续实验的 runner 应放在 `code/scripts/analysis/` 下以获得版本控制，
并相应调整 `REPO = Path(__file__).resolve().parents[N]` 的层数。）

## 实现要点对照协议

- 族导出 §2.1：按列名首个下划线段分组、成员 <2 并入 `singletons`；实测 94 列 → 10 组，
  与协议冻结的输出逐项一致；元数据型列检查为空。
- §12.5：**不做簇导出**，两个坐标在全部类对上聚合；`TAU` 未被使用。
- §2.3 两段制：代理量（逐类对 ΔAUC_tgt 求和）提议，接受须重训四模型并在 9 个跨轮次任务的
  macro-F1 上同时满足 Σ>0、G/|L|>=3、|W|<=0.5×A。
- §2.4 + §12.1 + §12.6：坐标 2 双触发器（`m_src_max>0` 仅 |S|>=2、`net_dis_src>0` 全档），
  接受证据按 |S| 分档（内层 LORO / 嵌套源域 OOF）。嵌套元层用同超参
  `LogisticRegression(max_iter=2000, class_weight="balanced")`。
- 最佳基模型一律按**源域 OOF** 准确率选，不读目标标签。
- `n_jobs=1`、`random_state=42`、线程钉死；双跑比对 6 个判定性产物的 md5。
