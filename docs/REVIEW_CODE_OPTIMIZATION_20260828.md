# 代码与工程优化建议（外部审阅）

**日期**：2026-08-28
**审阅者**：Claude（AI 辅助审阅）
**性质**：仅建议，未改动任何代码。所审代码为 `main` 分支 commit `97c2dec`。
**协议约束**：以下建议均不涉及实验设计变更（阈值、划分、指标定义、纳入排除标准），不触碰
`experiment_protocol_final.md` 的冻结条款；第 1、2 项属于"代码实现与协议不一致 / 实现缺陷"
范畴（冻结例外第 2 类），修改时按协议要求记录 Change Log。

总体评价：方法学纪律和代码质量高于平均水平——预注册协议、`cpd_core.py` 唯一实现 +
机器可验证的无泄漏断言、E1 三臂归因分解都很规范。以下问题按优先级排列。

---

## 一、可能影响实验结论（建议 9/7 P1/E1 截止前处理）

### 1. E1 的 A′ 臂折数硬编码为 2，归因分解只对旗舰任务成立

**位置**：`scripts/analysis/e1_oof_arms.py`，`ARMS` 常量：

```python
ARMS = {
    "A": ("random", 5),
    "A_prime": ("random", 2),   # <-- 固定 2 折
    "B": ("grouped", 5),
}
```

**问题**：A′ 的设计意图是"随机折叠、折数与 B 对齐"（docstring 与协议 §23 Change Log
2026-08-28 条均如此表述），但 B 的实际折数是动态的：

| 任务类型 | B 臂有效折数 | A′ 折数 | 是否对齐 |
|---|---|---|---|
| `loro_R2_R4_to_R3`（2 源轮次，旗舰） | 2 | 2 | ✅ |
| `loro_*`（其余 2 源轮次任务） | 2 | 2 | ✅ |
| `position/jitter_R2_R3_R4_to_*`（3 源轮次） | **3** | 2 | ❌ |
| `single_round_*` / `joint`（时间块） | **5** | 2 | ❌ |

用 `--tasks all` 跑全量 E1 时，"A vs A′ = 纯折数效应、A′ vs B = 纯分组效应"的分解在
3 源轮次和单轮任务上不成立。

**建议**：每个任务先计算 B 臂的有效折数（`folds_effective` 已在记录），A′ 的 `cv`
动态取该值，而不是硬编码 2。此改动不影响已跑完的旗舰任务预验证结果（该任务恰好对齐）。

### 2. Stacking 训练侧 OOF 概率缺少类别对齐

**位置**：`scripts/core/robust_iot_research.py`，`SimpleStackingClassifier.fit`：

```python
oof = np.zeros((len(x), len(self.classes_)))
for train_idx, val_idx in self._splitter(...):
    fold_model = clone(estimator)
    fold_model.fit(x.iloc[train_idx], y[train_idx])
    oof[val_idx] = fold_model.predict_proba(x.iloc[val_idx])   # <-- 假设 fold 见过全部类
```

**问题**：`fold_model.predict_proba` 的列对应 `fold_model.classes_`，若某折的训练数据
缺少某个类别，列数或列序将与 `self.classes_` 不一致——要么直接崩溃，要么概率列静默错位。
commit `6a3f678` 已在 `evaluate_model` 的**测试侧**按 `model.classes_` 回填缺失类，
但**训练侧 OOF** 未做同样处理。

自采数据 5 类均衡场景下风险低，但两个即将到来的场景风险高：

- **G0 环境组合网格**（协议 §8.5，约 156 任务）：部分环境组合可能缺类——
  `evaluate_model` 的注释自己也提到了这一点；
- **UNSW 18 类严重不均衡**（协议 §16.1：NestProtect 单日仅 7 个窗口），
  GroupKFold 折内缺类几乎必然。

**建议**：在 fit 内按 `fold_model.classes_` 做与测试侧相同的列回填。建议现在就修，
别等 P1/P2 批量任务跑到一半失败。可在 `test_oof_modes.py` 加一个"某折缺类"的构造用例。

### 3. 协议 §19.2 的持久化要求主管道尚未落地

**位置**：`robust_iot_research.py` 的 `evaluate_model` 与 `save_environment_report`。

**现状**：`save_environment_report` 只记录可选模块可用性；`evaluate_model` 存了
predictions / pred_proba / oof_meta / model.joblib，但**没有**：

- `train_idx.npy` / `test_idx.npy` / `split_metadata.json`（§19.2 明确要求；
  `robustness_scaling_20260706_v2/splits/` 有，主管道没有）；
- git commit hash、完整命令行（`sys.argv`）、种子、关键包版本。

**影响**：登记表要求"每个强结论可定位到配置、种子、输入"（§19.1、§19.3），
且当前代码和结果分在两个分支，commit ↔ 结果的对应关系只能靠手工登记，容易漏
（登记表中 HIST-gpu_capacity、HIST-cnn_contrast 两行的 commit 列已经是"—"）。

**建议**：在 `save_environment_report` 中追加 `git rev-parse HEAD`、`sys.argv`、
`random_state`、核心包版本（`pip freeze` 摘要）；在 `task_data` 返回处顺手把
train/test 索引落盘。成本约十分钟，收益是可复现性链闭环。

---

## 二、工程与可复现性

### 4. 仓库结构与路径假设互相矛盾——克隆件目前跑不起来

三层矛盾：

1. `main`（代码）与 `research-results`（docs + results）是**内容不相交的孤儿分支**，
   一次只能检出一个，但代码运行时要读 `results/robust_v2/raw_all/features_raw_all_w10.csv`；
2. `e1_oof_arms.py` 的 `REPO_ROOT = Path(__file__).resolve().parents[3]` 假设代码位于
   `<项目根>/code/scripts/analysis/`——即仓库必须被克隆为大项目里名为 `code` 的子目录；
3. 两个分支的 README 命令全部引用 `code/scripts/...`，在克隆件中该路径不存在。

**最省事的修复**（不改任何代码）：用 git worktree 恢复原始布局——

```bash
git clone <repo> iot-project && cd iot-project
git checkout research-results          # 项目根 = docs/ + results/
git worktree add code main             # code/ = 代码分支
```

这样 `docs/`、`results/`、`code/` 同时在盘上，所有路径假设自动成立。
建议把这段写进 README。长期看更干净的做法是把 `REPO_ROOT` 改为环境变量或
`--project-root` 参数。

### 5. 依赖未锁版本

`requirements-core.txt` / `requirements-cloud.txt` 只有裸包名。协议 §19.5 要求
"干净环境最小复现"、P1 验收容差 1e-6——sklearn / XGBoost 大版本差异足以打破。
建议补 `requirements-lock.txt`（`pip freeze` 即可）；投稿代码快照必备。

### 6. 有测试、没 CI

`test_cpd_core.py`（15 项，含三个历史值回归断言与 UDS 无泄漏签名断言）和
`test_oof_modes.py`（11 项）质量都很好，但没有 CI。建议加一个最小 GitHub Actions
（约 15 行）在每次 push 跑 pytest。理由：`_splitter` 上周刚出过"缺 `return` 静默降级"
的缺陷（Change Log 2026-08-28 第 2 条），这正是 CI 能自动拦住的问题类型——
协议最怕被无声改坏的两处恰好都有断言覆盖。

---

## 三、小问题（顺手记录，不紧急）

- `cpd_core.disagreement_matrix` 用 Python 逐样本循环。380 窗口无所谓；UNSW 单日
  Dropcam 8640 窗口 × 18 类 × 有序模型对时会慢，届时用 `np.add.at` 向量化即可；
- Stacking 元学习器 `LogisticRegression` 未设 `max_iter`（默认 100）。5 类 × 3 模型
  = 15 维 meta 特征没问题；UNSW 18 类 × 3 模型 = 54 维时可能出收敛告警，建议显式设置；
- `main` 分支 README 已过时（见另一份文档 `REVIEW_DOCS_AUDIT_20260828.md`）。

---

## 四、明确不建议做的事

- 不要为"结果更好看"动 `_splitter`、指标定义或任务划分——冻结条款管着；
- 不要用 `robustness_scaling_experiment.py` 跑多种子（协议 §14 已禁，这里只是重申）；
- 历史报告不删除，只加注（协议冻结条款；文档层面的具体清单见
  `REVIEW_DOCS_AUDIT_20260828.md`）。
