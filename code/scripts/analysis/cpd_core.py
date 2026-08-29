#!/usr/bin/env python3
"""cpd_core — 混淆模式漂移（CPD）与模型分歧漂移（UDS）的唯一实现。

协议依据：docs/experiment_protocol_final.md 第 4 节（定义）、第 11 节（本文件为唯一实现）。

本模块是 `CPD_y` / `CPD_dir` / `UDS` 的唯一合法实现。所有分析脚本必须 import 本模块，
不得保留私有 `compute_cpd` / `_normalize_cm` 副本（协议 §11、§20.2）。

三个指标的标签需求（协议 §4.1）：

    指标        需要目标环境真实标签   可用于部署前   依赖性质
    CPD_y       是                     否             performance-dependent
    CPD_dir     是                     否             已去误差幅度，非完全独立于性能
    UDS         否                     是             prediction-only

泄漏审计声明（协议 §17.1 条件 5）：
    `uds()` 的签名只有 `pred_src_oof` / `pred_tgt` 两个位置参数与一个 `class_order`
    关键字参数，**没有任何标签入参**，函数体内也不读取任何真实标签。
    `test_cpd_core.py::test_uds_signature_takes_no_labels` 用 `inspect.signature`
    对此做机器可验证的断言。

命名纪律（协议 §4.4）：已废弃 `cpd_env` 与"无标签 CPD"这两个叫法；无标签指标统一称 `UDS`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "normalize_cm",
    "off",
    "cpd_y",
    "cpd_dir",
    "uds",
    "CpdDirResult",
    "dir_matrix",
    "disagreement_matrix",
    "DEFAULT_MIN_ERR",
]

#: `CPD_dir` 的默认逐行最少误分类样本数门槛（协议 §4.3）。
DEFAULT_MIN_ERR = 20


# --------------------------------------------------------------------------
# 基础算子
# --------------------------------------------------------------------------

def normalize_cm(cm) -> np.ndarray:
    """行归一化混淆矩阵：`C̃_ij = C_ij / Σ_k C_ik`（协议 §4.2）。

    沿用 `cpd_comprehensive_analysis.py:88-92` 的历史口径：行和为 0 时该行除以 1
    （即保持全零行），而不是产生 NaN。此口径不得更改——三个历史数值的复现依赖它。
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，收到 shape={cm.shape}")
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return cm / row_sums


def off(cm) -> np.ndarray:
    """置零对角，只保留离对角（误分类）质量：`Off(·)`（协议 §4.2）。"""
    out = np.array(cm, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，收到 shape={out.shape}")
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------
# CPD_y
# --------------------------------------------------------------------------

def cpd_y(cm_ref, cm_tgt) -> float:
    """`CPD_y(C_ref, C_tgt) = || Off(C̃_ref) − Off(C̃_tgt) ||_F`（协议 §4.2）。

    需要目标环境真实标签，**不可用于部署前**（协议 §4.1）。

    注意其代数性质（协议 §4.3）：行归一化后每行离对角质量恰为 `1 − recall_i`，
    因此 `CPD_y` 在代数上内含误差幅度。用它预测 ΔF1 时，相关性有一部分是代数必然，
    结论只能作描述性证据（协议 §2.3）。
    """
    a = off(normalize_cm(cm_ref))
    b = off(normalize_cm(cm_tgt))
    if a.shape != b.shape:
        raise ValueError(f"两个混淆矩阵维度不一致：{a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b, ord="fro"))


# --------------------------------------------------------------------------
# CPD_dir
# --------------------------------------------------------------------------

@dataclass
class CpdDirResult:
    """`cpd_dir` 的返回值：数值 + 逐行纳入/缺失标注（协议 §4.3 要求"标注缺失行"）。"""

    value: float
    included_rows: tuple[int, ...] = ()
    excluded_rows: tuple[int, ...] = ()
    n_err_ref: tuple[float, ...] = ()
    n_err_tgt: tuple[float, ...] = ()
    min_err: int = DEFAULT_MIN_ERR
    n_classes: int = 0
    notes: tuple[str, ...] = field(default=())

    def __float__(self) -> float:
        return float(self.value)

    @property
    def is_defined(self) -> bool:
        """是否有任何一行通过 `min_err` 门槛。全部缺失时 `value` 为 NaN。"""
        return len(self.included_rows) > 0


def dir_matrix(cm, min_err: int = DEFAULT_MIN_ERR):
    """逐行方向分布 `D_ij = P(ŷ=j | y=i, ŷ≠i)`，以及每行的原始误分类计数。

    由原始计数直接算：`D_ij = C_ij / n_err_i`（`j ≠ i`，`n_err_i = Σ_k C_ik − C_ii`）。
    这与"先行归一化再按 L1 归一化离对角向量"在代数上等价（协议 §4.2），
    但保留了 `n_err_i` 以便施加 §4.3 的 `min_err` 门槛。

    返回 `(D, n_err)`；未达门槛的行在 `D` 中置为 NaN。

    Args:
        cm: **原始计数**混淆矩阵（不可传入已归一化的矩阵，否则 `min_err` 门槛失效）。
        min_err: 逐行最少误分类样本数门槛。
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，收到 shape={cm.shape}")
    if min_err < 0:
        raise ValueError(f"min_err 必须非负，收到 {min_err}")

    offd = off(cm)
    n_err = offd.sum(axis=1)

    denom = np.where(n_err > 0, n_err, 1.0)[:, None]
    d = offd / denom
    d[n_err < min_err, :] = np.nan
    return d, n_err


def cpd_dir(cm_ref, cm_tgt, min_err: int = DEFAULT_MIN_ERR) -> CpdDirResult:
    """`CPD_dir(C_ref, C_tgt) = || D_ref − D_tgt ||_F`（协议 §4.2、§4.3）。

    只保留误分类**方向**，已去除误差**幅度**。但**不得声称完全独立于性能**：
    误差计数带来的估计噪声未被去除——误差极少的行方向分布估计方差极大，
    会系统性抬高高准确率环境的 `CPD_dir`（协议 §4.1、§4.3）。

    因此逐行施加 `n_err ≥ min_err` 门槛；ref 或 tgt 任一侧不达标的行整行剔除，
    并在返回值中标注。Frobenius 范数只在共同纳入的行上计算。

    RQ1 使用本指标时必须附置换/随机标签参照，给出无真实漂移下的零分布（协议 §4.3）。

    Args:
        cm_ref, cm_tgt: **原始计数**混淆矩阵。
        min_err: 逐行最少误分类样本数门槛，默认 20（协议 §4.3）。

    Returns:
        `CpdDirResult`。全部行缺失时 `value` 为 `nan` 且 `is_defined` 为 False。
    """
    d_ref, n_err_ref = dir_matrix(cm_ref, min_err=min_err)
    d_tgt, n_err_tgt = dir_matrix(cm_tgt, min_err=min_err)
    if d_ref.shape != d_tgt.shape:
        raise ValueError(f"两个混淆矩阵维度不一致：{d_ref.shape} vs {d_tgt.shape}")

    n_classes = d_ref.shape[0]
    keep = (n_err_ref >= min_err) & (n_err_tgt >= min_err)
    included = tuple(int(i) for i in np.flatnonzero(keep))
    excluded = tuple(int(i) for i in np.flatnonzero(~keep))

    notes: list[str] = []
    if excluded:
        detail = ", ".join(
            f"row {i}(n_err_ref={n_err_ref[i]:.0f}, n_err_tgt={n_err_tgt[i]:.0f})"
            for i in excluded
        )
        notes.append(f"min_err={min_err} 门槛剔除 {len(excluded)} 行：{detail}")

    if not included:
        notes.append("无任何行达到 min_err 门槛，CPD_dir 未定义")
        value = float("nan")
    else:
        diff = d_ref[keep, :] - d_tgt[keep, :]
        value = float(np.linalg.norm(diff, ord="fro"))

    return CpdDirResult(
        value=value,
        included_rows=included,
        excluded_rows=excluded,
        n_err_ref=tuple(float(v) for v in n_err_ref),
        n_err_tgt=tuple(float(v) for v in n_err_tgt),
        min_err=min_err,
        n_classes=n_classes,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# UDS —— 无标签，部署前可算
# --------------------------------------------------------------------------

def _as_pred_mapping(preds, what: str) -> dict[str, np.ndarray]:
    """把 `{model: 预测数组}` 映射或 DataFrame（列=模型）统一成 dict。

    只接受**预测**。本函数不区分、也不接受真实标签列——调用方若把标签列混入，
    它会被当成"另一个模型"参与分歧统计，因此上游必须只传模型预测。
    """
    if hasattr(preds, "columns") and hasattr(preds, "to_numpy"):  # pandas DataFrame
        out = {str(c): np.asarray(preds[c]) for c in preds.columns}
    elif isinstance(preds, dict):
        out = {str(k): np.asarray(v) for k, v in preds.items()}
    else:
        raise TypeError(
            f"{what} 必须是 {{model_name: predictions}} 映射或 DataFrame（列=模型），"
            f"收到 {type(preds)!r}"
        )
    if len(out) < 2:
        raise ValueError(f"{what} 至少需要 2 个模型才能构成分歧矩阵，收到 {len(out)} 个")

    lengths = {k: len(v) for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{what} 各模型预测长度不一致：{lengths}")
    return out


def disagreement_matrix(pred_a, pred_b, class_order) -> np.ndarray:
    """分歧矩阵 `G^{ab}`（行归一化，协议 §4.2）。

    `G^{ab}_cj` = 在 `m_a` 预测为类 `c` 的样本上，`m_b` 预测为类 `j` 的比例。
    只用两个模型的预测，不读取任何真实标签。
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if pred_a.shape != pred_b.shape:
        raise ValueError(f"两个模型预测长度不一致：{pred_a.shape} vs {pred_b.shape}")

    index = {c: i for i, c in enumerate(class_order)}
    k = len(class_order)
    g = np.zeros((k, k), dtype=float)
    for a_val, b_val in zip(pred_a, pred_b):
        i = index.get(a_val)
        j = index.get(b_val)
        if i is None or j is None:
            raise ValueError(
                f"预测值 {a_val!r}/{b_val!r} 不在 class_order={list(class_order)} 中"
            )
        g[i, j] += 1.0
    return normalize_cm(g)


def uds(pred_src_oof, pred_tgt, *, class_order=None) -> float:
    """`UDS`：模型分歧漂移（协议 §4.2）。

        UDS = mean over (a,b) pairs of || Off(G^{ab}_src) − Off(G^{ab}_tgt) ||_F

    源域用 OOF 预测、目标域用测试预测。对所有**有序**模型对 `(a, b)`（`a ≠ b`）取均值——
    `G^{ab}` 与 `G^{ba}` 不同，两者都计入。

    **本函数不读取任何标签**（协议 §4.2、§17.1 条件 5）。签名中没有标签参数；
    若上游把真实标签列混入 `pred_*` 映射，它会被当作"另一个模型"，这属于调用方错误。

    若使用整批无标签目标样本，结果须标注为 transductive batch setting，
    并补目标批量大小敏感性（协议 §4.1、§7 第 8 条）。

    Args:
        pred_src_oof: 源域 OOF 预测，`{model_name: predictions}` 或 DataFrame（列=模型）。
        pred_tgt: 目标域测试预测，同结构、同模型集合。
        class_order: 类别轴顺序。默认取源域与目标域全部预测值的并集排序后的结果，
            以保证两侧矩阵同形同序。

    Returns:
        所有有序模型对的平均分歧漂移。
    """
    src = _as_pred_mapping(pred_src_oof, "pred_src_oof")
    tgt = _as_pred_mapping(pred_tgt, "pred_tgt")

    if set(src) != set(tgt):
        raise ValueError(
            f"源域与目标域模型集合必须一致：src={sorted(src)} vs tgt={sorted(tgt)}"
        )

    if class_order is None:
        values = set()
        for arr in list(src.values()) + list(tgt.values()):
            values.update(np.asarray(arr).tolist())
        class_order = sorted(values, key=str)
    class_order = list(class_order)

    models = sorted(src)
    distances = []
    for a in models:
        for b in models:
            if a == b:
                continue
            g_src = off(disagreement_matrix(src[a], src[b], class_order))
            g_tgt = off(disagreement_matrix(tgt[a], tgt[b], class_order))
            distances.append(float(np.linalg.norm(g_src - g_tgt, ord="fro")))

    if not distances:
        raise ValueError("模型对为空，无法计算 UDS")
    return float(np.mean(distances))
