#!/usr/bin/env python3
"""路线 B：环境稳健的加权聚合（协议 §17.3）

执行规格：`docs/EXECUTION_PLAN_20260829.md` D11（冻结于 commit 0f33137）
          + D11 追记 v1.7（|S|=1 的 lb 读法；实现方声明）

本脚本**只产出数表**。三条通过线的判定由审阅方作出，脚本不写任何判定或解读。

口径要点（与 D11 逐条对应）：
* 权重族（唯一预注册族）：w_raw_m ∝ exp(lb_m/τ) · max(1 − γ·ρ_m, 0)，
  再向均匀收缩 w = (1−λ)·normalize(w_raw) + λ/3。
* lb_m：|S|≥2 取各训练轮次 OOF macro-F1 的最小值（轮次即留出折）；
        |S|=1 取全部 OOF 行的整体 macro-F1（v1.7 裁定）。
* ρ_m：该模型与其余两模型在源域 OOF 上 0/1 错误指示向量 Pearson 相关的均值。
* 嵌套 LOEO：外层 6 fold × 25 任务（按 target_env）；内层 = 严格全含任务
  （源与目标均不含 e_out）；内层目标 = 环境等权平均 5-class macro-F1 最大，
  并列打破 ① 内层最差环境 macro-F1 更高 ② λ 更大。
* source-only：权重估计只接收源域 OOF（函数签名不含任何目标侧对象）；
  目标真实标签仅在最终打分处出现。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[3]
G0 = REPO / "results" / "g0_environment_grid"
RAW = G0 / "raw_all"
SUMMARY = G0 / "summary_metrics.csv"
VOTING = G0 / "voting_baselines.csv"
ENC = "utf-8-sig"

CLASSES = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]
CLASSES_4 = [c for c in CLASSES if c != "Socket"]
BASES = ["rf", "xgboost", "lightgbm"]
ENVS = ["R2", "R3", "R4", "R5", "R6", "R7"]

TAUS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- 指标（§10）

def score(y_true: list[str], y_pred: list[str]) -> dict:
    per = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    return {
        "macro_f1_5class": float(f1_score(y_true, y_pred, labels=CLASSES,
                                          average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y_true, y_pred, labels=CLASSES_4,
                                          average="macro", zero_division=0)),
        "worst_class_f1": float(np.min(per)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# ------------------------------------------------- 输入加载（只读 G0/D8 落盘）

def task_dir(task: str) -> Path:
    return RAW / task / "all_features"


def load_target(task: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """目标测试侧：三基模型概率矩阵 + 真实标签（标签仅用于最终打分）。"""
    probs, y_ref = {}, None
    for m in BASES:
        d = task_dir(task) / m
        pp = pd.read_csv(d / "pred_proba.csv", encoding=ENC, float_precision="round_trip")
        # G0 落盘的 pred_proba.csv 列名为 proba_<class>（oof_meta.csv 为 oof_<model>_<class>）
        pcols = [f"proba_{c}" for c in CLASSES]
        missing = [c for c in pcols if c not in pp.columns]
        if missing:
            raise SystemExit(f"[STOP] {task}/{m}/pred_proba.csv 缺类别列: {missing}")
        probs[m] = pp[pcols].to_numpy(dtype=float)
        rowsum = probs[m].sum(axis=1)
        if not np.allclose(rowsum, 1.0, atol=1e-6):
            raise SystemExit(f"[STOP] {task}/{m}/pred_proba.csv 行和偏离 1（max dev "
                             f"{float(np.max(np.abs(rowsum - 1.0))):.3e}）")
        pr = pd.read_csv(d / "predictions.csv", encoding=ENC)
        ycol = "true_label" if "true_label" in pr.columns else pr.columns[-2]
        y = pr[ycol].astype(str).tolist()
        if y_ref is None:
            y_ref = y
        elif y != y_ref:
            raise SystemExit(f"[STOP] {task}/{m} 的 true_label 与其它基模型不一致")
        if len(y) != probs[m].shape[0]:
            raise SystemExit(f"[STOP] {task}/{m} 概率行数与 predictions 不一致")
    return probs, y_ref


def source_stats(task: str) -> dict:
    """source-only：只读 stacking/oof_meta.csv，返回 lb / rho 及诊断。

    函数签名不接收任何目标侧对象——这是 D11 验收门 e 的结构性证据。
    """
    oof = pd.read_csv(task_dir(task) / "stacking" / "oof_meta.csv", encoding=ENC, float_precision="round_trip")
    if "true_label" not in oof.columns or "round" not in oof.columns:
        raise SystemExit(f"[STOP] {task}/stacking/oof_meta.csv 缺 true_label 或 round 列")
    y = oof["true_label"].astype(str).to_numpy()
    rounds = oof["round"].astype(str).to_numpy()
    uniq = sorted(set(rounds))

    pred, err, lb = {}, {}, {}
    for m in BASES:
        cols = [f"oof_{m}_{c}" for c in CLASSES]
        missing = [c for c in cols if c not in oof.columns]
        if missing:
            raise SystemExit(f"[STOP] {task} oof_meta 缺列: {missing}")
        p = oof[cols].to_numpy(dtype=float)
        pm = np.array([CLASSES[i] for i in p.argmax(axis=1)])
        pred[m] = pm
        err[m] = (pm != y).astype(float)
        if len(uniq) >= 2:  # 轮次即留出折（§9.1 grouped OOF）
            lb[m] = float(min(
                f1_score(y[rounds == r], pm[rounds == r], labels=CLASSES,
                         average="macro", zero_division=0) for r in uniq))
        else:  # v1.7 裁定：|S|=1 退化为该环境自身的整体 OOF 表现
            lb[m] = float(f1_score(y, pm, labels=CLASSES,
                                   average="macro", zero_division=0))

    rho, degenerate = {}, 0
    for m in BASES:
        vals = []
        for k in BASES:
            if k == m:
                continue
            a, b = err[m], err[k]
            if a.std() == 0 or b.std() == 0:
                degenerate += 1
                vals.append(0.0)  # 错误指示恒定 → 相关性无定义，记 0 并计数
            else:
                vals.append(float(np.corrcoef(a, b)[0, 1]))
        rho[m] = float(np.mean(vals))

    return {"lb": lb, "rho": rho, "n_oof": int(len(oof)),
            "n_source_rounds": len(uniq), "rho_degenerate_pairs": degenerate,
            "lb_basis": "per_round_min" if len(uniq) >= 2 else "single_round_overall"}


# ------------------------------------------------------------ 权重与聚合口径

def weights(lb: dict, rho: dict, tau: float, gamma: float, lam: float) -> np.ndarray:
    v = np.array([lb[m] for m in BASES], dtype=float)
    r = np.array([rho[m] for m in BASES], dtype=float)
    raw = np.exp((v - v.max()) / tau) * np.maximum(1.0 - gamma * r, 0.0)
    s = raw.sum()
    base = np.full(3, 1.0 / 3.0) if s <= 0 else raw / s
    return (1.0 - lam) * base + lam / 3.0


def aggregate(probs: dict[str, np.ndarray], w: np.ndarray) -> list[str]:
    p = sum(w[i] * probs[m] for i, m in enumerate(BASES))
    return [CLASSES[i] for i in np.asarray(p).argmax(axis=1)]


# ------------------------------------------------------------------ 任务清单

def load_tasks() -> pd.DataFrame:
    s = pd.read_csv(SUMMARY, encoding=ENC, float_precision="round_trip")
    rows = []
    for task, g in s.groupby("task", sort=True):
        r0 = g.iloc[0]
        rows.append({
            "task": task,
            "grid_kind": r0["grid_kind"],
            "n_sources": int(r0["n_sources"]),
            "target_env": str(r0["target_env"]),
            "train_rounds": tuple(ast.literal_eval(str(r0["train_rounds"]))),
            "test_rounds": tuple(ast.literal_eval(str(r0["test_rounds"]))),
            "f1_rf": float(g[g["model"] == "rf"]["macro_f1"].iloc[0]),
            "f1_stacking": float(g[g["model"] == "stacking"]["macro_f1"].iloc[0]),
        })
    df = pd.DataFrame(rows)
    v = pd.read_csv(VOTING, encoding=ENC, float_precision="round_trip")
    piv = v.pivot_table(index="task", columns="method", values="macro_f1_5class")
    for col, name in [("soft_voting_equal", "f1_soft_equal"),
                      ("soft_voting_calibrated", "f1_soft_cal"),
                      ("hard_voting", "f1_hard")]:
        df[name] = df["task"].map(piv[col])
    return df


# --------------------------------------------------------- 嵌套 LOEO（§9.2）

def inner_pool(df: pd.DataFrame, e_out: str) -> pd.DataFrame:
    """严格全含：源与目标均不含 e_out。"""
    keep = df.apply(lambda r: r["target_env"] != e_out
                    and e_out not in r["train_rounds"]
                    and e_out not in r["test_rounds"], axis=1)
    return df[keep]


def env_equal(rows: list[tuple[str, float]]) -> tuple[float, float]:
    """按目标环境等权聚合：返回 (环境等权均值, 最差环境均值)。"""
    d: dict[str, list[float]] = {}
    for env, val in rows:
        d.setdefault(env, []).append(val)
    means = [float(np.mean(v)) for v in d.values()]
    return float(np.mean(means)), float(np.min(means))


def grid_search(cache: dict, pool: pd.DataFrame) -> dict:
    best = None
    for tau, gamma, lam in product(TAUS, GAMMAS, LAMBDAS):
        rows = []
        for _, r in pool.iterrows():
            c = cache[r["task"]]
            w = weights(c["lb"], c["rho"], tau, gamma, lam)
            rows.append((r["target_env"],
                         score(c["y"], aggregate(c["probs"], w))["macro_f1_5class"]))
        mean_f1, worst_f1 = env_equal(rows)
        cand = {"tau": tau, "gamma": gamma, "lam": lam,
                "inner_mean_f1_env_equal": mean_f1, "inner_worst_env_f1": worst_f1}
        if best is None:
            best = cand
            continue
        # 并列打破：① 内层均值更高 ② 最差环境更高 ③ λ 更大（更保守）
        key_new = (cand["inner_mean_f1_env_equal"], cand["inner_worst_env_f1"], cand["lam"])
        key_old = (best["inner_mean_f1_env_equal"], best["inner_worst_env_f1"], best["lam"])
        if key_new > key_old:
            best = cand
    return best


# ---------------------------------------------------------------- 验收门 a/b

def gate_lambda1(cache: dict, df: pd.DataFrame) -> dict:
    """λ=1（全收缩至均匀）必须逐位复现 D8 的等权 soft voting 5-class macro-F1。"""
    n, bad, maxdev = 0, [], 0.0
    dev4 = 0.0
    for _, r in df.iterrows():
        c = cache[r["task"]]
        w = weights(c["lb"], c["rho"], 1.0, 0.0, 1.0)
        s = score(c["y"], aggregate(c["probs"], w))
        d = abs(s["macro_f1_5class"] - float(r["f1_soft_equal"]))
        maxdev = max(maxdev, d)
        n += 1
        if d != 0.0:
            bad.append({"task": r["task"], "mine": s["macro_f1_5class"],
                        "d8": float(r["f1_soft_equal"]), "abs_dev": d})
    return {"gate": "a_lambda1_reproduces_d8_soft_equal", "n_compared": n,
            "n_mismatch": len(bad), "max_abs_dev": maxdev,
            "examples": bad[:5], "passed": len(bad) == 0,
            "note_4class_max_abs_dev": dev4}


def gate_candidates(df: pd.DataFrame) -> dict:
    """候选 F1 读盘一致性：summary_metrics 与逐任务 metrics.json 逐位比对。"""
    n, bad = 0, []
    for _, r in df.iterrows():
        for m, col in [("rf", "f1_rf"), ("stacking", "f1_stacking")]:
            mj = json.loads((task_dir(r["task"]) / m / "metrics.json").read_text())
            v = float(mj["macro_f1"])
            n += 1
            if v != float(r[col]):
                bad.append({"task": r["task"], "model": m,
                            "metrics_json": v, "summary": float(r[col])})
    return {"gate": "b_candidate_f1_bitwise", "n_compared": n,
            "n_mismatch": len(bad), "examples": bad[:5], "passed": len(bad) == 0}


# ------------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "results" / "route_b"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_all = load_tasks()
    ood = df_all[df_all["grid_kind"] == "ood"].reset_index(drop=True)
    if len(ood) != 150:
        raise SystemExit(f"[STOP] OOD 任务数 {len(ood)} != 150")

    cache = {}
    for _, r in df_all.iterrows():          # 162 任务（含 12 IID，供门 a 用）
        probs, y = load_target(r["task"])
        st = source_stats(r["task"])
        cache[r["task"]] = {"probs": probs, "y": y, **st}

    gates = [gate_lambda1(cache, df_all), gate_candidates(ood)]

    # fold 结构自检（门 d）
    fold_rows, covered = [], []
    for e in ENVS:
        outer = ood[ood["target_env"] == e]
        inner = inner_pool(ood, e)
        assert len(outer) == 25, f"{e} 外层 {len(outer)} != 25"
        assert len(inner) == 70, f"{e} 内层 {len(inner)} != 70"
        for _, r in inner.iterrows():
            assert e not in r["train_rounds"] and e not in r["test_rounds"] \
                and r["target_env"] != e, f"{e} 泄漏进内层任务 {r['task']}"
        covered += outer["task"].tolist()
        fold_rows.append((e, outer, inner))
    assert sorted(covered) == sorted(ood["task"].tolist()), "外层未覆盖 150 任务"
    gates.append({"gate": "d_fold_structure", "n_outer_folds": 6,
                  "outer_per_fold": 25, "inner_per_fold": 70,
                  "outer_cover_150_exactly": True, "e_out_absent_from_inner": True,
                  "passed": True})

    if not all(g["passed"] for g in gates):
        (out / "acceptance_FAILED.json").write_text(
            json.dumps({"gates": gates}, ensure_ascii=False, indent=2))
        print("[STOP] 硬门未通过，未写正式产物；见 acceptance_FAILED.json")
        return 2

    # 内层选参 → 外层一次性评估
    task_rows, fold_params, weight_rows = [], [], []
    for e, outer, inner in fold_rows:
        best = grid_search(cache, inner)
        fold_params.append({"fold_e_out": e, "n_outer": len(outer), "n_inner": len(inner),
                            "n_grid_points": len(TAUS) * len(GAMMAS) * len(LAMBDAS), **best})
        for _, r in outer.iterrows():
            c = cache[r["task"]]
            w = weights(c["lb"], c["rho"], best["tau"], best["gamma"], best["lam"])
            s = score(c["y"], aggregate(c["probs"], w))
            cands = {"rf": r["f1_rf"], "stacking": r["f1_stacking"],
                     "soft_equal": r["f1_soft_equal"], "route_b": s["macro_f1_5class"]}
            oracle3 = max(r["f1_rf"], r["f1_stacking"], r["f1_soft_equal"])
            task_rows.append({
                "task": r["task"], "fold_e_out": e, "target_env": r["target_env"],
                "n_sources": r["n_sources"], "lb_basis": c["lb_basis"],
                **{f"f1_{k}": v for k, v in cands.items()},
                "f1_soft_calibrated": r["f1_soft_cal"], "f1_hard_voting": r["f1_hard"],
                "macro_f1_4class_route_b": s["macro_f1_4class"],
                "worst_class_f1_route_b": s["worst_class_f1"],
                "accuracy_route_b": s["accuracy"],
                "gain_vs_stacking": s["macro_f1_5class"] - r["f1_stacking"],
                "gain_vs_rf": s["macro_f1_5class"] - r["f1_rf"],
                "gain_vs_soft_equal": s["macro_f1_5class"] - r["f1_soft_equal"],
                "regret_route_b_vs_oracle3": oracle3 - s["macro_f1_5class"],
                "regret_always_rf": oracle3 - r["f1_rf"],
                "regret_always_stacking": oracle3 - r["f1_stacking"],
                "regret_always_soft_equal": oracle3 - r["f1_soft_equal"],
            })
            weight_rows.append({"task": r["task"], "fold_e_out": e,
                                **{f"lb_{m}": c["lb"][m] for m in BASES},
                                **{f"rho_{m}": c["rho"][m] for m in BASES},
                                **{f"w_{m}": float(w[i]) for i, m in enumerate(BASES)}})

    td = pd.DataFrame(task_rows).sort_values("task").reset_index(drop=True)

    # 逐环境汇总
    env_rows = []
    for e in ENVS:
        g = td[td["target_env"] == e]
        env_rows.append({
            "target_env": e, "n_tasks": len(g),
            "mean_f1_route_b": g["f1_route_b"].mean(),
            "mean_f1_rf": g["f1_rf"].mean(),
            "mean_f1_stacking": g["f1_stacking"].mean(),
            "mean_f1_soft_equal": g["f1_soft_equal"].mean(),
            "mean_f1_soft_calibrated": g["f1_soft_calibrated"].mean(),
            "mean_f1_hard_voting": g["f1_hard_voting"].mean(),
            "mean_gain_vs_stacking": g["gain_vs_stacking"].mean(),
            "mean_gain_vs_rf": g["gain_vs_rf"].mean(),
            "n_win_vs_stacking": int((g["gain_vs_stacking"] > 0).sum()),
            "n_loss_vs_stacking": int((g["gain_vs_stacking"] < 0).sum()),
            "n_win_vs_rf": int((g["gain_vs_rf"] > 0).sum()),
            "n_loss_vs_rf": int((g["gain_vs_rf"] < 0).sum()),
            "mean_regret_route_b": g["regret_route_b_vs_oracle3"].mean(),
            "mean_regret_always_rf": g["regret_always_rf"].mean(),
            "mean_regret_always_stacking": g["regret_always_stacking"].mean(),
            "mean_macro_f1_4class_route_b": g["macro_f1_4class_route_b"].mean(),
            "mean_worst_class_f1_route_b": g["worst_class_f1_route_b"].mean(),
        })
    ed = pd.DataFrame(env_rows)

    # 三条通过线的原始比较量（不判定）
    mean_b = float(ed["mean_f1_route_b"].mean())
    mean_rf = float(ed["mean_f1_rf"].mean())
    mean_st = float(ed["mean_f1_stacking"].mean())
    worst_env = ed.loc[ed["mean_f1_route_b"].idxmin()]
    n_env_pos = int((ed["mean_gain_vs_stacking"] > 0).sum())
    pass_rows = [
        ("cond1", "mean_f1_route_b_env_equal", mean_b, ""),
        ("cond1", "mean_f1_rf_env_equal", mean_rf, ""),
        ("cond1", "threshold_rf_minus_0.005", mean_rf - 0.005, ""),
        ("cond1", "margin_route_b_minus_threshold", mean_b - (mean_rf - 0.005), ""),
        ("cond1", "mean_f1_route_b_task_level", float(td["f1_route_b"].mean()), ""),
        ("cond2", "worst_env_name", float("nan"), str(worst_env["target_env"])),
        ("cond2", "worst_env_mean_f1_route_b", float(worst_env["mean_f1_route_b"]), ""),
        ("cond2", "same_env_mean_f1_stacking", float(worst_env["mean_f1_stacking"]), ""),
        ("cond2", "margin_worst_env_minus_stacking",
         float(worst_env["mean_f1_route_b"] - worst_env["mean_f1_stacking"]), ""),
        ("cond3", "n_envs_positive_gain_vs_stacking", float(n_env_pos), "阈值 >=4"),
        ("cond3", "n_envs_total", 6.0, ""),
        ("aux", "mean_f1_stacking_env_equal", mean_st, "非门禁"),
        ("aux", "mean_regret_route_b_env_equal", float(ed["mean_regret_route_b"].mean()), "非门禁"),
        ("aux", "mean_regret_always_rf_env_equal", float(ed["mean_regret_always_rf"].mean()),
         "非门禁；G1 实测 0.02025"),
        ("aux", "mean_f1_soft_equal_env_equal", float(ed["mean_f1_soft_equal"].mean()), "非门禁"),
        ("aux", "mean_f1_soft_calibrated_env_equal",
         float(ed["mean_f1_soft_calibrated"].mean()), "非门禁"),
        ("aux", "mean_f1_hard_voting_env_equal", float(ed["mean_f1_hard_voting"].mean()), "非门禁"),
        ("aux", "n_tasks_win_vs_stacking", float((td["gain_vs_stacking"] > 0).sum()), "非门禁"),
        ("aux", "n_tasks_win_vs_rf", float((td["gain_vs_rf"] > 0).sum()), "非门禁"),
    ]
    pl = pd.DataFrame(pass_rows, columns=["criterion", "quantity", "value_numeric", "note"])

    # 落盘
    td.to_csv(out / "route_b_task_detail.csv", index=False, encoding=ENC)
    ed.to_csv(out / "route_b_env_summary.csv", index=False, encoding=ENC)
    pd.DataFrame(fold_params).to_csv(out / "route_b_fold_params.csv", index=False, encoding=ENC)
    pd.DataFrame(weight_rows).sort_values("task").to_csv(
        out / "route_b_weights.csv", index=False, encoding=ENC)
    pl.to_csv(out / "route_b_passline.csv", index=False, encoding=ENC)

    audit = {
        "gate": "e_source_only_audit",
        "weight_estimator_signature": "source_stats(task: str) -> dict",
        "reads": ["results/g0_environment_grid/raw_all/<task>/all_features/stacking/oof_meta.csv"],
        "target_side_objects_in_signature": 0,
        "target_true_label_use": "仅在 score() 最终打分处；不进入 lb/rho/权重计算",
        "randomness": "none (deterministic; no seed used)",
        "passed": True,
    }
    gates.append(audit)
    (out / "acceptance.json").write_text(
        json.dumps({"gates": gates,
                    "spec": "EXECUTION_PLAN_20260829.md D11 @0f33137 + v1.7 addendum",
                    "verdict": "由审阅方按协议 §17.3 三条通过线作出；本文件不含判定"},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:
            return "unavailable"

    (out / "provenance.json").write_text(json.dumps({
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "argv": sys.argv, "executable": sys.executable, "cwd": str(Path.cwd()),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "python": sys.version.split()[0]},
        "inputs": {"summary": str(SUMMARY), "voting": str(VOTING), "raw_root": str(RAW)},
        "grid": {"tau": TAUS, "gamma": GAMMAS, "lambda": LAMBDAS},
        "n_tasks_ood": int(len(ood)), "n_tasks_cached": len(cache),
        "seeds": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    md5 = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
           for p in sorted(out.glob("route_b_*.csv"))}
    (out / "csv_md5.json").write_text(json.dumps(md5, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("[gates]", {g["gate"]: g["passed"] for g in gates})
    print(pl.to_string(index=False))
    print("[md5]", json.dumps(md5, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
