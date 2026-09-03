#!/usr/bin/env python3
"""CORAL 着色矩阵转置修复的构造性正确性检验。

只检验线性代数身份，不涉及任何实验数字：
  行向量约定下 transform 计算 X_c @ W @ C
  => Cov(X_aligned) = C.T @ (W.T @ Cs @ W) @ C = C.T @ C
  目标 Ct = L @ L.T   =>   必须 C = L.T
断言：修复版把 ||Cov(X_aligned) - Ct||_F / ||Ct||_F 压到 ~0；旧版不能。
"""
import importlib.util
import sys
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_domains(seed, n_src=800, n_tgt=600, d=12):
    rng = np.random.default_rng(seed)
    # 两个域用不同的随机满秩线性混合，保证 Cs != Ct 且都正定
    As = rng.normal(size=(d, d))
    At = rng.normal(size=(d, d))
    Xs = rng.normal(size=(n_src, d)) @ As + rng.normal(size=(1, d)) * 3
    Xt = rng.normal(size=(n_tgt, d)) @ At + rng.normal(size=(1, d)) * 3
    return Xs, Xt


def rel_err(aligner_cls, Xs, Xt, force_svd=False):
    al = aligner_cls()
    al.fit(Xs, Xt)
    if force_svd:
        # 走 SVD 分支：直接覆盖为对称平方根
        U, S, _ = np.linalg.svd(al.cov_tgt)
        al.coloring_mat = U @ np.diag(np.sqrt(S + 1e-5)) @ U.T
    Xa = al.transform(Xs, mode="source")
    cov_a = np.cov(Xa.T)
    return np.linalg.norm(cov_a - al.cov_tgt, ord="fro") / np.linalg.norm(al.cov_tgt, ord="fro")


def exact_identity_err(aligner_cls, Xs, Xt):
    """纯代数检验，不含抽样噪声：C.T @ C 是否等于 fit 时用的 cov_tgt。"""
    al = aligner_cls()
    al.fit(Xs, Xt)
    C = al.coloring_mat
    return np.linalg.norm(C.T @ C - al.cov_tgt, ord="fro") / np.linalg.norm(al.cov_tgt, ord="fro")


def main():
    old = load(sys.argv[1], "coral_old")
    new = load(sys.argv[2], "coral_new")
    rows, exact = [], []
    for seed in (0, 1, 2, 3, 4):
        Xs, Xt = make_domains(seed)
        rows.append((
            seed,
            rel_err(old.CORALAligner, Xs, Xt),
            rel_err(new.CORALAligner, Xs, Xt),
            rel_err(new.CORALAligner, Xs, Xt, force_svd=True),
        ))
        exact.append((
            seed,
            exact_identity_err(old.CORALAligner, Xs, Xt),
            exact_identity_err(new.CORALAligner, Xs, Xt),
        ))
    print("A. 代数身份 ||C.T @ C - Ct||_F / ||Ct||_F   （无抽样噪声，机器精度级）")
    print("seed   旧版(L_tgt)      修复版(L_tgt.T)")
    for s, o, n in exact:
        print("%4d   %12.6e   %12.6e" % (s, o, n))
    print()
    print("B. 端到端 ||Cov(X_aligned) - Ct||_F / ||Ct||_F   （残差 ~1e-5 来自 fit 里的 +1e-5·I 正则）")
    print("seed   旧版(L_tgt)      修复版(L_tgt.T)   SVD 兜底分支")
    for s, o, n, v in rows:
        print("%4d   %12.6e   %12.6e   %12.6e" % (s, o, n, v))
    ex_old = max(r[1] for r in exact)
    ex_new = max(r[2] for r in exact)
    old_max = max(r[1] for r in rows)
    new_max = max(r[2] for r in rows)
    svd_max = max(r[3] for r in rows)
    print()
    print("代数身份   旧版 %.3e   修复版 %.3e   （修复版须 < 1e-12）" % (ex_old, ex_new))
    print("端到端     旧版 %.3e   修复版 %.3e   SVD %.3e   （修复版须 < 1e-3 且优于旧版 1000 倍）"
          % (old_max, new_max, svd_max))
    print()
    ok = (ex_new < 1e-12) and (ex_old > 1e-2) \
        and (new_max < 1e-3) and (svd_max < 1e-3) and (old_max / new_max > 1000)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
