"""Materializa uma vez os modelos congelados e o pool de dígitos em frozen.npz.

Os adaptadores de todos os venvs leem só arrays numpy deste arquivo, então nenhum
objeto sklearn atravessa ambientes. Uso: .venv/bin/python -I prep.py
"""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
SEED = 20261015
D_LS, K_LS, SEP = 10, 3, 1.5  # label shift: 3 classes em d=10, médias SEP*e_c


def ls_sample(rng, y):
    x = rng.standard_normal((len(y), D_LS))
    x[np.arange(len(y)), y] += SEP
    return x


def main():
    rng = np.random.default_rng([SEED, 999])
    y = rng.integers(0, K_LS, 4000)
    x = ls_sample(rng, y)
    ls = LogisticRegression(max_iter=1000).fit(x[:3000], y[:3000])
    ls_acc = ls.score(x[3000:], y[3000:])

    X, Y = load_digits(return_X_y=True)
    perm = np.random.default_rng([SEED, 998]).permutation(len(Y))
    a, b = perm[: len(Y) // 2], perm[len(Y) // 2:]
    pca = PCA(10, random_state=0).fit(X[a])
    dg = LogisticRegression(max_iter=5000).fit(X[a] / 16, Y[a])
    dg_acc = dg.score(X[b] / 16, Y[b])

    out = HERE / "frozen.npz"
    np.savez(out, ls_W=ls.coef_.T, ls_b=ls.intercept_, dg_X=X[b], dg_y=Y[b],
             dg_pca_C=pca.components_.T, dg_pca_mu=pca.mean_, dg_W=dg.coef_.T, dg_b=dg.intercept_)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"frozen.npz sha={sha} label-shift LR acc={ls_acc:.3f} digits LR acc={dg_acc:.3f} "
          f"PCA var={pca.explained_variance_ratio_.sum():.3f} pool={len(b)}")


if __name__ == "__main__":
    main()
