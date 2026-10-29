"""Trilha batch multivariada (QP1/QP2/QP6): mean shift em k de d features, drift só de
correlação, label shift com BBSD e dígitos do sklearn corrompidos.

Mesmo desenho do piloto (../harness.py): um processo por venv, dados regenerados por
np.random.default_rng([SEED, cell, n, rep]), hash dos dados por célula, CSV longo.
Modelos congelados e pool de dígitos vêm de frozen.npz (prep.py), só como arrays numpy.

Uso:  <venv>/bin/python -I mv.py <adaptador> [--sizes 50,200,1000] [--r0 200 --r1 100]
      <venv>/bin/python -I mv.py selfcheck      (valida a referência MMD; .venv)
      <venv>/bin/python -I mv.py <adaptador> --cost N D   (tempo e pico de RSS, 5 testes)
"""
import argparse
import csv
import hashlib
import os
import resource
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPOS = HERE.parent.parent / "repos"
SEED = 20261015
ALPHA = 0.05
B = 200  # permutações em todos os detectores de permutação (regime harmonizado)
FZ = np.load(HERE / "frozen.npz")

# ------------------------------------------------------------------ células
# (id estável, cenário, d, k, magnitude, views). magnitude 0 = H0.
CELLS = []


def _add(scen, d, k, mag, views):
    CELLS.append((len(CELLS), scen, d, k, mag, views))


for d in (1, 10, 100):
    _add("h0", d, 0, 0.0, ("raw",))
for d, k in ((1, 1), (10, 1), (10, 10), (100, 1), (100, 10), (100, 100)):
    for mag in (0.2, 0.5):
        _add("mean", d, k, mag, ("raw",))
for rho in (0.2, 0.4, 0.6, 0.8):
    _add("corr", 10, 10, rho, ("raw",))
for gamma in (0.0, 100.0, 10.0, 1.0):  # 0 codifica γ = ∞ (H0)
    _add("label", 10, 3, gamma, ("raw", "bbsd"))
_add("digits_h0", 64, 0, 0.0, ("pca", "bbsd"))
for s in (2.0, 4.0, 8.0):
    _add("digits_noise", 64, 0, s, ("pca", "bbsd"))
for w in (0.7, 0.85, 1.0):  # fração do teste vinda das classes 0-4 (base 0,5)
    _add("digits_imb", 64, 5, w, ("pca", "bbsd"))


def is_h0(c):
    return c[4] == 0.0


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sample(cell, n, rep):
    """Devolve {view: (ref, test)} com arrays float64 2-D."""
    cid, scen, d, k, mag, views = cell
    rng = np.random.default_rng([SEED, cid, n, rep])
    if scen in ("h0", "mean"):
        ref, test = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        test[:, :k] += mag
        return {"raw": (ref, test)}
    if scen == "corr":
        ref, z = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        test = z.copy()
        test[:, 1::2] = mag * z[:, 0::2] + np.sqrt(1 - mag * mag) * z[:, 1::2]  # marginais N(0,1) mantidas
        return {"raw": (ref, test)}
    if scen == "label":
        p_ref = np.full(3, 1 / 3)
        p_test = p_ref if mag == 0 else rng.dirichlet(mag * p_ref)
        out = []
        for p in (p_ref, p_test):
            y = rng.choice(3, n, p=p)
            x = rng.standard_normal((n, 10))
            x[np.arange(n), y] += 1.5
            out.append(x)
        ref, test = out
        bb = lambda x: softmax(x @ FZ["ls_W"] + FZ["ls_b"])  # noqa: E731
        return {"raw": (ref, test), "bbsd": (bb(ref), bb(test))}
    # dígitos: ref e teste disjuntos dentro da repetição, sem reposição
    X, Y = FZ["dg_X"].astype(float), FZ["dg_y"]
    idx = rng.permutation(len(Y))
    ref_i, rest = idx[:n], idx[n:]
    if scen == "digits_imb":
        low = rest[Y[rest] < 5]
        high = rest[Y[rest] >= 5]
        n_low = int(round(mag * n))
        test_i = np.concatenate([low[: n_low], high[: n - n_low]])
    else:
        test_i = rest[:n]
    ref, test = X[ref_i], X[test_i].copy()
    if scen == "digits_noise":
        test = np.clip(test + rng.normal(0, mag, test.shape), 0, 16)
    pca = lambda x: (x - FZ["dg_pca_mu"]) @ FZ["dg_pca_C"]  # noqa: E731
    bb = lambda x: softmax((x / 16) @ FZ["dg_W"] + FZ["dg_b"])  # noqa: E731
    return {"pca": (pca(ref), pca(test)), "bbsd": (bb(ref), bb(test))}


# --------------------------------------------------------------- referências
def ref_mmd(x, y, rng, b=B):
    """MMD² não viesado (estatística U), kernel RBF com largura = mediana das distâncias
    do pool (sem a diagonal), p de permutação (k+1)/(B+1). Todas as permutações via K @ I."""
    from scipy.spatial.distance import pdist, squareform
    z = np.vstack([x, y])
    n, m, N = len(x), len(y), len(x) + len(y)
    dd = pdist(z, "sqeuclidean")
    sig2 = np.median(dd)  # mediana de ||.||², ou seja, largura = mediana das distâncias
    K = squareform(np.exp(-dd / (2 * sig2)))  # diagonal = 0 (já removida)
    I = np.zeros((N, b + 1))
    I[:n, 0] = 1
    for j in range(1, b + 1):
        I[rng.permutation(N)[:n], j] = 1
    KI, K1 = K @ I, K.sum(1, keepdims=True)
    sxx = (I * KI).sum(0)
    sxy = ((1 - I) * KI).sum(0)
    syy = K1.sum() - 2 * sxy - sxx
    t = sxx / (n * (n - 1)) + syy / (m * (m - 1)) - 2 * sxy / (n * m)
    return t[0], (1 + (t[1:] >= t[0]).sum()) / (b + 1)


def ref_c2st(x, y, seed):
    """C2ST (Lopez-Paz & Oquab 2017): metade treina, metade testa; p binomial unilateral."""
    from scipy.stats import binom
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    ix, iy = rng.permutation(len(x)), rng.permutation(len(y))
    hx, hy = len(x) // 2, len(y) // 2
    Xtr = np.vstack([x[ix[:hx]], y[iy[:hy]]]); ytr = np.r_[np.zeros(hx), np.ones(hy)]
    Xte = np.vstack([x[ix[hx:]], y[iy[hy:]]]); yte = np.r_[np.zeros(len(x) - hx), np.ones(len(y) - hy)]
    clf = RandomForestClassifier(100, n_jobs=1, random_state=seed).fit(Xtr, ytr)
    acc_k = int((clf.predict(Xte) == yte).sum())
    return acc_k / len(yte), float(binom.sf(acc_k - 1, len(yte), 0.5))


# ---------------------------------------------------------------- adaptadores
# Contrato: test(ref, x, seed) -> (estatística, p harmonizado ou nan, decisão nativa)
# "p" é o p-valor que o regime harmonizado compara com α = 0,05 (já corrigido por
# Bonferroni nos testes por feature); detectores sem p-valor só têm decisão nativa.

class RefMMD:
    def test(self, r, x, seed):
        s, p = ref_mmd(r, x, np.random.default_rng(seed))
        return s, p, p < ALPHA


class RefC2ST:
    def test(self, r, x, seed):
        s, p = ref_c2st(r, x, seed)
        return s, p, p < ALPHA


class RefKSBonf:  # controle por feature (scipy, Bonferroni)
    def test(self, r, x, seed):
        from scipy.stats import ks_2samp
        ps = [ks_2samp(r[:, j], x[:, j]).pvalue for j in range(r.shape[1])]
        p = min(1.0, min(ps) * r.shape[1])
        return min(ps), p, p < ALPHA


class RefChi2:  # BBSD: qui-quadrado de homogeneidade sobre a classe prevista
    only = "bbsd"

    def test(self, r, x, seed):
        from scipy.stats import chi2_contingency
        K = r.shape[1]
        t = np.vstack([np.bincount(r.argmax(1), minlength=K), np.bincount(x.argmax(1), minlength=K)])
        t = t[:, t.sum(0) > 0]
        if t.shape[1] < 2:
            return 0.0, 1.0, False
        s, p, _, _ = chi2_contingency(t)
        return s, p, p < ALPHA


class Evidently:  # DatasetDriftMetric padrão: drift se ≥50% das colunas derivam
    def __init__(self):
        import pandas as pd
        from evidently.legacy.metrics import DatasetDriftMetric
        from evidently.legacy.report import Report
        self.pd, self.M, self.R = pd, DatasetDriftMetric, Report

    def test(self, r, x, seed):
        cols = [f"f{j}" for j in range(r.shape[1])]
        rep = self.R(metrics=[self.M()])
        rep.run(reference_data=self.pd.DataFrame(r, columns=cols), current_data=self.pd.DataFrame(x, columns=cols))
        res = rep.as_dict()["metrics"][0]["result"]
        return res["share_of_drifted_columns"], np.nan, bool(res["dataset_drift"])


class _NannyChunked:
    """NannyML é por chunk: referência e análise divididas em 5 chunks de n/5; decisão
    nativa = algum chunk da análise com alerta (o que o usuário vê no relatório)."""

    def __init__(self):
        import logging
        import pandas as pd
        import nannyml as nml
        logging.disable(logging.CRITICAL)
        self.pd, self.nml = pd, nml

    def test(self, r, x, seed):
        cols = [f"f{j}" for j in range(r.shape[1])]
        ref, ana = self.pd.DataFrame(r, columns=cols), self.pd.DataFrame(x, columns=cols)
        res = self.calc(cols, len(r) // 5).fit(ref).calculate(ana).filter(period="analysis").to_df()
        return float(res[(self.key, "value")].mean()), np.nan, bool(res[(self.key, "alert")].any())


class NannyPCA(_NannyChunked):
    key = "reconstruction_error"

    def calc(self, cols, cs):
        return self.nml.DataReconstructionDriftCalculator(column_names=cols, chunk_size=cs)


class NannyDC(_NannyChunked):
    key = "domain_classifier_auroc"

    def calc(self, cols, cs):
        return self.nml.DomainClassifierCalculator(feature_column_names=cols, chunk_size=cs)


class _Alibi:
    def __init__(self):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        import alibi_detect.cd as cd
        self.cd = cd

    def test(self, r, x, seed):
        self.torch.manual_seed(seed); np.random.seed(seed % 2**32)
        det = self.make(r.astype(np.float32), seed)
        d = det.predict(x.astype(np.float32))["data"]
        return self.out(d, r.shape[1])

    def out(self, d, nf):  # detectores com um p-valor
        return float(d["distance"]), float(d["p_val"]), bool(d["is_drift"])


class AlibiMMD(_Alibi):
    def make(self, r, seed):
        return self.cd.MMDDrift(r, backend="pytorch", n_permutations=B, device="cpu")


class AlibiLSDD(_Alibi):
    def make(self, r, seed):
        return self.cd.LSDDDrift(r, backend="pytorch", n_permutations=B, device="cpu")


class AlibiClf(_Alibi):  # backend sklearn, mesmo RF da referência
    def make(self, r, seed):
        from sklearn.ensemble import RandomForestClassifier
        return self.cd.ClassifierDrift(r, model=RandomForestClassifier(100, n_jobs=1, random_state=seed),
                                       backend="sklearn", seed=seed)


class _AlibiFeat(_Alibi):  # testes por feature com Bonferroni (padrão do Alibi)
    def out(self, d, nf):
        p = np.atleast_1d(d["p_val"])
        return float(p.min()), float(min(1.0, p.min() * nf)), bool(d["is_drift"])


class AlibiKS(_AlibiFeat):
    def make(self, r, seed):
        return self.cd.KSDrift(r)


class AlibiCVM(_AlibiFeat):
    def make(self, r, seed):
        return self.cd.CVMDrift(r)


class AlibiChi2(_AlibiFeat):
    only = "bbsd"

    def test(self, r, x, seed):
        det = self.cd.ChiSquareDrift(r.argmax(1)[:, None])
        d = det.predict(x.argmax(1)[:, None])["data"]
        return self.out(d, 1)


class _Frouros:
    def __init__(self):
        sys.path.insert(0, str(REPOS / "frouros"))
        from frouros.callbacks.batch import PermutationTestDistanceBased
        from frouros.detectors.data_drift import MMD
        from frouros.utils.kernels import rbf_kernel
        self.MMD, self.Perm, self.rbf = MMD, PermutationTestDistanceBased, rbf_kernel

    def test(self, r, x, seed):
        from functools import partial
        kern = self.rbf if self.sigma is None else partial(self.rbf, sigma=self.sigma(r, x))
        det = self.MMD(kernel=kern, callbacks=[self.Perm(num_permutations=B, random_state=seed, num_jobs=1, name="perm")])
        det.fit(X=r)
        res, logs = det.compare(X=x)
        p = float(logs["perm"]["p_value"])
        return float(res.distance), p, p < ALPHA


class FrourosMMD(_Frouros):  # padrão: rbf_kernel com sigma = 1 fixo
    sigma = None


def _median_dist(r, x):
    from scipy.spatial.distance import pdist
    return float(np.sqrt(np.median(pdist(np.vstack([r, x]), "sqeuclidean"))))


class FrourosMMDMed(_Frouros):  # harmonizado: sigma = mediana das distâncias do pool
    sigma = staticmethod(_median_dist)


class TorchDriftMMD:
    def __init__(self):
        sys.path.insert(0, str(REPOS / "torchdrift"))
        import torch
        torch.set_num_threads(1)
        from torchdrift.detectors import KernelMMDDriftDetector
        self.torch, self.cls = torch, KernelMMDDriftDetector

    def test(self, r, x, seed):
        self.torch.manual_seed(seed)
        det = self.cls(n_perm=B)
        det.fit(self.torch.from_numpy(r))
        s, p = det.predict_shift_from_features(det.base_outputs, self.torch.from_numpy(x),
                                               compute_score=True, compute_p_value=True)
        p = float(p)
        return float(s), p, p < ALPHA


class DeepchecksMV:  # MultivariateDrift; condição padrão: score de drift < 0,25
    def __init__(self):
        import pandas as pd
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.checks import MultivariateDrift
        self.pd, self.D, self.C = pd, Dataset, MultivariateDrift

    def test(self, r, x, seed):
        cols = [f"f{j}" for j in range(r.shape[1])]
        ds = lambda a: self.D(self.pd.DataFrame(a, columns=cols), cat_features=[])  # noqa: E731
        res = self.C(random_state=seed % 2**31).add_condition_overall_drift_value_less_than().run(
            ds(r), ds(x), with_display=False)
        s = float(res.value["domain_classifier_drift_score"])
        return s, np.nan, not res.passed_conditions()


ADAPTERS = {"ref_mmd": RefMMD, "ref_c2st": RefC2ST, "ref_ksb": RefKSBonf, "ref_chi2": RefChi2,
            "evidently": Evidently, "nannyml_pca": NannyPCA, "nannyml_dc": NannyDC,
            "alibi_mmd": AlibiMMD, "alibi_lsdd": AlibiLSDD, "alibi_clf": AlibiClf, "alibi_ks": AlibiKS,
            "alibi_cvm": AlibiCVM, "alibi_chi2": AlibiChi2, "frouros_mmd": FrourosMMD,
            "frouros_mmd_med": FrourosMMDMed, "torchdrift_mmd": TorchDriftMMD, "deepchecks_mv": DeepchecksMV}


def run(name, sizes, r0, r1, cells, tag=""):
    det = ADAPTERS[name]()
    only = getattr(det, "only", None)
    out = HERE / "results" / f"{name}{tag}.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["adapter", "cell", "scen", "d", "k", "mag", "view", "n", "rep", "stat", "p", "native", "sec", "data_sha"])
        for n in sizes:
            for c in CELLS:
                if cells and c[1] not in cells:
                    continue
                if c[1].startswith("digits") and n > 200:
                    continue  # pool de 899 dígitos não comporta ref + teste disjuntos com n = 1000
                views = [v for v in c[5] if only in (None, v)]
                if not views:
                    continue
                R = r0 if is_h0(c) else r1
                h = {v: hashlib.sha256() for v in views}  # um hash por view
                rows = []
                for rep in range(R):
                    data = sample(c, n, rep)
                    for v in views:
                        ref, test = data[v]
                        h[v].update(ref.tobytes()); h[v].update(test.tobytes())
                        t0 = time.perf_counter()
                        try:
                            s, p, nat = det.test(ref, test, SEED + rep)
                            err = ""
                        except Exception as e:  # registrado, não silenciado
                            s, p, nat, err = np.nan, np.nan, False, type(e).__name__
                        rows.append([name, c[0], c[1], c[2], c[3], c[4], v, n, rep, s if not err else err, p, int(nat),
                                     time.perf_counter() - t0])
                sha = {v: hv.hexdigest()[:16] for v, hv in h.items()}
                w.writerows(r + [sha[r[6]]] for r in rows)
                fh.flush()
                print(name, n, c[1], c[2], c[3], c[4], f"rej={np.mean([r[11] for r in rows]):.3f}", flush=True)
    print(out)


def cost(name, n, d):
    det = ADAPTERS[name]()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rng = np.random.default_rng([SEED, 7, n, d])
    secs = []
    for i in range(5):
        r, x = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        if getattr(det, "only", None):
            r, x = softmax(r), softmax(x)
        t0 = time.perf_counter()
        det.test(r, x, SEED + i)
        secs.append(time.perf_counter() - t0)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"{name},{n},{d},{np.median(secs):.4f},{rss0/1024:.0f},{rss1/1024:.0f},{(rss1-rss0)/1024:.0f}")


def selfcheck():
    """Valida ref_mmd: (1) estatística U contra fórmula direta em caso pequeno;
    (2) média de MMD² com σ fixo contra a forma fechada para N(0,1) vs N(μ,1);
    (3) p de permutação fica em [1/(B+1), 1]."""
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((6, 2)), rng.standard_normal((5, 2)) + 1
    s, p = ref_mmd(x, y, rng, b=50)
    z = np.vstack([x, y]); D2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    sig2 = np.median(D2[np.triu_indices(11, 1)]); K = np.exp(-D2 / (2 * sig2))
    kxx = (K[:6, :6].sum() - 6) / 30; kyy = (K[6:, 6:].sum() - 5) / 20; kxy = K[:6, 6:].mean()
    assert abs(s - (kxx + kyy - 2 * kxy)) < 1e-12, (s, kxx + kyy - 2 * kxy)
    assert 1 / 51 <= p <= 1
    # forma fechada, σ = 1 fixo: MMD² = 2σ/√(σ²+2)·(1 − exp(−μ²/(2(σ²+2))))
    mu, sig, n = 0.7, 1.0, 400
    ests = []
    for _ in range(300):
        a, b = rng.standard_normal(n), rng.standard_normal(n) + mu
        Ka = np.exp(-(a[:, None] - a[None]) ** 2 / 2); Kb = np.exp(-(b[:, None] - b[None]) ** 2 / 2)
        Kab = np.exp(-(a[:, None] - b[None]) ** 2 / 2)
        ests.append((Ka.sum() - n) / (n * (n - 1)) + (Kb.sum() - n) / (n * (n - 1)) - 2 * Kab.mean())
    true = 2 * sig / np.sqrt(sig**2 + 2) * (1 - np.exp(-mu**2 / (2 * (sig**2 + 2))))
    se = np.std(ests) / np.sqrt(len(ests))
    print(f"selfcheck ok: U-stat igual à fórmula direta; MMD² médio {np.mean(ests):.5f} vs fechado {true:.5f} (EP {se:.5f})")
    assert abs(np.mean(ests) - true) < 4 * se


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", choices=list(ADAPTERS) + ["selfcheck"])
    ap.add_argument("--sizes", default="50,200,1000")
    ap.add_argument("--r0", type=int, default=200)
    ap.add_argument("--r1", type=int, default=100)
    ap.add_argument("--cells", default="", help="lista de cenários separados por vírgula (vazio = todos)")
    ap.add_argument("--cost", nargs=2, type=int, metavar=("N", "D"))
    ap.add_argument("--tag", default="", help="sufixo do CSV")
    a = ap.parse_args()
    if a.adapter == "selfcheck":
        selfcheck()
    elif a.cost:
        cost(a.adapter, *a.cost)
    else:
        run(a.adapter, [int(s) for s in a.sizes.split(",")], a.r0, a.r1, [c for c in a.cells.split(",") if c], a.tag)
