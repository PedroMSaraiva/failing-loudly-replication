"""Dano × detecção: para cada shift da replicação do Failing Loudly, mede quanto a acurácia
do modelo cai (com rótulos), quanto ela parece cair sem rótulos (confiança média do softmax)
e se os detectores disparam.

Reaproveita as células do notebook (build_notebook.py): mesmos dados, modelos, testes e shifts.
Uso: ../.venv/bin/python -I dano.py <mnist|cifar10> [--runs 3]
"""
import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=["mnist", "cifar10"])
ap.add_argument("--runs", type=int, default=3)
args = ap.parse_args()

# 1) executa as células de código 0–4 do notebook (setup, dados, modelos, testes, shifts)
from types import SimpleNamespace

fake = SimpleNamespace(v4=SimpleNamespace(new_markdown_cell=lambda s: None, new_code_cell=lambda s: s))
src = (FL / "build_notebook.py").read_text().split("nb = nbf.v4")[0].replace("import nbformat as nbf", "")
ns = {"nbf": fake, "__file__": str(FL / "build_notebook.py")}
exec(src, ns)
cells = [c for c in ns["cells"] if c is not None]
G = {}
setup = cells[0].replace('Path("data"), Path("resultados")', f'Path(r"{FL}/data"), Path(r"{HERE}/resultados")')
exec(setup, G)
G["DATASETS"] = []  # a célula de dados só carrega no loop; carregamos manualmente abaixo
for c in cells[1:5]:
    exec(c, G)

torch, np, pd, F = G["torch"], G["np"], G["pd"], G["F"]
DEV, apply, make_target, pick = G["DEV"], G["apply"], G["make_target"], G["pick"]
ks_bonf, mmd_perm, chi2_hard = G["ks_bonf"], G["mmd_perm"], G["chi2_hard"]

# 2) shifts candidatos a "inofensivos", além dos do artigo
SHIFTS = G["SHIFTS"] + [("contraste", 0.5), ("contraste", 1.0), ("so_0e1", 1.0)]


def target(kind, delta, X, y, g, clf):
    if kind == "contraste":  # imagem mais apagada: x * 0,7 numa fração δ
        X = X.clone(); i = pick(len(X), delta, g); X[i] = X[i] * 0.7
        return X, y.clone()
    if kind == "so_0e1":     # label shift extremo: só as classes 0 e 1
        k = (y == 0) | (y == 1)
        return X[k].clone(), y[k].clone()
    return make_target(kind, delta, X, y, g, clf)


name = args.dataset
X_all, y_all, idx_tr, idx_rest = G["load"](name)
DR, clf = G["build_dr"](name, X_all, y_all, idx_tr, idx_rest)
DR = {k: DR[k] for k in ("NoRed", "UAE", "BBSDs")}
SIZES = (200, 1000)
out = HERE / "resultados" / f"dano_{name}.csv"
out.parent.mkdir(exist_ok=True)
rows = []
for run in range(args.runs):
    p = np.random.default_rng(100 + run).permutation(idx_rest)
    Xv, yv = X_all[p[:10000]], y_all[p[:10000]]
    Xt0, yt0 = X_all[p[10000:]], y_all[p[10000:]]
    rep_v = {k: f(Xv) for k, f in DR.items()}
    acc_v = float((rep_v["BBSDs"].argmax(1) == G["np_"](yv)).mean())
    conf_v = float(rep_v["BBSDs"].max(1).mean())
    for j, (kind, delta) in enumerate(SHIFTS):
        t0 = time.time()
        g = torch.Generator().manual_seed(1000 * run + j)
        torch.manual_seed(1000 * run)
        Xt, yt = target(kind, delta, Xt0, yt0, g, clf)
        rep_t = {k: f(Xt) for k, f in DR.items()}
        sm = rep_t["BBSDs"]
        acc_t = float((sm.argmax(1) == G["np_"](yt)).mean())        # acurácia real (precisa de rótulo)
        conf_t = float(sm.max(1).mean())                              # estimativa sem rótulo
        ov = np.random.default_rng(run).permutation(len(Xv))
        ot = np.random.default_rng(run + 1).permutation(len(Xt))
        for s in SIZES:
            n = min(s, len(Xt))
            sv, st = ov[:n], ot[:n]
            base = dict(dataset=name, run=run, shift=kind, delta=delta, s=s, n=n,
                        acc_ref=acc_v, acc_alvo=acc_t, conf_ref=conf_v, conf_alvo=conf_t)
            ps = {"BBSD-KS": ks_bonf(rep_v["BBSDs"][sv], rep_t["BBSDs"][st]),
                  "BBSD-chi2": chi2_hard(rep_v["BBSDs"][sv].argmax(1), rep_t["BBSDs"][st].argmax(1)),
                  "Pixels-KS": ks_bonf(rep_v["NoRed"][sv], rep_t["NoRed"][st]),
                  "UAE-MMD": mmd_perm(rep_v["UAE"][sv], rep_t["UAE"][st])}
            rows += [{**base, "metodo": m, "p": pv} for m, pv in ps.items()]
        print(f"{name} r{run} {kind:9s} δ={delta:.1f} acc {acc_v:.3f}->{acc_t:.3f} "
              f"conf {conf_v:.3f}->{conf_t:.3f} {time.time() - t0:4.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
print(out)
