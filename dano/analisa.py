"""Resume resultados/dano_*.csv: o alarme acompanha o dano?

Dano = queda de acurácia (pontos percentuais) do alvo deslocado em relação ao alvo sem shift.
Prejudicial: queda >= LIMIAR pp. Detecção: p < 0,05.
Uso: ../.venv/bin/python -I analisa.py [--s 1000] [--limiar 2]
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--s", type=int, default=1000)
ap.add_argument("--limiar", type=float, default=2.0)
a = ap.parse_args()

df = pd.concat([pd.read_csv(f) for f in sorted((HERE / "resultados").glob("dano_*.csv"))])
df = df[df.s == a.s]
df["det"] = df.p < 0.05
base = df[df["shift"] == "no_shift"].groupby(["dataset", "run"])[["acc_alvo", "conf_alvo"]].first()
df = df.join(base, on=["dataset", "run"], rsuffix="_0")
df["dano_pp"] = 100 * (df.acc_alvo_0 - df.acc_alvo)        # queda real (precisa de rótulo)
df["dano_est_pp"] = 100 * (df.conf_alvo_0 - df.conf_alvo)   # queda estimada sem rótulo

k = ["dataset", "shift", "delta"]
tab = df.groupby(k + ["metodo"]).agg(dano=("dano_pp", "mean"), est=("dano_est_pp", "mean"), det=("det", "mean")).reset_index()
wide = tab.pivot_table(index=k + ["dano", "est"], columns="metodo", values="det").reset_index()
wide["prejudicial"] = wide.dano >= a.limiar
pd.set_option("display.width", 250, "display.max_rows", 200)

for ds, w in wide.groupby("dataset"):
    w = w[w["shift"] != "no_shift"].sort_values("dano", ascending=False)
    print(f"\n==== {ds} (s = {a.s}, prejudicial = queda >= {a.limiar} pp)")
    print(w.drop(columns="dataset").round(2).to_string(index=False))
    metodos = [c for c in w.columns if c not in k + ["dano", "est", "prejudicial"]]
    print("\nTaxa de detecção média | prejudiciais vs inofensivos | Spearman(detecção, dano)")
    for m in metodos:
        h, i = w[w.prejudicial][m].mean(), w[~w.prejudicial][m].mean()
        rho = spearmanr(w[m], w.dano).statistic
        print(f"  {m:10s} prejudiciais {h:.2f} | inofensivos {i:.2f} | rho {rho:+.2f}")
    print(f"  shifts prejudiciais: {w.prejudicial.sum()} de {len(w)}")
    rho = spearmanr(w.est, w.dano).statistic
    err = (w.est - w.dano)
    print(f"  Estimativa sem rótulo (confiança): Spearman com dano real {rho:+.2f}; "
          f"erro médio {err.mean():+.1f} pp (negativo = subestima o dano); "
          f"acerta prejudicial/inofensivo em {((w.est >= a.limiar) == w.prejudicial).mean():.0%}")
wide.to_csv(HERE / "resultados" / f"resumo_s{a.s}.csv", index=False)
