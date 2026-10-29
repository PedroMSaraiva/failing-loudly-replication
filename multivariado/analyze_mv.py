"""Agrega results/*.csv da trilha multivariada: rejeição nativa e harmonizada (p < 0,05)
com IC de Wilson 95%, verificação de hash entre adaptadores, tempo mediano.

Uso: <qualquer venv com pandas>/bin/python -I analyze_mv.py   -> results/summary.csv e tabelas em stdout
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
ALPHA = 0.05


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load():
    fs = [f for f in RES.glob("*.csv") if f.name not in ("summary.csv", "cost.csv") and "_smoke" not in f.name]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    st = df["stat"].astype(str)
    df["err"] = pd.to_numeric(st, errors="coerce").isna() & (st.str.lower() != "nan")  # nome da exceção no lugar da estatística
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    return df


def main():
    df = load()
    keys = ["adapter", "scen", "d", "k", "mag", "view", "n"]
    rows = []
    for key, g in df.groupby(keys):
        R = len(g)
        k_nat = int(g["native"].sum())
        lo, hi = wilson(k_nat, R)
        has_p = g["p"].notna().any()
        k_h = int((g["p"] < ALPHA).sum()) if has_p else np.nan
        hlo, hhi = wilson(k_h, R) if has_p else (np.nan, np.nan)
        rows.append(dict(zip(keys, key), reps=R, errors=int(g["err"].sum()), rej_native=k_nat / R, nat_lo=lo, nat_hi=hi,
                         rej_harm=(k_h / R if has_p else np.nan), harm_lo=hlo, harm_hi=hhi,
                         ms_median=1000 * g["sec"].median(), data_sha=g["data_sha"].iloc[0]))
    s = pd.DataFrame(rows)
    # mesmo hash de dados para todos os adaptadores na mesma célula/view/n
    shas = s.groupby(["scen", "d", "k", "mag", "view", "n", "reps"])["data_sha"].nunique()
    bad = shas[shas > 1]
    print("células com hash divergente entre adaptadores (mesmo R):", len(bad))
    s.to_csv(RES / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 2000, "display.max_columns", 30)
    for scen, g in s.groupby("scen"):
        print(f"\n== {scen}")
        print(g.drop(columns=["scen", "data_sha"]).round(3).to_string(index=False))


if __name__ == "__main__":
    assert abs(wilson(50, 1000)[0] - 0.0381) < 1e-3
    main()
