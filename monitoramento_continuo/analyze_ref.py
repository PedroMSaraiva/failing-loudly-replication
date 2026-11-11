"""E3: tamanho da referência × variação do falso alarme entre implantações (anos)."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
rows = []
for f in sorted((HERE / "results").glob("ano_*_ref*.csv")):
    if f.stat().st_size < 100:
        continue
    d = pd.read_csv(f)
    d = d[d.groupby("ano")["dia"].transform("size") == 365]
    por_ano = d.groupby("ano")["native"].sum()
    if len(por_ano) == 0:
        continue
    rows.append({"adapter": d.adapter.iat[0], "n_ref": int(f.stem.split("_ref")[-1]), "anos": len(por_ano),
                 "media": por_ano.mean(), "dp": por_ano.std(), "p10": por_ano.quantile(.1),
                 "p90": por_ano.quantile(.9), "max": por_ano.max(),
                 "anos_>36_%": 100 * (por_ano > 36).mean()})  # > 2× o esperado (18,25)
print(pd.DataFrame(rows).sort_values(["adapter", "n_ref"]).round(1).to_string(index=False))
