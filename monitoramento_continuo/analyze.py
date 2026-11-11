"""Resume results/ano_*.csv: falsos alarmes por ano, dias até o 1º alarme e mitigações.

Sem drift, todo alarme é falso. Referência teórica (testes independentes, α = 0,05, 365 dias):
18,25 alarmes/ano, desvio-padrão 4,2, 1º alarme em ~14 dias (mediana).
Mitigações aplicadas sobre os dados gravados (sem rodar de novo):
  - bonf_tempo: p < 0,05/365 (Bonferroni sobre o ano; só detectores com p-valor)
  - persist3: alarme só após 3 dias seguidos com decisão nativa positiva
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def persist(s, k=3):
    return (s.rolling(k).sum() == k).astype(int)


def first_day(s):
    idx = s.to_numpy().nonzero()[0]
    return idx[0] + 1 if len(idx) else None  # dia 1 = primeiro dia monitorado


def resume(df):
    df = df[df.groupby("ano")["dia"].transform("size") == 365]  # só anos completos
    df = df.sort_values(["ano", "dia"])
    tem_p = df["p"].notna().all()
    df["bonf"] = (df["p"] < 0.05 / 365).astype(int) if tem_p else pd.NA
    df["pers3"] = df.groupby("ano")["native"].transform(persist)
    g = df.groupby("ano")
    por_ano = g["native"].sum()
    primeiro = g["native"].apply(first_day)
    linha = {
        "adapter": df["adapter"].iat[0],
        "anos": df["ano"].nunique(),
        "taxa_diaria_%": round(100 * df["native"].mean(), 1),
        "alarmes_ano_media": round(por_ano.mean(), 1),
        "alarmes_ano_min": int(por_ano.min()),
        "alarmes_ano_max": int(por_ano.max()),
        "dp_entre_anos": round(por_ano.std(), 1),
        "dias_1o_alarme_mediana": primeiro.median() if primeiro.notna().any() else None,
        "anos_sem_alarme_%": round(100 * primeiro.isna().mean()),
        "pers3_alarmes_ano": round(g["pers3"].sum().mean(), 2),
        "pers3_anos_com_alarme_%": round(100 * (g["pers3"].sum() > 0).mean()),
        "bonf_alarmes_ano": round(g["bonf"].sum().mean(), 2) if tem_p else None,
        "bonf_anos_com_alarme_%": round(100 * (g["bonf"].sum() > 0).mean()) if tem_p else None,
        "seg_por_teste": round(df["sec"].median(), 3),
    }
    return linha


def md(df):
    rows = [df.columns.tolist(), ["---"] * df.shape[1]] + df.fillna("–").astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def selfcheck():
    s = pd.Series([0, 1, 1, 1, 0, 1, 1])
    assert persist(s).tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert first_day(pd.Series([0, 0, 1])) == 3 and first_day(pd.Series([0, 0])) is None


if __name__ == "__main__":
    selfcheck()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for cen in ("ano", "sazonal"):
        fs = [f for f in sorted((HERE / "results").glob(f"{cen}_*.csv")) if f.stat().st_size > 100 and "_ref" not in f.name]
        if not fs:
            continue
        todos = pd.concat([pd.read_csv(f) for f in fs])  # junta os pedaços adapter@inicio
        out = pd.DataFrame([resume(g.copy()) for _, g in todos.groupby("adapter")]).sort_values("alarmes_ano_media", ascending=False)
        out.to_csv(HERE / "results" / f"summary_{cen}.csv", index=False)
        print(f"\n== {cen}"); print(md(out) if "--md" in sys.argv else out.to_string(index=False))
