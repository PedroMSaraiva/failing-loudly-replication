"""Um ano simulado sem drift: cada detector roda uma vez por dia contra uma referência fixa.

Processo estacionário (N(0, I_d)); qualquer alarme é falso. Reaproveita os adaptadores de
../batch_multivariado/mv.py. Grava, por adaptador, p-valor harmonizado e decisão nativa
de cada (ano, dia) para o pós-processamento (analyze.py) aplicar mitigações sem rodar de novo.

Uso: <venv>/bin/python -I ano.py <adaptador> [--anos 20] [--dias 365] [--n 200] [--d 10]
     <venv>/bin/python -I ano.py <adaptador> --tempo      # custo de um teste
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "batch_multivariado"), str(HERE.parent / "multivariado")]  # local / repo
import mv  # noqa: E402

SEED = 20261111


class KSIngenuo:  # prática comum: KS por feature, alarme se qualquer p < α, sem correção
    def test(self, r, x, seed):
        from scipy.stats import ks_2samp
        p = min(ks_2samp(r[:, j], x[:, j]).pvalue for j in range(r.shape[1]))
        return p, p, p < mv.ALPHA


ADAPTERS = dict(mv.ADAPTERS, ks_ingenuo=KSIngenuo)


def ref(ano, n, d):
    return np.random.default_rng([SEED, ano, 0, n, d]).standard_normal((n, d))


def lote(ano, dia, n, d, cenario="estavel"):
    x = np.random.default_rng([SEED, ano, dia + 1, n, d]).standard_normal((n, d))
    if cenario == "sazonal" and dia % 7 in (5, 6):  # fim de semana: 2 features que o modelo não usa
        x[:, -2:] += 0.5
    return x


def run(name, anos, dias, n, d, cenario, inicio=0, nref=None):
    det = ADAPTERS[name]()
    out = HERE / "results" / ((f"ano_{name}" if cenario == "estavel" else f"{cenario}_{name}") + (f"@{inicio}" if inicio else "") + (f"_ref{nref}" if nref else "") + ".csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["adapter", "ano", "dia", "stat", "p", "native", "sec"])
        for a in range(inicio, inicio + anos):
            r = ref(a, nref or n, d)
            for t in range(dias):
                t0 = time.perf_counter()
                s, p, nat = det.test(r, lote(a, t, n, d, cenario), int(SEED + 1000 * a + t))
                w.writerow([name, a, t, s, p, int(nat), round(time.perf_counter() - t0, 4)])
            fh.flush()
            print(name, "ano", a, flush=True)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", choices=ADAPTERS)
    ap.add_argument("--anos", type=int, default=20)
    ap.add_argument("--dias", type=int, default=365)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--tempo", action="store_true")
    ap.add_argument("--nref", type=int, default=None, help="tamanho da referência (padrão: = n)")
    ap.add_argument("--inicio", type=int, default=0, help="primeiro ano (para paralelizar por anos)")
    ap.add_argument("--cenario", choices=("estavel", "sazonal"), default="estavel")
    a = ap.parse_args()
    if a.tempo:
        det = ADAPTERS[a.adapter]()
        det.test(ref(0, a.n, a.d), lote(0, 0, a.n, a.d), 1)  # aquecimento
        t0 = time.perf_counter()
        for i in range(3):
            det.test(ref(0, a.n, a.d), lote(0, i, a.n, a.d), i)
        print(f"{a.adapter},{(time.perf_counter() - t0) / 3:.3f}")
    else:
        run(a.adapter, a.anos, a.dias, a.n, a.d, a.cenario, a.inicio, a.nref)
