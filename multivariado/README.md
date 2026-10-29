# Drift em cenários multivariados

Comparação de detectores de drift de 6 bibliotecas Python (Evidently, Alibi Detect, NannyML, Frouros, TorchDrift, Deepchecks) com implementações de referência (scipy/sklearn), usando os mesmos dados sintéticos para todos.

Cenários: deslocamento de média em k de d variáveis, mudança só de correlação, label shift (bruto e BBSD) e dígitos do sklearn com ruído ou desbalanceamento.

**Resultados:** [RESULTADOS.md](RESULTADOS.md). Tabela agregada em `results/summary.csv`.

## Como rodar

```bash
python -I prep.py                      # gera frozen.npz (já incluído)
python -I mv.py selfcheck              # valida a referência MMD
python -I mv.py <adaptador> --sizes 50,200 --r0 100 --r1 30
python -I analyze_mv.py                # results/summary.csv
```

Cada biblioteca roda no próprio ambiente virtual por conflito de dependências. Frouros e TorchDrift são importados de clones dos repositórios em `../../repos/` (caminho fixo em `mv.py`).
