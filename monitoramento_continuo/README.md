# Um ano de monitoramento sem drift

Cada detector roda uma vez por dia, durante 365 dias, contra uma referência fixa (n = 200, d = 10, N(0, I)). Os lotes diários vêm do mesmo processo, então todo alarme é falso. α = 0,05 em cada teste. Adaptadores reaproveitados de `../multivariado/mv.py`, mais `ks_ingenuo` (KS por variável sem correção).

- **estável:** sem drift nenhum.
- **sazonal:** nos fins de semana (dias 5 e 6 de cada semana) as 2 últimas variáveis sobem 0,5σ. Drift real, mas em variáveis que o modelo hipotético não usa.

Colunas: `pers3` = alarme só após 3 dias seguidos; `bonf` = p < 0,05/365 (só detectores com p-valor). Referência teórica para testes independentes: 18,25 alarmes/ano, DP 4,2. Só anos completos entram na tabela; os detectores caros rodaram menos anos.

## Como rodar

```bash
python -I ano.py <adaptador> --anos 20 [--cenario sazonal]   # no venv do framework, como em mv.py
python -I ano.py <adaptador> --tempo                          # custo de um teste
python -I analyze.py --md                                     # results/summary_{ano,sazonal}.csv
```

## Resultados



### Estável (sem drift)
| adapter | anos | taxa_diaria_% | alarmes_ano_media | alarmes_ano_min | alarmes_ano_max | dp_entre_anos | dias_1o_alarme_mediana | anos_sem_alarme_% | pers3_alarmes_ano | pers3_anos_com_alarme_% | bonf_alarmes_ano | bonf_anos_com_alarme_% | seg_por_teste |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nannyml_pca | 20 | 80.0 | 291.9 | 99 | 365 | 100.6 | 1.0 | 0 | 241.4 | 100 | – | – | 0.061 |
| ks_ingenuo | 20 | 36.6 | 133.6 | 41 | 296 | 59.3 | 2.0 | 0 | 29.5 | 95 | 0.75 | 15.0 | 0.007 |
| alibi_mmd | 5 | 9.9 | 36.2 | 13 | 76 | 25.9 | 7.0 | 0 | 0.6 | 40 | 3.0 | 100.0 | 0.215 |
| torchdrift_mmd | 5 | 9.1 | 33.2 | 14 | 60 | 18.8 | 5.0 | 0 | 0.0 | 0 | 3.8 | 80.0 | 0.226 |
| ref_mmd | 20 | 5.4 | 19.8 | 0 | 60 | 19.8 | 17.0 | 5 | 0.15 | 10 | 0.0 | 0.0 | 0.054 |
| alibi_lsdd | 20 | 5.1 | 18.5 | 1 | 57 | 14.8 | 15.0 | 0 | 0.15 | 15 | 1.4 | 60.0 | 0.029 |
| ref_ksb | 20 | 5.0 | 18.4 | 2 | 114 | 25.1 | 20.5 | 0 | 0.7 | 15 | 0.15 | 5.0 | 0.008 |
| alibi_ks | 20 | 5.0 | 18.4 | 2 | 114 | 25.1 | 20.5 | 0 | 0.7 | 15 | 0.15 | 5.0 | 0.04 |
| ref_c2st | 9 | 4.6 | 16.8 | 1 | 35 | 11.6 | 11.0 | 0 | 0.11 | 11 | 0.11 | 11.0 | 0.13 |
| alibi_clf | 7 | 4.5 | 16.3 | 5 | 32 | 9.5 | 13.0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.154 |
| frouros_mmd | 4 | 3.9 | 14.2 | 2 | 39 | 17.4 | 98.0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.36 |
| deepchecks_mv | 4 | 2.0 | 7.2 | 2 | 17 | 6.7 | 58.0 | 0 | 0.0 | 0 | – | – | 0.273 |
| evidently | 19 | 0.0 | 0.1 | 0 | 1 | 0.2 | 257.0 | 95 | 0.0 | 0 | – | – | 0.061 |

### Sazonal (drift inofensivo no fim de semana)
| adapter | anos | taxa_diaria_% | alarmes_ano_media | alarmes_ano_min | alarmes_ano_max | dp_entre_anos | dias_1o_alarme_mediana | anos_sem_alarme_% | pers3_alarmes_ano | pers3_anos_com_alarme_% | bonf_alarmes_ano | bonf_anos_com_alarme_% | seg_por_teste |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nannyml_pca | 10 | 89.0 | 325.0 | 180 | 365 | 62.1 | 1.0 | 0 | 277.8 | 100 | – | – | 0.078 |
| ks_ingenuo | 10 | 57.1 | 208.4 | 133 | 317 | 54.4 | 2.0 | 0 | 80.6 | 100 | 82.1 | 100.0 | 0.013 |
| alibi_mmd | 5 | 35.2 | 128.4 | 112 | 155 | 17.1 | 6.0 | 0 | 10.2 | 100 | 106.6 | 100.0 | 0.211 |
| ref_ksb | 10 | 33.1 | 120.9 | 103 | 180 | 23.6 | 6.0 | 0 | 7.8 | 100 | 54.4 | 100.0 | 0.01 |
| alibi_ks | 10 | 33.1 | 120.9 | 103 | 180 | 23.6 | 6.0 | 0 | 7.8 | 100 | 54.4 | 100.0 | 0.04 |
| ref_mmd | 10 | 32.2 | 117.5 | 104 | 146 | 14.4 | 6.0 | 0 | 5.3 | 90 | 0.0 | 0.0 | 0.071 |
| evidently | 10 | 0.2 | 0.8 | 0 | 3 | 1.3 | 90.0 | 70 | 0.0 | 0 | – | – | 0.072 |
