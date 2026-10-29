# Trilha batch multivariada: resultados (versão reduzida)

Execução de 28/10/2026 com `mv.py` (código de 15/10), 17 adaptadores, todos com saída 0 e nenhum erro registrado. Os dados são idênticos entre adaptadores (0 células com hash divergente). O `selfcheck` da referência MMD passou (estatística U igual à fórmula direta; MMD² médio dentro de 4 EP da forma fechada).

**Regime reduzido:** n ∈ {50, 200} (n = 1000 cortado, MMD leva ~11 s por teste), R = 100 sob H0 e R = 30 sob drift (plano: 200/100). Com R = 30 o IC de Wilson tem ±~17 p.p., então só diferenças grandes são lidas abaixo. Tabelas completas: `results/summary.csv` (gerado por `analyze_mv.py`).

## Achados

1. **NannyML dispara quase sempre, mesmo sem drift.** Sob H0, o `DomainClassifierCalculator` alerta em 74–100% das repetições e o `DataReconstructionDriftCalculator` em 25–100% (α nominal 5%). Os limiares são calibrados com chunks da própria referência, o mesmo viés visto na trilha univariada. Ressalva: o adaptador usa 5 chunks de n/5 (10 ou 40 linhas) e decide "algum chunk alertou"; chunks maiores provavelmente reduzem a taxa, mas não foi testado.
2. **Drift só de correlação é invisível para testes por feature.** Com marginais N(0,1) preservadas, KS e CvM por feature (scipy e Alibi) ficam em ~5% para qualquer ρ, como esperado. Os testes multivariados detectam com n = 200 e ρ ≥ 0,6: MMD de referência 0,33→0,97, classificador (C2ST) 0,73→1,00. O Evidently fica em 0 em todas as células.
3. **A regra padrão do Evidently ("≥ 50% das colunas") não vê drift esparso.** Com 1 de 10 ou 1 de 100 features deslocada em 0,5 σ e n = 200, o Evidently detecta 0%, enquanto o KS com Bonferroni detecta 77–83%. Sob H0 o falso alarme é 0%, ou seja, a regra é conservadora demais.
4. **O kernel padrão do Frouros (σ = 1 fixo) perde poder em alta dimensão.** Em d = 100, k = 10, deslocamento 0,5, n = 200: 17% contra 100% com σ = mediana das distâncias (`frouros_mmd_med`, que reproduz a referência). No desbalanceamento de dígitos em PCA: 17% contra 100%. Em compensação, σ = 1 acerta melhor a escala do drift de correlação (n = 200, ρ = 0,4: 53% contra 17%). A escolha do kernel pesa mais que a biblioteca.
5. **LSDD do Alibi passa do α nas saídas softmax.** Falso alarme de 24% (IC 17–33%) em BBSD de dígitos com n = 50 e 10% com n = 200. Nas demais células fica perto de 5%.
6. **Deepchecks não é calibrado em n.** O limiar é sobre o score (< 0,25), não sobre um p-valor: falso alarme de 9–16% com n = 50 e 0–5% com n = 200, com poder abaixo do classificador de referência.
7. **Concordância com a referência.** MMD do Alibi, TorchDrift e Frouros com σ mediana ficam dentro do ruído da referência MMD; KS/CvM do Alibi batem com o KS do scipy; o qui-quadrado do Alibi é idêntico ao do scipy.

## Elo com o Failing Loudly

- **Label shift:** BBSD (qui-quadrado sobre a classe prevista) é o melhor nos dígitos desbalanceados (93% com 70/30 e n = 200, contra 33% do MMD em PCA), como no artigo.
- **Ruído gaussiano:** BBSD + KS por dimensão é o melhor com ruído fraco (60–77% com σ = 2 e n = 200); MMD em PCA fica em 13–17%. Mesma conclusão do artigo: a representação importa mais que o teste.

## Custo (mediana por teste, todos os cenários)

Qui-quadrado < 1 ms; KS/MMD de referência ~5 ms; Alibi KS/CvM/LSDD 9–16 ms; Evidently ~60 ms; MMD de bibliotecas e classificadores 130–170 ms; Frouros MMD ~400 ms; NannyML domain classifier ~640 ms.

## Limitações

- R reduzido e sem n = 1000; nenhum dado real (MNIST/CIFAR, tabular real, embeddings).
- Decisão do NannyML depende do tamanho de chunk escolhido no adaptador.
- Nenhum ajuste de hiperparâmetro além dos padrões e da variante σ mediana do Frouros.
