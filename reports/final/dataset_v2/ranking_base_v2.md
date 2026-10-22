# Ranking de Shorts na base v2

Run: `data/modeling/runs/viral-v2-ranking-20261021/` (gerado por `scripts/evaluate_viral_v2.py`).
O snapshot [ranking_base_v2.json](./ranking_base_v2.json) contém as métricas usadas pela interface, para que o ranking atual não dependa de arquivos locais ignorados pelo Git.
Como leitura exploratória pós-hoc, a interface também apresenta a média macro de AUC só de automotivo e beleza, excluindo moda temporariamente. A linha de moda continua visível na tabela; essa exclusão não representa nova promoção nem validação prospectiva.

## Protocolo

- **Base:** v2 (`docs/dicionario_base_v2.md`), 13.142 Shorts rotulados. Viral = views ≥ 3× a mediana dos
  30 Shorts anteriores do canal, com o vídeo maduro (≥ 60 dias).
- **Divisão:** corte de calendário por setor, com treino de 9.856 e teste de 3.286 vídeos em 23 canais.
  A taxa de virais no teste é de 13,9%.
- **Seleção interna:** as nove configurações de `evaluate_shorts_content_ranking.py`, fixadas antes desta
  base existir, foram escolhidas pela AUC média por canal em três dobras internas de calendário. Venceu o
  `ensemble_rank_mean` (média dos rankings de HistGB, CatBoost e CatBoost YetiRank com metadados,
  estilo, desvio do canal, novidade, título e5 e thumbnail ResNet), com AUC interna de 0,559 contra
  0,463 da referência.
- **Corte externo:** lido uma vez, com IC95% por bootstrap de 2.000 reamostragens de canais.

## Resultado no corte externo

| Recorte | AUC por canal (IC95%) | Referência | Δ AUC (IC95%) | Precision@10% | Lift@10 (IC95%) | ρ nota×idade |
|---|---|---|---|---|---|---|
| **Geral** | **0,588 [0,520; 0,660]** | 0,489 | **+0,099 [+0,041; +0,151]** | 24,9% | **1,80 [1,46; 2,15]** | +0,006 |
| **Automotivo + beleza (exploratório; sem moda)** | **0,641 [0,587; 0,709]** | 0,528 | **+0,114 [+0,058; +0,174]** | 24,9% | **1,92 [1,47; 2,33]** | +0,025 |
| Automotivo (8 canais) | 0,700 [0,638; 0,792] | 0,537 | +0,163 [+0,100; +0,222] | 26,5% | 2,17 [1,73; 2,47] | −0,004 |
| Beleza (7) | 0,574 [0,513; 0,640] | 0,516 | +0,057 [−0,026; +0,153] | 21,6% | 1,48 [0,89; 2,53] | +0,058 |
| Moda (5) | 0,430 [0,289; 0,571] | 0,375 | +0,055 [−0,090; +0,160] | 25,0% | 1,43 [0,85; 1,64] | −0,051 |

**Decisão do relatório original:** promovido pela regra aplicada aos três setores. A linha sem moda é uma análise exploratória pós-hoc, não uma nova promoção; a validação prospectiva continua necessária.

## Leitura

- **Há sinal geral, e ele não é idade.** A AUC tem IC acima de 0,5, o ganho sobre a referência tem
  IC acima de zero e a correlação da nota com a idade é praticamente nula.
- **Os 10% que o modelo aponta acertam 24,9% de virais**, contra 13,9% de base: 1,8× um sorteio.
- **Automotivo** sustenta o resultado, com AUC de 0,700.
- **Beleza** tem ganho cujo intervalo cruza o zero.
- **Moda** ordena abaixo do acaso em todos os modelos, inclusive nas dobras internas. O padrão
  aprendido nos outros setores parece se inverter em moda. É o próximo ponto a investigar, não a
  corrigir olhando este teste.

## Limites

- **Uma única semente e uma única leitura.** Com 20 canais com pares, os intervalos são largos.
- **O corte externo não é virgem para o protocolo.** O mesmo período de calendário já tinha sido lido
  na base v1 com esta mesma lista de configurações (que não foi alterada). A prova definitiva é a
  validação prospectiva (vídeos desde 21/10/2026, rótulo com 30 dias).
- **Hipóteses do rótulo:** 60 dias de maturidade e k = 3 são hipóteses declaradas, não medidas.
- **A diferença para a v1 vem da base e do alvo, não de um modelo novo.** As mesmas configurações
  ficavam em 0,529 contra 0,543 com o rótulo antigo.
