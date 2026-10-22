# Fase 1 — protocolo corrigido — 21/10/2026

Esta fase corrige os problemas de protocolo de
`auditoria-vazamentos-20261021.md` e reexecuta as duas linhas de avaliação.
Ela não abre nenhum dado novo. Os números abaixo continuam sendo de
**desenvolvimento**: a mesma base já foi consultada antes, e a validação final
passa a ser o período prospectivo (`docs/validacao_final_prospectiva.md`).

## O que mudou no código

| Problema | Correção | Onde |
|---|---|---|
| Vencedor da regressão escolhido pelo MAE do holdout | O modelo reportado é a receita escolhida nas dobras internas do treino. Os outros candidatos aparecem só como diagnóstico (`used_for_selection=False`) | `frozen_evaluation.evaluate_outer_split` |
| Dobras internas sempre temporais | No holdout de canal, as dobras internas separam canais inteiros (`GroupKFold`) | `_inner_splits` |
| Conformal de outro modelo e calibração por posição | O conformal usa a receita reportada. A calibração é a parte mais recente do treino (temporal) ou canais inteiros do treino (canal) | `_conformal` |
| Alvo usava canais reservados na curva setorial | No split por canal, o alvo é recalculado sem esses canais na curva e na mediana do setor | `causal_relative_targets(..., excluded_curve_channels)` e `_split_target` |
| Definição do alvo escolhida olhando todas as linhas | `residual_age_channel` fixado a priori; os diagnósticos de idade são só descritivos | `run_frozen_model_evaluation.py` |
| OOD comparava cada ponto com ele mesmo | O limiar ignora o próprio ponto | `_ood_scores(exclude_self=True)` |
| Colunas vazias decididas com treino e teste | Decididas só pelo treino | `prepare_feature_frame` |
| Ranking cortado por canal | Corte de calendário por setor, agora o padrão (`--split calendar`) | `evaluation.calendar_split` |
| Sem dado intocado para a validação final | Vídeos publicados a partir de 21/10/2026 00:00 UTC ficam bloqueados no desenvolvimento | `galaxies/prospective.py` |

Testes novos em `tests/test_phase1_protocol.py`. A suíte completa passa (104
testes) e o `ruff check` está limpo.

## Ranking pré-publicação

Execução: `data/modeling/runs/known-channel-calendar-phase1-20261021/`.
Cortes por setor: automobilismo em 29/08/2026, moda em 18/07/2026 e beleza em
28/05/2026. Treino com 1.897 vídeos; teste com 571 vídeos e 83 positivos, em 18
canais, dos quais 15 têm pares positivo/negativo.

| Modelo | AUC intra (IC95%) | Precisão@10 | Lift@10 (IC95%) | Δ AUC vs. logística (IC95%) |
|---|---:|---:|---:|---:|
| Logística pooled, top-10 (referência) | 0,722 [0,561; 0,829] | 29,2% | 2,01 [1,05; 3,04] | — |
| Logística pooled, contraste | 0,769 [0,671; 0,858] | 36,9% | 2,54 [1,57; 3,53] | +0,047 [−0,017; +0,137] |
| HistGB pooled, contraste | 0,726 [0,574; 0,869] | 38,5% | 2,65 [1,67; 3,71] | +0,004 [−0,098; +0,100] |
| MLP pequena pooled | 0,725 [0,528; 0,851] | 24,6% | 1,69 [0,68; 2,84] | +0,002 [−0,142; +0,098] |
| HistGB por setor, contraste | 0,741 [0,603; 0,821] | 36,9% | 2,54 [1,35; 3,50] | +0,019 [−0,058; +0,092] |
| CatBoost Ranker por setor | 0,756 [0,601; 0,885] | 32,3% | 2,22 [1,33; 3,30] | +0,034 [−0,075; +0,133] |

**Leitura.** Nenhum candidato supera a referência: todos os deltas cruzam zero.
A vantagem do CatBoost Ranker no corte antigo (Lift@10 de 2,33 contra 1,34) não
se repete no corte de calendário, onde a ordem entre os candidatos muda.

**Por que todas as AUCs subiram juntas.** Não foi melhora de modelo. No período
de teste do corte de calendário, **a duração sozinha dá AUC intra-canal de
0,734**, contra 0,557 no corte antigo. O comprimento do título dá 0,672. Nesse
período, dentro do mesmo canal, vídeos longos superam os curtos, e qualquer
modelo que use a duração herda esse sinal. Isso mostra um defeito do rótulo do
ranking: ele compara um Short com os vídeos longos do canal. É exatamente a
comparação que o escopo da regressão evitou ao usar o formato como estrato
(`docs/escopo_avaliacao_final.md`, decisão 4). O ranking nunca aplicou essa
decisão, e a pergunta do produto ("qual criativo priorizar entre comparáveis")
não é respondida por "vídeo longo ganha de Short".

Também pesa o tamanho: só 15 canais têm pares avaliáveis, e os intervalos têm
de 0,2 a 0,3 de largura.

### Investigação: o salto de AUC é vazamento?

Reprodução: `python scripts/diagnose_format_confounding.py
known-channel-calendar-phase1-20261021 known-channel-asof-candidates-ci-20261021`
(tabela em `known-channel-calendar-phase1-20261021/format_confounding.csv`).

**Não é vazamento de informação do futuro para as features.** A duração é
conhecida antes da publicação, e o título quase não mudou entre as duas
leituras (1 troca em 3.425 vídeos). O problema é que **o rótulo e a métrica
medem o formato**, não o criativo:

- No treino, a duração sozinha não tem sinal (AUC intra 0,51), e Shorts e
  vídeos longos têm taxas de positivos parecidas (9,5% e 11,4%).
- No teste do corte de calendário, 31,6% dos vídeos longos são positivos,
  contra 7,1% dos Shorts. O teste tem 70% de Shorts, contra 44% no treino: os
  canais passaram a publicar mais Shorts, e o limiar de cada canal vem de um
  histórico com mais vídeos longos. "Positivo" virou "é vídeo longo".
- Dentro do mesmo formato, a duração não discrimina (0,545), e o comprimento do
  título também não (0,587 no corte de calendário e 0,467 no antigo).
- O rótulo também carrega um pouco de idade: no teste, ordenar pelo vídeo mais
  novo já dá AUC de 0,571. A curva única de idade, ajustada em vídeos mais
  velhos, é extrapolada para os novos.

AUC intra-canal comparando só pares do mesmo canal **e do mesmo formato**:

| Modelo | Calendário: todos os pares | Calendário: mesmo formato | Corte antigo: todos os pares | Corte antigo: mesmo formato |
|---|---:|---:|---:|---:|
| Logística top-10 (referência) | 0,722 | 0,560 | 0,524 | 0,443 |
| Logística contraste | 0,769 | 0,597 | 0,559 | 0,464 |
| HistGB pooled | 0,726 | 0,510 | 0,665 | 0,596 |
| HistGB por setor | 0,741 | 0,581 | 0,664 | 0,584 |
| CatBoost Ranker por setor | 0,756 | 0,580 | 0,666 | 0,608 |
| MLP pequena | 0,725 | 0,514 | 0,564 | 0,483 |

**Conclusão.** Comparando criativos do mesmo formato, que é a decisão real do
produto, a capacidade de ordenar fica entre 0,51 e 0,61 nos dois cortes.
- O 0,666 do CatBoost Ranker já incluía parte do atalho de formato: cai para
  0,608.
- O 0,72–0,77 do corte de calendário é quase todo formato: cai para 0,51–0,60.
- Esses números não têm intervalo de confiança e entram só como diagnóstico. A
  próxima avaliação precisa nascer com o rótulo por formato (Fase 2, item 1).

## Regressão contínua (base de 581)

Execução: `reports/final/modelagem/final-frozen-evaluation-phase1/`. MAE em
log, com alvo `residual_age_channel`. O "melhor no teste" é só para medir o viés
de seleção do protocolo antigo.

| Setor | Holdout | n teste | Escolhido internamente | MAE | Baseline | Δ MAE IC95% | Melhor no teste (antigo) |
|---|---|---:|---|---:|---:|---|---|
| Automobilismo | temporal | 33 | Random Forest | 1,038 | 1,144 | [−0,31; +0,10] | ElasticNet 0,994 |
| Automobilismo | canal novo | 41 | MLP | 1,131 | 0,969 | [−0,07; +0,41] | Random Forest 0,956 |
| Moda | temporal | 43 | ElasticNet | 0,926 | 1,186 | [−0,58; +0,06] | HistGB 0,912 |
| Moda | canal novo | 65 | Random Forest | 1,290 | 1,764 | **[−0,79; −0,16]** | HistGB 1,189 |
| Beleza | temporal | 41 | Random Forest | 0,930 | 1,192 | **[−0,45; −0,07]** | Random Forest 0,930 |
| Beleza | canal novo | 25 | Random Forest | 1,155 | 1,093 | [−0,11; +0,21] | MLP 1,052 |

**Leitura.**

- Os MAE "melhores" relatados antes eram otimistas: em 4 dos 6 cortes, o melhor
  modelo no teste não era o escolhido internamente. O número honesto de
  automobilismo no corte temporal é 1,038, não 0,994.
- Só dois cortes têm ganho confirmado sobre o baseline: moda em canal novo e
  beleza no temporal. Nenhum setor confirma ganho nos dois cortes.
- Os intervalos conformais de 90% têm de 3,2 a 6,3 unidades de log de largura.
  Na escala de views, a faixa vai de cerca de 5× a 23× para cima ou para baixo
  em torno do previsto. A cobertura observada ficou entre 79% e 95%.
  Na prática, a regressão não diz ao usuário quantas views esperar.

## Decisão

**Nenhum modelo promovido.**
- **Ranking:** a logística de referência continua como baseline, e o CatBoost
  Ranker volta a ser só um candidato.
- **Regressão:** o baseline histórico continua como referência operacional nos
  três setores.

## Próximos passos (Fase 2)

1. **Rótulo do ranking por formato.** Calcular o limiar do canal dentro do
   formato (até 180 s e acima de 180 s) e medir AUC só entre pares do mesmo
   formato. Sem isso, a métrica premia o formato e não o criativo.
2. **Seleção aninhada no ranking.** Escolher entre os candidatos com dobras de
   calendário dentro do treino, congelar uma configuração por setor e não
   comparar mais nada no corte externo atual.
3. **Pooling ou setor**, decidido dentro dessas dobras internas.
4. **Fila de novos uploads para a reserva prospectiva.** O coletor diário só
   acompanha vídeos já listados. Sem listar os uploads novos dos 30 canais, a
   validação final não acontece.
5. **Mais canais** em automobilismo e beleza, que são o limite real dos dois
   holdouts (de 25 a 41 vídeos de teste por corte nesses setores).
