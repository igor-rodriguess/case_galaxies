# Fase 2 — ranking só de Shorts, com seleção aninhada — 21/10/2026

## Decisões desta fase

- **Só Shorts.** O produto ranqueia Shorts. Vídeos longos não são treinados,
  rotulados nem premiados (`docs/escopo_avaliacao_final.md`, revisão de 21/10).
  Critério operacional: duração até 180 s, com até 60 s como sensibilidade. A
  Data API não marca Short, então vídeos horizontais curtos também passam.
- **Rótulo por canal, só entre Shorts.** A curva de idade e o limiar top 10% são
  ajustados em cada canal com os Shorts do treino. Canais com menos de 10 Shorts
  no treino saem dos dois lados.
- **Seleção aninhada.** Três dobras crescentes de calendário dentro do treino de
  cada setor, com cortes nos quantis 55%, 70% e 85% e rótulos recalculados em
  cada dobra. Ganha a maior AUC intra-canal somando as dobras; no empate, fica a
  spec mais simples. A escolha foi gravada em `frozen_selection.json` (sha256
  `6447c3955183…`) antes de pontuar o corte externo.
- **Uma leitura do corte externo.** Só a escolha de cada setor e a referência
  fixada a priori (logística pooled top-10). Os outros candidatos não foram
  pontuados no corte externo.
- **Intervalos** por bootstrap de canais reais (1.000 reamostras, seed 42).

Código: `src/galaxies/agent/format_ranking.py`,
`scripts/evaluate_shorts_ranking.py`, `tests/test_format_ranking.py`.
Artefatos: `data/modeling/runs/shorts-ranking-phase2-180s/` e
`shorts-ranking-phase2-60s/`.

## Resultado principal (até 180 s)

Treino com 919 Shorts. Teste com 312 Shorts, **35 positivos**, em 15 canais.
Cortes externos: automobilismo em 01/09/2026, moda em 16/08/2026 e beleza em
04/06/2026.

| Setor | Escolhido nas dobras internas | Canais | Positivos | AUC (IC95%) | Lift@10 (IC95%) | AUC da referência | Δ AUC (IC95%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Automobilismo | MLP pooled, contraste | 6 | 11 | 0,532 [0,146; 0,812] | 1,91 [0,00; 3,55] | 0,312 | +0,220 [−0,292; +0,792] |
| Beleza | Logística pooled top-10 (a própria referência) | 5 | 14 | 0,584 [0,371; 0,662] | 0,60 [0,00; 0,99] | 0,584 | — |
| Moda | HistGB pooled, contraste | 4 | 10 | 0,541 [0,458; 0,821] | 0,87 [0,00; 1,09] | 0,225 | +0,317 [+0,014; +0,786] |
| **Três setores** | — | 15 | 35 | **0,564 [0,428; 0,640]** | **1,14 [0,34; 1,96]** | 0,450 | [−0,001; +0,322] |

## Leitura honesta

1. **Não há sinal confiável para Shorts com metadados e cadência.**
   - A AUC conjunta de 0,564 tem intervalo que inclui 0,5.
   - O Lift@10 de 1,14 também tem intervalo que inclui 1, o mesmo que sortear.
   - Em beleza, o Lift@10 foi 0,60: os escolhidos acertaram menos que o acaso.
2. **A seleção interna é praticamente ruído.**
   - Nas dobras, os pares por setor foram poucos: 102 em automobilismo, 195 em
     moda e 265 em beleza.
   - A AUC de um mesmo candidato pula de 0,3 a 0,9 entre dobras
     (`inner_selection.csv`).
   - Em beleza, nenhum candidato passou de 0,52 nas dobras internas.
3. **O "ganho" de moda não é ganho.** O IC do delta fica acima de zero, mas a
   referência teve AUC de 0,225, muito abaixo do acaso, com 4 canais e 10
   positivos. Um baseline pior que sortear não serve de comparação.
4. **A sensibilidade de 60 s não é avaliável.** O teste ficou com 87 vídeos,
   **3 positivos** e 7 canais; automobilismo não teve nenhum par.
5. **Comparação com as fases anteriores.** Os números antigos (CatBoost Ranker
   com AUC de 0,666 e Lift@10 de 2,33) misturavam vídeos longos e não medem a
   pergunta do produto. Para Shorts, eles não valem.

## Decisão

**Nenhum modelo promovido para o ranking de Shorts.** A logística pooled top-10
continua como referência, mas não há evidência de que ela, ou qualquer
candidato, ordene Shorts melhor que o acaso. O produto não deve exibir
"potencial" ou "prioridade" como previsão do modelo com a base atual.

## O que limita e o que fazer

O gargalo agora é **tamanho**, não algoritmo: 35 positivos no teste e cerca de
20 canais com Shorts suficientes. Nenhum ajuste de modelo resolve isso.

1. **Mais canais com Shorts**, priorizando automobilismo e moda, que têm só de
   4 a 6 canais avaliáveis no teste. Canal novo é o que mais estreita o IC.
2. **Todos os Shorts dos canais atuais.** A coleta tem cerca de 110 uploads por
   canal. Buscar o histórico completo de Shorts aumenta treino e teste sem
   misturar formato.
3. **Rótulo prospectivo de 30 dias só para Shorts**, com a fila de novos uploads
   (`docs/validacao_final_prospectiva.md`). É também a validação final.
4. **Testar conteúdo depois do tamanho.** Thumbnail, VLM harmonizado e
   transcrição só fazem sentido quando o teste tiver pares suficientes para
   detectar um ganho de 0,05 de AUC. Com 35 positivos, o IC tem cerca de 0,2 de
   largura.
5. **Refazer a regressão só com Shorts.** A base de 581 inclui vídeos longos e
   não serve ao produto como está.
