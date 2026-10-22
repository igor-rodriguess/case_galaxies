# Decisão final de modelos — Galaxies

Base avaliada: **1000 vídeos com VLM, sem IDs duplicados**.

A seleção foi feita por validação interna. Os holdouts temporal e de canais foram abertos somente para confirmar generalização.

| Setor | Candidato interno | Temporal | Canal novo | Decisão |
|---|---|---:|---:|---|
| Automotivo | random_forest | 15.8% (R² 0.184) | 14.5% (R² 0.025) | historical_median_baseline |
| Moda | ensemble_equal | 0.9% (R² -0.323) | -0.6% (R² -0.484) | historical_median_baseline |
| Beleza | ensemble_equal | 12.1% (R² -1.253) | 5.5% (R² 0.034) | historical_median_baseline |

## Resultado

Nenhum algoritmo de ML cumpriu a regra completa nos dois holdouts. O baseline histórico permanece como campeão operacional nos três setores. Isso é um resultado negativo válido e evita apresentar overfitting como avanço.

O melhor sinal parcial apareceu em Automotivo com Random Forest: ganho de 15,8% no corte temporal e 14,5% em canal novo, mas o IC temporal tocou zero e o R² de canal novo ficou abaixo do limiar. Em Beleza, o ensemble melhorou MAE nos dois cortes, porém os R² permaneceram insuficientes. Em Moda, o ensemble não generalizou.

Embeddings de thumbnail, kNN temporal e CatBoost também não foram promovidos: seus intervalos pareados cruzaram zero, e CatBoost ficou abaixo da referência em AUC.

## Próxima melhoria com maior valor

Coletar alvo em horizonte fixo e ampliar canais independentes. Novos algoritmos sem corrigir o alvo e a diversidade de canais provavelmente só aumentariam a flexibilidade da busca, não a generalização.
