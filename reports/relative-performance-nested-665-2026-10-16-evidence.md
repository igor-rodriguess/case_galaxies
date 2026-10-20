# Evidência pós-seleção dos modelos

Execução avaliada: `relative-performance-nested-665-2026-10-16`.

O vencedor de cada setor foi definido antes de consultar os holdouts. A tabela compara seu erro absoluto ao baseline de mediana nos mesmos vídeos, com intervalo de confiança de 95% por bootstrap pareado.

| Setor | Vencedor | Corte | N | MAE vencedor | MAE baseline | Ganho | IC95% da diferença | R² | Evidência positiva |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Automotivo | gradient_boosting | temporal | 46 | 1.006 | 1.351 | 25.6% | [-0.504, -0.181] | -0.177 | não |
| Automotivo | gradient_boosting | channel_holdout | 46 | 0.758 | 0.849 | 10.7% | [-0.169, -0.018] | -0.180 | não |
| Moda | extra_trees | temporal | 46 | 1.688 | 1.786 | 5.5% | [-0.225, 0.035] | -0.780 | não |
| Moda | extra_trees | channel_holdout | 49 | 1.434 | 1.546 | 7.2% | [-0.246, 0.024] | -0.268 | não |
| Beleza | ensemble_equal | temporal | 43 | 1.360 | 1.430 | 4.9% | [-0.135, -0.003] | -0.421 | não |
| Beleza | ensemble_equal | channel_holdout | 47 | 1.028 | 1.154 | 10.9% | [-0.183, -0.067] | 0.122 | sim |

## Regra de confirmação

Um setor só é confirmado quando o limite superior do IC95% da diferença de MAE fica abaixo de zero, o ganho é de pelo menos 5% e o R² é de pelo menos 0.05 nos dois cortes. Falhar nessa regra não prova que o algoritmo é inútil; indica que a base atual ainda não sustenta a afirmação de generalização.
