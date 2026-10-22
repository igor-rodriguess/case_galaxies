# Evidência pós-seleção dos modelos

Execução avaliada: `optimized-vlm1001-relative-metadata-av-20261020`.

O vencedor de cada setor foi definido antes de consultar os holdouts. A tabela compara seu erro absoluto ao baseline de mediana nos mesmos vídeos, com intervalo de confiança de 95% por bootstrap pareado.

| Setor | Vencedor | Corte | N | MAE vencedor | MAE baseline | Ganho | IC95% da diferença | R² | Evidência positiva |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Automotivo | random_forest | temporal | 70 | 0.869 | 1.033 | 15.8% | [-0.322, 0.001] | 0.184 | não |
| Automotivo | random_forest | channel_holdout | 68 | 0.746 | 0.873 | 14.5% | [-0.188, -0.063] | 0.025 | não |
| Moda | ensemble_equal | temporal | 66 | 1.188 | 1.199 | 0.9% | [-0.060, 0.038] | -0.323 | não |
| Moda | ensemble_equal | channel_holdout | 72 | 1.557 | 1.547 | -0.6% | [-0.039, 0.060] | -0.484 | não |
| Beleza | ensemble_equal | temporal | 66 | 1.994 | 2.268 | 12.1% | [-0.340, -0.206] | -1.253 | não |
| Beleza | ensemble_equal | channel_holdout | 71 | 1.105 | 1.170 | 5.5% | [-0.129, -0.001] | 0.034 | não |

## Regra de confirmação

Um setor só é confirmado quando o limite superior do IC95% da diferença de MAE fica abaixo de zero, o ganho é de pelo menos 5% e o R² é de pelo menos 0.05 nos dois cortes. Falhar nessa regra não prova que o algoritmo é inútil; indica que a base atual ainda não sustenta a afirmação de generalização.
