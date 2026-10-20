# Evidência pós-seleção dos modelos

Execução avaliada: `relative-performance-nested-314-2026-10-16`.

O vencedor de cada setor foi definido antes de consultar os holdouts. A tabela compara seu erro absoluto ao baseline de mediana nos mesmos vídeos, com intervalo de confiança de 95% por bootstrap pareado.

| Setor | Vencedor | Corte | N | MAE vencedor | MAE baseline | Ganho | IC95% da diferença | R² | Evidência positiva |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Automotivo | ensemble_weighted | temporal | 21 | 0.999 | 1.359 | 26.5% | [-0.589, -0.119] | 0.119 | sim |
| Automotivo | ensemble_weighted | channel_holdout | 22 | 0.661 | 0.829 | 20.2% | [-0.303, -0.044] | 0.001 | não |
| Moda | ridge | temporal | 22 | 1.347 | 1.321 | -2.0% | [-0.021, 0.069] | -0.189 | não |
| Moda | ridge | channel_holdout | 24 | 1.712 | 1.783 | 4.0% | [-0.119, -0.025] | -2.035 | não |
| Beleza | ridge | temporal | 21 | 0.961 | 0.946 | -1.7% | [-0.016, 0.046] | -0.197 | não |
| Beleza | ridge | channel_holdout | 24 | 1.231 | 1.235 | 0.4% | [-0.048, 0.039] | -0.040 | não |

## Regra de confirmação

Um setor só é confirmado quando o limite superior do IC95% da diferença de MAE fica abaixo de zero, o ganho é de pelo menos 5% e o R² é de pelo menos 0.05 nos dois cortes. Falhar nessa regra não prova que o algoritmo é inútil; indica que a base atual ainda não sustenta a afirmação de generalização.
