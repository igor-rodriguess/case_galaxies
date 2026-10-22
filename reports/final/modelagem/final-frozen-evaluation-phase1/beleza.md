# Avaliação final — Beleza

Alvo principal: `residual_age_channel`.

## Comparação temporal

As métricas de holdout dos demais candidatos são diagnóstico: o modelo reportado é o escolhido nas dobras internas do treino.

| Modelo | MAE | RMSE | R² | Spearman | Lift@K |
|---|---:|---:|---:|---:|---:|
| historical_baseline | 1.192 | 1.422 | -0.020 | — | 0.51 |
| ridge_no_selection | 1.215 | 1.460 | -0.075 | 0.3339721254355401 | 1.52 |
| ridge_f_test | 1.139 | 1.375 | 0.046 | 0.33153310104529626 | 1.52 |
| elastic_net_mi | 1.196 | 1.421 | -0.019 | 0.2837979094076655 | 1.52 |
| random_forest (escolhido nas dobras internas) | 0.930 | 1.129 | 0.356 | 0.6475609756097562 | 3.04 |
| hist_gradient_boosting | 1.039 | 1.206 | 0.265 | 0.4689895470383275 | 2.02 |
| mlp_regularized | 1.215 | 1.425 | -0.025 | 0.19756097560975616 | 1.01 |
| ensemble_oof | 0.980 | 1.188 | 0.288 | 0.5649825783972127 | 2.02 |

Vencedor selecionado internamente: **random_forest**.
Intervalo conformal: `{"available": true, "recipe": "random_forest", "calibration": "temporal", "calibration_rows": 32, "absolute_residual_quantile_90": 1.953027190749947, "coverage_test": 0.9512195121951219, "mean_interval_width": 3.906054381499894}`.
Classificação secundária: macro-F1 `0.477`, PR-AUC alto `0.6327629475641156`.

A promoção exige vencer o baseline em temporal e por canal com IC bootstrap favorável; caso contrário, o baseline permanece a referência.
