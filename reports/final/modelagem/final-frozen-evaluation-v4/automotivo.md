# Avaliação final — Automotivo

Alvo principal: `residual_age_channel`.

## Comparação temporal

| Modelo | MAE | RMSE | R² | Spearman | Lift@K |
|---|---:|---:|---:|---:|---:|
| historical_baseline | 1.144 | 1.375 | -0.204 | — | 0.67 |
| ridge_no_selection | 1.387 | 2.445 | -2.807 | 0.41778074866310166 | 1.35 |
| ridge_f_test | 1.206 | 1.679 | -0.796 | 0.06851604278074867 | 1.35 |
| elastic_net_mi | 0.994 | 1.305 | -0.085 | 0.3275401069518717 | 1.35 |
| random_forest | 1.038 | 1.281 | -0.045 | 0.2469919786096257 | 0.67 |
| hist_gradient_boosting | 1.097 | 1.347 | -0.156 | 0.2573529411764706 | 1.35 |
| mlp_regularized | 1.014 | 1.283 | -0.048 | 0.1681149732620321 | 1.35 |
| ensemble_oof | 1.038 | 1.281 | -0.045 | 0.2469919786096257 | 0.67 |

Vencedor selecionado internamente: **elastic_net_mi**.
Intervalo conformal: `{"available": true, "absolute_residual_quantile_90": 1.583342776099557, "coverage_test": 0.7878787878787878, "mean_interval_width": 3.166685552199114}`.
Classificação secundária: macro-F1 `0.337`, PR-AUC alto `0.5566605807985118`.

A promoção exige vencer o baseline em temporal e por canal com IC bootstrap favorável; caso contrário, o baseline permanece a referência.
