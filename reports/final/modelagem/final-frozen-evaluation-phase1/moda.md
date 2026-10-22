# Avaliação final — Moda

Alvo principal: `residual_age_channel`.

## Comparação temporal

As métricas de holdout dos demais candidatos são diagnóstico: o modelo reportado é o escolhido nas dobras internas do treino.

| Modelo | MAE | RMSE | R² | Spearman | Lift@K |
|---|---:|---:|---:|---:|---:|
| historical_baseline | 1.186 | 1.753 | -0.495 | — | 2.12 |
| ridge_no_selection | 1.027 | 1.380 | 0.074 | 0.21745696164300818 | 2.65 |
| ridge_f_test | 1.013 | 1.395 | 0.054 | 0.2902446390818484 | 2.12 |
| elastic_net_mi (escolhido nas dobras internas) | 0.926 | 1.310 | 0.165 | 0.36031410450015094 | 2.12 |
| random_forest | 0.947 | 1.447 | -0.019 | 0.1478405315614618 | 1.06 |
| hist_gradient_boosting | 0.912 | 1.343 | 0.123 | 0.34838417396556925 | 1.06 |
| mlp_regularized | 1.270 | 1.715 | -0.430 | -0.058290546662639685 | 2.12 |
| ensemble_oof | 0.951 | 1.413 | 0.030 | 0.21066143159166414 | 1.59 |

Vencedor selecionado internamente: **elastic_net_mi**.
Intervalo conformal: `{"available": true, "recipe": "elastic_net_mi", "calibration": "temporal", "calibration_rows": 35, "absolute_residual_quantile_90": 2.7940989416315882, "coverage_test": 0.9302325581395349, "mean_interval_width": 5.5881978832631765}`.
Classificação secundária: macro-F1 `0.379`, PR-AUC alto `0.512879117164107`.

A promoção exige vencer o baseline em temporal e por canal com IC bootstrap favorável; caso contrário, o baseline permanece a referência.
