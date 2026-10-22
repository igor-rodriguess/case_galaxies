# Relatório final de avaliação — base congelada

## Escopo e integridade

- Base: 581 vídeos únicos (Automotivo 165, Moda 215, Beleza 201), congelada em 2026-10-20T01:45:00Z.
- Não entram como entrada: views, likes, comentários, inscritos, IDs, rótulos de caso/controle ou qualquer métrica pós-publicação.
- VLM: abertura de 0-3 s; thumbnail embedding disponível para 100%; transcrição completa ainda não está disponível e, portanto, não é promovida.

## Alvo

Alvo selecionado antes dos holdouts: `residual_age_channel`. A escolha usou a menor correlação absoluta de Spearman com idade, e não desempenho de modelo.

| Definição | Linhas | Spearman com idade | IQR |
|---|---:|---:|---:|
| residual_age_channel | 581 | -0.022 | 1.875 |
| smoothed_ratio | 581 | -0.023 | 1.804 |

`residual_age_channel` é log(views) menos uma expectativa estimada somente com vídeos anteriores do mesmo setor/canal. `smoothed_ratio` é a razão suavizada views/mediana histórica prévia; ela foi mantida como sensibilidade, não como alvo principal.

## Decisão de promoção

Um modelo só é promovido se for o mesmo vencedor no holdout temporal e no holdout de canais e se ambos os ICs bootstrap confirmarem MAE menor que o baseline. Essa regra evita escolher um método que apenas se adaptou a um corte específico.

| Setor | Melhor temporal (MAE) | Baseline temporal | Melhor canal (MAE) | Baseline canal | Decisão |
|---|---|---:|---|---:|---|
| automotivo | elastic_net_mi (0.994) | 1.144 | hist_gradient_boosting (0.872) | 0.889 | não promovido: vencedor muda entre holdouts |
| moda | hist_gradient_boosting (0.912) | 1.186 | elastic_net_mi (1.128) | 1.556 | não promovido: vencedor muda entre holdouts |
| beleza | random_forest (0.930) | 1.192 | mlp_regularized (1.034) | 1.032 | não promovido: vencedor muda entre holdouts |

## Leitura responsável

- **Moda** traz o sinal mais consistente de ganho sobre o baseline, mas muda de HistGradientBoosting no futuro para ElasticNet no canal inédito. É o melhor setor para aprofundar em uma próxima coleta, não para congelar um único algoritmo ainda.
- **Automotivo** melhora pontualmente, mas os ganhos não têm IC bootstrap inteiramente abaixo de zero; além disso, é o setor com menos canais maduros.
- **Beleza** tem Random Forest forte no corte temporal, porém não generaliza melhor que o baseline para canal novo.
- A MLP foi testada regularizada; não foi promovida. Em pelo menos um fold ela alcançou o máximo de épocas sem convergir, um sinal adicional para não tratá-la como campeã.

## Arquivos

- `model_metrics.csv`: métricas de regressão por candidato, setor, holdout e ablação.
- `predictions.csv`: previsões, faixas conformais e marcação OOD por vídeo avaliado.
- `manual_error_review.csv`: maiores erros para conferência humana.
- `feature_stability_external_splits.csv`: repetição das top features de permutação entre os dois holdouts.
- Figuras em `reports/figures/final-frozen-evaluation-v4/` incluem comparação de modelos, previsão vs. real, resíduos, calibração, matriz de confusão, SHAP quando aplicável e importância por permutação.
