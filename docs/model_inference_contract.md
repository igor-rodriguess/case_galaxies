# Contrato de inferência da baseline congelada

## O que está em produção experimental

A referência atual é uma regressão logística com metadados pré-publicação e
cadência de publicação. Ela produz um **ranking de potencial relativo** e uma
probabilidade calibrada para o rótulo histórico `top-10% intra-canal`. Ela não
prevê visualizações absolutas e não garante viralização.

Artefatos usados:

- `data/modeling/runs/baseline-freeze-calibrated-20261020/baseline_pipeline.joblib`;
- `data/modeling/runs/baseline-freeze-calibrated-20261020/probability_calibrator.joblib`;
- `data/modeling/runs/baseline-freeze-calibrated-20261020/confidence_profile.json`.

## Entrada

O preditor aceita as features do modelo diretamente ou calcula derivações
determinísticas a partir de `title`, `description`, `tags`, `published_at` e
`duration_seconds`. Para a confiança normal, informe também a cadência anterior
do canal.

```json
{
  "duration_seconds": 47,
  "title": "Título do criativo",
  "description": "Descrição planejada",
  "tags": ["tag-1", "tag-2"],
  "published_at": "2026-10-21T18:00:00Z",
  "category_id": "22",
  "sector": "beleza",
  "days_since_previous_upload": 3,
  "prior_upload_interval_median_days": 5
}
```

Os campos pós-publicação — visualizações, curtidas, comentários e inscritos
observados — são ignorados. Eles não podem entrar na previsão de um criativo
antes da publicação.

## Saída e confiança

`ranking_score_raw` é a nota não calibrada, útil para ordenar candidatos.
`viral_probability_calibrated` passa pelo calibrador Platt ajustado com previsões
fora da dobra por canal. `confidence.level` vira `low` quando há entrada ausente,
categoria inédita ou valor fora do intervalo p01–p99 do treino. A interface deve
exibir esse alerta, e não ocultá-lo.

`top_local_contributions` mostra os termos que elevaram ou reduziram o log-odds
do modelo para aquele cenário. São associações do classificador, não explicações
causais.

## Uso local

```powershell
.\.venv\Scripts\python.exe scripts\predict_frozen_baseline.py `
  --artifact-dir data\modeling\runs\baseline-freeze-calibrated-20261020 `
  --input caminho\para\criativo.json
```

Para uma tela de comparação, passe o segundo cenário. O resultado traz ambos os
scores e seus deltas:

```powershell
.\.venv\Scripts\python.exe scripts\predict_frozen_baseline.py `
  --artifact-dir data\modeling\runs\baseline-freeze-calibrated-20261020 `
  --input caminho\para\atual.json `
  --compare-with caminho\para\alternativa.json
```

A comparação deve ser apresentada como **"simulação associativa do modelo"**.
Ela não permite concluir que uma edição isolada causará aumento de desempenho.
