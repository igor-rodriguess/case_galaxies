# Modelos por setor

Cada pasta contém a configuração declarativa de um modelo especialista. O código
de treino é compartilhado em `src/galaxies/modeling/` para evitar três cópias do
mesmo pipeline.

```text
models/
├── automotivo/config.json
├── moda/config.json
└── beleza/config.json
```

Quando houver dados suficientes, cada treino criará localmente:

```text
models/<setor>/artifacts/<run_id>/
├── model_bundle.joblib
├── metrics.json
├── feature_importance.json
└── training_manifest.json
```

Os artefatos ficam no `.gitignore` até existir uma decisão explícita sobre como
versionar modelos e dados. As configurações e o código permanecem no Git.

## Por que a regressão logística está separada

`view_count` é uma variável contínua. Portanto, regressão logística não compete
diretamente com os regressores. Ela está preparada para um experimento secundário
de classificação em faixas (`abaixo`, `na_média`, `acima`, `viral`). Os limiares
ainda não foram definidos e a trilha fica desabilitada até essa decisão.

## Regra de promoção

Um resultado exploratório pode testar se o código funciona, mas não pode ser
promovido a modelo final. O treino final exige alvo em horizonte fixo, histórico
temporalmente válido, cobertura multimodal mínima e os splits temporal e por canal.
O candidato vencedor também precisa superar os baselines nos dois cortes. Caso
contrário, os artefatos ficam marcados como exploratórios e não atualizam
`latest.json`.

Os ponteiros usam caminhos relativos dentro de `artifacts/`. A camada de serviço
recusa referências que escapem dessa pasta e rejeita payloads com features ausentes
acima do limite definido em `serving.max_missing_fraction`.
