# Estado atual da modelagem — 21/10/2026

## Decisão de produto

Não há modelo preditivo promovido. A interface pode mostrar atributos,
comparações descritivas, cobertura de dados e a fila prospectiva, mas não pode
atribuir uma nota de potencial, ordenar criativos ou declarar um vencedor
predito.

## Evidência vigente

O único resultado de ranking que valia como referência era
`data/modeling/runs/shorts-ranking-final-14k/`, sob a regra temporal anterior.
Ele será substituído por uma nova execução com o critério operacional único de
até 180 segundos:

| População | Modelo candidato | AUC média por canal (IC95%) | Referência logística | Lift@10 |
|---|---:|---:|---:|---:|
| 13.685 Shorts maduros, 26 canais | seleção aninhada por setor | 0,491 [0,423; 0,560] | 0,540 | 1,15 [0,92; 1,44] |

O intervalo da diferença contra a referência é `[-0,150; +0,028]`. Portanto,
o candidato não supera nem o acaso de maneira demonstrável, nem a referência.

Resultados mais altos de execuções anteriores não são comparáveis: misturavam
Shorts e vídeos longos, permitiam que duração explicasse o rótulo, ou escolhiam
vencedores olhando o próprio holdout. Eles permanecem como histórico técnico,
não como métrica do produto.

## O que está sendo construído

1. **Embedding visual em toda a população.** O extrator retomável produz
   `data/features/embeddings/thumb_resnet18.parquet` para os 14.421 Shorts e
   preserva o CLIP anterior em arquivo separado. O experimento seguinte compara
   metadados/cadência contra metadados/cadência + embedding, no mesmo corte
   externo e com redução de dimensão ajustada apenas no treino.
2. **VLM em amostra neutra.** Os 1.001 VLMs existentes cruzam 452 Shorts da
   população (3,1%) e foram selecionados pela fila caso-controle. Servem para
   checagem exploratória; não serão promovidos. A próxima coleta deve ser
   estratificada por setor, canal e período, sem consultar desempenho.
3. **Validação prospectiva.** Vídeos a partir de 21/10/2026 estão bloqueados
   para desenvolvimento. Seus snapshots de horizonte fixo vão confirmar ou
   rejeitar o modelo congelado quando amadurecerem.

## Regra de promoção

Um candidato só pode ser promovido se, no mesmo corte externo e depois no
período prospectivo:

- superar a referência em AUC média por canal, com IC95% do delta inteiramente
  acima de zero;
- não piorar materialmente algum setor;
- tiver Lift@10 acima de 1 com intervalo favorável;
- não usar views, likes, comentários, IDs, métricas coletadas depois da
  publicação ou rótulos definidos no conjunto avaliado.

Até isso acontecer, `creative_battle` retorna `model_not_ready` por padrão.
