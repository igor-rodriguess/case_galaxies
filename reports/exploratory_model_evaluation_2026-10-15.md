# Avaliação exploratória dos modelos — 15/10/2026

## Escopo

Esta rodada inicia a avaliação dos três modelos setoriais usando o mapa
`data/modeling/runs/exploratory-eval-2026-10-15/feature_map.jsonl`.

Os resultados são exploratórios. A variável-alvo ainda é o snapshot atual de
visualizações, e não visualizações em um horizonte fixo. Além disso, o histórico
dos vídeos antigos foi calculado com métricas coletadas depois da publicação.
Essas limitações impedem promover qualquer artefato como modelo final.

## Protocolo executado

- um modelo separado para automotivo, moda e beleza;
- alvo transformado com `log1p`;
- holdout temporal para avaliar publicações mais recentes;
- holdout por canal para avaliar generalização a criadores não vistos;
- comparação com mediana global e mediana histórica do canal;
- candidatos executados: regressão linear, Ridge, Gradient Boosting, Random
  Forest e MLP;
- XGBoost, CatBoost e LightGBM foram ignorados porque os extras opcionais ainda
  não estão instalados;
- seleção primária por `MAE` no espaço logarítmico;
- métricas complementares: `RMSE log`, `R² log`, Spearman e acurácia de ranking
  dentro do canal.

## Resultado resumido

| Setor | Vencedor exploratório | MAE log temporal | Baseline temporal | MAE log canal novo | Baseline canal novo | Spearman temporal |
|---|---:|---:|---:|---:|---:|---:|
| Automotivo | Random Forest | 1,659 | 1,688 | 0,860 | 0,818 | 0,525 |
| Moda | Random Forest | 1,178 | 1,197 | 2,333 | 1,424 | 0,580 |
| Beleza | Gradient Boosting | 1,064 | 1,325 | 1,134 | 1,051 | 0,686 |

Nenhum vencedor superou o melhor baseline simultaneamente nos dois cortes. Os
três melhoraram no holdout temporal, mas perderam para a mediana histórica quando
receberam canais não vistos. O problema é especialmente forte em moda.

Os intervalos de 10% a 90% cobriram apenas 47,5% dos casos em automotivo, 60% em
moda e 55% em beleza. Um intervalo nominal de 80% com essa cobertura está
subcalibrado e não deve ser apresentado como incerteza confiável.

## Features que apareceram no primeiro benchmark

As importâncias abaixo são por permutação no holdout temporal. Elas mostram
associação preditiva dentro deste experimento, não causalidade.

- Automotivo: duração, média anterior de visualizações, mediana anterior de
  comentários, quantidade de tags, mediana anterior de curtidas e cadência de
  publicação.
- Moda: média anterior de visualizações, duração, categoria, mediana anterior de
  visualizações, cadência e engajamento histórico. Entre as features
  audiovisuais, contagem de pessoas e confiança do VLM aparecem com ganho pequeno.
- Beleza: duração, dias desde o upload anterior, média anterior de visualizações,
  horário, tamanho do título e categoria. Confiança do VLM e intensidade de
  movimento aparecem entre as dez primeiras, mas abaixo das variáveis de canal.

Como apenas cerca de metade da coleta possui VLM, os valores audiovisuais ausentes
foram imputados. Portanto, este benchmark combinado ainda não mede isoladamente o
ganho do criativo. A próxima avaliação deve usar a mesma amostra com VLM para
comparar grupos de features de forma justa.

## Matriz de experimentos proposta

Para cada setor e usando exatamente os mesmos vídeos em cada comparação:

1. `baseline`: mediana histórica do canal;
2. `metadata`: duração, publicação, título, descrição, tags e categoria;
3. `channel_history`: métricas e cadência anteriores ao vídeo;
4. `audiovisual`: somente características da abertura extraídas pelo VLM;
5. `metadata + channel_history`;
6. `metadata + audiovisual`;
7. `combined`: todos os grupos.

O ganho do audiovisual será a diferença entre os experimentos 5 e 7 nos dois
holdouts. Se ele melhorar apenas o corte temporal e piorar canais novos, não deve
ser considerado uma contribuição generalizável.

## Resultado da ablação na amostra com VLM

A matriz foi executada em uma amostra comum de 101 vídeos automotivos, 105 de
moda e 100 de beleza. Assim, a comparação não favorece um grupo por ter mais
linhas disponíveis.

| Setor | Melhor grupo temporal | MAE log | Melhor grupo em canais novos | MAE log |
|---|---|---:|---|---:|
| Automotivo | metadados + histórico | 1,575 | histórico do canal | 0,817 |
| Moda | combinado | 1,425 | histórico do canal | 1,599 |
| Beleza | combinado | 0,945 | metadados + histórico | 0,955 |

O audiovisual isolado não foi o melhor grupo em nenhum setor. Em moda, adicioná-lo
ao conjunto completo melhorou o corte temporal em relação a metadados + histórico,
mas não superou o histórico isolado em canais novos. Em beleza, o conjunto
completo teve o melhor resultado temporal, mas piorou no holdout por canal. Em
automotivo, o audiovisual piorou os dois cortes em relação a metadados + histórico.

### Teste da rede neural

A MLP possui duas camadas ocultas, com 64 e 32 neurônios, padronização, early
stopping e limite de 500 iterações. Ela não foi o melhor modelo global em nenhum
setor ou grupo. Em algumas execuções, chegou ao limite sem convergir.

O único recorte em que venceu os outros algoritmos foi audiovisual isolado de
beleza no holdout por canal (`MAE log = 1,149`). Mesmo nesse caso, Random Forest
com metadados + histórico foi melhor (`0,955`). A conclusão atual é que a rede
neural continua como challenger, mas árvores são mais adequadas ao tamanho e à
estrutura tabular da amostra atual.

## AutoML e rastreamento

AutoML deve entrar como candidato, e não controlar o split ou decidir sozinho a
métrica. A recomendação inicial é FLAML por ser leve, aceitar orçamento de tempo,
métrica customizada e estratégias de split por tempo ou grupo. O resultado dele
deve enfrentar os mesmos baselines e os mesmos holdouts do código manual.

Para registrar experimentos, métricas, parâmetros, artefatos e SHAP, MLflow é
mais adequado do que um framework de avaliação de agentes. A avaliação do LLM e
do MCP deve ficar em uma trilha separada, baseada em um conjunto dourado de
requisições, chamadas esperadas de tools, validade do JSON, latência e ausência de
previsões inventadas.

Não foi encontrada uma única “API Jarvis” reconhecida que substitua a avaliação
de regressão tabular. OpenJarvis avalia modelos e agentes em benchmarks; NVIDIA
Jarvis foi renomeado para Riva e trata de fala; Microsoft JARVIS é uma arquitetura
de orquestração de ferramentas. É necessário confirmar o nome ou link citado
antes de adicioná-lo ao projeto.

## Próximas decisões

1. O alvo principal foi definido como `views_30d`.
2. Definir como serão obtidos snapshots aos 30 dias e como tratar vídeos que já
   passaram desse horizonte sem observação histórica correspondente.
3. Aprovar FLAML como challenger de AutoML e MLflow como registro/eval.
4. Só depois instalar extras, executar busca de hiperparâmetros e gerar SHAP do
   vencedor validado.
