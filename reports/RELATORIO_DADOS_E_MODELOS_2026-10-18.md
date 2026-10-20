# Relatório de dados e diagnóstico de modelagem — Galaxies

Data de referência: 18 de outubro de 2026. Este relatório descreve o estado
real da coleta e dos experimentos; não promove nenhum resultado exploratório
como modelo de produção.

## Resumo executivo

O projeto já possui uma base ampla de metadados e uma camada audiovisual em
expansão, com rastreabilidade por `video_id`. A maior limitação atual não é a
falta de algoritmos: é a comparabilidade do alvo e a cobertura audiovisual.

- **3.425 vídeos** de **30 canais** foram coletados pela YouTube Data API v3.
- A distribuição é: **1.200 automotivo**, **1.098 moda** e **1.127 beleza**.
- Existem **696 análises audiovisuais válidas e únicas**; a meta operacional é
  1.000, faltando 304. A coleta é retomável, balanceada por setor/canal e não
  repete IDs concluídos.
- Os modelos atuais são **regressões exploratórias de desempenho relativo**,
  não previsores finais de visualizações em 30 dias.
- Nenhum setor passou os dois gates de generalização (tempo e canal novo). Por
  isso, a API/MCP deve continuar retornando `model_not_ready` para previsão.

## 1. O que foi extraído

### 1.1 Metadados e métricas da YouTube Data API

Cada registro bruto preserva a origem, o instante da coleta e o `video_id`.
O mapa de features conecta os registros sem alterar os JSONs originais.

| Grupo | Exemplos de campos |
|---|---|
| Identificação e proveniência | `video_id`, `channel_id`, setor, caminho do JSON de origem, `published_at`, `collected_at` |
| Vídeo | duração, definição, categoria, tamanho do título e da descrição, quantidade de tags, dia/hora de publicação |
| Canal (snapshot) | inscritos e quantidade de vídeos no instante de coleta |
| Métrica observada | visualizações, curtidas e comentários no snapshot; data e idade da observação |
| Qualidade | valores ausentes explícitos, origem do atributo e versão do extrator |

**Cuidado de causalidade:** curtidas e comentários atuais são ótimos para
análise descritiva, mas não podem ser feature de uma previsão antes da
publicação. Da mesma forma, inscritos atuais não representam inscritos na data
de publicação. O pipeline de modelagem seguro exclui histórico de canal cuja
validade temporal não foi comprovada.

### 1.2 Análise audiovisual da abertura

O extrator validado envia os primeiros **3 segundos**, a **4 FPS**, ao modelo
visual. Os resultados guardam modelo, versão do prompt, intervalo analisado,
evidências, incertezas e caminho do resultado.

Os atributos observáveis incluem:

- pessoas, rostos, produto e marca visível;
- texto na tela, enquadramento, câmera, movimento e mudanças de cena;
- fala, início da fala, idioma, música e efeitos sonoros;
- tipo de abertura, destaque do produto e chamada para ação;
- atributos específicos dos setores: close de carro, som de motor, POV e
  interior; aplicação de beleza, swatch e antes/depois; try-on e detalhes de
  vestimenta.

Na auditoria da base de 665 resultados usada nos experimentos, a cobertura VLM
era **226/1.200 (18,8%)** em automotivo, **226/1.098 (20,6%)** em moda e
**213/1.127 (18,9%)** em beleza. A coleta atual chegou a 696 e só entrará em
nova avaliação depois de reconstruir o mapa de features.

### 1.3 Fala e transcrição inicial

Foram derivadas features compactas do trecho inicial: presença de fala,
quantidade de palavras/caracteres, densidade de fala, diversidade lexical,
interrogações, exclamações, números, sinais de segunda pessoa e CTA. Elas são
rastreáveis, mas ainda não substituem uma transcrição segmentada do vídeo todo.

O teste de ablação com essas features simples **não melhorou a generalização de
forma consistente**. Logo, elas permanecem como experimento, não como requisito
de produção. A transcrição completa exige uma fonte de áudio autorizada; a
YouTube Data API não fornece o arquivo audiovisual.

## 2. Alvo e validade temporal

O alvo ideal do produto é `views_30d`: visualizações observadas em uma janela
fixa próxima de 30 dias após a publicação. A base pública atual tem snapshots
de visualização em idades muito diferentes, portanto não permite reconstruir
as views históricas do dia 30.

Para não interromper a pesquisa, os experimentos usam o alvo exploratório
`relative_performance`: resíduo de `log1p(views)` após ajuste de idade do vídeo
e efeito suavizado do canal, calculado exclusivamente no treino. Isso reduz
parte do ruído, mas **não torna o alvo equivalente a views aos 30 dias**.

| Auditoria do alvo final | Resultado |
|---|---:|
| Linhas com visualização observada | 3.425 |
| Linhas com horizonte fixo comprovado | 0 |
| Histórico de canal temporalmente válido | 0 |
| Estado para treino final | Bloqueado |

Para um produto final, temos duas opções corretas:

1. Coletar diariamente, daqui para frente, os vídeos novos e congelar o
   snapshot entre os dias 29 e 31; ou
2. Usar YouTube Analytics com OAuth dos proprietários dos canais, quando houver
   autorização, para obter a série histórica real.

## 3. Como os modelos são testados

Há três pipelines independentes, um por setor. São comparados:

- baseline da mediana;
- regressão linear e Ridge;
- Random Forest, Extra Trees, Gradient Boosting e HistGradientBoosting;
- MLP (rede neural) e ensembles de média/ponderados.

O pré-processamento é aprendido dentro de cada fold: imputação mediana com
indicadores de ausência para numéricos, imputação de categoria ausente e
One-Hot Encoding para categorias. A busca remove colunas constantes,
compara `StandardScaler` com `RobustScaler` e seleciona features por F-test,
informação mútua e importância de Extra Trees. Isso evita que o holdout
participe do ajuste de hiperparâmetros.

A seleção acontece por validação interna temporal (4 folds expanding-window).
Depois, o vencedor é checado em dois conjuntos que não participam da seleção:

1. **Holdout temporal:** vídeos posteriores ao treino.
2. **Holdout por canal:** canais totalmente fora do treino.

As métricas de regressão são MAE e RMSE no espaço logarítmico, R², Spearman,
erros relativos e acurácia de pares dentro do canal. AUC e matriz de confusão
ficam para uma futura tarefa de classificação por faixas de desempenho; não são
métricas corretas para a regressão atual.

## 4. Resultado dos melhores candidatos atuais

Os números abaixo são da execução `relative-performance-nested-665-2026-10-16`.
`Melhora MAE` compara o candidato com a baseline da mediana. Mesmo quando há
melhora de MAE, um R² negativo ou um holdout reprovado impede a promoção.

| Setor | Escolhido por CV interna | Holdout temporal | Holdout canal novo | Situação |
|---|---|---|---|---|
| Automotivo | Gradient Boosting | MAE 1,006 vs. 1,351 (**+25,6%**); R² −0,177 | MAE 0,758 vs. 0,849 (**+10,7%**); R² −0,180 | MAE melhor, mas R² negativo nos dois cortes; não promove |
| Moda | Extra Trees | MAE 1,688 vs. 1,786 (**+5,5%**); R² −0,780 | MAE 1,434 vs. 1,546 (**+7,2%**); R² −0,268 | Ganho sem confirmação estatística nos dois cortes; não promove |
| Beleza | Ensemble médio | MAE 1,360 vs. 1,430 (**+4,9%**); R² −0,421 | MAE 1,028 vs. 1,154 (**+10,9%**); R² **0,122** | Canal novo passa, temporal não; não promove |

### Leitura correta dos resultados

- **Automotivo** é o sinal mais promissor em erro absoluto: supera a baseline
  nos dois holdouts, mas não explica a variância com estabilidade suficiente.
- **Moda** é o setor mais difícil no alvo atual. O ganho é pequeno e os
  intervalos de confiança da melhora incluem piora possível.
- **Beleza** é o único caso com R² positivo no holdout de canal, mas falha no
  corte temporal; ainda não é uma generalização confiável.
- A **MLP** foi testada, mas não venceu de forma estável fora da amostra. Com
  poucas centenas de linhas por setor, uma rede maior tende a sobreajustar.
- O ensemble não deve ser adotado por estética: ele só entra se superar os
  modelos simples nos dois holdouts.

## 5. Diagnóstico: por que os resultados ainda são limitados

1. **Alvo heterogêneo:** vídeos com idades muito distintas têm snapshots que
   não representam o mesmo momento de vida.
2. **Cobertura audiovisual parcial:** 696 análises ainda representam uma parte
   da base de 3.425. A meta de 1.000 melhora o experimento, mas não cumpre o
   mínimo de 80% de cobertura definido para um treino final por setor.
3. **Poucas linhas por setor após os filtros:** cada holdout tem cerca de
   43–49 vídeos; a incerteza é naturalmente alta.
4. **Histórico de canal não recuperável retrospectivamente:** a API pública
   oferece o snapshot atual, não inscritos e views no instante passado.
5. **Cardinalidade de categorias do VLM:** descrições livres de enquadramento e
   câmera podem gerar muitas colunas após One-Hot. A seleção dentro dos folds
   reduz isso, mas a solução estrutural é taxonomia fechada/normalizada.

## 6. Plano priorizado de melhoria

### Prioridade 1 — não negociar o alvo

- Criar coleta diária de vídeos novos e materializar `views_30d` em janela
  fixa; manter data de publicação, data da leitura e política de tolerância.
- Enquanto isso, nomear os resultados atuais como **pesquisa exploratória de
  desempenho relativo**, não como previsão final de views.

### Prioridade 2 — aumentar cobertura com qualidade

- Concluir as 1.000 análises atuais, mantendo equilíbrio entre setores e
  canais.
- Para treino final com features audiovisuais, definir uma amostra fechada por
  setor e buscar cobertura VLM alta nela (idealmente >= 80%), em vez de misturar
  milhares de linhas sem VLM com poucas linhas analisadas.
- Auditar manualmente uma amostra aleatória por setor para medir consistência do
  extrator visual.

### Prioridade 3 — melhorar as features, não só o algoritmo

- Converter saídas livres do VLM em categorias controladas, por exemplo:
  `close_up|medium|wide|pov`, `static|pan|cut|dynamic` e tipos fixos de hook.
- Só ampliar transcrição segmentada após validar timestamps e autorização de
  uso do áudio. Comparar embeddings e sinais textuais em ablação controlada.
- Separar features disponíveis antes da publicação de métricas que só existem
  depois dela; manter a lista de features do modelo como contrato versionado.

### Prioridade 4 — modelo e avaliação

- Repetir a busca atual depois de reconstruir a base; manter baseline, holdout
  temporal e holdout por canal intactos.
- Favorecer Gradient/HistGradientBoosting e Ridge como referências fortes para
  tabelas pequenas; testar MLP apenas com regularização forte e dados suficientes.
- Criar também uma classificação secundária (baixo/médio/alto) somente após
  definir faixas sem vazamento. Aí entram AUC, F1, recall e matriz de confusão.
- Produzir intervalo de previsão, curva de resíduos e calibração antes de
  qualquer recomendação ao usuário.

## 7. O que já pode entrar no dashboard

- Cobertura por setor/canal e progresso audiovisual;
- versão do dataset, do prompt e do extrator;
- MAE, RMSE, R², Spearman, baseline e intervalos de confiança;
- comparação temporal vs. canal novo;
- estado do gate (`exploratory`, `model_not_ready` ou `promoted`);
- trilha MCP: entrada sanitizada, validação de features, versão do modelo,
  previsão/intervalo e explicação SHAP quando houver modelo promovido.

SHAP e PDP devem ser apresentados depois que um modelo passe pelos gates. Eles
explicam a contribuição dentro do modelo; não corrigem alvo ruim nem provam
causalidade.

## Conclusão

Temos uma arquitetura e uma coleta bem rastreáveis, além de um experimento
comparativo sério entre modelos. O melhor próximo investimento é **qualidade e
comparabilidade do dado**, sobretudo `views_30d` e cobertura audiovisual, antes
de aumentar a complexidade da rede neural. Hoje a conclusão metodológica correta
é: há sinais exploratórios úteis, mas ainda não há modelo confiável para prever
resultado de criativos em produção.
