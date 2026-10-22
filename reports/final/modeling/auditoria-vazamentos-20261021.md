# Auditoria de vazamentos e de seleção — 21/10/2026

Escopo: o ranking temporal (`scripts/evaluate_known_channel.py`,
`src/galaxies/agent/{protocol,evaluation,channel_history}.py`) e a regressão
congelada (`scripts/run_frozen_model_evaluation.py`,
`src/galaxies/modeling/frozen_evaluation.py`). Nenhum código, dado ou artefato
foi alterado por esta auditoria. Os números de verificação vêm de
`data/modeling/runs/feature-map-vlm1001-20261020/feature_map.jsonl` com o mesmo
split usado em `known-channel-asof-candidates-ci-20261021`.

## Gravidade alta

### A1. Regressão: o "vencedor" é escolhido no próprio holdout

`evaluate_outer_split` (linha 582) faz
`winner = min(available, key=lambda row: row["metrics"]["mae"])`, e essas
métricas foram calculadas no teste externo. A seleção interna (`_select_recipe`)
existe, mas o resultado reportado e as previsões, o OOD e a classificação usam o
melhor de cerca de 7 candidatos no teste. `run_frozen_model_evaluation.py:270`
escreve "Vencedor selecionado internamente", o que não corresponde ao código.

Consequência: os MAE de 0,994 (automobilismo), 0,912 (moda) e 0,930 (beleza)
são o mínimo sobre os candidatos no holdout, portanto otimistas. Também explica
parte de "o vencedor muda entre holdouts": cada holdout escolhe o seu.

### A2. Ranking: o corte "temporal" não é um corte de calendário

`temporal_split` reserva os 25% mais recentes **de cada canal**. Como cada canal
tem seu próprio corte, um modelo por setor treina com vídeos de outros canais
publicados depois de vídeos do teste.

| Setor | Linhas de treino publicadas depois do primeiro vídeo de teste do setor |
|---|---:|
| Automobilismo | 84% |
| Moda | 66% |
| Beleza | 71% |

Não é vazamento de feature do próprio vídeo, mas viola "nenhum dado do futuro".
Tendências de formato, de plataforma e de pauta do período de teste entram no
treino. A métrica intra-canal reduz o efeito, mas não o elimina.

### A3. Holdouts já usados no desenvolvimento

O próprio relatório de 21/10 reconhece que o corte temporal virou conjunto de
desenvolvimento. Os holdouts de canal da regressão são um único
`GroupShuffleSplit` com cerca de 2 canais por setor, também já consultados em
várias versões (`final-frozen-evaluation-v1` a `v4`). Nenhum dado intocado resta
na base atual para a validação final.

## Gravidade média

### M1. Rótulo do teste extrapola a curva de idade

Há uma única leitura de views (16/10). A curva de idade é ajustada no treino, que
é mais velho, e aplicada ao teste, que é mais novo. A taxa de positivos no teste
foi de 15,3%, contra 10,5% no treino: 21,3% em automobilismo, 14,5% em beleza e
10,4% em moda. Dentro do canal, o rho mediano entre o rótulo do teste e a idade
é +0,13. Fica abaixo do limite de 0,20, mas mostra que o limiar não tem a mesma
escala nos dois períodos.

### M2. Metadados lidos no snapshot, não na publicação

`title_character_count`, `description_character_count` e `tag_count` vêm do
estado atual do vídeo, que pode ter sido editado depois de publicado. A segunda
leitura encontrou uma troca de título em 3,6 dias. O risco é pequeno, mas a
afirmação "não usam título nem descrição" é imprecisa: o modelo usa o
comprimento de ambos.

### M3. Alvo da regressão usa canais do holdout

`causal_relative_targets(history, frame)` recebe como `history` todos os
3.425 vídeos. A curva setorial do alvo de um vídeo de treino usa, portanto,
vídeos anteriores dos canais reservados. O alvo é só rótulo, mas a escolha do
alvo também é feita olhando todas as linhas: `target_diagnostics(frame)` ordena
as definições pelo rho com a idade calculado na base inteira, incluindo o teste.

### M4. Intervalo conformal desalinhado

`_conformal` usa a receita de `_select_recipe`, não o vencedor reportado, e
separa a calibração por posição. No split por canal, a calibração compartilha
canais com o treino próprio, e a cobertura em canal novo tende a ficar abaixo
do nominal.

## Gravidade baixa

- `prepare_feature_frame` descarta colunas totalmente vazias olhando treino e
  teste juntos. Isso decide a estrutura, não o valor, mas deveria olhar só o
  treino.
- O limiar de OOD compara o treino com ele mesmo, e cada ponto é o próprio
  vizinho mais próximo. O limiar fica otimista.
- Cadência (`days_since_previous_upload`) só existe dentro da janela coletada.
  Os primeiros vídeos de cada canal ficam sem valor, e a ausência marca posição
  no tempo. Hoje isso afeta só o treino.

## O que foi verificado e está correto

- Views, likes e comentários do próprio vídeo não entram como feature;
  `is_forbidden` barra prefixos `target_` e `prior_views_*`.
- As taxas históricas por formato e duração exigem um timestamp de observação do
  resultado. Sem ele, voltam ao prior e marcam `history_available=0`.
- Na regressão, PCA de thumbnail, imputação, escala, seleção de features e pesos
  do ensemble são ajustados só no treino ou em dobras internas do treino.
- No ranking, a curva de idade e o limiar top 10% são ajustados só no treino.
- `prior_video_count` está bloqueado como índice de tempo.
