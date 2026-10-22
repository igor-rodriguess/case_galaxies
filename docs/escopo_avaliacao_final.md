# Escopo da avaliação final

Congelamento da auditoria: `2026-10-20T01:45:00Z`. As decisões abaixo foram
fixadas antes de qualquer modelagem desta etapa. Nenhuma foi escolhida olhando o
desempenho de um modelo; mudar uma delas cria uma nova versão do escopo.

## Base de partida

- 3.425 vídeos de 30 canais, dez por setor, em
  `data/youtube/runs/expansion-1000-per-sector-balanced-2026-10-16`.
- Primeiro snapshot em `2026-10-16T11:16:34Z` e segundo snapshot completo em
  `2026-10-20`, ambos preservados com proveniência.
- 775 análises VLM únicas existem no armazenamento; a auditoria usa somente as
  concluídas até o congelamento acima, o que torna a base reprodutível mesmo se a
  coleta continuar depois.

## Decisões

| # | Decisão | Motivo | Efeito |
|---|---|---|---|
| 1 | Excluir `synthetic-lipstick-opening` | Teste sintético do Qwen, sem setor e sem registro na Data API | 770 → 769 |
| 2 | Tirar os 55 vídeos da fila caso-controle da regressão | Foram escolhidos pelo resultado: 29 virais e 26 controles, 53% de virais contra cerca de 10% na população. São os mesmos 55 do prompt v0.2.0: 41 do Pegasus 1.6 com janela de 0–15 s e 14 do Gemini 3.8 Flash. Ficam guardados para o Creative Battle e a validação manual | 769 → 714 |
| 3 | Protocolo principal: Gemini 3.5 Flash-Lite, `galaxies-opening-v0.1.0`, 0–3 s, 4 FPS | Extratores diferentes rotulam de formas diferentes. O extrator mudou entre lotes e poderia virar atalho para a data. Ele nunca entra como feature | Reextração concluída: 714 vídeos fora da fila caso-controle seguem exatamente esse protocolo |
| 4 | Manter todos os formatos e usar o formato como estrato da referência histórica do canal | A Data API não confirma Short, e a duração sozinha também não. Dentro do setor, vídeos com mais de 3 min têm mediana de views de 3,5× a 33× a dos vídeos de até 60 s, e todos os canais misturam formatos. As faixas de duração e o fallback ficam definidos na Parte 2, antes de olhar o alvo | O escopo passa a ser "criativos curtos e longos dos canais selecionados". Sensibilidade: só vídeos de até 180 s |
| 5 | Idade mínima de 7 dias no primeiro snapshot | A regra pré-definida escolhe o primeiro corte cuja mediana de crescimento diário fica em até 0,01 log-ponto/dia. Aos 7 dias a mediana foi 0,0098; o IC95% cruza o limite | Sensibilidades obrigatórias: 14 e 30 dias |

O critério de revisão da idade mínima usa apenas o crescimento das views entre
as duas leituras do mesmo vídeo. O desempenho de modelos nunca entra nessa escolha.
A regra foi fixada antes de abrir o segundo snapshot:

- crescimento diário de cada vídeo:
  `(log1p(views_2) - log1p(views_1)) / dias entre as leituras`;
- a idade mínima é o menor valor em {7, 14, 30} dias em que a mediana desse
  crescimento, entre os vídeos dos três setores com idade de `m` a `m + 7` dias
  na primeira leitura, fica em até 0,01 por dia (cerca de 1% de views a mais por dia);
- se nenhum dos três cortes cumprir a regra, fica o de 30 dias.

O limite de 0,01 por dia é uma convenção, não um resultado. A auditoria mostra a
curva inteira para que o corte possa ser contestado. Como o IC95% da mediana aos
7 dias cruza o limite, as sensibilidades de 14 e 30 dias são obrigatórias.

## Tamanho congelado da base principal

Contagens finais após a reextração e o segundo snapshot.

| Recorte | Automotivo | Beleza | Moda | Total |
|---|---|---|---|---:|
| Principal: idade ≥ 7 dias | 165 (9 canais, 6 com 10+ vídeos) | 201 (10 canais, 10 com 10+) | 215 (10 canais, 10 com 10+) | 581 |
| Sensibilidade: idade ≥ 14 dias | 117 (8 canais, 4 com 10+) | 176 (10 canais, 9 com 10+) | 192 (10 canais, 9 com 10+) | 485 |
| Sensibilidade: idade ≥ 30 dias | 88 (3 canais, 3 com 10+) | 133 (9 canais, 7 com 10+) | 142 (6 canais, 6 com 10+) | 363 |
| Sensibilidade: idade ≥ 7 dias e até 180 s | 62 (8 canais, 2 com 10+) | 123 (10 canais, 7 com 10+) | 103 (7 canais, 6 com 10+) | 288 |

Com 30 dias, automotivo fica com 3 canais. Nesse recorte, o teste por canal novo
não é viável em automotivo, e o setor só pode ser avaliado no corte temporal.

## Estado antes da modelagem

O segundo snapshot e a reextração já foram concluídos. A auditoria final está em
`reports/final/auditoria/` e o resumo interpretado em
`reports/final/01_auditoria_base.md`.

Ainda não é permitido promover um modelo. Antes disso, a Parte 2 deve:

1. Aplicar a normalização fechada e versionada às categorias VLM antes do split;
2. Remover colunas 100% ausentes ou constantes por setor e recodificar as colunas
   "constante nos observados" (por exemplo, `brand_legible` como booleano);
3. Avaliar apenas com os recortes congelados, incluindo 14 e 30 dias;
4. Relatar separadamente a fragilidade de automotivo, no qual o recorte de 30 dias
   deixa apenas três canais e inviabiliza holdout por canal.

## Correção de integridade temporal — 21/10/2026

Uma avaliação exploratória de **canal conhecido** chegou a AUC intra-canal de
0,663. Ela não é elegível para promoção. Embora os vídeos antecedentes fossem
filtrados por data de publicação, as taxas de formato e duração reutilizavam o
rótulo `viral`; esse rótulo é top 10% calculado sobre a história completa do
canal e, portanto, pode depender de vídeos publicados depois do criativo-alvo.

O executor agora só libera taxas de resultado histórico se houver uma coluna
que registre que o resultado já estava observado antes da publicação do alvo.
Os snapshots disponíveis foram coletados posteriormente e não satisfazem essa
condição. Até existir uma coleta prospectiva com horizonte fixo (por exemplo,
views no dia 30), o modo seguro usa somente conteúdo, metadados e cadência de
publicação; a avaliação retrospectiva fica restrita a diagnóstico e não pode
ser usada para escolher modelo.

O backtest temporal também recalcula a curva de idade e o limiar top 10% usando
apenas a janela de treino e os aplica ao período seguinte. O rótulo congelado
da população completa continua útil para análises descritivas e para o cofre,
mas não é usado para selecionar um modelo no corte temporal.

## Revisão de 21/10/2026: só Shorts

A decisão 4 foi revista pelo responsável pelo produto: **o Galaxies ranqueia
só Shorts**. Vídeos longos não são treinados, rotulados nem premiados. A Fase 1
mostrou por quê: com os dois formatos no mesmo rótulo, a métrica do ranking
passava a medir "vídeo longo ganha de Short" (AUC 0,734 só com a duração,
`reports/final/modeling/fase1-protocolo-corrigido-20261021.md`).

- População do ranking: vídeos com duração conhecida até 180 s, o limite atual
  dos Shorts. A Data API não marca Short, então até 180 s é um critério
  operacional: vídeos horizontais curtos também passam por ele.
- Sensibilidade: até 60 s, faixa em que quase todo vídeo é Short.
- Implementação: `galaxies.agent.format_ranking.shorts_population` e
  `scripts/evaluate_shorts_ranking.py`.
- A regressão de 581 vídeos (todos os formatos) continua documentada como
  estava e precisa ser refeita só com Shorts antes de qualquer uso no produto.

### Histórico completo de Shorts — 21/10/2026

`scripts/collect_channel_shorts_history.py` enumerou a playlist de uploads
inteira dos 30 canais (55 mil vídeos, cerca de 2.500 unidades de cota) e
guardou os detalhes dos vídeos de até 180 s em
`data/youtube/shorts_history/shorts-history-2026-10-21/` (fora do git).

A elegibilidade como Short depende da data de publicação
(`format_ranking.shorts_population`): nada antes de 12/07/2021, que é o
lançamento global; até 60 s entre essa data e 15/10/2024; até 180 s depois.
Com essa regra e idade de 30 dias ou mais no snapshot, a base de Shorts passa de
1.271 para **14.042 vídeos**: automobilismo 7.442 (10 canais com 10 ou mais),
beleza 3.697 (8) e moda 2.903 (6).

Cuidados:
- Em moda, um canal sozinho tem 62% dos Shorts; em beleza, 42%. Treino e
  métricas precisam pesar canais, não vídeos.
- Os números da Fase 2 usaram até 180 s sem olhar a data e a base antiga.
  Precisam ser refeitos com esta base e esta regra.
- O snapshot de métricas desta coleta é de 21/10/2026, diferente do de
  16/10/2026 da base antiga. As duas não devem ser misturadas no mesmo rótulo.
