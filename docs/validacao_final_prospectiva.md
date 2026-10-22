# Validação final prospectiva

Data da reserva: 21/10/2026. Código: `src/galaxies/prospective.py`.

## Por que

A auditoria de 21/10 (`reports/final/modeling/auditoria-vazamentos-20261021.md`)
mostrou que todos os holdouts da base atual já foram consultados durante o
desenvolvimento: o corte temporal do ranking, os holdouts temporal e de canal da
regressão e as versões `final-frozen-evaluation-v1` a `v4`. Nenhum desses
conjuntos pode mais confirmar um campeão. A validação final passa a ser um
período que ainda não existe na base.

## O que fica reservado

- Vídeos dos 30 canais atuais publicados a partir de **21/10/2026 00:00 UTC**.
- Rótulo: views observadas perto do dia 30 (30 dias ± 18 h), lidas por coleta
  real. Nada é reconstruído nem extrapolado.
- Features: as mesmas do candidato congelado. Se o candidato usar VLM, a
  abertura desses vídeos é extraída com o protocolo principal (Gemini 3.5
  Flash-Lite, `galaxies-opening-v0.1.0`, 0–3 s, 4 FPS) **antes** de qualquer
  leitura do rótulo.

`exclude_prospective` remove esses vídeos em `galaxies.agent.protocol.load_dataset`
e no histórico de `scripts/run_frozen_model_evaluation.py`. Eles não entram em
treino, seleção, alvo, diagnóstico nem gráfico de desenvolvimento.

## Regras de abertura

1. Antes de abrir, congelar e registrar cada candidato com: código (commit),
   features, hiperparâmetros, setor/pooling, seed e o hash do arquivo de
   configuração. Também congelar o baseline e as métricas de decisão.
2. Abrir **uma única vez**, quando cada setor tiver pelo menos
   `MIN_VIDEOS_PER_SECTOR = 100` vídeos com a leitura de 30 dias.
3. Métricas e critério fixados agora:
   - ranking: AUC intra-canal, Precision@10%, Recall@10% e Lift@10%, com IC95%
     por bootstrap de canais e delta pareado contra o baseline;
   - regressão: MAE e RMSE em log, Spearman e cobertura conformal;
   - promoção por setor só se o IC95% do delta contra o baseline não cruzar zero.
4. Não há segunda tentativa no mesmo período. Uma nova versão do modelo exige
   um novo período reservado, com nova data de início.

## Prazo estimado

O histórico completo de Shorts (`data/youtube/shorts_history/shorts-history-2026-10-21/`)
registra, entre 16/09 e 16/10/2026, 165 Shorts em automobilismo, 119 em moda e
80 em beleza. Mantido esse ritmo, beleza é o gargalo: cerca de 38 dias de
publicações (até perto de 28/11) mais 30 dias de maturação. A abertura fica
perto do fim de dezembro de 2026. É uma estimativa pelo ritmo passado, não uma
data garantida.

## Fila de novos uploads

Implementada em `scripts/track_prospective_shorts.py`, com estado em
`data/youtube/prospective/`. Rodar **uma vez por dia**:

- descobre os uploads dos 30 canais publicados a partir da reserva e coloca na
  fila só os Shorts (até 180 s);
- guarda título, descrição, tags e thumbnail no primeiro contato, o mais perto
  possível da publicação, para que as features não dependam de edições
  posteriores;
- lê as métricas nas idades de 1, 3, 7, 14 e 30 dias, cada uma com ±18 h. Uma
  janela perdida fica marcada como perdida (`missed_ages`) e não é
  substituída;
- `--dry-run` mostra quantos Shorts têm rótulo de 30 dias por setor e se a regra
  de abertura já foi cumprida.

Primeira execução em 21/10/2026: 13 uploads novos, 8 Shorts na fila
(automobilismo 4, moda 2, beleza 2).

Agendamento sugerido no Windows (não foi criado automaticamente):

```powershell
schtasks /Create /SC DAILY /ST 09:00 /TN "Galaxies fila prospectiva" /TR "C:\Users\Inteli\Documents\case_galaxies\.venv\Scripts\python.exe C:\Users\Inteli\Documents\case_galaxies\scripts\track_prospective_shorts.py"
```

Com uma execução por dia, a janela de ±18 h sempre contém uma leitura. Um dia
sem execução pode perder as janelas curtas (1 e 3 dias); a de 30 dias só se
perde se a falha cobrir as 36 h em volta do dia 30.
