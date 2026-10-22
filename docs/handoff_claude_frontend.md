# Prompt de handoff — Claude no front-end

Trabalhe exclusivamente na camada de front-end/integração visual do projeto
Galaxies. Não continue experimentos, refatorações ou alterações no pipeline de
Machine Learning.

Antes de editar:

1. Consulte `git status` e `git diff`.
2. Desfaça **somente** alterações de ML feitas por você nesta sessão e que você
   consiga identificar com segurança. Não use `git reset --hard`, restauração em
   massa ou comandos que descartem mudanças de outras pessoas/agentes.
3. Se uma alteração sua estiver misturada com trabalho alheio no mesmo arquivo,
   preserve o arquivo e informe o conflito; não tente adivinhar autoria.

Considere protegidos e não modifique estes arquivos/artefatos:

- `src/galaxies/mcp_server.py`
- `src/galaxies/schemas.py`
- `src/galaxies/modeling/inference.py`
- `src/galaxies/agent/channel_history.py`
- `src/galaxies/agent/protocol.py`
- `src/galaxies/agent/evaluation.py`
- `src/galaxies/agent/rolling.py`
- `src/galaxies/agent/format_ranking.py`
- `src/galaxies/prospective.py`
- `scripts/evaluate_known_channel.py`
- `scripts/evaluate_known_channel_rolling.py`
- `scripts/evaluate_rolling_ensemble.py`
- `scripts/plot_rolling_backtest.py`
- `scripts/build_shorts_feature_map.py`
- `scripts/collect_channel_shorts_history.py`
- `scripts/evaluate_shorts_ranking.py`
- `scripts/plot_shorts_ranking.py`
- `tests/test_mcp_server.py`
- `tests/test_inference.py`
- `tests/test_channel_history.py`
- `tests/test_agent.py`
- `tests/test_rolling.py`
- `docs/creative_battle_mcp.md`
- `docs/escopo_avaliacao_final.md`
- `docs/validacao_final_prospectiva.md`
- `docs/status_modelagem_atual.md`
- `reports/final/RELATORIO_FINAL_2026-10-21.md`
- `data/modeling/runs/known-channel-rolling-fold-aware-20261021/`
- `data/modeling/runs/shorts-ranking-final-14k/`
- `reports/figures/known-channel-rolling-fold-aware-20261021/`
- `reports/figures/shorts-ranking-final-14k/`

Objetivo do seu trabalho:

- finalizar o front-end existente e deixá-lo executável;
- manter o chat/MCP como elemento principal e visualmente destacado;
- integrar as tools reais `model_status`, `predict_performance`,
  `rank_creatives`, `creative_battle` e `recent_mcp_calls`;
- criar a tela Creative Battle com versão A e B, setor e canal comuns, mudanças
  de features e o estado `model_not_ready`; só mostre delta, confiança e vencedor
  se a tool os retornar. O modelo misto antigo está aposentado por viés de formato;
  não simule nem invente uma recomendação;
- exibir métricas reais dos JSONs/CSVs de relatório: AUC intra-canal, IC95%,
  Lift@10, Precision@10, estabilidade temporal, calibração e SHAP quando houver;
- nunca inventar métrica. Quando um artefato estiver ausente, mostrar estado
  `indisponível`/`model_not_ready`;
- distinguir claramente `modelo de referência`, `candidato exploratório` e
  `modelo promovido`;
- incluir rastreabilidade: `model_run_id`, horário da chamada, trace MCP,
  confiança, alertas OOD e features ausentes;
- deixar responsivo e acessível, com estados de carregamento, erro e vazio.

Use como fontes:

- `docs/creative_battle_mcp.md`
- `data/modeling/runs/shorts-ranking-final-14k/report.json`
- `reports/figures/shorts-ranking-final-14k/`
- `reports/final/RELATORIO_FINAL_2026-10-21.md`
- `data/mcp/traces/mcp_calls.jsonl`

Ao finalizar:

1. Execute lint, testes e build do front-end.
2. Inicie o front e valide as telas principais.
3. Não faça commit/push sem autorização explícita.
4. Entregue uma lista objetiva dos arquivos alterados, comandos executados,
   resultado dos testes/build e qualquer integração ainda bloqueada.
