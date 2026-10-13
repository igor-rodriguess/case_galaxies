# Dados de modelagem

`scripts/build_modeling_dataset.py` cria uma execução em `runs/<timestamp>/`:

- `feature_map.jsonl`: uma linha achatada por `video_id`;
- `manifest.json`: fontes e protocolo da junção;
- `readiness_audit.json`: cobertura, validade temporal e bloqueios por setor.

Essas execuções são locais e ignoradas pelo Git. O mapa não altera os JSONs de
origem; ele apenas os conecta e deriva colunas reproduzíveis.

Uma execução usa diretório imutável: repetir um `run_id` existente gera erro em vez
de sobrescrever silenciosamente uma auditoria anterior.

O campo `target_is_fixed_horizon` diferencia um alvo comparável (por exemplo,
views no dia 30) do snapshot atual. O treino final recusa snapshots variáveis.
