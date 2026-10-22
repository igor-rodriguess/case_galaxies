# Galaxies — interface

Next.js 16, React 19, TypeScript e Tailwind 4, no mesmo padrão do `volta/apps/web`:
tokens em `src/app/globals.css` (tema claro e escuro), `Shell` com barra lateral,
primitivos em `src/components/ui.tsx`, i18n PT/EN em `src/lib/i18n.tsx` e testes
com Vitest. Identidade da Galaxies: azul `#0065FF`, DM Sans e JetBrains Mono, e o
logo em `public/`.

## Rodar

São dois processos: a ponte Python, que fala com o servidor MCP real, e o Next.

```powershell
# terminal 1, na raiz do repositório
.venv\Scripts\python.exe scripts\run_frontend.py          # http://127.0.0.1:8765/api

# terminal 2
cd frontend
npm install
npm run dev                                               # http://127.0.0.1:3000
```

O Next encaminha `/api/*` para a ponte (`next.config.ts`). Para outra porta, use
`GALAXIES_BRIDGE_URL`.

## Verificar

```powershell
npm run typecheck
npm run lint
npm test
npm run build
```

## Telas

| Rota | O que mostra |
|---|---|
| `/` | Visão geral: estado real dos modelos, avaliação só de Shorts e o caminho de uma comparação |
| `/console` | Console MCP: chama as cinco tools e mostra resposta, trace, horário, duração e JSON |
| `/battle` | Creative Battle: A × B no mesmo canal e setor, veredito, confiança, alertas e inputs alterados |
| `/models` | Status por setor, backtests rolling (AUC, IC95%, Lift@10, Precision@10, estabilidade), SHAP e calibração |
| `/traces` | Auditoria de `recent_mcp_calls` com filtro por tool |

O trilho "MCP ao vivo" fica à direita em telas largas.

## Regras de exibição

- Todo número vem de uma tool MCP ou de um arquivo em `data/modeling/runs`.
  Valor ausente aparece como "—" ou "indisponível". Célula vazia de CSV nunca
  vira 0 (`cellNumber`).
- "Modelo promovido" só aparece quando `model_status` devolve `ready: true`. A
  referência é `report.reference`. Os demais modelos são candidatos
  exploratórios.
- No Creative Battle, B só aparece como mais bem pontuado com delta positivo e
  confiança diferente de `low` (`docs/creative_battle_mcp.md`). O aviso de que a
  comparação é associativa, não causal, fica sempre visível.
