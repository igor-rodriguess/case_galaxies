"use client";

/** Interface em dois idiomas, como no VOLTA: chave em inglês, tradução PT no mesmo arquivo. */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type Lang = "pt" | "en";

export const PT: Record<string, string> = {
  // Navegação e moldura
  Overview: "Visão geral",
  "MCP Console": "Console MCP",
  "Creative Battle": "Creative Battle",
  "Models & evidence": "Modelos e evidências",
  "MCP traces": "Traces MCP",
  Product: "Produto",
  Evidence: "Evidência",
  Sections: "Seções",
  "Creative intelligence": "Creative intelligence",
  "Skip to content": "Pular para o conteúdo",
  "Open menu": "Abrir menu",
  "Close menu": "Fechar menu",
  Language: "Idioma",
  "Switch to dark theme": "Usar tema escuro",
  "Switch to light theme": "Usar tema claro",
  "MCP connected": "MCP conectado",
  "Bridge offline": "Ponte desligada",
  "Checking bridge": "Verificando a ponte",
  "No model promoted: predictions answer model_not_ready.": "Nenhum modelo promovido: as previsões respondem model_not_ready.",
  "Reading model_status…": "Lendo model_status…",
  Promoted: "Promovido",
  "Association, not causation. Every number comes from an MCP tool or from the reports in data/modeling/runs.":
    "Associação, não causalidade. Todo número vem de uma tool MCP ou dos relatórios em data/modeling/runs.",
  "Live MCP calls": "Chamadas MCP ao vivo",
  "MCP live": "MCP ao vivo",
  "Call trail": "Trilha de chamadas",
  "No calls recorded yet.": "Nenhuma chamada registrada ainda.",
  "Full audit": "Auditoria completa",

  // Estados
  Loading: "Carregando",
  "Something failed": "Algo falhou",
  "Try again": "Tentar de novo",
  unavailable: "indisponível",
  "Could not reach the local bridge. Is it running?": "Não foi possível falar com a ponte local. Ela está rodando?",

  // Níveis do modelo
  "Promoted model": "Modelo promovido",
  "Reference model": "Modelo de referência",
  "Exploratory candidate": "Candidato exploratório",
  "Passed the final gates and is cleared for the product.": "Passou pelos gates finais e está liberado para o produto.",
  "Baseline fixed before the comparison: a yardstick, not a champion.": "Baseline fixada antes da comparação: é régua, não campeão.",
  "Research result: not promoted and should not drive a decision alone.":
    "Resultado de pesquisa: não promovido e não deve orientar decisão sozinho.",

  // Visão geral
  "Decide which creative": "Decida qual criativo",
  "to test first.": "testar primeiro.",
  "Compare versions of a Short before publishing, ask the MCP and see exactly what the evidence supports, and what it does not support yet.":
    "Compare versões de um Short antes de publicar, consulte o MCP e veja exatamente o que a evidência sustenta e o que ainda não sustenta.",
  "Open Creative Battle": "Abrir o Creative Battle",
  "Ask the MCP": "Perguntar ao MCP",
  "No model promoted": "Nenhum modelo promovido",
  "Association, not causation": "Associação, não causalidade",
  "Where the evidence stands": "Onde a evidência está",
  "Honest numbers, straight from the reports": "Números honestos, direto dos relatórios",
  // Base v2
  "Gain over reference": "Ganho sobre a referência",
  reference: "referência",
  "base rate": "taxa base",
  "Offline CI rule passed": "Regra offline de IC passou",
  "Not promoted": "Não promovido",
  "Viral = 3× the channel median": "Viral = 3× a mediana do canal",
  "age shortcut": "atalho de idade",
  Reference: "Referência",
  "Gain [95% CI]": "Ganho [IC95%]",
  "Ranking on dataset v2": "Ranking na base v2",
  "This is retrospective evidence; prospective validation is still required.":
    "Esta é uma evidência retrospectiva; ainda é necessária validação prospectiva.",
  "With the cleaned dataset and the outlier label, the ranking beats the reference with a favorable CI. Automotive is strongest; fashion's point estimate is below chance, but uncertain. This is retrospective evidence and still needs prospective validation.":
    "Com a base limpa e o rótulo de outlier, o ranking supera a referência com IC favorável. Automotivo é o mais forte; a estimativa pontual de moda fica abaixo do acaso, mas é incerta. Esta é uma evidência retrospectiva e ainda precisa de validação prospectiva.",
  "Current dataset v2 report unavailable.": "Relatório atual da base v2 indisponível.",
  "Lift@10": "Lift@10",
  "Precision@10": "Precision@10",
  "See all evidence": "Ver toda a evidência",
  "Shorts ranking · AUC per channel": "Ranking de Shorts · AUC por canal",
  "Shorts ranking · Lift@10": "Ranking de Shorts · Lift@10",
  "Shorts test set": "Teste com Shorts",
  "Battle model · temporal AUC": "Modelo do Battle · AUC temporal",
  "95% CI": "IC95%",
  "0.5 is chance": "0,5 é o acaso",
  "1.0 is chance": "1,0 é o acaso",
  positives: "positivos",
  channels: "canais",
  "Shorts-only evaluation not found.": "Avaliação só de Shorts não encontrada.",
  "Battle model report not found.": "Relatório do modelo do Battle não encontrado.",
  "Reading:": "Leitura:",
  "How a comparison travels": "Como uma comparação percorre o sistema",
  "From the draft to the auditable trace": "Do rascunho ao trace auditável",
  "Pre-publication creative": "Criativo antes de publicar",
  "Only fields known before publishing: duration, title, category, schedule and channel cadence.":
    "Só campos conhecidos antes de publicar: duração, título, categoria, horário e cadência do canal.",
  "Typed MCP tools": "Tools MCP tipadas",
  "The server validates sector, channel and features, and refuses lists or inconsistent scenarios.":
    "O servidor valida setor, canal e features, e recusa listas ou cenários inconsistentes.",
  "Frozen reference model": "Modelo de referência congelado",
  "Calibrated probability of the historical top-10% label, with confidence and out-of-range alerts.":
    "Probabilidade calibrada do rótulo histórico top 10%, com confiança e alertas de fora da faixa.",
  "Auditable trace": "Trace auditável",
  "Each call records sector, channel, status, winner and confidence, never the creative itself.":
    "Cada chamada registra setor, canal, status, vencedor e confiança, nunca o criativo em si.",
  "Call model_status, predict_performance, rank_creatives, creative_battle and recent_mcp_calls.":
    "Chame model_status, predict_performance, rank_creatives, creative_battle e recent_mcp_calls.",
  "Version A against version B, same channel and sector, with delta, confidence and the non-causal warning.":
    "Versão A contra versão B, mesmo canal e setor, com delta, confiança e o aviso de não causalidade.",
  "AUC, 95% CI, Lift@10, Precision@10, stability per window, SHAP and calibration.":
    "AUC, IC95%, Lift@10, Precision@10, estabilidade por janela, SHAP e calibração.",
  Open: "Abrir",

  // Console
  "Every answer is the raw output of a real tool, with time, duration, trace and the full JSON. No language model writes text or numbers here.":
    "Cada resposta é a saída crua de uma tool real, com horário, duração, trace e o JSON completo. Nenhum modelo de linguagem escreve texto ou número aqui.",
  "Conversation with the MCP": "Conversa com o MCP",
  "Connected to the galaxies-performance server.": "Conectado ao servidor galaxies-performance.",
  "Try one of these, or pick a tool below to load a ready-made call.":
    "Experimente um destes ou escolha uma tool abaixo para carregar uma chamada pronta.",
  Calling: "Chamando",
  "Calling…": "Chamando…",
  "Command for the MCP": "Comando para o MCP",
  "Insert a call template": "Inserir modelo de chamada",
  "Send to MCP": "Enviar ao MCP",
  "This console only runs MCP tools. Commands:": "Este console só executa tools MCP. Comandos:",
  "status moda — model_status for one sector (no sector: all three)": "status moda — model_status de um setor (sem setor: os três)",
  "traces 10 — recent_mcp_calls": "traces 10 — recent_mcp_calls",
  "/<tool> {json} — calls any interface tool with JSON arguments": "/<tool> {json} — chama qualquer tool da interface com argumentos JSON",
  "Type a command.": "Digite um comando.",
  "Arguments must be valid JSON.": "Os argumentos precisam ser um JSON válido.",
  "Arguments must be a JSON object.": "Os argumentos precisam ser um objeto JSON.",
  "Response from": "Resposta de",
  "via MCP": "via MCP",
  "MCP rejected the call": "O MCP recusou a chamada",
  confidence: "confiança",
  Confidence: "Confiança",
  creative: "criativo",
  "No creative was ranked.": "Nenhum criativo foi ordenado.",
  "calls recorded": "chamadas registradas",
  "not recorded": "não registrado",
  time: "horário",
  transport: "transporte",
  model_run_id: "model_run_id",
  "MCP trace": "Trace MCP",
  model: "modelo",
  "Arguments sent": "Argumentos enviados",
  "Full response": "Resposta completa",

  // Creative Battle
  "Version A against version B": "Versão A contra versão B",
  "Same channel, same sector, two versions of the creative. The frozen model says which scenario scores higher and how confident it is.":
    "Mesmo canal, mesmo setor, duas versões do criativo. O modelo congelado diz qual cenário pontua melhor e com que confiança.",
  "Sector (shared by A and B)": "Setor (comum a A e B)",
  "Channel (shared by A and B)": "Canal (comum a A e B)",
  "no channel available": "nenhum canal disponível",
  "Copy A into B": "Copiar A para B",
  Automotive: "Automobilismo",
  Fashion: "Moda",
  Beauty: "Beleza",
  "A · current version": "A · versão atual",
  "B · alternative version": "B · versão alternativa",
  Title: "Título",
  "Duration (s)": "Duração (s)",
  Description: "Descrição",
  "Number of tags": "Quantidade de tags",
  "YouTube category": "Categoria do YouTube",
  "Planned publication (UTC)": "Publicação planejada (UTC)",
  "Days since previous upload": "Dias desde o upload anterior",
  "Median interval between uploads (days)": "Intervalo mediano entre uploads (dias)",
  "Only the character count enters the model.": "Só o número de caracteres entra no modelo.",
  "1 · Film & Animation": "1 · Filmes e animação",
  "2 · Autos & Vehicles": "2 · Autos e veículos",
  "17 · Sports": "17 · Esportes",
  "22 · People & Blogs": "22 · Pessoas e blogs",
  "24 · Entertainment": "24 · Entretenimento",
  "26 · Howto & Style": "26 · Estilo e instruções",
  changed: "alterado",
  "field(s) differ between A and B": "campo(s) diferente(s) entre A e B",
  "A and B are identical: the model will see no difference": "A e B estão iguais: o modelo não verá diferença",
  "Compare A × B in the MCP": "Comparar A × B no MCP",
  "Comparing…": "Comparando…",
  "Comparing in the MCP": "Comparando no MCP",
  "Build both versions and compare": "Monte as duas versões e compare",
  "The result shows delta, confidence, out-of-range alerts and the call trace.":
    "O resultado mostra delta, confiança, alertas de fora da faixa e o trace da chamada.",
  "No comparison yet": "Nenhuma comparação ainda",
  "Invalid input": "Entrada inválida",
  "Delta unavailable": "Delta indisponível",
  "Inconclusive: low confidence": "Inconclusivo: confiança baixa",
  "At least one version left the range seen in training or has a missing feature.":
    "Ao menos uma versão saiu da faixa vista no treino ou tem feature ausente.",
  "B is the higher-scored scenario": "B é o cenário mais bem pontuado pelo modelo",
  "A remains the higher-scored scenario": "A continua mais bem pontuado pelo modelo",
  "Tie in the model": "Empate no modelo",
  "Frozen model verdict": "Veredito do modelo congelado",
  probability: "probabilidade",
  score: "escore",
  low: "baixa",
  standard: "padrão",
  normal: "normal",
  "Associative comparison, not causal.": "Comparação associativa, não causal.",
  "Validate the winning version in a controlled test before attributing an effect to the change.":
    "Valide a versão vencedora num teste controlado antes de atribuir efeito à mudança.",
  "Calibrated probability of landing in the channel's top 10% (historical label).":
    "Probabilidade calibrada de ficar no top 10% do canal (rótulo histórico).",
  "ranking score": "escore de ranking",
  "Feature outside the training p01–p99 range, unseen category or missing value":
    "Feature fora da faixa p01–p99 do treino, categoria nunca vista ou valor ausente",
  "confidence alert(s)": "alerta(s) de confiança",
  "missing feature(s)": "feature(s) ausente(s)",
  "Largest local contributions (log-odds)": "Maiores contribuições locais (log-odds)",
  "Model inputs that changed": "Inputs do modelo que mudaram",
  "No feature used by the model changed": "Nenhuma feature usada pelo modelo mudou",
  "Edit fields that enter the model, such as duration or category.": "Edite campos que entram no modelo, como duração ou categoria.",
  "Model behind this comparison": "Modelo usado nesta comparação",
  "not promoted": "não promovido",
  "Evaluation of the frozen model": "Avaliação do modelo congelado",
  Evaluation: "Avaliação",
  "Within-channel AUC": "AUC intra-canal",
  Channels: "Canais",
  "New channel (LOCO)": "Canal novo (LOCO)",
  "Temporal cut": "Corte temporal",
  Vault: "Cofre",
  "evaluation unavailable": "avaliação indisponível",
  "The frozen model report was not found.": "O relatório do modelo congelado não foi encontrado.",

  // Modelos
  "What the numbers actually say": "O que os números realmente dizem",
  "Everything on this page comes from the JSON/CSV reports and the MCP tools. A missing value shows as unavailable.":
    "Tudo nesta página vem dos relatórios JSON/CSV e das tools MCP. Valor ausente aparece como indisponível.",
  "Status per sector": "Status por setor",
  "Latest evaluation · Shorts only": "Avaliação mais recente · só Shorts",
  "AUC per channel": "AUC por canal",
  "Test set": "Teste",
  Sector: "Setor",
  "Chosen in inner folds": "Escolhido nas dobras internas",
  "AUC per channel [95% CI]": "AUC por canal [IC95%]",
  Positives: "Positivos",
  "Ranking backtests": "Backtests de ranking",
  "Rolling chronological windows; age curve and threshold fitted on training only.":
    "Janelas cronológicas crescentes; curva de idade e limiar ajustados só no treino.",
  "Evaluation run": "Execução de avaliação",
  "Rolling fold-aware": "Rolling fold-aware",
  "Rolling expanded": "Rolling expandido",
  "OOF ensembles": "Ensembles OOF",
  "report unavailable": "relatório indisponível",
  "These backtests mix Shorts and long videos.": "Estes backtests misturam Shorts e vídeos longos.",
  "base: all formats": "base: todos os formatos",
  "Within-channel AUC with 95% CI per model": "AUC intra-canal com IC95% por modelo",
  chance: "acaso",
  "Dot = within-channel AUC · bar = 95% CI (bootstrap over channels)":
    "Ponto = AUC intra-canal · traço = IC95% (bootstrap de canais)",
  Model: "Modelo",
  Status: "Status",
  "AUC within channel [95% CI]": "AUC intra-canal [IC95%]",
  "Δ AUC vs reference [95% CI]": "Δ AUC vs. referência [IC95%]",
  "Temporal stability": "Estabilidade temporal",
  "ρ age": "ρ idade",
  "windows above ref.": "janelas acima da ref.",
  "worst Δ": "pior Δ",
  "Temporal stability: within-channel AUC per window": "Estabilidade temporal: AUC intra-canal por janela",
  Window: "Janela",
  "Values below 0.5 (orange) are worse than chance in that window.": "Valores abaixo de 0,5 (laranja) são piores que o acaso naquela janela.",
  "Explanation and calibration of the reference model": "Explicação e calibração do modelo de referência",
  "Global SHAP": "SHAP global",
  "SHAP unavailable": "SHAP indisponível",
  "Probability calibration": "Calibração da probabilidade",
  "calibration unavailable": "calibração indisponível",
  "OOF Brier, raw": "Brier OOF sem calibrar",
  "OOF Brier, calibrated": "Brier OOF calibrado",
  "Lower is better. A reliability curve per bin does not exist in the artifacts and is not shown.":
    "Menor é melhor. A curva de confiabilidade por faixa não existe nos artefatos e não é exibida.",
  Figure: "Figura",
  "figures unavailable": "figuras indisponíveis",

  // Revisão de layout
  "See evaluation": "Ver avaliação",
  Channel: "Canal",
  "A and B always share sector and channel: only the creative changes.":
    "A e B sempre compartilham setor e canal: só o criativo muda.",
  Field: "Campo",
  "Current version": "Versão atual",
  "Alternative version": "Versão alternativa",
  "Reset example": "Restaurar exemplo",
  "Comparison result": "Resultado da comparação",
  "Open full size": "Abrir em tamanho real",
  "Nested selection on calendar folds, one external read, Shorts only.":
    "Seleção aninhada em dobras de calendário, uma leitura externa, só Shorts.",
  "Windows above ref.": "Janelas acima da ref.",
  "Where to start": "Por onde começar",
  Close: "Fechar",
  "Same consecutive calls are grouped.": "Chamadas iguais em sequência aparecem agrupadas.",
  "Shorts · current": "Shorts · atual",
  "AUC per channel · Shorts": "AUC por canal · Shorts",
  "Lift@10 · Shorts": "Lift@10 · Shorts",
  "Compare two versions": "Comparar duas versões",
  "Edit A and B, send to the MCP and read verdict, confidence and alerts.":
    "Edite A e B, envie ao MCP e leia veredito, confiança e alertas.",
  "Ask the MCP directly": "Perguntar direto ao MCP",
  "Run any of the five tools and inspect the raw answer and its trace.":
    "Rode qualquer uma das cinco tools e veja a resposta crua e o trace.",
  "Check the evidence": "Conferir a evidência",
  "See what each model scored, with confidence intervals.": "Veja quanto cada modelo marcou, com intervalos de confiança.",

  // Traces
  Audit: "Auditoria",
  "The server records only sector, channel, counts, status, winner and confidence. Creative values never enter the log.":
    "O servidor registra só setor, canal, contagens, status, vencedor e confiança. Valores do criativo nunca entram no log.",
  Reload: "Recarregar",
  "Filter by tool": "Filtrar por tool",
  "all tools": "todas as tools",
  "No calls for this filter": "Nenhuma chamada para este filtro",
  Time: "Horário",
  "Input (sanitized)": "Entrada (sanitizada)",
  Output: "Saída",
};

interface I18n {
  lang: Lang;
  locale: string;
  setLang: (lang: Lang) => void;
  t: (key: string) => string;
}

const Ctx = createContext<I18n | null>(null);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>("pt");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("galaxies.lang");
      // eslint-disable-next-line react-hooks/set-state-in-effect -- localStorage só existe depois da montagem
      if (saved === "en" || saved === "pt") setLangState(saved);
    } catch {}
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";
  }, [lang]);
  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    try {
      localStorage.setItem("galaxies.lang", next);
    } catch {}
  }, []);
  const t = useCallback((key: string) => (lang === "pt" ? (PT[key] ?? key) : key), [lang]);
  const value = useMemo(() => ({ lang, locale: lang === "pt" ? "pt-BR" : "en-US", setLang, t }), [lang, setLang, t]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useT(): I18n {
  const value = useContext(Ctx);
  if (!value) throw new Error("useT precisa estar dentro de I18nProvider.");
  return value;
}
