// Console MCP: texto → chamadas de tool. Determinístico; não há LLM gerando texto.

import { SECTORS, type Sector, type ToolName } from "./types";

export const TOOLS: ToolName[] = [
  "model_status",
  "predict_performance",
  "rank_creatives",
  "creative_battle",
  "recent_mcp_calls",
];

export const TEMPLATES: Record<ToolName, Record<string, unknown>> = {
  model_status: { sector: "moda" },
  predict_performance: { sector: "moda", features: { duration_seconds: 30, title_character_count: 42 } },
  rank_creatives: {
    sector: "beleza",
    creatives: [
      { video_id: "draft-a", features: { duration_seconds: 25 } },
      { video_id: "draft-b", features: { duration_seconds: 55 } },
    ],
  },
  creative_battle: {
    sector: "moda",
    channel_id: "UC7eUDymR_ThtyGV1IxUSpcw",
    baseline: { features: { duration_seconds: 45, title_character_count: 40, description_character_count: 80, tag_count: 3, category_id: "26" } },
    candidate: { features: { duration_seconds: 20, title_character_count: 22, description_character_count: 80, tag_count: 3, category_id: "26" } },
  },
  recent_mcp_calls: { limit: 10 },
};

export interface Call {
  tool: ToolName;
  arguments: Record<string, unknown>;
}

export type Parsed = { calls: Call[] } | { help: true } | { error: string };

export function parseCommand(text: string): Parsed {
  const input = String(text ?? "").trim();
  if (!input) return { error: "Type a command." };
  const slash = input.match(/^\/(\w+)\s*([\s\S]*)$/);
  if (slash) {
    const [, tool, rest] = slash;
    if (!TOOLS.includes(tool as ToolName)) return { error: `Tool "${tool}" is not available in this interface.` };
    let args: unknown = {};
    if (rest.trim()) {
      try {
        args = JSON.parse(rest);
      } catch {
        return { error: "Arguments must be valid JSON." };
      }
    }
    if (typeof args !== "object" || args === null || Array.isArray(args)) {
      return { error: "Arguments must be a JSON object." };
    }
    return { calls: [{ tool: tool as ToolName, arguments: args as Record<string, unknown> }] };
  }
  const words = input.toLowerCase().split(/\s+/);
  if (["status", "modelo", "modelos", "model", "models"].includes(words[0])) {
    const sector = SECTORS.find((item) => words.includes(item));
    const sectors: Sector[] = sector ? [sector] : SECTORS;
    return { calls: sectors.map((item) => ({ tool: "model_status", arguments: { sector: item } })) };
  }
  if (["traces", "trace", "chamadas", "calls"].includes(words[0])) {
    const limit = Number.parseInt(words[1] ?? "10", 10);
    return {
      calls: [{ tool: "recent_mcp_calls", arguments: { limit: Number.isFinite(limit) ? Math.min(Math.max(limit, 1), 100) : 10 } }],
    };
  }
  return { help: true };
}

export function templateCommand(tool: ToolName): string {
  return `/${tool} ${JSON.stringify(TEMPLATES[tool], null, 2)}`;
}
