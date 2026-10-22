// Agrupa chamadas MCP iguais em sequência para a trilha não virar uma parede.

import type { TraceEvent } from "./types";

export type Tone = "ok" | "warn" | "danger" | "neutral";

export function outcomeTone(output: Record<string, unknown> | undefined): Tone {
  if (output?.status === "ok" || output?.ready === true) return "ok";
  if (output?.status === "invalid_input") return "danger";
  if (output?.status === "model_not_ready" || output?.ready === false) return "warn";
  return "neutral";
}

export function outcomeLabel(output: Record<string, unknown> | undefined): string {
  if (typeof output?.status === "string") return output.status;
  if (output?.ready === true) return "promoted";
  if (output?.ready === false) return "model_not_ready";
  return "—";
}

export interface TraceGroup {
  key: string;
  tool: string;
  label: string;
  tone: Tone;
  count: number;
  latest: TraceEvent;
  sectors: string[];
}

/** Junta eventos consecutivos com a mesma tool e o mesmo resultado. */
export function groupEvents(events: TraceEvent[]): TraceGroup[] {
  const groups: TraceGroup[] = [];
  for (const event of events) {
    const label = outcomeLabel(event.output_summary);
    const sector = typeof event.input_summary?.sector === "string" ? event.input_summary.sector : null;
    const last = groups[groups.length - 1];
    if (last && last.tool === event.tool_name && last.label === label) {
      last.count += 1;
      if (sector && !last.sectors.includes(sector)) last.sectors.push(sector);
      continue;
    }
    groups.push({
      key: event.trace_id,
      tool: event.tool_name,
      label,
      tone: outcomeTone(event.output_summary),
      count: 1,
      latest: event,
      sectors: sector ? [sector] : [],
    });
  }
  return groups;
}
