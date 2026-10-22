// Creative Battle: payload a partir do formulário e leitura da resposta da tool
// `creative_battle`. Nenhuma probabilidade é recalculada aqui.

import type { Battle, BattleSide, Sector } from "./types";

export type FieldKey =
  | "title"
  | "duration_seconds"
  | "description"
  | "tag_count"
  | "category_id"
  | "published_at"
  | "days_since_previous_upload"
  | "prior_upload_interval_median_days";

export type Version = Record<FieldKey, string>;

export interface FieldSpec {
  key: FieldKey;
  label: string;
  type: "text" | "number" | "textarea" | "select" | "datetime-local";
  hint?: string;
  min?: number;
  step?: number;
}

export const BATTLE_FIELDS: FieldSpec[] = [
  { key: "title", label: "Title", type: "text", hint: "Only the character count enters the model." },
  { key: "duration_seconds", label: "Duration (s)", type: "number", min: 1, step: 1 },
  { key: "description", label: "Description", type: "textarea", hint: "Only the character count enters the model." },
  { key: "tag_count", label: "Number of tags", type: "number", min: 0, step: 1 },
  { key: "category_id", label: "YouTube category", type: "select" },
  { key: "published_at", label: "Planned publication (UTC)", type: "datetime-local" },
  { key: "days_since_previous_upload", label: "Days since previous upload", type: "number", min: 0, step: 0.1 },
  {
    key: "prior_upload_interval_median_days",
    label: "Median interval between uploads (days)",
    type: "number",
    min: 0,
    step: 0.1,
  },
];

/** Categorias vistas no treino da baseline congelada (confidence_profile.json). */
export const CATEGORIES: { value: string; label: string }[] = [
  { value: "1", label: "1 · Film & Animation" },
  { value: "2", label: "2 · Autos & Vehicles" },
  { value: "17", label: "17 · Sports" },
  { value: "22", label: "22 · People & Blogs" },
  { value: "24", label: "24 · Entertainment" },
  { value: "26", label: "26 · Howto & Style" },
];

const NUMERIC = new Set<FieldKey>([
  "duration_seconds",
  "tag_count",
  "days_since_previous_upload",
  "prior_upload_interval_median_days",
]);

/** Só escalares chegam ao MCP: listas são recusadas pelo schema da tool. */
export function toFeatures(version: Partial<Version>): Record<string, string | number> {
  const features: Record<string, string | number> = {};
  for (const [key, raw] of Object.entries(version) as [FieldKey, string | undefined][]) {
    if (raw === undefined || raw === null || raw === "") continue;
    if (NUMERIC.has(key)) {
      const value = Number(raw);
      if (Number.isFinite(value)) features[key] = value;
    } else if (key === "published_at") {
      features.published_at = raw.length === 16 ? `${raw}:00Z` : raw;
    } else {
      features[key] = String(raw);
    }
  }
  return features;
}

export function battleArguments(input: {
  sector: Sector;
  channelId: string;
  baseline: Partial<Version>;
  candidate: Partial<Version>;
}) {
  return {
    sector: input.sector,
    channel_id: input.channelId,
    baseline: { features: toFeatures(input.baseline) },
    candidate: { features: toFeatures(input.candidate) },
  };
}

export function editedFields(baseline: Partial<Version>, candidate: Partial<Version>): FieldKey[] {
  return BATTLE_FIELDS.filter(({ key }) => String(baseline[key] ?? "") !== String(candidate[key] ?? "")).map(
    ({ key }) => key,
  );
}

export type VerdictKind = "empty" | "unavailable" | "error" | "inconclusive" | "candidate" | "baseline" | "tie";

/**
 * Regra de docs/creative_battle_mcp.md: B só aparece como cenário mais bem
 * pontuado quando o delta é positivo e a confiança não é baixa.
 */
export function verdict(response: Battle | null | undefined): { kind: VerdictKind; title: string; detail?: string } {
  if (!response) return { kind: "empty", title: "No comparison yet" };
  if (response.status === "model_not_ready") {
    return { kind: "unavailable", title: "model_not_ready", detail: response.comparison_scope };
  }
  if (response.status === "invalid_input") {
    return { kind: "error", title: "Invalid input", detail: (response.warnings ?? []).join(" ") };
  }
  const delta = response.delta?.viral_probability_calibrated;
  if (typeof delta !== "number" || !Number.isFinite(delta)) return { kind: "unavailable", title: "Delta unavailable" };
  if (response.confidence_flag === "low") {
    return {
      kind: "inconclusive",
      title: "Inconclusive: low confidence",
      detail: "At least one version left the range seen in training or has a missing feature.",
    };
  }
  if (delta > 0) return { kind: "candidate", title: "B is the higher-scored scenario" };
  if (delta < 0) return { kind: "baseline", title: "A remains the higher-scored scenario" };
  return { kind: "tie", title: "Tie in the model" };
}

export function sideConfidence(side: BattleSide | null | undefined) {
  const confidence = side?.confidence ?? {};
  return {
    level: confidence.level ?? "unavailable",
    alerts: confidence.alerts ?? [],
    missing: confidence.missing_model_inputs ?? [],
  };
}
