// Reorganiza os relatórios para a interface. Só lê campos existentes.

import { cellNumber, isNumber } from "./format";
import type { Metrics, ModelStatus, Paired, RollingReport, Stability } from "./types";

export type Tier = "promoted" | "reference" | "exploratory";

export const TIER_TEXT: Record<Tier, { label: string; hint: string }> = {
  promoted: {
    label: "Promoted model",
    hint: "Passed the final gates and is cleared for the product.",
  },
  reference: {
    label: "Reference model",
    hint: "Baseline fixed before the comparison: a yardstick, not a champion.",
  },
  exploratory: {
    label: "Exploratory candidate",
    hint: "Research result: not promoted and should not drive a decision alone.",
  },
};

/** Promovido só quando model_status confirma; referência vem do relatório. */
export function tierFor(name: string, opts: { referenceNames?: string[]; promotedNames?: string[] } = {}): Tier {
  if (opts.promotedNames?.includes(name)) return "promoted";
  if (opts.referenceNames?.includes(name)) return "reference";
  return "exploratory";
}

export interface MetricBlock {
  aucIntra: number | null;
  aucIntraCi: [number, number] | null;
  lift10: number | null;
  lift10Ci: [number, number] | null;
  precision10: number | null;
  precision10Ci: [number, number] | null;
  recall10: number | null;
  ageRho: number | null;
  rows: number | null;
  positives: number | null;
  channels: number | null;
}

function range(low: unknown, high: unknown): [number, number] | null {
  return isNumber(low) && isNumber(high) ? [low, high] : null;
}

export function metricBlock(metrics: Metrics | undefined | null): MetricBlock | null {
  if (!metrics) return null;
  return {
    aucIntra: metrics.auc_intra ?? null,
    aucIntraCi: range(metrics.auc_intra_ci_low, metrics.auc_intra_ci_high),
    lift10: metrics.lift_top10 ?? null,
    lift10Ci: range(metrics.lift_ci_low, metrics.lift_ci_high),
    precision10: metrics.precision_top10 ?? null,
    precision10Ci: range(metrics.precision_ci_low, metrics.precision_ci_high),
    recall10: metrics.recall_top10 ?? null,
    ageRho: metrics.age_rho ?? null,
    rows: metrics.rows ?? null,
    positives: metrics.positives ?? null,
    channels: metrics.channels ?? null,
  };
}

export interface RollingRow {
  name: string;
  tier: Tier;
  metrics: MetricBlock | null;
  paired: { diff: number; ci: [number, number] | null; pBetter: number | null } | null;
  stability: Stability | null;
  weights: Record<string, number> | null;
}

function paired(value: Paired | undefined): RollingRow["paired"] {
  if (!value || !isNumber(value.diff)) return null;
  return { diff: value.diff, ci: range(value.ci_low, value.ci_high), pBetter: value.p_better ?? null };
}

/** Linhas da tabela de um relatório rolling (fold-aware, expandido ou ensemble). */
export function rollingRows(report: RollingReport | null | undefined, promotedNames: string[] = []): RollingRow[] {
  if (!report || !Array.isArray(report.results)) return [];
  const rows: RollingRow[] = [];
  const ensembleReference = typeof report.reference === "object" && report.reference !== null;
  if (ensembleReference) {
    for (const [name, metrics] of Object.entries(report.reference as Record<string, Metrics>)) {
      rows.push({
        name,
        tier: name === "logistic" ? "reference" : "exploratory",
        metrics: metricBlock(metrics),
        paired: null,
        stability: null,
        weights: null,
      });
    }
  }
  const referenceNames = ensembleReference ? [] : [report.reference as string];
  for (const result of report.results) {
    rows.push({
      name: result.name,
      tier: tierFor(result.name, { referenceNames, promotedNames }),
      metrics: metricBlock(result.metrics),
      paired: paired(result.paired_vs_reference ?? result.paired_vs_logistic),
      stability: result.stability ?? null,
      weights: result.weights ?? null,
    });
  }
  return rows;
}

export interface FoldPoint {
  fold: number;
  auc: number;
  rows: number | null;
}

/** AUC intra-canal por janela e modelo (fold_metrics.csv); célula vazia continua ausente. */
export function foldSeries(rows: Record<string, string>[] | undefined): Map<string, FoldPoint[]> {
  const series = new Map<string, FoldPoint[]>();
  for (const row of rows ?? []) {
    const auc = cellNumber(row.auc_intra);
    const fold = cellNumber(row.fold);
    if (auc === null || fold === null) continue;
    const list = series.get(row.model) ?? [];
    list.push({ fold, auc, rows: cellNumber(row.rows) });
    series.set(row.model, list);
  }
  for (const list of series.values()) list.sort((a, b) => a.fold - b.fold);
  return series;
}

export function shapRows(rows: Record<string, string>[] | undefined, limit = 12): { feature: string; value: number }[] {
  return (rows ?? [])
    .map((row) => ({ feature: row.feature, value: cellNumber(row.mean_abs_shap) }))
    .filter((row): row is { feature: string; value: number } => row.value !== null)
    .sort((a, b) => b.value - a.value)
    .slice(0, limit);
}

export function promotionSummary(statuses: (ModelStatus | null | undefined)[]): {
  state: "unknown" | "none" | "partial";
  promoted: string[];
} {
  const known = statuses.filter((status): status is ModelStatus => !!status && typeof status.ready === "boolean");
  if (!known.length) return { state: "unknown", promoted: [] };
  const promoted = known.filter((status) => status.ready).map((status) => status.sector);
  return { state: promoted.length ? "partial" : "none", promoted };
}
