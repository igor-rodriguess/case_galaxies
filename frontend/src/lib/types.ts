// Formas das respostas da ponte e das tools MCP. Espelham src/galaxies/schemas.py
// e os relatórios em data/modeling/runs; campos opcionais podem faltar.

export type Sector = "automotivo" | "moda" | "beleza";
export const SECTORS: Sector[] = ["automotivo", "moda", "beleza"];

export type ToolName =
  | "model_status"
  | "predict_performance"
  | "rank_creatives"
  | "creative_battle"
  | "recent_mcp_calls";

export interface TraceEvent {
  schema_version?: string;
  trace_id: string;
  recorded_at: string;
  tool_name: string;
  input_summary?: Record<string, unknown>;
  output_summary?: Record<string, unknown>;
}

export interface ToolCall<T = unknown> {
  tool: ToolName;
  arguments: Record<string, unknown>;
  transport: string;
  started_at: string;
  duration_ms: number;
  is_error: boolean;
  result: T | null;
  error_text: string | null;
  trace: TraceEvent | null;
}

export interface ModelStatus {
  sector: Sector;
  ready: boolean;
  run_id?: string | null;
  model_name?: string | null;
  target?: string | null;
  blocking_reasons?: string[];
}

export interface Driver {
  feature: string;
  importance: number;
  value?: unknown;
  direction?: string;
}

export interface Prediction {
  status: "ok" | "model_not_ready" | "invalid_input";
  sector: Sector;
  video_id?: string | null;
  predicted_metric?: number | null;
  metric_name?: string;
  interval_low?: number | null;
  interval_high?: number | null;
  confidence_flag: "normal" | "low" | "unavailable";
  relative_to_channel?: number | null;
  top_drivers?: Driver[];
  explanation: string;
  model_run_id?: string | null;
  warnings?: string[];
}

export interface Ranking {
  sector: Sector;
  status: "ok" | "model_not_ready" | "partial" | "invalid_input";
  items: { rank: number; request_index: number; prediction: Prediction }[];
  warnings?: string[];
}

export interface ConfidenceAlert {
  feature: string;
  type: string;
  value?: unknown;
}

export interface BattleSide {
  ranking_score_raw?: number;
  viral_probability_calibrated?: number;
  prediction_scope?: string;
  confidence?: {
    level?: string;
    rule?: string;
    alerts?: ConfidenceAlert[];
    missing_model_inputs?: string[];
  };
  top_local_contributions?: { feature: string; log_odds_contribution: number; direction: string }[];
}

export interface Battle {
  status: "ok" | "model_not_ready" | "invalid_input";
  sector: Sector;
  channel_id: string;
  model_run_id?: string | null;
  model_name?: string | null;
  comparison_scope: string;
  changed_model_inputs?: { feature: string; baseline: unknown; candidate: unknown }[];
  baseline?: BattleSide | null;
  candidate?: BattleSide | null;
  delta?: Record<string, number>;
  winner?: "baseline" | "candidate" | "tie" | "undetermined";
  confidence_flag?: "normal" | "low" | "unavailable";
  warnings?: string[];
}

export interface Traces {
  events: TraceEvent[];
  event_count: number;
}

export interface Metrics {
  auc_intra?: number;
  auc_intra_ci_low?: number;
  auc_intra_ci_high?: number;
  lift_top10?: number;
  lift_ci_low?: number;
  lift_ci_high?: number;
  precision_top10?: number;
  precision_ci_low?: number;
  precision_ci_high?: number;
  recall_top10?: number;
  age_rho?: number;
  rows?: number;
  positives?: number;
  channels?: number;
}

export interface Paired {
  diff: number;
  ci_low: number;
  ci_high: number;
  p_better?: number;
}

export interface Stability {
  folds: number;
  folds_better_than_reference: number;
  worst_fold_auc_delta: number;
  per_fold_auc_delta?: Record<string, number>;
}

export interface RollingResult {
  name: string;
  metrics: Metrics;
  stability?: Stability;
  paired_vs_reference?: Paired;
  paired_vs_logistic?: Paired;
  weights?: Record<string, number>;
}

export interface RollingReport {
  protocol?: string;
  feature_map?: string;
  source_run?: string;
  reference: string | Record<string, Metrics>;
  results: RollingResult[];
}

export type Loaded<T> =
  | { status: "ok"; path: string; data: T }
  | { status: "unavailable"; path: string; reason?: string };

export type LoadedCsv = { status: "ok"; path: string; rows: Record<string, string>[] } | { status: "unavailable"; path: string };

export interface Figure {
  run: string;
  file: string;
  url: string;
}

export interface BattleModelReport {
  decision?: string;
  evaluation?: {
    loco?: Metrics;
    temporal?: Metrics;
    vault?: Metrics;
    oof_brier_raw?: number;
    oof_brier_cross_calibrated?: number;
  };
  explainability?: { method?: string; interpretation?: string };
}

export interface ShortsReport {
  population?: { test_rows?: number; test_positives?: number; test_channels?: number; train_rows?: number };
  overall?: {
    auc_macro_channel?: number;
    lift_top10?: number;
    precision_top10?: number;
    recall_top10?: number;
    reference?: { auc_macro_channel?: number };
    ci?: Record<string, [number, number]>;
  };
  sectors?: {
    sector: Sector;
    chosen: string;
    channels: number;
    positives: number;
    auc_macro_channel: number;
    lift_top10: number;
    ci: Record<string, [number, number]>;
  }[];
}

export interface ViralMetrics {
  auc_macro_channel?: number;
  reference_auc?: number;
  lift_top10?: number;
  reference_lift_top10?: number;
  precision_top10?: number;
  recall_top10?: number;
  age_rho?: number;
  channels_with_pairs?: number;
  ci?: Record<string, [number, number]>;
}

/** Ranking na base v2: rótulo de outlier (3x a mediana dos Shorts anteriores do canal). */
export interface ViralReport {
  chosen?: string;
  promoted?: boolean;
  promotion_rule?: string;
  population?: { train_rows?: number; test_rows?: number; test_channels?: number; test_viral_rate?: number };
  overall?: ViralMetrics;
  sectors?: (ViralMetrics & { sector: Sector })[];
}

export interface Artifacts {
  loaded_at: string;
  rolling: Record<string, { report: Loaded<RollingReport>; fold_metrics: LoadedCsv; figures: Figure[] }>;
  battle_model: { run_id: string; report: Loaded<BattleModelReport>; shap_global: LoadedCsv };
  shorts_only: { run_id: string; report: Loaded<ShortsReport>; figures: Figure[] };
  viral_v2?: { run_id: string; report: Loaded<ViralReport>; figures: Figure[] };
}

export interface Channel {
  sector: Sector;
  label: string;
  channel_id: string;
}
