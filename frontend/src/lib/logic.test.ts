import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { BATTLE_FIELDS, CATEGORIES, battleArguments, editedFields, toFeatures, verdict } from "./battle";
import { parseCommand, TOOLS } from "./commands";
import { cellNumber, ci, num, pct, points, signed, MISSING } from "./format";
import { PT } from "./i18n";
import { foldSeries, promotionSummary, rollingRows, shapRows, TIER_TEXT, tierFor } from "./metrics";
import type { Battle, RollingReport } from "./types";

describe("formatação nunca inventa valor", () => {
  it("ausente vira traço e CSV vazio não vira zero", () => {
    expect(num(null)).toBe(MISSING);
    expect(num(Number.NaN)).toBe(MISSING);
    expect(pct(undefined)).toBe(MISSING);
    expect(points("0.1")).toBe(MISSING);
    expect(ci([0.4, null])).toBe("");
    expect(cellNumber("")).toBeNull();
    expect(cellNumber("  ")).toBeNull();
    expect(cellNumber("0.25")).toBe(0.25);
    expect(signed(-0.0123, 3)).toBe("−0,012");
    expect(points(0.0123)).toBe("+1,23 p.p.");
  });
});

describe("Creative Battle", () => {
  const ok: Battle = {
    status: "ok",
    sector: "moda",
    channel_id: "c",
    comparison_scope: "",
    delta: { viral_probability_calibrated: 0.02 },
    confidence_flag: "normal",
  };
  it("B só vence com delta positivo e confiança não baixa", () => {
    expect(verdict(ok).kind).toBe("candidate");
    expect(verdict({ ...ok, confidence_flag: "low" }).kind).toBe("inconclusive");
    expect(verdict({ ...ok, delta: { viral_probability_calibrated: -0.01 } }).kind).toBe("baseline");
    expect(verdict({ ...ok, delta: { viral_probability_calibrated: 0 } }).kind).toBe("tie");
    expect(verdict({ ...ok, status: "model_not_ready" }).kind).toBe("unavailable");
    expect(verdict({ ...ok, status: "invalid_input", warnings: ["canal"] }).kind).toBe("error");
    expect(verdict({ ...ok, delta: {} }).kind).toBe("unavailable");
    expect(verdict(null).kind).toBe("empty");
  });
  it("payload leva só escalares e mantém setor e canal comuns", () => {
    const args = battleArguments({
      sector: "moda",
      channelId: "UC1",
      baseline: { title: "A", duration_seconds: "45", tag_count: "", published_at: "2026-10-23T15:00" },
      candidate: { title: "B", duration_seconds: "abc", category_id: "26" },
    });
    expect(args.channel_id).toBe("UC1");
    expect(args.baseline.features).toEqual({ title: "A", duration_seconds: 45, published_at: "2026-10-23T15:00:00Z" });
    expect(args.candidate.features).toEqual({ title: "B", category_id: "26" });
    for (const value of Object.values(toFeatures({ title: "x", tag_count: "2" }))) {
      expect(["string", "number"]).toContain(typeof value);
    }
  });
  it("campos editados comparam A e B", () => {
    expect(editedFields({ title: "a", duration_seconds: "10" }, { title: "a", duration_seconds: "12" })).toEqual(["duration_seconds"]);
  });
});

describe("console", () => {
  it("aceita só as tools da interface", () => {
    expect(parseCommand("status moda")).toEqual({ calls: [{ tool: "model_status", arguments: { sector: "moda" } }] });
    const all = parseCommand("status");
    expect("calls" in all && all.calls.length).toBe(3);
    const traces = parseCommand("traces 500");
    expect("calls" in traces && traces.calls[0].arguments).toEqual({ limit: 100 });
    expect(parseCommand("/get_video_feature_map {}")).toHaveProperty("error");
    expect(parseCommand("/model_status {x")).toEqual({ error: "Arguments must be valid JSON." });
    expect(parseCommand("vai viralizar?")).toEqual({ help: true });
    expect(TOOLS).toEqual(["model_status", "predict_performance", "rank_creatives", "creative_battle", "recent_mcp_calls"]);
  });
});

describe("métricas", () => {
  it("promovido só via model_status; referência vem do relatório", () => {
    expect(tierFor("x", { promotedNames: ["x"] })).toBe("promoted");
    expect(tierFor("logistic_pooled_top10", { referenceNames: ["logistic_pooled_top10"] })).toBe("reference");
    expect(tierFor("catboost", { referenceNames: ["logistic_pooled_top10"] })).toBe("exploratory");
    expect(promotionSummary([{ sector: "moda", ready: false }])).toEqual({ state: "none", promoted: [] });
    expect(promotionSummary([])).toEqual({ state: "unknown", promoted: [] });
  });
  it("relatório rolling real preserva os números do JSON", () => {
    const path = join(process.cwd(), "..", "data", "modeling", "runs", "known-channel-rolling-fold-aware-20261021", "report.json");
    let report: RollingReport;
    try {
      report = JSON.parse(readFileSync(path, "utf8"));
    } catch {
      return; // artefato ausente nesta máquina
    }
    const rows = rollingRows(report);
    expect(rows).toHaveLength(report.results.length);
    const reference = rows.find((row) => row.name === report.reference);
    expect(reference?.tier).toBe("reference");
    expect(reference?.metrics?.aucIntra).toBe(report.results[0].metrics.auc_intra);
    expect(rows.filter((row) => row.name !== report.reference).every((row) => row.tier === "exploratory")).toBe(true);
  });
  it("ensemble: logística é referência, blends são exploratórios", () => {
    const rows = rollingRows({
      source_run: "base",
      reference: { logistic: { auc_intra: 0.6 }, hgb_sector: { auc_intra: 0.68 } },
      results: [{ name: "blend", metrics: { auc_intra: 0.69 }, paired_vs_logistic: { diff: 0.05, ci_low: 0.01, ci_high: 0.1 } }],
    });
    expect(rows.map((row) => row.tier)).toEqual(["reference", "exploratory", "exploratory"]);
    expect(rows[2].paired?.diff).toBe(0.05);
  });
  it("janelas e SHAP ignoram células vazias", () => {
    const series = foldSeries([
      { fold: "2", model: "m", auc_intra: "0.6", rows: "10" },
      { fold: "1", model: "m", auc_intra: "0.5", rows: "12" },
      { fold: "3", model: "m", auc_intra: "", rows: "0" },
    ]);
    expect(series.get("m")?.map((point) => point.fold)).toEqual([1, 2]);
    expect(shapRows([{ feature: "a", mean_abs_shap: "" }, { feature: "b", mean_abs_shap: "0.3" }])).toEqual([{ feature: "b", value: 0.3 }]);
  });
});

function sourceFiles(directory: string): string[] {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(name) && !name.endsWith(".test.ts") ? [path] : [];
  });
}

describe("i18n", () => {
  it("toda chave visível tem tradução PT", () => {
    const keys = new Set<string>();
    for (const file of sourceFiles(join(process.cwd(), "src"))) {
      for (const match of readFileSync(file, "utf8").matchAll(/\bt\("((?:[^"\\]|\\.)*)"\)/g)) keys.add(match[1]);
    }
    for (const field of BATTLE_FIELDS) {
      keys.add(field.label);
      if (field.hint) keys.add(field.hint);
    }
    for (const category of CATEGORIES) keys.add(category.label);
    for (const tier of Object.values(TIER_TEXT)) {
      keys.add(tier.label);
      keys.add(tier.hint);
    }
    const missing = [...keys].filter((key) => !(key in PT));
    expect(missing).toEqual([]);
  });
});

describe("trilha MCP", () => {
  it("agrupa chamadas iguais em sequência e preserva as diferentes", async () => {
    const { groupEvents } = await import("./traces");
    const event = (id: string, tool: string, output: Record<string, unknown>, sector?: string) => ({
      trace_id: id,
      recorded_at: "2026-10-21T10:00:00Z",
      tool_name: tool,
      input_summary: sector ? { sector } : {},
      output_summary: output,
    });
    const groups = groupEvents([
      event("1", "model_status", { ready: false }, "moda"),
      event("2", "model_status", { ready: false }, "beleza"),
      event("3", "creative_battle", { status: "ok" }),
      event("4", "model_status", { ready: false }, "moda"),
    ]);
    expect(groups.map((group) => [group.tool, group.count])).toEqual([
      ["model_status", 2],
      ["creative_battle", 1],
      ["model_status", 1],
    ]);
    expect(groups[0].sectors).toEqual(["moda", "beleza"]);
    expect(groups[0].label).toBe("model_not_ready");
  });
});
