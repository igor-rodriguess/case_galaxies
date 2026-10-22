"use client";

import Image from "next/image";
import { useState } from "react";

import { ForestPlot, HBars } from "@/components/charts";
import { ErrorBox, Loading, PageHeader, Pill, TierPill, Unavailable } from "@/components/ui";
import { useData } from "@/lib/data";
import { ci, humanize, isNumber, num, pct, signed, signedCi } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { foldSeries, rollingRows, shapRows, TIER_TEXT, type Tier } from "@/lib/metrics";
import type { Artifacts, Figure } from "@/lib/types";

const TABS = [
  { id: "shorts", label: "Shorts · current" },
  { id: "backtests", label: "Ranking backtests" },
  { id: "reference", label: "Reference model" },
  { id: "status", label: "Status per sector" },
] as const;
type TabId = (typeof TABS)[number]["id"];

const RUN_LABEL: Record<string, string> = {
  "known-channel-rolling-fold-aware-20261021": "Rolling fold-aware",
  "known-channel-rolling-expanded-20261021": "Rolling expanded",
  "known-channel-rolling-ensemble-20261021": "OOF ensembles",
};

function Figures({ figures }: { figures: Figure[] }) {
  const { t } = useT();
  if (!figures.length) return null;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {figures.map((figure) => (
        <a key={figure.url} href={figure.url} target="_blank" rel="noopener" className="card group block overflow-hidden p-3 transition hover:border-brand-line">
          <Image src={figure.url} alt={`${t("Figure")} ${humanize(figure.file.replace(".png", ""))}`} width={1200} height={800} unoptimized className="h-auto w-full rounded-lg bg-white" />
          <p className="mono mt-2 flex justify-between gap-2 text-[11px] text-muted group-hover:text-brand">
            <span className="truncate">{figure.file}</span>
            <span className="shrink-0">{t("Open full size")} ↗</span>
          </p>
        </a>
      ))}
    </div>
  );
}

function ShortsTab({ artifacts }: { artifacts: Artifacts }) {
  const { t, locale } = useT();
  const loaded = artifacts.shorts_only.report;
  if (loaded.status !== "ok") return <Unavailable detail="Shorts-only evaluation not found." />;
  const report = loaded.data;
  const overall = report.overall;
  return (
    <div className="space-y-6">
      <div className="card border-warn/40 p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="font-bold text-warn">{t("With metadata and cadence, the order of Shorts was at chance level")}</p>
            <p className="mt-1 text-sm text-muted">{t("Nested selection on calendar folds, one external read, Shorts only.")}</p>
          </div>
          <span className="mono text-xs text-muted">{loaded.path}</span>
        </div>
        <dl className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["AUC per channel", num(overall?.auc_macro_channel, 3, locale), `${t("95% CI")} ${ci(overall?.ci?.auc_macro_channel, 3, locale)}`],
            ["Lift@10", num(overall?.lift_top10, 2, locale), `${t("95% CI")} ${ci(overall?.ci?.lift_top10, 2, locale)}`],
            ["Precision@10", pct(overall?.precision_top10, 1, locale), ""],
            ["Test set", report.population?.test_rows?.toLocaleString(locale) ?? "—", `${report.population?.test_positives ?? "—"} ${t("positives")} · ${report.population?.test_channels ?? "—"} ${t("channels")}`],
          ].map(([label, value, detail]) => (
            <div key={label}>
              <dt className="label">{t(label)}</dt>
              <dd className="display mt-1 text-3xl tabular-nums">{value}</dd>
              {detail && <dd className="mono mt-1 text-xs text-muted">{detail}</dd>}
            </div>
          ))}
        </dl>
      </div>
      {report.sectors?.length ? (
        <div className="card overflow-x-auto">
          <table className="table min-w-[720px]">
            <thead>
              <tr>
                <th scope="col">{t("Sector")}</th>
                <th scope="col">{t("Chosen in inner folds")}</th>
                <th scope="col">{t("AUC per channel [95% CI]")}</th>
                <th scope="col">Lift@10 [IC95%]</th>
                <th scope="col">{t("Channels")}</th>
                <th scope="col">{t("Positives")}</th>
              </tr>
            </thead>
            <tbody>
              {report.sectors.map((sector) => (
                <tr key={sector.sector}>
                  <th scope="row" className="font-semibold capitalize">{sector.sector}</th>
                  <td className="mono text-xs">{sector.chosen}</td>
                  <td className="num">
                    {num(sector.auc_macro_channel, 3, locale)} <span className="text-muted">{ci(sector.ci?.auc_macro_channel, 3, locale)}</span>
                  </td>
                  <td className="num">
                    {num(sector.lift_top10, 2, locale)} <span className="text-muted">{ci(sector.ci?.lift_top10, 2, locale)}</span>
                  </td>
                  <td className="num">{sector.channels}</td>
                  <td className="num">{sector.positives}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <Figures figures={artifacts.shorts_only.figures} />
    </div>
  );
}

function BacktestsTab({ artifacts }: { artifacts: Artifacts }) {
  const { t, locale } = useT();
  const runIds = Object.keys(artifacts.rolling);
  const [active, setActive] = useState(runIds[0] ?? "");
  const run = artifacts.rolling[active];
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2" role="group" aria-label={t("Evaluation run")}>
          {runIds.map((id) => (
            <button
              key={id}
              type="button"
              aria-pressed={id === active}
              onClick={() => setActive(id)}
              className={`rounded-xl px-3.5 py-1.5 text-sm font-semibold transition ${id === active ? "bg-text text-surface" : "border border-border bg-surface text-text-2 hover:border-brand-line"}`}
            >
              {t(RUN_LABEL[id] ?? id)}
            </button>
          ))}
        </div>
        <Pill tone="warn" title={t("These backtests mix Shorts and long videos.")}>
          {t("base: all formats")}
        </Pill>
      </div>
      {!run || run.report.status !== "ok" ? (
        <Unavailable label="report unavailable" detail={run?.report.path} />
      ) : (
        (() => {
          const report = run.report.data;
          const rows = rollingRows(report);
          const series = run.fold_metrics.status === "ok" ? foldSeries(run.fold_metrics.rows) : null;
          const folds = series ? [...new Set([...series.values()].flat().map((point) => point.fold))].sort((x, y) => x - y) : [];
          return (
            <>
              {report.protocol && <p className="text-sm leading-relaxed text-muted">{report.protocol}</p>}
              <ForestPlot rows={rows} />
              <div className="card overflow-x-auto">
                <table className="table min-w-[980px]">
                  <thead>
                    <tr>
                      <th scope="col">{t("Model")}</th>
                      <th scope="col">{t("AUC within channel [95% CI]")}</th>
                      <th scope="col">Lift@10 [IC95%]</th>
                      <th scope="col">Precision@10</th>
                      <th scope="col">{t("Δ AUC vs reference [95% CI]")}</th>
                      <th scope="col">{t("Windows above ref.")}</th>
                      <th scope="col">{t("ρ age")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => {
                      const m = row.metrics;
                      return (
                        <tr key={row.name}>
                          <th scope="row">
                            <span className="mono block text-xs font-semibold">{row.name}</span>
                            <span className="mt-1.5 block">
                              <TierPill tier={row.tier} />
                            </span>
                          </th>
                          <td className="num">
                            <strong>{num(m?.aucIntra, 3, locale)}</strong> <span className="text-muted">{ci(m?.aucIntraCi, 3, locale)}</span>
                          </td>
                          <td className="num">
                            {num(m?.lift10, 2, locale)} <span className="text-muted">{ci(m?.lift10Ci, 2, locale)}</span>
                          </td>
                          <td className="num">{pct(m?.precision10, 1, locale)}</td>
                          <td className="num">
                            {row.paired ? (
                              <>
                                <span className={row.paired.ci && row.paired.ci[0] > 0 ? "font-semibold text-ok" : ""}>{signed(row.paired.diff, 3, locale)}</span>{" "}
                                <span className="text-muted">{signedCi(row.paired.ci, 3, locale)}</span>
                              </>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="num">{row.stability ? `${row.stability.folds_better_than_reference}/${row.stability.folds}` : "—"}</td>
                          <td className="num">{num(m?.ageRho, 3, locale)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {series && (
                <div className="card overflow-x-auto p-5">
                  <p className="mb-1 font-bold">{t("Temporal stability: within-channel AUC per window")}</p>
                  <p className="mb-4 text-xs text-muted">{t("Values below 0.5 (orange) are worse than chance in that window.")}</p>
                  <table className="table min-w-[560px]">
                    <thead>
                      <tr>
                        <th scope="col">{t("Model")}</th>
                        {folds.map((fold) => (
                          <th key={fold} scope="col">
                            {t("Window")} {fold}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...series.entries()].map(([model, points]) => (
                        <tr key={model}>
                          <th scope="row" className="mono text-xs font-semibold">{model}</th>
                          {folds.map((fold) => {
                            const point = points.find((item) => item.fold === fold);
                            return (
                              <td key={fold} className={`num ${point && point.auc < 0.5 ? "font-semibold text-warn" : ""}`}>
                                {point ? num(point.auc, 3, locale) : "—"}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <Figures figures={run.figures} />
            </>
          );
        })()
      )}
    </div>
  );
}

function ReferenceTab({ artifacts }: { artifacts: Artifacts }) {
  const { t, locale } = useT();
  const model = artifacts.battle_model;
  const report = model.report.status === "ok" ? model.report.data : null;
  const evaluation = report?.evaluation;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <TierPill tier="reference" />
        <Pill tone="ghost">{t("not promoted")}</Pill>
        <span className="mono text-xs text-muted">{model.run_id}</span>
      </div>
      {report?.decision && <p className="text-sm text-text-2">{report.decision}</p>}
      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5">
          <p className="font-bold">{t("Global SHAP")}</p>
          <div className="mt-4">
            {model.shap_global.status === "ok" ? (
              <HBars rows={shapRows(model.shap_global.rows).map((row) => ({ label: row.feature, value: row.value }))} />
            ) : (
              <Unavailable label="SHAP unavailable" />
            )}
          </div>
          {report?.explainability?.interpretation && <p className="mt-4 text-xs text-muted">{report.explainability.interpretation}</p>}
        </div>
        <div className="card p-5">
          <p className="font-bold">{t("Probability calibration")}</p>
          {evaluation && isNumber(evaluation.oof_brier_raw) ? (
            <dl className="mt-5 grid grid-cols-2 gap-6">
              <div>
                <dt className="label">{t("OOF Brier, raw")}</dt>
                <dd className="display mt-1 text-3xl tabular-nums">{num(evaluation.oof_brier_raw, 4, locale)}</dd>
              </div>
              <div>
                <dt className="label">{t("OOF Brier, calibrated")}</dt>
                <dd className="display mt-1 text-3xl tabular-nums text-brand">{num(evaluation.oof_brier_cross_calibrated, 4, locale)}</dd>
              </div>
            </dl>
          ) : (
            <div className="mt-4">
              <Unavailable label="calibration unavailable" />
            </div>
          )}
          <p className="mt-5 text-xs text-muted">{t("Lower is better. A reliability curve per bin does not exist in the artifacts and is not shown.")}</p>
        </div>
      </div>
    </div>
  );
}

function StatusTab() {
  const { t } = useT();
  const { statuses, reload } = useData();
  if (statuses.state === "loading") return <Loading rows={3} />;
  if (statuses.state === "error") return <ErrorBox message={statuses.message} onRetry={reload} />;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        {statuses.data.map((status) => (
          <div key={status.sector} className="card p-5">
            <div className="flex items-center justify-between gap-2">
              <p className="text-lg font-bold capitalize">{status.sector}</p>
              {status.ready ? <Pill tone="ok">{t("Promoted model")}</Pill> : <Pill tone="warn">model_not_ready</Pill>}
            </div>
            {status.run_id && <p className="mono mt-2 text-xs text-muted">{status.run_id}</p>}
            <ul className="mt-3 space-y-1.5 text-sm text-text-2">
              {(status.blocking_reasons ?? []).map((reason) => (
                <li key={reason} className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-muted" aria-hidden />
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {(Object.keys(TIER_TEXT) as Tier[]).map((tier) => (
          <div key={tier} className="card-quiet p-4">
            <TierPill tier={tier} />
            <p className="mt-2 text-xs leading-relaxed text-muted">{t(TIER_TEXT[tier].hint)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const { t } = useT();
  const { artifacts, reload } = useData();
  const [tab, setTab] = useState<TabId>("shorts");
  return (
    <div>
      <PageHeader
        eyebrow={`03 · ${t("Models & evidence")}`}
        title={t("What the numbers actually say")}
        lede={t("Everything on this page comes from the JSON/CSV reports and the MCP tools. A missing value shows as unavailable.")}
      />
      <div className="mb-6 flex gap-1 overflow-x-auto border-b border-border" role="tablist" aria-label={t("Sections")}>
        {TABS.map((item) => (
          <button
            key={item.id}
            id={`tab-${item.id}`}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            aria-controls={`panel-${item.id}`}
            onClick={() => setTab(item.id)}
            className={`-mb-px shrink-0 border-b-2 px-4 py-3 text-sm font-semibold transition ${tab === item.id ? "border-brand text-brand" : "border-transparent text-muted hover:text-text"}`}
          >
            {t(item.label)}
          </button>
        ))}
      </div>
      <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {tab === "status" ? (
          <StatusTab />
        ) : artifacts.state === "loading" ? (
          <Loading rows={6} />
        ) : artifacts.state === "error" ? (
          <ErrorBox message={artifacts.message} onRetry={reload} />
        ) : tab === "shorts" ? (
          <ShortsTab artifacts={artifacts.data} />
        ) : tab === "backtests" ? (
          <BacktestsTab artifacts={artifacts.data} />
        ) : (
          <ReferenceTab artifacts={artifacts.data} />
        )}
      </div>
    </div>
  );
}
