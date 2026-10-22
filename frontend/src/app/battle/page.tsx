"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Diverging } from "@/components/charts";
import { IconArrow, IconShield } from "@/components/icons";
import { CopyId, Empty, ErrorBox, JsonDetails, Loading, Meta, PageHeader, Pill, TierPill, Unavailable } from "@/components/ui";
import { BATTLE_FIELDS, CATEGORIES, battleArguments, editedFields, sideConfidence, verdict, type FieldSpec, type Version } from "@/lib/battle";
import { useData } from "@/lib/data";
import { ci, dateTime, humanize, num, pct, points, signed } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { SECTORS, type Battle, type BattleModelReport, type BattleSide, type Metrics, type Sector, type ToolCall } from "@/lib/types";

const SECTOR_LABEL: Record<Sector, string> = { automotivo: "Automotive", moda: "Fashion", beleza: "Beauty" };

function example(title: string, duration: number): Version {
  return {
    title,
    duration_seconds: String(duration),
    description: "Combinação de peças básicas para o calor, com lista de preços no fim.",
    tag_count: "3",
    category_id: "26",
    published_at: "2026-10-23T15:00",
    days_since_previous_upload: "1",
    prior_upload_interval_median_days: "1.5",
  };
}

const EXAMPLE_A = example("Look de verão com 3 peças básicas", 45);
const EXAMPLE_B = example("3 peças, 1 look", 20);

function Control({ spec, id, value, edited, onChange, label }: { spec: FieldSpec; id: string; value: string; edited: boolean; onChange: (value: string) => void; label: string }) {
  const { t } = useT();
  const className = `input ${edited ? "is-edited" : ""}`;
  if (spec.type === "textarea") {
    return <textarea id={id} aria-label={label} rows={2} className={`${className} resize-y`} value={value} onChange={(event) => onChange(event.target.value)} />;
  }
  if (spec.type === "select") {
    return (
      <select id={id} aria-label={label} className={className} value={value} onChange={(event) => onChange(event.target.value)}>
        {CATEGORIES.map((item) => (
          <option key={item.value} value={item.value}>
            {t(item.label)}
          </option>
        ))}
      </select>
    );
  }
  return <input id={id} aria-label={label} type={spec.type} min={spec.min} step={spec.step} className={className} value={value} onChange={(event) => onChange(event.target.value)} />;
}

function ProbabilityBars({ a, b }: { a?: number; b?: number }) {
  const { t, locale } = useT();
  const max = Math.max(a ?? 0, b ?? 0, 0.0001) * 1.15;
  return (
    <div className="space-y-3">
      {(
        [
          ["A", a, "bg-text-2"],
          ["B", b, "bg-gradient-to-r from-brand to-brand-2"],
        ] as const
      ).map(([label, value, color]) => (
        <div key={label} className="grid grid-cols-[20px_1fr_64px] items-center gap-3">
          <span className="font-bold">{label}</span>
          <span className="h-3 overflow-hidden rounded-full bg-surface-3" aria-hidden>
            <span className={`block h-full rounded-full ${color}`} style={{ width: `${((value ?? 0) / max) * 100}%` }} />
          </span>
          <span className="mono text-right tabular-nums">{pct(value, 2, locale)}</span>
        </div>
      ))}
      <p className="text-xs text-muted">{t("Calibrated probability of landing in the channel's top 10% (historical label).")}</p>
    </div>
  );
}

function SideCard({ label, side, accent }: { label: string; side: BattleSide | null | undefined; accent?: boolean }) {
  const { t, locale } = useT();
  if (!side) return <Unavailable />;
  const confidence = sideConfidence(side);
  return (
    <div className={`card p-5 ${accent ? "border-brand-line" : ""}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="label">{t(label)}</p>
          <p className={`display mt-2 text-4xl tabular-nums ${accent ? "text-brand" : ""}`}>{pct(side.viral_probability_calibrated, 2, locale)}</p>
          <p className="mono mt-1 text-xs text-muted">
            {t("ranking score")} {num(side.ranking_score_raw, 4, locale)}
          </p>
        </div>
        <Pill tone={confidence.level === "low" ? "warn" : "ok"}>
          {t("confidence")} {t(confidence.level)}
        </Pill>
      </div>
      {(confidence.alerts.length > 0 || confidence.missing.length > 0) && (
        <div className="mt-4 rounded-xl bg-warn-soft p-3 text-xs text-text-2">
          <p className="font-semibold text-warn">
            {confidence.alerts.length} {t("confidence alert(s)")}
            {confidence.missing.length ? ` · ${confidence.missing.length} ${t("missing feature(s)")}` : ""}
          </p>
          <ul className="mono mt-1 space-y-0.5">
            {confidence.alerts.map((alert) => (
              <li key={`${alert.feature}-${alert.type}`}>
                {humanize(alert.feature)} · {humanize(alert.type)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {side.top_local_contributions?.length ? (
        <div className="mt-5 border-t border-border pt-4">
          <p className="mb-3 text-xs font-medium text-muted">{t("Largest local contributions (log-odds)")}</p>
          <Diverging items={side.top_local_contributions.slice(0, 5)} />
        </div>
      ) : null}
    </div>
  );
}

function ModelEvidence({ runId, report }: { runId: string; report: BattleModelReport | null }) {
  const { t, locale } = useT();
  const evaluation = report?.evaluation;
  const rows: [string, Metrics | undefined][] = [
    ["New channel (LOCO)", evaluation?.loco],
    ["Temporal cut", evaluation?.temporal],
    ["Vault", evaluation?.vault],
  ];
  return (
    <details className="card group p-5">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2">
        <IconShield className="h-5 w-5 text-brand" />
        <span className="font-bold">{t("Model behind this comparison")}</span>
        <TierPill tier="reference" />
        <Pill tone="ghost">{t("not promoted")}</Pill>
        <span className="ml-auto text-xs font-semibold text-brand group-open:hidden">{t("See evaluation")}</span>
      </summary>
      {!report ? (
        <div className="mt-4">
          <Unavailable label="evaluation unavailable" detail="The frozen model report was not found." />
        </div>
      ) : (
        <div className="mt-4">
          <p className="mono text-xs text-muted">{runId}</p>
          <div className="mt-3 overflow-x-auto">
            <table className="table">
              <caption className="sr-only">{t("Evaluation of the frozen model")}</caption>
              <thead>
                <tr>
                  <th scope="col">{t("Evaluation")}</th>
                  <th scope="col">{t("Within-channel AUC")}</th>
                  <th scope="col">{t("95% CI")}</th>
                  <th scope="col">Lift@10</th>
                  <th scope="col">{t("Channels")}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([label, m]) => (
                  <tr key={label}>
                    <th scope="row" className="font-medium">{t(label)}</th>
                    <td className="num">{num(m?.auc_intra, 3, locale)}</td>
                    <td className="num text-muted">{ci([m?.auc_intra_ci_low, m?.auc_intra_ci_high], 3, locale) || "—"}</td>
                    <td className="num">{num(m?.lift_top10, 2, locale)}</td>
                    <td className="num">{m?.channels ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {report.decision && <p className="mt-3 text-xs text-muted">{report.decision}</p>}
        </div>
      )}
    </details>
  );
}

const VERDICT_STYLE: Record<string, string> = {
  candidate: "border-brand-line bg-[radial-gradient(700px_220px_at_0%_0%,var(--brand-soft),transparent)]",
  inconclusive: "border-warn/40 bg-[radial-gradient(700px_220px_at_0%_0%,var(--warn-soft),transparent)]",
};

function Result({ call }: { call: ToolCall<Battle> }) {
  const { t, locale } = useT();
  if (call.is_error) return <ErrorBox message={`${t("MCP rejected the call")}: ${call.error_text}`} />;
  const response = call.result as Battle;
  const decision = verdict(response);
  if (decision.kind === "error") return <ErrorBox message={decision.detail || decision.title} />;
  if (decision.kind === "unavailable") return <Unavailable label={decision.title} detail={decision.detail} />;
  return (
    <div className="space-y-5">
      <div className={`card p-6 ${VERDICT_STYLE[decision.kind] ?? ""}`}>
        <div className="grid gap-6 md:grid-cols-[1.1fr_1fr] md:items-center">
          <div>
            <p className="eyebrow">{t("Frozen model verdict")}</p>
            <h2 className="display mt-2 text-[clamp(1.5rem,2.6vw,2.1rem)]">{t(decision.title)}</h2>
            {decision.detail && <p className="mt-2 text-sm text-text-2">{t(decision.detail)}</p>}
            <div className="mt-4 flex flex-wrap gap-2">
              <Pill tone="brand">
                Δ {t("probability")} {points(response.delta?.viral_probability_calibrated, 2, locale)}
              </Pill>
              <Pill>
                Δ {t("score")} {signed(response.delta?.ranking_score_raw, 4, locale)}
              </Pill>
            </div>
          </div>
          <ProbabilityBars a={response.baseline?.viral_probability_calibrated} b={response.candidate?.viral_probability_calibrated} />
        </div>
        <p className="mt-5 flex gap-2 border-t border-border pt-4 text-xs text-text-2">
          <IconShield className="h-4 w-4 shrink-0 text-warn" />
          <span>
            <strong className="text-warn">{t("Associative comparison, not causal.")}</strong>{" "}
            {t("Validate the winning version in a controlled test before attributing an effect to the change.")}
          </span>
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <SideCard label="A · current version" side={response.baseline} />
        <SideCard label="B · alternative version" side={response.candidate} accent />
      </div>

      <div className="card p-5">
        <p className="font-bold">{t("Model inputs that changed")}</p>
        {response.changed_model_inputs?.length ? (
          <div className="mt-3 overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Feature</th>
                  <th scope="col">A</th>
                  <th scope="col">B</th>
                </tr>
              </thead>
              <tbody>
                {response.changed_model_inputs.map((item) => (
                  <tr key={item.feature}>
                    <th scope="row" className="mono font-medium">{item.feature}</th>
                    <td className="num">{typeof item.baseline === "number" ? num(item.baseline, 3, locale) : String(item.baseline)}</td>
                    <td className="num font-semibold text-brand">{typeof item.candidate === "number" ? num(item.candidate, 3, locale) : String(item.candidate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="mt-3">
            <Empty title="No feature used by the model changed" detail="Edit fields that enter the model, such as duration or category." />
          </div>
        )}
        <Meta
          items={[
            ["model_run_id", response.model_run_id ?? "—"],
            ["model", response.model_name ?? "—"],
            ["time", dateTime(call.started_at, locale)],
            ["MCP trace", <CopyId key="trace" value={call.trace?.trace_id} />],
          ]}
        />
        <JsonDetails summary="Full response" value={response} />
      </div>
    </div>
  );
}

export default function BattlePage() {
  const { t } = useT();
  const { channels, artifacts, call, reload } = useData();
  const [sector, setSector] = useState<Sector>("moda");
  const [channelId, setChannelId] = useState("");
  const [a, setA] = useState<Version>(EXAMPLE_A);
  const [b, setB] = useState<Version>(EXAMPLE_B);
  const [state, setState] = useState<{ status: "idle" } | { status: "loading" } | { status: "done"; call: ToolCall<Battle> } | { status: "error"; message: string }>({ status: "idle" });
  const resultRef = useRef<HTMLElement>(null);

  const options = useMemo(() => (channels.state === "ok" ? channels.data.filter((item) => item.sector === sector) : []), [channels, sector]);
  const selected = options.some((item) => item.channel_id === channelId) ? channelId : (options[0]?.channel_id ?? "");
  const edited = useMemo(() => new Set(editedFields(a, b)), [a, b]);
  const model = artifacts.state === "ok" ? artifacts.data.battle_model : null;

  useEffect(() => {
    if (state.status === "done" || state.status === "error") resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [state.status]);

  async function compare(event: React.FormEvent) {
    event.preventDefault();
    if (!selected) return;
    setState({ status: "loading" });
    try {
      const response = await call<Battle>("creative_battle", battleArguments({ sector, channelId: selected, baseline: a, candidate: b }));
      setState({ status: "done", call: response });
    } catch (reason) {
      setState({ status: "error", message: reason instanceof Error ? reason.message : String(reason) });
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="02 · Creative Battle"
        title={t("Version A against version B")}
        lede={t("Same channel, same sector, two versions of the creative. The frozen model says which scenario scores higher and how confident it is.")}
      />

      <form onSubmit={compare} className="card overflow-hidden">
        <div className="flex flex-wrap items-end gap-4 border-b border-border bg-surface-2 p-5">
          <fieldset>
            <legend className="label mb-2">{t("Sector")}</legend>
            <div className="flex rounded-xl border border-border bg-surface p-1">
              {SECTORS.map((item) => (
                <button
                  key={item}
                  type="button"
                  aria-pressed={sector === item}
                  onClick={() => setSector(item)}
                  className={`rounded-lg px-3.5 py-1.5 text-sm font-semibold transition ${sector === item ? "bg-brand text-white" : "text-muted hover:text-text"}`}
                >
                  {t(SECTOR_LABEL[item])}
                </button>
              ))}
            </div>
          </fieldset>
          <div className="min-w-[220px] flex-1">
            <label htmlFor="channel" className="label mb-2 block">
              {t("Channel")}
            </label>
            {channels.state === "loading" ? (
              <div className="skeleton h-10" />
            ) : channels.state === "error" ? (
              <ErrorBox message={channels.message} onRetry={reload} />
            ) : (
              <select id="channel" className="input" value={selected} onChange={(event) => setChannelId(event.target.value)}>
                {options.length ? options.map((item) => <option key={item.channel_id} value={item.channel_id}>{item.label}</option>) : <option value="">{t("no channel available")}</option>}
              </select>
            )}
          </div>
          <button type="button" className="btn md:hidden" onClick={() => setB({ ...a })}>
            {t("Copy A into B")}
          </button>
          <p className="w-full text-xs text-muted sm:w-auto sm:max-w-[220px]">{t("A and B always share sector and channel: only the creative changes.")}</p>
        </div>

        <div className="hidden grid-cols-[minmax(170px,0.8fr)_1fr_1fr] gap-4 border-b border-border px-5 py-3 md:grid">
          <span className="label self-end">{t("Field")}</span>
          <span className="flex items-center gap-2 font-bold">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-surface-3 text-xs">A</span>
            {t("Current version")}
          </span>
          <span className="flex items-center justify-between gap-2 font-bold">
            <span className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-brand to-brand-2 text-xs text-white">B</span>
              {t("Alternative version")}
            </span>
            <button type="button" className="text-xs font-semibold text-brand hover:underline" onClick={() => setB({ ...a })}>
              {t("Copy A into B")}
            </button>
          </span>
        </div>

        <div className="divide-y divide-border">
          {BATTLE_FIELDS.map((spec) => {
            const changed = edited.has(spec.key);
            return (
              <div key={spec.key} className={`grid gap-3 px-5 py-4 md:grid-cols-[minmax(170px,0.8fr)_1fr_1fr] md:gap-4 ${changed ? "bg-brand-soft/40" : ""}`}>
                <div>
                  <p className="flex items-center gap-2 text-sm font-semibold">
                    {t(spec.label)}
                    {changed && <span className="mono rounded-md bg-brand px-1.5 py-0.5 text-[10px] font-bold text-white">≠</span>}
                  </p>
                  {spec.hint && <p className="mt-0.5 text-xs text-muted">{t(spec.hint)}</p>}
                </div>
                <div className="grid grid-cols-[18px_1fr] items-start gap-2 md:block">
                  <span className="pt-2.5 text-xs font-bold text-muted md:hidden">A</span>
                  <Control spec={spec} id={`a-${spec.key}`} label={`A · ${t(spec.label)}`} value={a[spec.key]} edited={false} onChange={(value) => setA((current) => ({ ...current, [spec.key]: value }))} />
                </div>
                <div className="grid grid-cols-[18px_1fr] items-start gap-2 md:block">
                  <span className="pt-2.5 text-xs font-bold text-brand md:hidden">B</span>
                  <Control spec={spec} id={`b-${spec.key}`} label={`B · ${t(spec.label)}`} value={b[spec.key]} edited={changed} onChange={(value) => setB((current) => ({ ...current, [spec.key]: value }))} />
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-surface-2 px-5 py-4">
          <p className="text-sm" role="status">
            {edited.size ? (
              <>
                <strong className="text-brand">{edited.size}</strong> {t("field(s) differ between A and B")}
              </>
            ) : (
              <span className="text-warn">{t("A and B are identical: the model will see no difference")}</span>
            )}
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="btn"
              onClick={() => {
                setA(EXAMPLE_A);
                setB(EXAMPLE_B);
                setState({ status: "idle" });
              }}
            >
              {t("Reset example")}
            </button>
            <button type="submit" className="btn btn-primary" disabled={state.status === "loading" || !selected}>
              {state.status === "loading" ? t("Comparing…") : t("Compare A × B in the MCP")} <IconArrow className="h-4 w-4" />
            </button>
          </div>
        </div>
      </form>

      <section ref={resultRef} className="mt-8 scroll-mt-24 space-y-5" aria-live="polite" aria-label={t("Comparison result")}>
        {state.status === "idle" && <Empty title="Build both versions and compare" detail="The result shows delta, confidence, out-of-range alerts and the call trace." />}
        {state.status === "loading" && <Loading rows={5} label={t("Comparing in the MCP")} />}
        {state.status === "error" && <ErrorBox message={state.message} />}
        {state.status === "done" && <Result call={state.call} />}
        {model && <ModelEvidence runId={model.run_id} report={model.report.status === "ok" ? model.report.data : null} />}
      </section>
    </div>
  );
}
