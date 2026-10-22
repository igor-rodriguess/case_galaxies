"use client";

import { CopyId, JsonDetails, Meta, Pill, StatusPill } from "@/components/ui";
import { sideConfidence, verdict } from "@/lib/battle";
import { dateTime, num, pct, points } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { Battle, ModelStatus, Prediction, Ranking, ToolCall, Traces } from "@/lib/types";

function Reasons({ items }: { items?: string[] }) {
  if (!items?.length) return null;
  return (
    <ul className="mt-2 space-y-1 text-sm text-text-2">
      {items.map((item) => (
        <li key={item} className="flex gap-2">
          <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-muted" aria-hidden />
          {item}
        </li>
      ))}
    </ul>
  );
}

function Body({ call }: { call: ToolCall }) {
  const { t, locale } = useT();
  if (call.is_error) {
    return (
      <>
        <Pill tone="danger">{t("MCP rejected the call")}</Pill>
        <pre className="mono mt-3 max-h-60 overflow-auto rounded-xl bg-danger-soft p-3 text-[12px] whitespace-pre-wrap text-danger">{call.error_text}</pre>
      </>
    );
  }
  if (call.tool === "model_status") {
    const result = call.result as ModelStatus;
    return (
      <>
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone="brand">{result.sector}</Pill>
          {result.ready ? <Pill tone="ok">{t("Promoted model")}</Pill> : <Pill tone="warn">model_not_ready</Pill>}
          {result.model_name && <span className="mono text-text-2">{result.model_name}</span>}
        </div>
        <Reasons items={result.blocking_reasons} />
      </>
    );
  }
  if (call.tool === "predict_performance") {
    const result = call.result as Prediction;
    return (
      <>
        <div className="flex flex-wrap gap-2">
          <StatusPill status={result.status} />
          <Pill tone={result.confidence_flag === "normal" ? "ok" : "warn"}>
            {t("confidence")}: {result.confidence_flag}
          </Pill>
        </div>
        {result.status === "ok" && (
          <p className="mt-3">
            <span className="display text-3xl">{num(result.predicted_metric, 0, locale)}</span>{" "}
            <span className="text-sm text-muted">
              {result.metric_name} · {num(result.interval_low, 0, locale)}–{num(result.interval_high, 0, locale)}
            </span>
          </p>
        )}
        <p className="mt-2 text-sm text-text-2">{result.explanation}</p>
        <Reasons items={result.warnings} />
      </>
    );
  }
  if (call.tool === "rank_creatives") {
    const result = call.result as Ranking;
    return (
      <>
        <StatusPill status={result.status} />
        {result.items?.length ? (
          <ol className="mt-2 space-y-1 text-sm">
            {result.items.map((item) => (
              <li key={item.request_index}>
                #{item.rank} · {t("creative")} {item.request_index + 1} ·{" "}
                {item.prediction.status === "ok" ? num(item.prediction.predicted_metric, 0, locale) : item.prediction.status}
              </li>
            ))}
          </ol>
        ) : (
          <p className="mt-2 text-sm text-muted">{t("No creative was ranked.")}</p>
        )}
        <Reasons items={result.warnings} />
      </>
    );
  }
  if (call.tool === "creative_battle") {
    const result = call.result as Battle;
    const decision = verdict(result);
    const a = sideConfidence(result.baseline);
    const b = sideConfidence(result.candidate);
    return (
      <>
        <p className="font-semibold">{t(decision.title)}</p>
        {result.status === "ok" && (
          <p className="mono mt-1 text-text-2">
            P(top 10%) A {pct(result.baseline?.viral_probability_calibrated, 2, locale)} → B{" "}
            {pct(result.candidate?.viral_probability_calibrated, 2, locale)} ({points(result.delta?.viral_probability_calibrated, 2, locale)})
          </p>
        )}
        <p className="mt-1 text-xs text-muted">
          {t("Confidence")} A: {a.level} · B: {b.level}
        </p>
        <Reasons items={result.warnings} />
      </>
    );
  }
  const result = call.result as Traces;
  return (
    <>
      <p className="text-sm">
        {result.events?.length ?? 0} {t("calls recorded")}
      </p>
      <ul className="mt-2 space-y-1 text-xs text-muted">
        {result.events?.slice(0, 6).map((event) => (
          <li key={event.trace_id} className="mono">
            {dateTime(event.recorded_at, locale)} · {event.tool_name}
          </li>
        ))}
      </ul>
    </>
  );
}

export function ToolCallCard({ call }: { call: ToolCall }) {
  const { t, locale } = useT();
  const runId = (call.result as { model_run_id?: string; run_id?: string } | null)?.model_run_id ??
    (call.result as { run_id?: string } | null)?.run_id;
  return (
    <article className="card overflow-hidden" aria-label={`${t("Response from")} ${call.tool}`}>
      <div className="flex items-center justify-between gap-3 border-b border-border bg-brand-soft/60 px-5 py-3">
        <span className="mono font-semibold text-brand-ink">{call.tool}</span>
        <span className="text-xs text-muted">{t("via MCP")} · {num(call.duration_ms, 0, locale)} ms</span>
      </div>
      <div className="p-5">
        <Body call={call} />
        <Meta
          items={[
            ["time", dateTime(call.started_at, locale)],
            ["transport", call.transport],
            ["model_run_id", runId ?? "—"],
            ["MCP trace", call.trace ? <CopyId value={call.trace.trace_id} /> : t("not recorded")],
          ]}
        />
        <JsonDetails summary="Arguments sent" value={call.arguments} />
        {call.result !== null && <JsonDetails summary="Full response" value={call.result} />}
        {call.trace && <JsonDetails summary="MCP trace" value={call.trace} />}
      </div>
    </article>
  );
}
