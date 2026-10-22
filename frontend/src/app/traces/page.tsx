"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { outcomeLabel, outcomeTone } from "@/components/McpRail";
import { CopyId, Empty, ErrorBox, Loading, PageHeader, Pill } from "@/components/ui";
import { TOOLS } from "@/lib/commands";
import { useData } from "@/lib/data";
import { dateTime } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { TraceEvent, Traces } from "@/lib/types";

function summary(values: Record<string, unknown> | undefined): string {
  return Object.entries(values ?? {})
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`)
    .join(" · ");
}

export default function TracesPage() {
  const { t, locale } = useT();
  const { call } = useData();
  const [events, setEvents] = useState<TraceEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const load = useCallback(async () => {
    setError(null);
    try {
      const response = await call<Traces>("recent_mcp_calls", { limit: 100 });
      if (response.is_error) throw new Error(response.error_text ?? "MCP error");
      setEvents(response.result?.events ?? []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [call]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- primeira leitura dos traces
    load();
  }, [load]);

  const visible = useMemo(() => (events ?? []).filter((event) => !filter || event.tool_name === filter), [events, filter]);
  const counts = useMemo(() => {
    const map = new Map<string, number>();
    for (const event of events ?? []) map.set(event.tool_name, (map.get(event.tool_name) ?? 0) + 1);
    return map;
  }, [events]);

  return (
    <div>
      <PageHeader
        eyebrow={`04 · ${t("Audit")}`}
        title={t("MCP traces")}
        lede={t("The server records only sector, channel, counts, status, winner and confidence. Creative values never enter the log.")}
        right={
          <button type="button" className="btn" onClick={load}>
            {t("Reload")}
          </button>
        }
      />
      <div className="mb-5 flex flex-wrap gap-2" role="group" aria-label={t("Filter by tool")}>
        {["", ...TOOLS.filter((tool) => tool !== "recent_mcp_calls")].map((tool) => (
          <button
            key={tool || "all"}
            type="button"
            aria-pressed={filter === tool}
            onClick={() => setFilter(tool)}
            className={`mono rounded-xl px-3 py-1.5 text-xs font-semibold transition ${filter === tool ? "bg-brand text-white" : "border border-border bg-surface text-text-2 hover:border-brand-line"}`}
          >
            {tool || t("all tools")}
            {tool && counts.get(tool) ? <span className="ml-1.5 opacity-70">{counts.get(tool)}</span> : null}
          </button>
        ))}
      </div>
      {error && <ErrorBox message={error} onRetry={load} />}
      {!events && !error && <Loading rows={6} />}
      {events && visible.length === 0 && <Empty title="No calls for this filter" />}
      {visible.length > 0 && (
        <div className="card overflow-x-auto">
          <table className="table min-w-[860px]">
            <thead>
              <tr>
                <th scope="col">{t("Time")}</th>
                <th scope="col">Tool</th>
                <th scope="col">{t("Input (sanitized)")}</th>
                <th scope="col">{t("Output")}</th>
                <th scope="col">Trace</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((event) => (
                <tr key={event.trace_id}>
                  <td className="num text-xs">{dateTime(event.recorded_at, locale)}</td>
                  <th scope="row" className="mono text-xs font-semibold">{event.tool_name}</th>
                  <td className="text-xs text-text-2">{summary(event.input_summary) || "—"}</td>
                  <td className="text-xs">
                    <Pill tone={outcomeTone(event.output_summary) === "neutral" ? "neutral" : outcomeTone(event.output_summary)}>{outcomeLabel(event.output_summary)}</Pill>
                    <p className="mt-1.5 text-text-2">{summary(event.output_summary)}</p>
                  </td>
                  <td><CopyId value={event.trace_id} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
