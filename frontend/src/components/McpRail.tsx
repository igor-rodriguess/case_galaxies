"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { IconArrow, IconClose, IconTrace } from "@/components/icons";
import { api } from "@/lib/api";
import { useData } from "@/lib/data";
import { dateTime, shortId } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { groupEvents } from "@/lib/traces";
import type { TraceEvent, Traces } from "@/lib/types";

export { outcomeLabel, outcomeTone } from "@/lib/traces";

const DOT: Record<string, string> = { ok: "bg-ok", warn: "bg-warn", danger: "bg-danger", neutral: "bg-muted" };

function useTrail(limit: number, active: boolean) {
  const { pulse, bridge } = useData();
  const [events, setEvents] = useState<TraceEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      const response = await api.call<Traces>("recent_mcp_calls", { limit });
      if (response.is_error) throw new Error(response.error_text ?? "MCP error");
      setEvents(response.result?.events ?? []);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [limit]);
  useEffect(() => {
    if (!active || bridge !== "ok") return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- a trilha relê quando chega uma chamada nova
    refresh();
    const timer = setInterval(() => document.visibilityState === "visible" && refresh(), 15000);
    return () => clearInterval(timer);
  }, [active, bridge, pulse, refresh]);
  return { events, error, bridge };
}

/** Lista da trilha: agrupa chamadas iguais seguidas. */
export function TrailList({ limit = 30, active = true }: { limit?: number; active?: boolean }) {
  const { t, locale } = useT();
  const { events, error, bridge } = useTrail(limit, active);
  if (bridge === "down") return <p className="p-4 text-sm text-danger">{t("Bridge offline")}</p>;
  if (error) return <p className="p-4 text-sm text-danger">{error}</p>;
  if (!events)
    return (
      <div className="space-y-4 p-4">
        {Array.from({ length: 4 }, (_, index) => (
          <div key={index}>
            <div className="skeleton h-3 w-2/3" />
            <div className="skeleton mt-2 h-2.5 w-1/2" />
          </div>
        ))}
      </div>
    );
  if (!events.length) return <p className="p-4 text-sm text-muted">{t("No calls recorded yet.")}</p>;
  return (
    <ol className="divide-y divide-border" aria-live="polite">
      {groupEvents(events).map((group) => (
        <li key={group.key} className="flex gap-3 px-4 py-3">
          <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${DOT[group.tone]}`} aria-hidden />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-2">
              <span className="mono truncate font-semibold text-text">{group.tool}</span>
              {group.count > 1 && <span className="mono shrink-0 rounded-md bg-surface-3 px-1.5 text-[11px] text-text-2">×{group.count}</span>}
            </div>
            <p className="mt-0.5 truncate text-xs text-muted">
              {group.label}
              {group.sectors.length ? ` · ${group.sectors.join(", ")}` : ""}
            </p>
            <p className="mono mt-0.5 truncate text-[11px] text-muted" title={group.latest.trace_id}>
              {dateTime(group.latest.recorded_at, locale)} · {shortId(group.latest.trace_id)}
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}

/** Botão do cabeçalho/barra lateral que abre a gaveta da trilha. */
export function McpButton({ compact = false }: { compact?: boolean }) {
  const { t } = useT();
  const { bridge, setTrailOpen, trailOpen } = useData();
  return (
    <button
      type="button"
      onClick={() => setTrailOpen(!trailOpen)}
      aria-expanded={trailOpen}
      aria-controls="mcp-drawer"
      aria-label={t("MCP live")}
      className={`inline-flex items-center gap-2 rounded-xl border border-brand-line bg-brand-soft font-semibold text-brand-ink transition hover:border-brand ${compact ? "px-2.5 py-2" : "w-full px-3 py-2.5 text-sm"}`}
    >
      <span className={`live-dot h-2 w-2 shrink-0 rounded-full ${bridge === "ok" ? "bg-ok text-ok" : bridge === "down" ? "bg-danger text-danger" : "bg-muted text-muted"}`} />
      <IconTrace className="h-4 w-4" />
      {!compact && <span className="flex-1 text-left">{t("MCP live")}</span>}
    </button>
  );
}

/** Gaveta lateral com a trilha de chamadas. */
export function McpDrawer() {
  const { t } = useT();
  const { trailOpen, setTrailOpen } = useData();
  useEffect(() => {
    if (!trailOpen) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setTrailOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [trailOpen, setTrailOpen]);
  if (!trailOpen) return null;
  return (
    <div className="fixed inset-0 z-40">
      <button type="button" aria-label={t("Close")} className="absolute inset-0 bg-[rgba(15,23,41,0.35)] backdrop-blur-[2px]" onClick={() => setTrailOpen(false)} />
      <aside
        id="mcp-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={t("Live MCP calls")}
        className="absolute inset-y-0 right-0 flex w-full max-w-sm flex-col border-l border-border bg-surface shadow-[var(--shadow-lg)]"
      >
        <div className="flex items-start justify-between gap-3 bg-gradient-to-br from-brand to-brand-2 px-5 py-5 text-white">
          <div>
            <p className="font-mono text-[11px] font-semibold tracking-[0.14em] uppercase opacity-85">{t("MCP live")}</p>
            <p className="mt-1 text-lg font-bold">{t("Call trail")}</p>
            <p className="mt-1 text-xs opacity-85">{t("Same consecutive calls are grouped.")}</p>
          </div>
          <button type="button" onClick={() => setTrailOpen(false)} className="rounded-lg p-1.5 hover:bg-white/15" aria-label={t("Close")}>
            <IconClose />
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          <TrailList limit={40} active={trailOpen} />
        </div>
        <Link
          href="/traces"
          onClick={() => setTrailOpen(false)}
          className="flex items-center justify-between border-t border-border px-5 py-3.5 text-sm font-semibold text-brand hover:bg-brand-soft"
        >
          {t("Full audit")} <IconArrow className="h-4 w-4" />
        </Link>
      </aside>
    </div>
  );
}
