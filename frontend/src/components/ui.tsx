"use client";

import { useState } from "react";

import { useT } from "@/lib/i18n";
import { TIER_TEXT, type Tier } from "@/lib/metrics";

export function PageHeader({
  eyebrow,
  title,
  lede,
  right,
}: {
  eyebrow: string;
  title: string;
  lede?: string;
  right?: React.ReactNode;
}) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="display mt-3 text-balance text-[clamp(2rem,4.2vw,3.25rem)]">{title}</h1>
        {lede && <p className="mt-3 text-[15.5px] leading-relaxed text-text-2">{lede}</p>}
      </div>
      {right}
    </header>
  );
}

export function SectionTitle({ title, hint, right }: { title: string; hint?: string; right?: React.ReactNode }) {
  return (
    <div className="mb-4 mt-12 flex flex-wrap items-end justify-between gap-3 border-t border-border pt-8 first:mt-0 first:border-0 first:pt-0">
      <div>
        <h2 className="text-xl font-bold tracking-tight">{title}</h2>
        {hint && <p className="mt-1 text-sm text-muted">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

export type Tone = "brand" | "ok" | "warn" | "danger" | "magenta" | "ghost" | "neutral";

export function Pill({ tone = "neutral", children, title }: { tone?: Tone; children: React.ReactNode; title?: string }) {
  return (
    <span className={`pill ${tone === "neutral" ? "" : `pill-${tone}`}`} title={title}>
      {children}
    </span>
  );
}

export function TierPill({ tier }: { tier: Tier }) {
  const { t } = useT();
  const tone: Record<Tier, Tone> = { promoted: "ok", reference: "brand", exploratory: "magenta" };
  return (
    <Pill tone={tone[tier]} title={t(TIER_TEXT[tier].hint)}>
      {t(TIER_TEXT[tier].label)}
    </Pill>
  );
}

export function StatusPill({ status }: { status: string | undefined | null }) {
  const tone: Record<string, Tone> = {
    ok: "ok",
    model_not_ready: "warn",
    invalid_input: "danger",
    partial: "warn",
    data_not_ready: "warn",
  };
  return <Pill tone={tone[status ?? ""] ?? "neutral"}>{status ?? "—"}</Pill>;
}

export function Stat({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
  tone?: "brand" | "warn" | "ok";
}) {
  const color = tone === "brand" ? "text-brand" : tone === "warn" ? "text-warn" : tone === "ok" ? "text-ok" : "";
  return (
    <div className="card p-5">
      <div className="label">{label}</div>
      <div className={`display mt-2 text-3xl tabular-nums ${color}`}>{value}</div>
      {detail && <div className="mt-2 text-xs leading-relaxed text-muted">{detail}</div>}
    </div>
  );
}

export function Loading({ label, rows = 3 }: { label?: string; rows?: number }) {
  const { t } = useT();
  return (
    <div className="card space-y-3 p-5" role="status" aria-live="polite">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="skeleton h-3" style={{ width: `${90 - index * 18}%` }} />
      ))}
      <span className="sr-only">{label ?? t("Loading")}</span>
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  const { t } = useT();
  return (
    <div role="alert" className="rounded-2xl border border-danger/40 bg-danger-soft p-5 text-sm">
      <p className="font-semibold text-danger">{t("Something failed")}</p>
      <p className="mt-1 text-text-2">{t(message)}</p>
      {onRetry && (
        <button type="button" className="btn mt-3" onClick={onRetry}>
          {t("Try again")}
        </button>
      )}
    </div>
  );
}

export function Empty({ title, detail }: { title: string; detail?: string }) {
  const { t } = useT();
  return (
    <div className="rounded-2xl border border-dashed border-border-strong p-6 text-center">
      <p className="font-semibold">{t(title)}</p>
      {detail && <p className="mx-auto mt-1 max-w-md text-sm text-muted">{t(detail)}</p>}
    </div>
  );
}

export function Unavailable({ label = "unavailable", detail }: { label?: string; detail?: string }) {
  const { t } = useT();
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-dashed border-border-strong p-4 text-sm">
      <Pill tone="ghost">{t(label)}</Pill>
      {detail && <span className="text-muted">{t(detail)}</span>}
    </div>
  );
}

export function JsonDetails({ summary, value }: { summary: string; value: unknown }) {
  const { t } = useT();
  return (
    <details className="group mt-2">
      <summary className="cursor-pointer select-none text-xs font-medium text-muted hover:text-text">{t(summary)}</summary>
      <pre
        tabIndex={0}
        className="mono mt-2 max-h-80 overflow-auto rounded-xl border border-border bg-surface-2 p-3 text-[12px] leading-relaxed whitespace-pre-wrap break-words text-text-2"
      >
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

export function Meta({ items }: { items: [string, React.ReactNode][] }) {
  const { t } = useT();
  return (
    <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 border-t border-dashed border-border pt-4 text-xs sm:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="min-w-0">
          <dt className="label">{t(label)}</dt>
          <dd className="mono mt-1 truncate text-text-2">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function CopyId({ value }: { value: string | null | undefined }) {
  const [copied, setCopied] = useState(false);
  if (!value) return <span className="text-muted">—</span>;
  return (
    <button
      type="button"
      title={value}
      className="mono inline-flex max-w-full items-center gap-1 truncate text-text-2 hover:text-brand"
      onClick={() => {
        navigator.clipboard?.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
    >
      {value.slice(0, 8)}
      <span aria-hidden>{copied ? "✓" : "⧉"}</span>
    </button>
  );
}
