"use client";

import { ci, humanize, isNumber, num } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { RollingRow } from "@/lib/metrics";

const DOT = { reference: "var(--brand)", exploratory: "var(--magenta)", promoted: "var(--ok)" } as const;

/** AUC intra-canal com IC95% por modelo; tracejado = acaso, linha azul = referência. */
export function ForestPlot({ rows }: { rows: RollingRow[] }) {
  const { t, locale } = useT();
  const valid = rows.filter((row) => isNumber(row.metrics?.aucIntra));
  if (!valid.length) return null;
  const lows = valid.map((row) => row.metrics?.aucIntraCi?.[0] ?? (row.metrics?.aucIntra as number));
  const highs = valid.map((row) => row.metrics?.aucIntraCi?.[1] ?? (row.metrics?.aucIntra as number));
  const min = Math.min(0.45, ...lows) - 0.02;
  const max = Math.max(...highs) + 0.02;
  const width = 760;
  const left = 250;
  const rowHeight = 34;
  const height = valid.length * rowHeight + 54;
  const x = (value: number) => left + ((value - min) / (max - min)) * (width - left - 24);
  const reference = valid.find((row) => row.tier === "reference");
  const ticks: number[] = [];
  for (let tick = Math.ceil(min * 20) / 20; tick <= max + 1e-9; tick += 0.05) ticks.push(Number(tick.toFixed(2)));
  return (
    <figure className="card p-5">
      <div className="-mx-1 overflow-x-auto px-1">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t("Within-channel AUC with 95% CI per model")} className="h-auto w-full min-w-[620px]">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={18} y2={height - 32} stroke="var(--border)" />
            <text x={x(tick)} y={height - 12} textAnchor="middle" fill="var(--muted)" fontSize={11} fontFamily="var(--font-mono)">
              {num(tick, 2, locale)}
            </text>
          </g>
        ))}
        <line x1={x(0.5)} x2={x(0.5)} y1={10} y2={height - 32} stroke="var(--muted)" strokeDasharray="4 4" />
        <text x={x(0.5) + 6} y={14} fill="var(--muted)" fontSize={11} fontFamily="var(--font-mono)">
          {t("chance")} 0,5
        </text>
        {reference?.metrics?.aucIntra != null && (
          <line x1={x(reference.metrics.aucIntra)} x2={x(reference.metrics.aucIntra)} y1={10} y2={height - 32} stroke="var(--brand)" strokeWidth={1.5} opacity={0.6} />
        )}
        {valid.map((row, index) => {
          const y = 36 + index * rowHeight;
          const auc = row.metrics?.aucIntra as number;
          const [low, high] = row.metrics?.aucIntraCi ?? [auc, auc];
          return (
            <g key={row.name}>
              <text x={left - 14} y={y + 4} textAnchor="end" fill="var(--text-2)" fontSize={12.5}>
                {humanize(row.name).slice(0, 34)}
              </text>
              <line x1={x(low)} x2={x(high)} y1={y} y2={y} stroke="var(--border-strong)" strokeWidth={3} strokeLinecap="round" />
              <circle cx={x(auc)} cy={y} r={6.5} fill={DOT[row.tier]} stroke="var(--surface)" strokeWidth={2.5}>
                <title>{`${row.name}: ${num(auc, 3, locale)} ${ci([low, high], 3, locale)}`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      </div>
      <figcaption className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
        <span>{t("Dot = within-channel AUC · bar = 95% CI (bootstrap over channels)")}</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-full bg-brand" /> {t("Reference model")}</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-full bg-magenta" /> {t("Exploratory candidate")}</span>
      </figcaption>
    </figure>
  );
}

export function HBars({ rows, digits = 3 }: { rows: { label: string; value: number }[]; digits?: number }) {
  const { locale } = useT();
  const max = Math.max(...rows.map((row) => row.value), 0) || 1;
  return (
    <ul className="space-y-2.5">
      {rows.map((row) => (
        <li key={row.label} className="grid grid-cols-[minmax(0,1.3fr)_1fr_auto] items-center gap-3">
          <span className="mono truncate text-xs text-text-2" title={row.label}>
            {row.label}
          </span>
          <span className="h-2 overflow-hidden rounded-full bg-surface-3" aria-hidden>
            <span className="block h-full rounded-full bg-gradient-to-r from-brand to-brand-2" style={{ width: `${(row.value / max) * 100}%` }} />
          </span>
          <span className="mono text-xs tabular-nums text-text-2">{num(row.value, digits, locale)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Contribuições locais (log-odds) em barras divergentes a partir do zero. */
export function Diverging({ items }: { items: { feature: string; log_odds_contribution: number }[] }) {
  const { locale } = useT();
  const max = Math.max(...items.map((item) => Math.abs(item.log_odds_contribution)), 0) || 1;
  return (
    <ul className="space-y-2">
      {items.map((item) => {
        const value = item.log_odds_contribution;
        const width = (Math.abs(value) / max) * 50;
        return (
          <li key={item.feature} className="grid grid-cols-[minmax(0,1.2fr)_1fr_auto] items-center gap-3 text-xs">
            <span className="truncate text-text-2" title={item.feature}>
              {humanize(item.feature)}
            </span>
            <span className="relative h-2 rounded-full bg-surface-3" aria-hidden>
              <span className="absolute inset-y-[-3px] left-1/2 w-px bg-border-strong" />
              <span
                className={`absolute inset-y-0 rounded-full ${value >= 0 ? "bg-brand" : "bg-warn"}`}
                style={value >= 0 ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
              />
            </span>
            <span className="mono tabular-nums text-text-2">{num(value, 3, locale)}</span>
          </li>
        );
      })}
    </ul>
  );
}
