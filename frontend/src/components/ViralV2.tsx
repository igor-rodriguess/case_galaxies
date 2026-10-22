"use client";

import { Pill } from "@/components/ui";
import { ci, num, pct, signed, signedCi } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { ViralMetrics, ViralReport } from "@/lib/types";

/** Diferença contra a referência, calculada só a partir dos dois valores do relatório. */
function delta(metrics: ViralMetrics | undefined): number | null {
  if (metrics?.auc_macro_channel == null || metrics.reference_auc == null) return null;
  return metrics.auc_macro_channel - metrics.reference_auc;
}

/** Números da base v2, lidos do report.json sem arredondar nem completar. */
export function ViralHeadline({ report, path }: { report: ViralReport; path?: string }) {
  const { t, locale } = useT();
  const overall = report.overall;
  const items: [string, string, string][] = [
    ["AUC per channel", num(overall?.auc_macro_channel, 3, locale), `${t("95% CI")} ${ci(overall?.ci?.auc_macro_channel, 3, locale)} · ${t("0.5 is chance")}`],
    ["Gain over reference", signed(delta(overall), 3, locale), `${t("95% CI")} ${signedCi(overall?.ci?.delta_auc_macro, 3, locale)} · ${t("reference")} ${num(overall?.reference_auc, 3, locale)}`],
    ["Lift@10", num(overall?.lift_top10, 2, locale), `${t("95% CI")} ${ci(overall?.ci?.lift_top10, 2, locale)} · ${t("1.0 is chance")}`],
    ["Precision@10", pct(overall?.precision_top10, 1, locale), `${t("base rate")} ${pct(report.population?.test_viral_rate, 1, locale)}`],
  ];
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        {report.promoted ? <Pill tone="ok">{t("Offline CI rule passed")}</Pill> : <Pill tone="warn">{t("Not promoted")}</Pill>}
        <Pill tone="brand">{t("Viral = 3× the channel median")}</Pill>
        <span className="mono text-xs text-muted">
          {t("age shortcut")} ρ {signed(overall?.age_rho, 3, locale)}
          {path ? ` · ${path}` : ""}
        </span>
      </div>
      <dl className="mt-5 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {items.map(([label, value, detail]) => (
          <div key={label}>
            <dt className="label">{t(label)}</dt>
            <dd className={`display mt-1 text-3xl tabular-nums ${report.promoted ? "text-ok" : "text-warn"}`}>{value}</dd>
            <dd className="mono mt-1 text-xs text-muted">{detail}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function ViralSectors({ report }: { report: ViralReport }) {
  const { t, locale } = useT();
  if (!report.sectors?.length) return null;
  return (
    <div className="card overflow-x-auto">
      <table className="table min-w-[760px]">
        <thead>
          <tr>
            <th scope="col">{t("Sector")}</th>
            <th scope="col">{t("AUC per channel [95% CI]")}</th>
            <th scope="col">{t("Reference")}</th>
            <th scope="col">{t("Gain [95% CI]")}</th>
            <th scope="col">Lift@10 [IC95%]</th>
            <th scope="col">{t("Channels")}</th>
          </tr>
        </thead>
        <tbody>
          {report.sectors.map((sector) => {
            const significant = (sector.ci?.delta_auc_macro?.[0] ?? 0) > 0;
            return (
              <tr key={sector.sector}>
                <th scope="row" className="font-semibold capitalize">{sector.sector}</th>
                <td className="num">
                  {num(sector.auc_macro_channel, 3, locale)} <span className="text-muted">{ci(sector.ci?.auc_macro_channel, 3, locale)}</span>
                </td>
                <td className="num">{num(sector.reference_auc, 3, locale)}</td>
                <td className={`num ${significant ? "font-semibold text-ok" : ""}`}>
                  {signed(delta(sector), 3, locale)} <span className="text-muted">{signedCi(sector.ci?.delta_auc_macro, 3, locale)}</span>
                </td>
                <td className="num">
                  {num(sector.lift_top10, 2, locale)} <span className="text-muted">{ci(sector.ci?.lift_top10, 2, locale)}</span>
                </td>
                <td className="num">{sector.channels_with_pairs ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
