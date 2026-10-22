"use client";

import { Pill } from "@/components/ui";
import { ci, num, pct, signed, signedCi } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { Sector, ViralMetrics, ViralReport } from "@/lib/types";

/** Diferença contra a referência, calculada só a partir dos dois valores do relatório. */
function delta(metrics: ViralMetrics | undefined): number | null {
  if (metrics?.auc_macro_channel == null || metrics.reference_auc == null) return null;
  return metrics.auc_macro_channel - metrics.reference_auc;
}

/** Números da base v2, lidos do report.json sem arredondar nem completar. */
export function ViralHeadline({
  report,
  path,
  exploratory = false,
}: {
  report: ViralReport;
  path?: string;
  exploratory?: boolean;
}) {
  const { t, locale } = useT();
  const summary = exploratory ? report.exploratory_excluding_fashion : undefined;
  const overall = summary?.overall ?? report.overall;
  const population = summary?.population ?? report.population;
  const promoted = !exploratory && report.promoted;
  const items: [string, string, string][] = [
    ["AUC per channel", num(overall?.auc_macro_channel, 3, locale), `${t("95% CI")} ${ci(overall?.ci?.auc_macro_channel, 3, locale)} · ${t("0.5 is chance")}`],
    ["Gain over reference", signed(delta(overall), 3, locale), `${t("95% CI")} ${signedCi(overall?.ci?.delta_auc_macro, 3, locale)} · ${t("reference")} ${num(overall?.reference_auc, 3, locale)}`],
    ["Lift@10", num(overall?.lift_top10, 2, locale), `${t("95% CI")} ${ci(overall?.ci?.lift_top10, 2, locale)} · ${t("1.0 is chance")}`],
    ["Precision@10", pct(overall?.precision_top10, 1, locale), `${t("base rate")} ${pct(population?.test_viral_rate, 1, locale)}`],
  ];
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        {exploratory ? (
          <Pill tone="warn">{t("Exploratory mean · fashion excluded")}</Pill>
        ) : promoted ? (
          <Pill tone="ok">{t("Offline CI rule passed")}</Pill>
        ) : (
          <Pill tone="warn">{t("Not promoted")}</Pill>
        )}
        <Pill tone="brand">{t("Viral = 3× the channel median")}</Pill>
        <span className="mono text-xs text-muted">
          {t("age shortcut")} ρ {signed(overall?.age_rho, 3, locale)}
          {population?.test_rows ? ` · ${population.test_rows.toLocaleString(locale)} ${t("test videos")}` : ""}
          {path ? ` · ${path}` : ""}
        </span>
      </div>
      <dl className="mt-5 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {items.map(([label, value, detail]) => (
          <div key={label}>
            <dt className="label">{t(label)}</dt>
            <dd className={`display mt-1 text-3xl tabular-nums ${promoted ? "text-ok" : "text-warn"}`}>{value}</dd>
            <dd className="mono mt-1 text-xs text-muted">{detail}</dd>
          </div>
        ))}
      </dl>
      {exploratory && summary?.note && <p className="mt-4 text-sm text-muted">{summary.note}</p>}
    </div>
  );
}

export function ViralSectors({
  report,
  excludedSectors = [],
}: {
  report: ViralReport;
  excludedSectors?: Sector[];
}) {
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
            const excluded = excludedSectors.includes(sector.sector);
            const significant = (sector.ci?.delta_auc_macro?.[0] ?? 0) > 0;
            return (
              <tr key={sector.sector}>
                <th scope="row" className="font-semibold capitalize">
                  {sector.sector}
                  {excluded && <span className="ml-2 text-xs font-normal text-muted">{t("Excluded from mean")}</span>}
                </th>
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
