"use client";

import Image from "next/image";
import Link from "next/link";

import { IconArrow, IconBattle, IconChart, IconChat } from "@/components/icons";
import { ErrorBox, Loading, Pill, Unavailable } from "@/components/ui";
import { useData } from "@/lib/data";
import { ci, num, pct } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { promotionSummary } from "@/lib/metrics";

function Orbit() {
  return (
    <div className="relative mx-auto aspect-square w-full max-w-[360px]" aria-hidden>
      <div className="grid-bg absolute inset-0 rounded-[32px] [mask-image:radial-gradient(circle_at_center,black,transparent_72%)]" />
      <div className="absolute inset-[6%] rounded-full border border-brand-line" />
      <div className="absolute inset-[20%] rounded-full border border-dashed border-brand-line motion-safe:animate-[spin-slow_60s_linear_infinite]" />
      <div className="absolute inset-[34%] rounded-full bg-[radial-gradient(circle,var(--brand-soft),transparent_70%)]" />
      <div className="absolute inset-0 motion-safe:animate-[spin-slow_24s_linear_infinite]">
        <span className="absolute left-1/2 top-[6%] h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand shadow-[0_0_0_6px_var(--brand-soft)]" />
      </div>
      <div className="absolute inset-0 motion-safe:animate-[spin-slow_40s_linear_infinite_reverse]">
        <span className="absolute bottom-[20%] right-[20%] h-2 w-2 rounded-full bg-magenta" />
      </div>
      <Image src="/galaxies-mark.png" alt="" width={160} height={160} priority className="absolute left-1/2 top-1/2 h-[30%] w-[30%] -translate-x-1/2 -translate-y-1/2 drop-shadow-[0_18px_30px_rgba(0,101,255,0.35)]" />
    </div>
  );
}

export default function Overview() {
  const { t, locale } = useT();
  const { statuses, artifacts, reload } = useData();
  const summary = statuses.state === "ok" ? promotionSummary(statuses.data) : null;
  const shorts = artifacts.state === "ok" && artifacts.data.shorts_only.report.status === "ok" ? artifacts.data.shorts_only.report.data : null;
  const battle = artifacts.state === "ok" && artifacts.data.battle_model.report.status === "ok" ? artifacts.data.battle_model.report.data : null;
  const overall = shorts?.overall;

  return (
    <div className="space-y-10">
      <section className="relative grid items-center gap-10 overflow-hidden rounded-[28px] border border-border bg-surface px-6 py-10 shadow-[var(--shadow)] sm:px-10 lg:grid-cols-[1.5fr_1fr] lg:py-14">
        <div className="relative z-10">
          <p className="eyebrow">Galaxies · {t("Creative intelligence")}</p>
          <h1 className="display mt-4 text-[clamp(2.2rem,3.7vw,3.6rem)]">
            {t("Decide which creative")} <span className="bg-gradient-to-r from-brand to-brand-2 bg-clip-text text-transparent">{t("to test first.")}</span>
          </h1>
          <p className="mt-5 max-w-xl text-[16px] leading-relaxed text-text-2">
            {t("Compare versions of a Short before publishing, ask the MCP and see exactly what the evidence supports, and what it does not support yet.")}
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/battle" className="btn btn-primary">
              {t("Open Creative Battle")} <IconArrow className="h-4 w-4" />
            </Link>
            <Link href="/console" className="btn">
              <IconChat className="h-4 w-4 text-brand" /> {t("Ask the MCP")}
            </Link>
          </div>
          <div className="mt-7 flex flex-wrap items-center gap-2 text-sm">
            {summary?.state === "none" && <Pill tone="warn">{t("No model promoted")}</Pill>}
            {summary?.state === "partial" && <Pill tone="ok">{t("Promoted")}: {summary.promoted.join(", ")}</Pill>}
            <Pill tone="brand">{t("Association, not causation")}</Pill>
          </div>
        </div>
        <Orbit />
      </section>

      <section aria-labelledby="evidence-title" className="card overflow-hidden">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b border-border px-6 py-5">
          <div>
            <p className="eyebrow">{t("Where the evidence stands")}</p>
            <h2 id="evidence-title" className="mt-1.5 text-xl font-bold tracking-tight">{t("Honest numbers, straight from the reports")}</h2>
          </div>
          <Link href="/models" className="inline-flex items-center gap-1.5 text-sm font-semibold text-brand">
            {t("See all evidence")} <IconArrow className="h-4 w-4" />
          </Link>
        </div>
        {artifacts.state === "loading" && (
          <div className="p-6">
            <Loading rows={3} />
          </div>
        )}
        {artifacts.state === "error" && (
          <div className="p-6">
            <ErrorBox message={artifacts.message} onRetry={reload} />
          </div>
        )}
        {artifacts.state === "ok" && (
          <>
            <dl className="grid divide-y divide-border sm:grid-cols-3 sm:divide-x sm:divide-y-0">
              {[
                {
                  label: "AUC per channel · Shorts",
                  value: num(overall?.auc_macro_channel, 3, locale),
                  detail: overall ? `${t("95% CI")} ${ci(overall.ci?.auc_macro_channel, 3, locale)} · ${t("0.5 is chance")}` : t("unavailable"),
                },
                {
                  label: "Lift@10 · Shorts",
                  value: num(overall?.lift_top10, 2, locale),
                  detail: overall ? `${t("95% CI")} ${ci(overall.ci?.lift_top10, 2, locale)} · ${t("1.0 is chance")}` : t("unavailable"),
                },
                {
                  label: "Battle model · temporal AUC",
                  value: num(battle?.evaluation?.temporal?.auc_intra, 3, locale),
                  detail: battle?.evaluation?.temporal
                    ? `${t("95% CI")} ${ci([battle.evaluation.temporal.auc_intra_ci_low, battle.evaluation.temporal.auc_intra_ci_high], 3, locale)}`
                    : t("unavailable"),
                },
              ].map((item) => (
                <div key={item.label} className="px-6 py-5">
                  <dt className="label">{t(item.label)}</dt>
                  <dd className="display mt-2 text-4xl tabular-nums text-warn">{item.value}</dd>
                  <dd className="mono mt-2 text-xs text-muted">{item.detail}</dd>
                </div>
              ))}
            </dl>
            {overall ? (
              <p className="border-t border-border bg-warn-soft px-6 py-4 text-sm text-text-2">
                <strong className="text-warn">{t("Reading:")}</strong>{" "}
                {t("with metadata and cadence alone, the order of Shorts within a channel is at chance level. Use the tools to compare scenarios and plan tests, not to promise results.")}{" "}
                <span className="mono text-xs text-muted">
                  {shorts?.population?.test_rows?.toLocaleString(locale)} Shorts · P@10 {pct(overall.precision_top10, 1, locale)}
                </span>
              </p>
            ) : (
              <div className="p-6">
                <Unavailable detail="Shorts-only evaluation not found." />
              </div>
            )}
          </>
        )}
      </section>

      <section aria-labelledby="start-title">
        <h2 id="start-title" className="mb-4 text-xl font-bold tracking-tight">{t("Where to start")}</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { href: "/battle", icon: IconBattle, title: "Compare two versions", text: "Edit A and B, send to the MCP and read verdict, confidence and alerts.", primary: true },
            { href: "/console", icon: IconChat, title: "Ask the MCP directly", text: "Run any of the five tools and inspect the raw answer and its trace." },
            { href: "/models", icon: IconChart, title: "Check the evidence", text: "See what each model scored, with confidence intervals." },
          ].map(({ href, icon: Icon, title, text, primary }) => (
            <Link
              key={href}
              href={href}
              className={`group flex items-start gap-4 rounded-2xl border p-5 transition hover:-translate-y-0.5 hover:shadow-[var(--shadow-lg)] ${primary ? "border-brand-line bg-brand-soft" : "border-border bg-surface"}`}
            >
              <span className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${primary ? "bg-gradient-to-br from-brand to-brand-2 text-white" : "bg-brand-soft text-brand"}`}>
                <Icon />
              </span>
              <span className="min-w-0">
                <span className="flex items-center gap-1.5 font-bold">
                  {t(title)} <IconArrow className="h-4 w-4 text-brand transition group-hover:translate-x-0.5" />
                </span>
                <span className="mt-1 block text-sm leading-relaxed text-muted">{t(text)}</span>
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
