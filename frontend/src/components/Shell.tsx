"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { IconBattle, IconChart, IconChat, IconClose, IconHome, IconMenu, IconMoon, IconSun, IconTrace } from "@/components/icons";
import { McpButton, McpDrawer } from "@/components/McpRail";
import { useData } from "@/lib/data";
import { useT } from "@/lib/i18n";
import { promotionSummary } from "@/lib/metrics";

const NAV = [
  { href: "/", label: "Overview", icon: IconHome, group: "Product" },
  { href: "/console", label: "MCP Console", icon: IconChat, group: "Product" },
  { href: "/battle", label: "Creative Battle", icon: IconBattle, group: "Product" },
  { href: "/models", label: "Models & evidence", icon: IconChart, group: "Evidence" },
  { href: "/traces", label: "MCP traces", icon: IconTrace, group: "Evidence" },
] as const;

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-3" aria-label="Galaxies">
      <Image src="/galaxies-logo.png" alt="Galaxies" width={150} height={26} priority className="logo-light h-[22px] w-auto" />
      <Image src="/galaxies-logo-white.png" alt="Galaxies" width={150} height={26} priority className="logo-dark h-[22px] w-auto" />
    </Link>
  );
}

function SystemStatus() {
  const { t } = useT();
  const { bridge, statuses } = useData();
  const summary = statuses.state === "ok" ? promotionSummary(statuses.data) : null;
  const text =
    bridge === "down"
      ? t("Bridge offline")
      : summary?.state === "none"
        ? t("No model promoted: predictions answer model_not_ready.")
        : summary?.state === "partial"
          ? `${t("Promoted")}: ${summary.promoted.join(", ")}`
          : t("Reading model_status…");
  return <p className={`px-1 text-xs leading-snug ${bridge === "down" ? "text-danger" : "text-muted"}`}>{text}</p>;
}

function Toggles() {
  const { lang, setLang, t } = useT();
  const [theme, setTheme] = useState<"light" | "dark" | "auto">("auto");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("galaxies.theme") as "light" | "dark" | null;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- localStorage só existe depois da montagem
      if (saved) setTheme(saved);
    } catch {}
  }, []);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "auto") localStorage.removeItem("galaxies.theme");
      else localStorage.setItem("galaxies.theme", theme);
    } catch {}
  }, [theme]);
  const dark = theme === "dark";
  return (
    <div className="flex items-center gap-2">
      <div className="flex rounded-xl border border-border p-0.5 text-xs font-semibold" role="group" aria-label={t("Language")}>
        {(["pt", "en"] as const).map((item) => (
          <button
            key={item}
            type="button"
            aria-pressed={lang === item}
            onClick={() => setLang(item)}
            className={`rounded-lg px-2.5 py-1 uppercase ${lang === item ? "bg-brand text-white" : "text-muted hover:text-text"}`}
          >
            {item}
          </button>
        ))}
      </div>
      <button
        type="button"
        className="rounded-xl border border-border p-1.5 text-muted hover:text-text"
        aria-label={dark ? t("Switch to light theme") : t("Switch to dark theme")}
        onClick={() => setTheme(dark ? "light" : "dark")}
      >
        {dark ? <IconSun className="h-4 w-4" /> : <IconMoon className="h-4 w-4" />}
      </button>
    </div>
  );
}

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const { t } = useT();
  const path = usePathname();
  const groups = ["Product", "Evidence"] as const;
  return (
    <nav aria-label={t("Sections")} className="space-y-6">
      {groups.map((group) => (
        <div key={group}>
          <p className="label mb-2 px-3">{t(group)}</p>
          <ul className="space-y-1">
            {NAV.filter((item) => item.group === group).map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? path === "/" : path?.startsWith(href);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    onClick={onNavigate}
                    aria-current={active ? "page" : undefined}
                    className={`group flex items-center gap-3 rounded-xl px-3 py-2.5 text-[14px] font-medium transition ${
                      active
                        ? "bg-brand text-white shadow-[0_10px_24px_-14px_var(--brand)]"
                        : "text-text-2 hover:bg-surface-2 hover:text-text"
                    }`}
                  >
                    <Icon className={`h-[18px] w-[18px] ${active ? "" : "text-muted group-hover:text-brand"}`} />
                    {t(label)}
                    {href === "/console" && !active && (
                      <span className="ml-auto rounded-md bg-brand-soft px-1.5 py-0.5 text-[10px] font-bold text-brand-ink">MCP</span>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const { t } = useT();
  const [open, setOpen] = useState(false);
  const path = usePathname();
  return (
    <div className="flex min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-brand focus:px-4 focus:py-2 focus:text-white">
        {t("Skip to content")}
      </a>
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col gap-8 border-r border-border bg-surface/80 px-4 py-6 backdrop-blur lg:flex">
        <div className="px-3">
          <Brand />
          <p className="label mt-3">{t("Creative intelligence")}</p>
        </div>
        <Nav />
        <div className="mt-auto space-y-3">
          <McpButton />
          <SystemStatus />
          <Toggles />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-border bg-surface/85 px-4 py-3 backdrop-blur lg:hidden">
          <Brand />
          <div className="flex items-center gap-2">
          <McpButton compact />
          <button
            type="button"
            className="rounded-xl border border-border p-2"
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label={open ? t("Close menu") : t("Open menu")}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? <IconClose /> : <IconMenu />}
          </button>
          </div>
        </header>
        {open && (
          <div id="mobile-nav" className="fixed inset-x-0 top-[57px] bottom-0 z-20 space-y-6 overflow-auto border-t border-border bg-surface p-5 lg:hidden">
            <Nav onNavigate={() => setOpen(false)} />
            <SystemStatus />
            <Toggles />
          </div>
        )}

        <main id="main" key={path} tabIndex={-1} className="reveal mx-auto w-full max-w-[1180px] flex-1 px-4 py-8 outline-none sm:px-8 lg:py-10">
          {children}
        </main>
        <McpDrawer />
        <footer className="border-t border-border px-4 py-5 text-xs text-muted sm:px-8">
          {t("Association, not causation. Every number comes from an MCP tool or from the reports in data/modeling/runs.")}
        </footer>
      </div>
    </div>
  );
}
