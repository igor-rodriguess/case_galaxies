import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

import { Shell } from "@/components/Shell";
import { DataProvider } from "@/lib/data";
import { I18nProvider } from "@/lib/i18n";

const dmSans = DM_Sans({ variable: "--font-dm-sans", subsets: ["latin"], weight: ["400", "500", "600", "700"] });
const jetbrains = JetBrains_Mono({ variable: "--font-jetbrains", subsets: ["latin"], weight: ["400", "500", "600"] });

export const metadata: Metadata = {
  title: { default: "Galaxies · Creative intelligence", template: "%s · Galaxies" },
  description: "Console MCP, Creative Battle e evidências dos modelos de criativos da Galaxies.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR" className={`${dmSans.variable} ${jetbrains.variable} h-full antialiased`}>
      <body className="min-h-full font-sans">
        <I18nProvider>
          <DataProvider>
            <Shell>{children}</Shell>
          </DataProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
