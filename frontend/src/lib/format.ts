// Formatação pura. Nada aqui cria ou completa métricas: valor ausente vira "—".

export const MISSING = "—";

export function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Converte célula de CSV em número sem transformar vazio em 0. */
export function cellNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  if (!text) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

export function num(value: unknown, digits = 3, locale = "pt-BR"): string {
  return isNumber(value)
    ? value.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : MISSING;
}

export function pct(value: unknown, digits = 1, locale = "pt-BR"): string {
  return isNumber(value) ? `${num(value * 100, digits, locale)}%` : MISSING;
}

export function signed(value: unknown, digits = 3, locale = "pt-BR"): string {
  if (!isNumber(value)) return MISSING;
  const text = num(Math.abs(value), digits, locale);
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

export function points(value: unknown, digits = 2, locale = "pt-BR"): string {
  return isNumber(value) ? `${signed(value * 100, digits, locale)} p.p.` : MISSING;
}

export function ci(range: readonly [unknown, unknown] | null | undefined, digits = 3, locale = "pt-BR"): string {
  if (!range || !isNumber(range[0]) || !isNumber(range[1])) return "";
  return `[${num(range[0], digits, locale)}; ${num(range[1], digits, locale)}]`;
}

export function signedCi(range: readonly [unknown, unknown] | null | undefined, digits = 3, locale = "pt-BR"): string {
  if (!range || !isNumber(range[0]) || !isNumber(range[1])) return "";
  return `[${signed(range[0], digits, locale)}; ${signed(range[1], digits, locale)}]`;
}

export function dateTime(value: unknown, locale = "pt-BR"): string {
  if (typeof value !== "string" || !value) return MISSING;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return MISSING;
  return parsed.toLocaleString(locale, {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function shortId(value: unknown, size = 8): string {
  return typeof value === "string" && value ? value.slice(0, size) : MISSING;
}

export function humanize(name: unknown): string {
  return String(name ?? "")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
}
