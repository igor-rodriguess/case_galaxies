"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

import { IconSend } from "@/components/icons";
import { TrailList } from "@/components/McpRail";
import { ToolCallCard } from "@/components/ToolCallCard";
import { ErrorBox, PageHeader } from "@/components/ui";
import { parseCommand, templateCommand, TOOLS } from "@/lib/commands";
import { useData } from "@/lib/data";
import { useT } from "@/lib/i18n";
import type { ToolCall } from "@/lib/types";

type Message =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "mcp"; kind: "call"; call: ToolCall }
  | { id: number; role: "mcp"; kind: "pending"; tool: string }
  | { id: number; role: "mcp"; kind: "error"; tool: string; message: string }
  | { id: number; role: "mcp"; kind: "note"; text: string; help?: boolean };

type WithoutId<T> = T extends unknown ? Omit<T, "id"> : never;

const HELP = [
  "status moda — model_status for one sector (no sector: all three)",
  "traces 10 — recent_mcp_calls",
  "/<tool> {json} — calls any interface tool with JSON arguments",
];

const SUGGESTIONS = ["status", "status beleza", "traces 5"];

export default function ConsolePage() {
  const { t } = useT();
  const { call, bridge } = useData();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const counter = useRef(0);
  const end = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function push(message: WithoutId<Message>) {
    counter.current += 1;
    const id = counter.current;
    setMessages((list) => [...list, { ...message, id } as Message]);
    return id;
  }

  function replace(id: number, message: WithoutId<Message>) {
    setMessages((list) => list.map((item) => (item.id === id ? ({ ...message, id } as Message) : item)));
  }

  async function run(text: string) {
    const parsed = parseCommand(text);
    push({ role: "user", text });
    if ("error" in parsed) {
      push({ role: "mcp", kind: "note", text: t(parsed.error) });
      return;
    }
    if ("help" in parsed) {
      push({ role: "mcp", kind: "note", text: t("This console only runs MCP tools. Commands:"), help: true });
      return;
    }
    setBusy(true);
    for (const item of parsed.calls) {
      const id = push({ role: "mcp", kind: "pending", tool: item.tool });
      try {
        const response = await call(item.tool, item.arguments);
        replace(id, { role: "mcp", kind: "call", call: response });
      } catch (reason) {
        replace(id, { role: "mcp", kind: "error", tool: item.tool, message: reason instanceof Error ? reason.message : String(reason) });
      }
    }
    setBusy(false);
  }

  function submit(event?: React.FormEvent) {
    event?.preventDefault();
    if (!input.trim() || busy) return;
    const text = input;
    setInput("");
    run(text);
  }

  return (
    <div className="flex min-h-[calc(100vh-9rem)] flex-col">
      <PageHeader
        eyebrow={`01 · ${t("MCP Console")}`}
        title={t("Ask the MCP")}
        lede={t("Every answer is the raw output of a real tool, with time, duration, trace and the full JSON. No language model writes text or numbers here.")}
      />

      {bridge === "down" && <ErrorBox message="Could not reach the local bridge. Is it running?" />}

      <div className="grid flex-1 items-start gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="flex min-w-0 flex-col">
      <ol className="flex-1 space-y-5" aria-live="polite" aria-label={t("Conversation with the MCP")}>
        {messages.length === 0 && (
          <li className="card-quiet flex flex-col items-start gap-4 p-6 sm:flex-row sm:items-center">
            <Image src="/galaxies-mark.png" alt="" width={48} height={48} className="h-12 w-12" />
            <div className="flex-1">
              <p className="font-semibold">{t("Connected to the galaxies-performance server.")}</p>
              <p className="mt-1 text-sm text-muted">{t("Try one of these, or pick a tool below to load a ready-made call.")}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button key={suggestion} type="button" className="pill pill-brand cursor-pointer px-3 py-1.5 text-[13px]" onClick={() => run(suggestion)}>
                  {suggestion}
                </button>
              ))}
            </div>
          </li>
        )}
        {messages.map((message) =>
          message.role === "user" ? (
            <li key={message.id} className="flex justify-end">
              <pre className="mono max-w-[85%] overflow-x-auto rounded-2xl rounded-br-md bg-gradient-to-br from-brand to-brand-2 px-4 py-3 text-[13px] whitespace-pre-wrap text-white shadow-[0_10px_24px_-14px_var(--brand)]">
                {message.text}
              </pre>
            </li>
          ) : (
            <li key={message.id} className="flex gap-3">
              <Image src="/galaxies-mark.png" alt="MCP" width={32} height={32} className="mt-1 h-8 w-8 shrink-0" />
              <div className="min-w-0 flex-1">
                {message.kind === "call" && <ToolCallCard call={message.call} />}
                {message.kind === "pending" && (
                  <div className="card flex items-center gap-3 p-4 text-sm text-muted" role="status">
                    <span className="live-dot h-2 w-2 rounded-full bg-brand text-brand" />
                    {t("Calling")} <span className="mono text-text">{message.tool}</span>…
                  </div>
                )}
                {message.kind === "error" && <ErrorBox message={`${message.tool}: ${message.message}`} />}
                {message.kind === "note" && (
                  <div className="card p-4 text-sm">
                    <p>{message.text}</p>
                    {message.help && (
                      <ul className="mono mt-2 space-y-1 text-xs text-muted">
                        {HELP.map((line) => (
                          <li key={line}>{t(line)}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </li>
          ),
        )}
        <div ref={end} />
      </ol>

      <form onSubmit={submit} className="mt-8 rounded-2xl border border-brand-line sm:sticky sm:bottom-4 bg-surface/95 p-3 shadow-[var(--shadow-lg)] backdrop-blur">
        <label htmlFor="composer" className="sr-only">
          {t("Command for the MCP")}
        </label>
        <textarea
          id="composer"
          ref={field}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          rows={input.includes("\n") ? 6 : 2}
          placeholder="status moda · traces 10 · /creative_battle {…}"
          className="mono w-full resize-none bg-transparent px-2 py-1.5 text-[13.5px] outline-none placeholder:text-muted"
        />
        <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-border pt-3">
          <div className="-mx-1 flex w-full flex-1 gap-1.5 overflow-x-auto px-1 pb-1 sm:w-auto sm:flex-wrap sm:overflow-visible sm:pb-0" role="group" aria-label={t("Insert a call template")}>
            {TOOLS.map((tool) => (
              <button
                key={tool}
                type="button"
                className="mono shrink-0 rounded-lg border border-border px-2.5 py-1 text-[11.5px] text-text-2 transition hover:border-brand hover:text-brand"
                onClick={() => {
                  setInput(templateCommand(tool));
                  field.current?.focus();
                }}
              >
                {tool}
              </button>
            ))}
          </div>
          <button type="submit" className="btn btn-primary w-full sm:w-auto" disabled={busy || !input.trim()}>
            <IconSend className="h-4 w-4" /> {busy ? t("Calling…") : t("Send to MCP")}
          </button>
        </div>
      </form>
        </div>
        <aside className="sticky top-8 hidden overflow-hidden rounded-2xl border border-border bg-surface shadow-[var(--shadow)] xl:block">
          <div className="border-b border-border bg-brand-soft px-4 py-3">
            <p className="font-mono text-[11px] font-semibold tracking-[0.14em] text-brand uppercase">{t("MCP live")}</p>
            <p className="text-sm font-bold">{t("Call trail")}</p>
          </div>
          <div className="max-h-[calc(100vh-12rem)] overflow-auto">
            <TrailList limit={30} />
          </div>
        </aside>
      </div>
    </div>
  );
}
