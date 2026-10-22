"use client";

/** Estado compartilhado: artefatos, canais, model_status e o pulso das chamadas MCP. */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api } from "./api";
import { SECTORS, type Artifacts, type Channel, type ModelStatus, type ToolCall, type ToolName } from "./types";

type Async<T> = { state: "loading" } | { state: "ok"; data: T } | { state: "error"; message: string };

interface DataContext {
  bridge: "checking" | "ok" | "down";
  artifacts: Async<Artifacts>;
  channels: Async<Channel[]>;
  statuses: Async<ModelStatus[]>;
  pulse: number;
  reload: () => void;
  trailOpen: boolean;
  setTrailOpen: (open: boolean) => void;
  call: <T = unknown>(tool: ToolName, args: Record<string, unknown>) => Promise<ToolCall<T>>;
}

const Ctx = createContext<DataContext | null>(null);

const STATUS_CACHE = "galaxies.model_status";
const STATUS_TTL_MS = 10 * 60 * 1000;

/** model_status vale por 10 minutos na sessão: evita encher a trilha com chamadas iguais. */
function readCachedStatuses(): ModelStatus[] | null {
  try {
    const raw = sessionStorage.getItem(STATUS_CACHE);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { at: number; data: ModelStatus[] };
    return Date.now() - parsed.at < STATUS_TTL_MS && Array.isArray(parsed.data) ? parsed.data : null;
  } catch {
    return null;
  }
}

function writeCachedStatuses(data: ModelStatus[]) {
  try {
    sessionStorage.setItem(STATUS_CACHE, JSON.stringify({ at: Date.now(), data }));
  } catch {}
}

function message(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

export function DataProvider({ children }: { children: React.ReactNode }) {
  const [bridge, setBridge] = useState<DataContext["bridge"]>("checking");
  const [artifacts, setArtifacts] = useState<Async<Artifacts>>({ state: "loading" });
  const [channels, setChannels] = useState<Async<Channel[]>>({ state: "loading" });
  const [statuses, setStatuses] = useState<Async<ModelStatus[]>>({ state: "loading" });
  const [pulse, setPulse] = useState(0);
  const [generation, setGeneration] = useState(0);
  const [trailOpen, setTrailOpen] = useState(false);

  const call = useCallback(async <T,>(tool: ToolName, args: Record<string, unknown>) => {
    try {
      return await api.call<T>(tool, args);
    } finally {
      if (tool !== "recent_mcp_calls") setPulse((value) => value + 1);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await api.health();
        if (alive) setBridge("ok");
      } catch {
        if (!alive) return;
        setBridge("down");
        const down = { state: "error" as const, message: "Could not reach the local bridge. Is it running?" };
        setArtifacts(down);
        setChannels(down);
        setStatuses(down);
        return;
      }
      api
        .artifacts()
        .then((data) => alive && setArtifacts({ state: "ok", data }))
        .catch((error) => alive && setArtifacts({ state: "error", message: message(error) }));
      api
        .channels()
        .then((data) => alive && setChannels({ state: "ok", data: data.channels ?? [] }))
        .catch((error) => alive && setChannels({ state: "error", message: message(error) }));
      const cached = generation === 0 ? readCachedStatuses() : null;
      if (cached) {
        if (alive) setStatuses({ state: "ok", data: cached });
        return;
      }
      const results: ModelStatus[] = [];
      let lastError: string | null = null;
      for (const sector of SECTORS) {
        // Uma nova tentativa por setor: uma falha isolada não apaga os outros.
        for (let attempt = 0; attempt < 2; attempt += 1) {
          try {
            const response = await api.call<ModelStatus>("model_status", { sector });
            if (!response.is_error && response.result) results.push(response.result);
            break;
          } catch (error) {
            lastError = message(error);
          }
        }
      }
      if (!alive) return;
      if (results.length) {
        writeCachedStatuses(results);
        setStatuses({ state: "ok", data: results });
        setPulse((value) => value + 1);
      } else {
        setStatuses({ state: "error", message: lastError ?? "model_status unavailable" });
      }
    })();
    return () => {
      alive = false;
    };
  }, [generation]);

  const reload = useCallback(() => {
    setBridge("checking");
    setArtifacts({ state: "loading" });
    setChannels({ state: "loading" });
    setStatuses({ state: "loading" });
    setGeneration((value) => value + 1);
  }, []);

  const value = useMemo(
    () => ({ bridge, artifacts, channels, statuses, pulse, reload, call, trailOpen, setTrailOpen }),
    [bridge, artifacts, channels, statuses, pulse, reload, call, trailOpen],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useData(): DataContext {
  const value = useContext(Ctx);
  if (!value) throw new Error("useData precisa estar dentro de DataProvider.");
  return value;
}
