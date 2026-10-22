// Cliente da ponte Python. Em dev, /api é reescrito para http://127.0.0.1:8765.

import type { Artifacts, Channel, ToolCall, ToolName } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status = 0,
    public readonly code = "network",
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { ...init, headers: { Accept: "application/json", ...(init?.headers ?? {}) } });
  } catch {
    throw new ApiError("Could not reach the local bridge. Is it running?");
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError(`Invalid bridge response (HTTP ${response.status}).`, response.status, "parse");
  }
  if (!response.ok) {
    const body = (payload ?? {}) as { message?: string; error?: string };
    throw new ApiError(body.message ?? body.error ?? `HTTP ${response.status}`, response.status, body.error ?? "http");
  }
  return payload as T;
}

export const api = {
  health: () => request<{ status: string; time: string }>("/api/health"),
  artifacts: () => request<Artifacts>("/api/artifacts"),
  channels: () => request<{ status: string; channels?: Channel[] }>("/api/channels"),
  call: <T = unknown>(tool: ToolName, args: Record<string, unknown>) =>
    request<ToolCall<T>>("/api/mcp/call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool, arguments: args }),
    }),
};
