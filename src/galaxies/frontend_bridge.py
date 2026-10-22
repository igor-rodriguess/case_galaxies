"""Ponte HTTP entre o front-end e o servidor MCP real do Galaxies.

O navegador não fala MCP diretamente. Esta ponte:

- chama as tools pelo cliente MCP oficial (JSON-RPC em memória ou por URL), sem
  atalhos para as funções Python; cada chamada passa pela validação do servidor
  e grava o trace normal em ``data/mcp/traces``;
- expõe só as tools da interface (``ALLOWED_TOOLS``);
- lê os relatórios de avaliação em modo leitura. Artefato ausente vira
  ``{"status": "unavailable"}``; a ponte nunca calcula nem completa métricas;
- expõe só ``/api``; a interface é o app Next.js em ``frontend/``, que encaminha
  ``/api`` para cá (``next.config.ts``).

Só escuta em 127.0.0.1 por padrão. Não há autenticação: não exponha em rede.
"""

from __future__ import annotations

import asyncio
import csv
import json
import mimetypes
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mcp.client import Client

from .paths import PROJECT_ROOT

ALLOWED_TOOLS = (
    "model_status",
    "predict_performance",
    "rank_creatives",
    "creative_battle",
    "recent_mcp_calls",
)
ROLLING_RUNS = (
    "known-channel-rolling-fold-aware-20261021",
    "known-channel-rolling-expanded-20261021",
    "known-channel-rolling-ensemble-20261021",
)
BATTLE_RUN = "baseline-freeze-calibrated-20261020"
SHORTS_RUN = "shorts-ranking-final-14k"
# Ranking na base v2 (rótulo de outlier por canal), promovido pela regra de IC.
VIRAL_RUN = "viral-v2-ranking-20261021"
FIGURE_RUNS = ("known-channel-rolling-fold-aware-20261021", SHORTS_RUN, VIRAL_RUN)
RUNS_DIR = PROJECT_ROOT / "data" / "modeling" / "runs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SELECTED_CHANNELS = PROJECT_ROOT / "config" / "selected_channels.json"
MAX_BODY_BYTES = 256 * 1024


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable", "path": _relative(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "path": _relative(path), "reason": str(exc)}
    return {"status": "ok", "path": _relative(path), "data": payload}


def _read_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable", "path": _relative(path)}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {"status": "ok", "path": _relative(path), "rows": rows}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _figures(run: str) -> list[dict[str, str]]:
    directory = FIGURES_DIR / run
    if not directory.is_dir():
        return []
    return [
        {"run": run, "file": path.name, "url": f"/api/figures/{run}/{path.name}"}
        for path in sorted(directory.glob("*.png"))
    ]


def load_artifacts(runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    """Lê os relatórios exibidos pela interface, sem transformar valores."""
    return {
        "loaded_at": utc_now(),
        "rolling": {
            run: {
                "report": _read_json(runs_dir / run / "report.json"),
                "fold_metrics": _read_csv(runs_dir / run / "fold_metrics.csv"),
                "figures": _figures(run),
            }
            for run in ROLLING_RUNS
        },
        "battle_model": {
            "run_id": BATTLE_RUN,
            "report": _read_json(runs_dir / BATTLE_RUN / "report.json"),
            "shap_global": _read_csv(runs_dir / BATTLE_RUN / "shap_global.csv"),
        },
        "shorts_only": {
            "run_id": SHORTS_RUN,
            "report": _read_json(runs_dir / SHORTS_RUN / "report.json"),
            "figures": _figures(SHORTS_RUN),
        },
        "viral_v2": {
            "run_id": VIRAL_RUN,
            "report": _read_json(runs_dir / VIRAL_RUN / "report.json"),
            "figures": _figures(VIRAL_RUN),
        },
    }


def load_channels(path: Path = SELECTED_CHANNELS) -> dict[str, Any]:
    payload = _read_json(path)
    if payload["status"] != "ok":
        return payload
    channels = [
        {"sector": item["sector"], "label": item["label"], "channel_id": item["channel_id"]}
        for item in payload["data"].get("channels", [])
    ]
    return {"status": "ok", "channels": channels}


class McpGateway:
    """Chama tools pelo cliente MCP oficial e anexa o trace correspondente."""

    def __init__(self, target: Any, *, allowed: tuple[str, ...] = ALLOWED_TOOLS) -> None:
        self.target = target
        self.allowed = allowed
        self._lock = threading.Lock()

    async def _session_call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with Client(self.target) as client:
            result = await client.call_tool(tool, arguments)
            trace = None
            if tool != "recent_mcp_calls" and not result.is_error:
                recent = await client.call_tool("recent_mcp_calls", {"limit": 1})
                events = (recent.structured_content or {}).get("events", [])
                trace = events[0] if events and events[0].get("tool_name") == tool else None
        text = " ".join(
            getattr(item, "text", "") for item in result.content if getattr(item, "text", "")
        )
        return {
            "is_error": bool(result.is_error),
            "result": result.structured_content,
            "error_text": text if result.is_error else None,
            "trace": trace,
        }

    async def _list(self) -> list[dict[str, Any]]:
        async with Client(self.target) as client:
            listed = await client.list_tools()
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in listed.tools
            if tool.name in self.allowed
        ]

    def list_tools(self) -> list[dict[str, Any]]:
        with self._lock:
            return asyncio.run(self._list())

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool not in self.allowed:
            raise PermissionError(f"Tool fora da interface: {tool}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments deve ser um objeto JSON.")
        started_at, start = utc_now(), time.perf_counter()
        with self._lock:
            outcome = asyncio.run(self._session_call(tool, arguments))
        return {
            "tool": tool,
            "arguments": arguments,
            "transport": "mcp-url" if isinstance(self.target, str) else "mcp-in-memory",
            "started_at": started_at,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            **outcome,
        }


def _static_root() -> Path:
    built = FRONTEND_DIR / "dist"
    return built if (built / "index.html").is_file() else FRONTEND_DIR / "src"


def make_handler(
    gateway: McpGateway,
    artifacts: Callable[[], dict[str, Any]] = load_artifacts,
    static_root: Callable[[], Path] = _static_root,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "GalaxiesBridge/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type.endswith("javascript"):
                content_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, path.read_bytes(), content_type)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/api/health":
                    self._json(HTTPStatus.OK, {"status": "ok", "time": utc_now()})
                elif path == "/api/tools":
                    self._json(HTTPStatus.OK, {"tools": gateway.list_tools()})
                elif path == "/api/artifacts":
                    self._json(HTTPStatus.OK, artifacts())
                elif path == "/api/channels":
                    self._json(HTTPStatus.OK, load_channels())
                elif path.startswith("/api/figures/"):
                    self._figure(path.removeprefix("/api/figures/"))
                elif path.startswith("/api/"):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "rota desconhecida"})
                else:
                    self._static(path)
            except Exception as exc:  # resposta explícita em vez de conexão quebrada
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__, "message": str(exc)},
                )

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/mcp/call":
                self._json(HTTPStatus.NOT_FOUND, {"error": "rota desconhecida"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "corpo vazio ou grande demais"}
                )
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = gateway.call(str(payload.get("tool")), payload.get("arguments") or {})
            except PermissionError as exc:
                self._json(HTTPStatus.FORBIDDEN, {"error": "tool_not_allowed", "message": str(exc)})
                return
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(
                    HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)}
                )
                return
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "mcp_unavailable", "message": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._json(HTTPStatus.OK, response)

        def _figure(self, rest: str) -> None:
            run, _, name = rest.partition("/")
            target = (FIGURES_DIR / run / name).resolve()
            allowed = run in FIGURE_RUNS and name.endswith(".png") and "/" not in name
            if (
                not allowed
                or not target.is_file()
                or target.parent != (FIGURES_DIR / run).resolve()
            ):
                self._json(HTTPStatus.NOT_FOUND, {"error": "figura indisponível"})
                return
            self._file(target)

        def _static(self, path: str) -> None:
            root = static_root().resolve()
            if not (root / "index.html").is_file():
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "error": "frontend_elsewhere",
                        "message": "A interface roda no Next.js: cd frontend && npm run dev "
                        "(http://127.0.0.1:3000). Esta porta serve só /api.",
                    },
                )
                return
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (root / relative).resolve()
            if root not in target.parents and target != root / "index.html":
                self._json(HTTPStatus.NOT_FOUND, {"error": "arquivo indisponível"})
                return
            if not target.is_file():
                target = root / "index.html"
            self._file(target)

    return Handler


def serve(host: str, port: int, mcp_url: str | None = None) -> None:
    from .mcp_server import mcp

    gateway = McpGateway(mcp_url or mcp)
    server = ThreadingHTTPServer((host, port), make_handler(gateway))
    print(f"Galaxies em http://{host}:{port} (MCP: {mcp_url or 'em memória'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
