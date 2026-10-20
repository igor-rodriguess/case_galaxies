"""Registro local e sanitizado das chamadas do servidor MCP."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import MCP_TRACES_DIR


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class McpTraceStore:
    """Persiste eventos JSONL para auditoria e para uma futura interface.

    O log guarda apenas resumos de entrada: valores de features não são escritos.
    """

    def __init__(self, directory: Path = MCP_TRACES_DIR):
        self.directory = directory
        self.path = directory / "mcp_calls.jsonl"

    def record(
        self,
        *,
        tool_name: str,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
    ) -> dict[str, Any]:
        event = {
            "schema_version": "0.1.0",
            "trace_id": str(uuid4()),
            "recorded_at": utc_now(),
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1 or limit > 100:
            raise ValueError("limit deve ficar entre 1 e 100.")
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return list(reversed(events[-limit:]))
