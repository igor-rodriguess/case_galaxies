from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from galaxies.mcp_server import mcp


def test_mcp_lists_expected_tools_and_reports_not_ready() -> None:
    async def scenario() -> None:
        async with Client(mcp) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {
                "modeling_contract",
                "model_status",
                "validate_feature_payload",
                "get_video_feature_map",
                "predict_performance",
                "rank_creatives",
            } <= names
            result = await client.call_tool("model_status", {"sector": "automotivo"})
            assert result.structured_content["ready"] is False

    asyncio.run(scenario())


def test_mcp_stdio_launcher_exposes_tools() -> None:
    async def scenario() -> None:
        project_root = Path(__file__).resolve().parents[1]
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                str(project_root / "scripts" / "run_mcp_server.py"),
                "--transport",
                "stdio",
            ],
            cwd=str(project_root),
        )
        async with Client(parameters) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert "predict_performance" in names
            assert "rank_creatives" in names

    asyncio.run(scenario())
