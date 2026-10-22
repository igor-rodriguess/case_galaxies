from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from galaxies.mcp_server import create_mcp_server, mcp


class StubBattlePredictor:
    def __init__(self) -> None:
        self.artifact_dir = Path("baseline-freeze-test")

    def compare(self, baseline, candidate):
        return {
            "comparison_scope": "Comparação associativa, não causal.",
            "changed_model_inputs": [
                {"feature": "duration_seconds", "baseline": 20.0, "candidate": 35.0}
            ],
            "baseline": {
                "confidence": {"level": "standard"},
                "ranking_score_raw": 0.2,
                "viral_probability_calibrated": 0.1,
            },
            "candidate": {
                "confidence": {"level": "low"},
                "ranking_score_raw": 0.4,
                "viral_probability_calibrated": 0.3,
            },
            "delta": {"ranking_score_raw": 0.2, "viral_probability_calibrated": 0.2},
        }


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
                "creative_battle",
                "recent_mcp_calls",
            } <= names
            result = await client.call_tool("model_status", {"sector": "automotivo"})
            assert result.structured_content["ready"] is False

    asyncio.run(scenario())


def test_creative_battle_compares_same_channel_versions_without_causal_claim() -> None:
    async def scenario() -> None:
        server = create_mcp_server(creative_battle_predictor=StubBattlePredictor())
        async with Client(server) as client:
            result = await client.call_tool(
                "creative_battle",
                {
                    "sector": "moda",
                    "channel_id": "channel-123",
                    "baseline": {"features": {"duration_seconds": 20.0}},
                    "candidate": {"features": {"duration_seconds": 35.0}},
                },
            )
        payload = result.structured_content
        assert payload["status"] == "ok"
        assert payload["winner"] == "candidate"
        assert payload["confidence_flag"] == "low"
        assert payload["changed_model_inputs"][0]["feature"] == "duration_seconds"
        assert "não causal" in payload["comparison_scope"]

    asyncio.run(scenario())


def test_default_creative_battle_does_not_load_the_retired_mixed_format_model() -> None:
    async def scenario() -> None:
        server = create_mcp_server()
        async with Client(server) as client:
            result = await client.call_tool(
                "creative_battle",
                {
                    "sector": "moda",
                    "channel_id": "channel-123",
                    "baseline": {"features": {"duration_seconds": 20.0}},
                    "candidate": {"features": {"duration_seconds": 35.0}},
                },
            )
        payload = result.structured_content
        assert payload["status"] == "model_not_ready"
        assert payload["winner"] == "undetermined"
        assert "Shorts-only" in payload["warnings"][0]

    asyncio.run(scenario())


def test_creative_battle_rejects_mixed_sector() -> None:
    async def scenario() -> None:
        server = create_mcp_server(creative_battle_predictor=StubBattlePredictor())
        async with Client(server) as client:
            result = await client.call_tool(
                "creative_battle",
                {
                    "sector": "moda",
                    "channel_id": "channel-123",
                    "baseline": {"features": {"sector": "beleza"}},
                    "candidate": {"features": {"duration_seconds": 35.0}},
                },
            )
        payload = result.structured_content
        assert payload["status"] == "invalid_input"
        assert "mesmo setor" in payload["comparison_scope"]

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
