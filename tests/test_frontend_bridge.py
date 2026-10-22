from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from galaxies.frontend_bridge import McpGateway, load_artifacts, make_handler
from galaxies.mcp_server import create_mcp_server
from galaxies.observability import McpTraceStore


@pytest.fixture()
def gateway(tmp_path: Path) -> McpGateway:
    return McpGateway(create_mcp_server(trace_store=McpTraceStore(tmp_path)))


def test_tool_outside_interface_is_rejected(gateway: McpGateway) -> None:
    with pytest.raises(PermissionError):
        gateway.call("get_video_feature_map", {"video_id": "x"})


def test_call_goes_through_mcp_and_returns_its_trace(gateway: McpGateway) -> None:
    response = gateway.call("model_status", {"sector": "moda"})
    assert response["transport"] == "mcp-in-memory"
    assert response["is_error"] is False
    assert response["result"]["sector"] == "moda"
    assert response["trace"]["tool_name"] == "model_status"
    assert response["trace"]["input_summary"] == {"sector": "moda"}


def test_mcp_validation_error_is_reported_not_hidden(gateway: McpGateway) -> None:
    response = gateway.call(
        "creative_battle",
        {
            "sector": "moda",
            "channel_id": "c",
            "baseline": {"features": {"tags": ["lista"]}},
            "candidate": {"features": {"duration_seconds": 10}},
        },
    )
    assert response["is_error"] is True
    assert response["result"] is None
    assert "validation" in response["error_text"]
    assert response["trace"] is None


def test_missing_artifacts_are_unavailable(tmp_path: Path) -> None:
    artifacts = load_artifacts(tmp_path)
    run = next(iter(artifacts["rolling"].values()))
    assert run["report"]["status"] == "unavailable"
    assert "data" not in run["report"]
    assert artifacts["battle_model"]["report"]["status"] == "unavailable"


def _serve(gateway: McpGateway, tmp_path: Path) -> tuple[ThreadingHTTPServer, str]:
    (tmp_path / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    handler = make_handler(gateway, artifacts=lambda: {"ok": True}, static_root=lambda: tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_http_routes_and_guards(gateway: McpGateway, tmp_path: Path) -> None:
    server, base = _serve(gateway, tmp_path)
    try:
        with urlopen(f"{base}/api/health") as response:
            assert json.loads(response.read())["status"] == "ok"
        with pytest.raises(HTTPError) as figure:
            urlopen(
                f"{base}/api/figures/known-channel-rolling-fold-aware-20261021/..%2F..%2Fsecret.png"
            )
        assert figure.value.code == 404
        request = Request(
            f"{base}/api/mcp/call",
            data=json.dumps({"tool": "predict_performance_x", "arguments": {}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as forbidden:
            urlopen(request)
        assert forbidden.value.code == 403
        with urlopen(f"{base}/qualquer/rota") as response:
            assert b"<title>t</title>" in response.read()
    finally:
        server.shutdown()
