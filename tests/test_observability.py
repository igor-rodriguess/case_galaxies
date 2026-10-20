from __future__ import annotations

from galaxies.observability import McpTraceStore


def test_trace_store_persists_sanitized_events(tmp_path) -> None:
    store = McpTraceStore(tmp_path)

    event = store.record(
        tool_name="predict_performance",
        input_summary={"sector": "moda", "feature_count": 12},
        output_summary={"status": "model_not_ready"},
    )

    assert event["tool_name"] == "predict_performance"
    assert store.recent() == [event]
