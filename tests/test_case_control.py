from __future__ import annotations

from datetime import UTC, datetime, timedelta

from galaxies.case_control import build_case_control_queue, parse_duration_seconds


def _record(video_id: str, channel: str, days_ago: int, views: int, duration: str = "PT30S") -> dict:
    collected = datetime(2026, 10, 18, tzinfo=UTC)
    return {
        "video_id": video_id,
        "sector": "moda",
        "channel_id": channel,
        "channel_title": channel,
        "published_at": (collected - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z"),
        "collected_at": collected.isoformat().replace("+00:00", "Z"),
        "duration": duration,
        "view_count": str(views),
    }


def test_parse_duration_seconds() -> None:
    assert parse_duration_seconds("PT1M2.5S") == 62.5
    assert parse_duration_seconds("PT2H") == 7200
    assert parse_duration_seconds("invalid") is None


def test_case_control_keeps_same_channel_format_and_never_reuses_controls() -> None:
    records = []
    for channel in ("a", "b"):
        for index in range(30):
            records.append(_record(f"{channel}-{index}", channel, 40 + index, 10 + index))
        records.append(_record(f"{channel}-hit", channel, 80, 1_000_000))
        records.append(_record(f"{channel}-long", channel, 80, 20, "PT10M"))
    queue, detail = build_case_control_queue(records, analyzed_video_ids=set(), controls_per_case=3)
    assert detail["summary"]["cases_pending"] >= 2
    controls = [item["video_id"] for item in queue if item["role"] == "control"]
    assert len(controls) == len(set(controls))
    for group in detail["groups"]:
        assert all(control["channel_id"] == group["channel_id"] for control in group["controls"])
        assert all(control["format"] == group["format"] for control in group["controls"])
        assert group["case"]["selection_audit"]["observed_label"] == "viral"
