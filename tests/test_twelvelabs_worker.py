from __future__ import annotations

import json

from scripts.maintain_twelvelabs_workers import (
    is_deterministic_download_failure,
    option_value,
    pending_ids,
    record_source_block,
    record_source_issue,
    worker_state,
)


def test_pending_ids_preserves_queue_order_and_excludes_terminal_or_active_items() -> None:
    result = pending_ids(
        ["a", "b", "c", "d"],
        completed={"a"},
        blocked={"b"},
        active={"c"},
    )
    assert result == ["d"]


def test_option_value_accepts_a_youtube_id_that_starts_with_hyphen() -> None:
    assert option_value("--video-id", "-0kEAmYHXdo") == "--video-id=-0kEAmYHXdo"


def test_source_403_becomes_a_permanent_block(tmp_path) -> None:
    record_source_block(tmp_path, "video", "download failed: HTTP Error 403: Forbidden")
    active, blocked = worker_state(tmp_path)
    assert active == []
    assert blocked == {"video"}
    saved = json.loads((tmp_path / "videos" / "video" / "error.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "source_permission_denied"


def test_repeated_missing_source_file_becomes_terminal_for_the_worker(tmp_path) -> None:
    message = "Esperava um único arquivo fonte em media, encontrei: []"
    assert is_deterministic_download_failure(message)
    record_source_issue(tmp_path, "unavailable", "source_download_unavailable", message)
    _, blocked = worker_state(tmp_path)
    assert blocked == {"unavailable"}
