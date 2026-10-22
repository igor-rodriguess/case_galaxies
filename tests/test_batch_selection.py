import json
from datetime import UTC, datetime

import scripts.gemini_opening_batch as opening_batch
from scripts.gemini_opening_batch import balance_pending, existing_permission_denials


def test_balanced_selection_interleaves_sectors_and_channels() -> None:
    videos = [
        {"video_id": "a1", "sector": "automotivo", "channel_id": "a"},
        {"video_id": "a2", "sector": "automotivo", "channel_id": "a"},
        {"video_id": "a3", "sector": "automotivo", "channel_id": "b"},
        {"video_id": "m1", "sector": "moda", "channel_id": "m"},
        {"video_id": "m2", "sector": "moda", "channel_id": "n"},
        {"video_id": "b1", "sector": "beleza", "channel_id": "x"},
        {"video_id": "b2", "sector": "beleza", "channel_id": "y"},
    ]

    ordered = balance_pending(videos)

    assert [item["sector"] for item in ordered[:3]] == [
        "automotivo",
        "moda",
        "beleza",
    ]
    assert [item["video_id"] for item in ordered] == [
        "a1",
        "m1",
        "b1",
        "a3",
        "m2",
        "b2",
        "a2",
    ]


def test_existing_permission_denials_only_keeps_permanent_403(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(opening_batch, "PROJECT_ROOT", tmp_path)
    blocked_error = tmp_path / "run-a" / "videos" / "blocked" / "error.json"
    blocked_error.parent.mkdir(parents=True)
    blocked_error.write_text(
        json.dumps({"message": "HTTP 403: PERMISSION_DENIED"}),
        encoding="utf-8",
    )

    transient_error = tmp_path / "run-b" / "videos" / "retry" / "error.json"
    transient_error.parent.mkdir(parents=True)
    transient_error.write_text(
        json.dumps({"message": "HTTP 503: service unavailable"}),
        encoding="utf-8",
    )

    result = existing_permission_denials(tmp_path)

    assert set(result) == {"blocked"}
    assert result["blocked"]["error"].endswith(
        "run-a/videos/blocked/error.json"
    )


def test_execution_lock_is_exclusive_and_recovers_after_expiration(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "gemini.lock.json"
    monkeypatch.setattr(opening_batch, "COLLECTOR_LOCK_PATH", lock_path)
    now = datetime(2026, 10, 20, tzinfo=UTC)

    first_token, _ = opening_batch.acquire_execution_lock("first", now=now)
    second_token, active_lock = opening_batch.acquire_execution_lock("second", now=now)

    assert first_token is not None
    assert second_token is None
    assert active_lock is not None
    assert active_lock["owner"] == "first"

    later = datetime(2026, 10, 20, 1, tzinfo=UTC)
    recovered_token, recovered_lock = opening_batch.acquire_execution_lock("recovered", now=later)

    assert recovered_token is not None
    assert recovered_lock is not None
    assert recovered_lock["owner"] == "recovered"


def test_video_claim_is_exclusive_and_released_by_its_owner(tmp_path, monkeypatch) -> None:
    claims_dir = tmp_path / "claims"
    monkeypatch.setattr(opening_batch, "VIDEO_CLAIMS_DIR", claims_dir)
    now = datetime(2026, 10, 20, tzinfo=UTC)

    first_token, _ = opening_batch.claim_video("video-123", "batch-a", now=now)
    second_token, active_claim = opening_batch.claim_video("video-123", "batch-b", now=now)

    assert first_token is not None
    assert second_token is None
    assert active_claim is not None
    assert active_claim["owner"] == "batch-a"

    opening_batch.release_video_claim("video-123", first_token)
    third_token, _ = opening_batch.claim_video("video-123", "batch-c", now=now)

    assert third_token is not None
