from __future__ import annotations

import json
from pathlib import Path

from galaxies.dataset import build_feature_rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def youtube_document(video_id: str, published_at: str, views: int) -> dict:
    return {
        "schema_version": "0.1.0",
        "identity": {
            "video_id": video_id,
            "channel_id": "channel-1",
            "youtube_watch_url": f"https://youtube.test/watch?v={video_id}",
        },
        "classification": {"sector": "automotivo"},
        "publication": {
            "published_at": published_at,
            "title": "Teste",
            "description": "Descrição",
            "tags": ["a", "b"],
            "category_id": "2",
        },
        "content_details": {"duration_seconds": 30, "definition": "hd"},
        "metrics_snapshot": {
            "observed_at": "2026-10-12T00:00:00Z",
            "view_count": views,
            "like_count": views // 10,
            "comment_count": views // 100,
        },
        "channel_snapshot": {
            "subscriber_count_observed_at_collection": 1000,
            "video_count_observed_at_collection": 50,
        },
        "provenance": {"collected_at": "2026-10-12T00:00:00Z"},
    }


def test_feature_map_joins_vlm_and_uses_only_prior_posts(tmp_path: Path) -> None:
    youtube_run = tmp_path / "youtube-run"
    video_dir = youtube_run / "curated" / "videos" / "automotivo" / "canal"
    write_json(
        video_dir / "old.json",
        youtube_document("old", "2026-01-01T12:00:00Z", 100),
    )
    write_json(
        video_dir / "middle.json",
        youtube_document("middle", "2026-02-01T12:00:00Z", 200),
    )
    write_json(
        video_dir / "new.json",
        youtube_document("new", "2026-03-01T12:00:00Z", 400),
    )
    feature_dir = tmp_path / "feature-runs"
    write_json(
        feature_dir / "run" / "videos" / "middle" / "opening_vlm.json",
        {
            "identity": {"video_id": "middle"},
            "provenance": {
                "completed_at": "2026-10-12T01:00:00Z",
                "extractor": {"prompt_version": "test-v1"},
                "clip": {"start_seconds": 0, "end_seconds": 3, "requested_fps": 4},
            },
            "analysis": {
                "visual": {
                    "people": {
                        "present": True,
                        "count_estimate": 1,
                        "faces_visible": [{"timestamp_seconds": 0.0}],
                    },
                    "on_screen_text": [{"text": "oi"}],
                    "scene_changes": [],
                },
                "audio": {
                    "speech_present": True,
                    "speech_start_seconds": 0.5,
                    "transcript_excerpt": "Você quer conferir 3 carros? Compre agora!",
                    "sound_effects": [],
                },
                "interpretation": {
                    "opening_type": {"label": "person_speaking"},
                    "product_prominence": {"label": "none"},
                    "call_to_action": {"present": False},
                },
                "evidence": [],
                "uncertainties": [],
            },
        },
    )

    rows, manifest = build_feature_rows(youtube_run, feature_dir)
    by_id = {row["video_id"]: row for row in rows}

    assert manifest["record_count"] == 3
    assert by_id["old"]["prior_video_count"] == 0
    assert by_id["middle"]["prior_video_count"] == 1
    assert by_id["middle"]["prior_views_median_observed"] == 100
    assert by_id["new"]["prior_views_median_observed"] == 150
    assert by_id["middle"]["vlm_available"] is True
    assert by_id["middle"]["faces_visible"] is True
    assert by_id["middle"]["opening_type"] == "person_speaking"
    assert by_id["middle"]["transcript_available"] is True
    assert by_id["middle"]["transcript_excerpt_word_count"] == 7
    assert by_id["middle"]["transcript_number_token_count"] == 1
    assert by_id["middle"]["transcript_second_person_cue"] is True
    assert by_id["middle"]["transcript_call_to_action_cue"] is True
    assert by_id["middle"]["transcript_question_mark_count"] == 1
    assert by_id["middle"]["transcript_exclamation_mark_count"] == 1
    assert by_id["old"]["target_is_fixed_horizon"] is False
    assert by_id["new"]["channel_history_temporally_valid"] is False
