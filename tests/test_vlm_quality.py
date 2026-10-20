from __future__ import annotations

from galaxies.vlm_quality import audit_result


def _complete() -> dict:
    return {
        "identity": {"video_id": "video"},
        "provenance": {"extractor": {"prompt_version": "galaxies-opening-v0.2.0"}, "clip": {"start_seconds": 0, "end_seconds": 3}},
        "validation": {"status": "complete"},
        "analysis": {
            "analysis_scope": {},
            "visual": {"first_visible_element": {}, "people": {}, "product_or_object_in_focus": {}, "on_screen_text": [{"timestamp_seconds": 1}], "setting": {}, "shot_and_camera": {"framing": "wide", "camera_movement": "static"}, "scene_changes": [], "motion_intensity": {"label": "low"}},
            "audio": {"speech_present": False, "transcript_excerpt": None, "language": None, "music_present": False, "sound_effects": []},
            "interpretation": {"opening_type": {"label": "other"}, "product_prominence": {"label": "none"}, "call_to_action": {"present": False}, "sector_specific": {}},
            "evidence": [{"timestamp_seconds": 2}],
            "uncertainties": [],
        },
    }


def test_audit_accepts_complete_taxonomy_and_timestamps() -> None:
    assert audit_result(_complete()) == []


def test_audit_reports_taxonomy_and_timestamp_problems() -> None:
    result = _complete()
    result["analysis"]["visual"]["shot_and_camera"]["framing"] = "super close"
    result["analysis"]["evidence"][0]["timestamp_seconds"] = 4
    kinds = {issue["kind"] for issue in audit_result(result)}
    assert kinds == {"taxonomy_drift", "timestamp_outside_clip"}
