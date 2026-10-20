from __future__ import annotations

import pytest

from scripts.smolvlm_opening_pilot import (
    extract_json_object,
    synthetic_opening_frames,
    validate_analysis_shape,
    validate_compact_shape,
)


def valid_analysis() -> dict:
    return {
        "visual": {},
        "audio": {"analyzed": False},
        "interpretation": {"opening_type": "product_reveal"},
        "evidence": [],
        "uncertainties": [],
    }


def test_synthetic_opening_has_three_seconds_at_four_fps() -> None:
    frames, timestamps = synthetic_opening_frames(4)

    assert len(frames) == 12
    assert timestamps[0] == 0
    assert timestamps[-1] == 2.75
    assert all(frame.size == (384, 216) for frame in frames)


def test_extract_json_accepts_markdown_fence() -> None:
    parsed = extract_json_object('```json\n{"visual": {}}\n```')

    assert parsed == {"visual": {}}


def test_extract_json_rejects_non_json_response() -> None:
    with pytest.raises(ValueError, match="objeto JSON"):
        extract_json_object("não consigo analisar")


def test_minimum_shape_accepts_expected_contract() -> None:
    assert validate_analysis_shape(valid_analysis()) == []


def test_minimum_shape_rejects_audio_hallucination() -> None:
    payload = valid_analysis()
    payload["audio"] = {"analyzed": True, "transcript": "inventado"}

    errors = validate_analysis_shape(payload)

    assert "audio deve registrar exatamente que não foi analisado" in errors


def test_compact_shape_requires_all_fields() -> None:
    errors = validate_compact_shape({"visible_text": []})

    assert errors
    assert "cta_text" in errors[0]
