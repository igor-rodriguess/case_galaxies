from scripts.gemini_transcription_pilot import validate_transcript


def test_validate_transcript_accepts_bounded_segments() -> None:
    analysis = {
        "speech_present": True,
        "segments": [
            {
                "start_seconds": 0.5,
                "end_seconds": 2.1,
                "text": "Olá, tudo bem?",
            }
        ],
    }

    assert validate_transcript(analysis, start_seconds=0, end_seconds=15) == []


def test_validate_transcript_rejects_segment_outside_requested_interval() -> None:
    analysis = {
        "speech_present": True,
        "segments": [
            {
                "start_seconds": 14.0,
                "end_seconds": 16.0,
                "text": "fora do trecho",
            }
        ],
    }

    errors = validate_transcript(analysis, start_seconds=0, end_seconds=15)

    assert errors == ["segments[0] está fora do intervalo solicitado"]
