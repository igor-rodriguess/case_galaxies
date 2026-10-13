from scripts.gemini_opening_batch import balance_pending


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
