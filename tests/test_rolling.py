from __future__ import annotations

import pandas as pd

from galaxies.agent.rolling import fold_aware_channels, rolling_calendar_splits


def test_rolling_calendar_splits_preserve_time_and_known_channels() -> None:
    rows = pd.DataFrame(
        {
            "video_id": [f"v{index}" for index in range(20)],
            "sector": ["moda"] * 20,
            "channel_id": ["a", "b"] * 10,
            "published_at": pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC"),
        }
    )
    splits = rolling_calendar_splits(rows, boundaries=(0.50, 0.75, 1.0))
    assert len(splits) == 2
    for split in splits:
        assert split.train["published_at"].max() < split.test["published_at"].min()
        assert set(split.test["channel_id"]).issubset(set(split.train["channel_id"]))


def test_rolling_calendar_splits_requires_valid_boundaries() -> None:
    rows = pd.DataFrame(
        {
            "sector": ["moda", "moda"],
            "channel_id": ["a", "a"],
            "published_at": ["2025-01-01", "2025-01-02"],
        }
    )
    try:
        rolling_calendar_splits(rows, boundaries=(0.5, 0.8))
    except ValueError as error:
        assert "terminar" in str(error)
    else:  # pragma: no cover - deixa a mensagem de falha explícita.
        raise AssertionError("A fronteira final inválida deveria falhar.")


def test_fold_aware_channels_do_not_mix_a_channel_across_windows() -> None:
    frame = pd.DataFrame({"fold": [1, 2], "channel_id": ["channel-a", "channel-a"]})
    result = fold_aware_channels(frame)
    assert result["channel_id"].tolist() == ["1::channel-a", "2::channel-a"]
