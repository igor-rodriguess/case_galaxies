from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxies.agent.channel_history import known_channel_history_features


def test_known_history_uses_only_same_channel_and_30_day_old_rows() -> None:
    history = pd.DataFrame(
        {
            "video_id": ["old-viral", "recent-nonviral", "other-channel"],
            "channel_id": ["a", "a", "b"],
            "published_at": ["2026-01-01", "2026-02-20", "2026-01-01"],
            "duration_seconds": [30, 30, 30],
            "viral": [1, 0, 0],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        }
    )
    targets = pd.DataFrame(
        {
            "video_id": ["target"],
            "channel_id": ["a"],
            "published_at": ["2026-03-01"],
            "duration_seconds": [30],
            "viral": [0],
            "embedding": [np.array([1.0, 0.0])],
        }
    )
    result = known_channel_history_features(targets, history, embedding_column="embedding")

    assert result.loc[0, "known_channel_history_available"] == 1.0
    assert result.loc[0, "known_channel_format_viral_rate"] == pytest.approx((1 + 20 * 0.1) / 21)
    assert pd.isna(result.loc[0, "known_channel_similarity_margin"])


def test_known_history_has_prior_when_no_eligible_past_video() -> None:
    rows = pd.DataFrame(
        {
            "video_id": ["current"],
            "channel_id": ["a"],
            "published_at": ["2026-01-01"],
            "duration_seconds": [60],
            "viral": [0],
        }
    )
    result = known_channel_history_features(rows, rows)
    assert result.loc[0, "known_channel_history_available"] == 0.0
    assert result.loc[0, "known_channel_format_viral_rate"] == pytest.approx(0.1)
