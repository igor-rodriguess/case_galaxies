from __future__ import annotations

import numpy as np
import pandas as pd

from galaxies.agent.temporal_knn import temporal_other_channel_knn


def test_knn_excludes_same_channel_and_future_candidates() -> None:
    targets = pd.DataFrame(
        {
            "video_id": ["target"],
            "channel_id": ["a"],
            "published_at": ["2026-04-01T00:00:00Z"],
            "embedding": [np.array([1.0, 0.0])],
        }
    )
    candidates = pd.DataFrame(
        {
            "video_id": ["same-channel", "past-other", "future-other"],
            "channel_id": ["a", "b", "c"],
            "published_at": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-03-20T00:00:00Z"],
            "embedding": [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])],
            "outcome": [1.0, 0.25, 0.9],
        }
    )

    result = temporal_other_channel_knn(
        targets, candidates, embedding_column="embedding", outcome_column="outcome", k=10
    )

    assert result.loc[0, "neighbor_count"] == 1
    assert result.loc[0, "neighbor_outcome"] == 0.25
