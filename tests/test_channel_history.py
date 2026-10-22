from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxies.agent.channel_history import known_channel_history_features
from galaxies.agent.protocol import temporal_as_of_labels


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
    result = known_channel_history_features(
        targets,
        history,
        embedding_column="embedding",
        allow_retrospective_outcome_labels=True,
    )

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
    result = known_channel_history_features(rows, rows, allow_retrospective_outcome_labels=True)
    assert result.loc[0, "known_channel_history_available"] == 0.0
    assert result.loc[0, "known_channel_format_viral_rate"] == pytest.approx(0.1)


def test_known_history_keeps_format_rate_distinct_from_duration_rate() -> None:
    history = pd.DataFrame(
        {
            "video_id": ["short-viral", "short-nonviral", "long-viral"],
            "channel_id": ["a", "a", "a"],
            "published_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "duration_seconds": [30, 120, 600],
            "viral": [1, 0, 1],
        }
    )
    target = pd.DataFrame(
        {
            "video_id": ["target"],
            "channel_id": ["a"],
            "published_at": ["2026-03-01"],
            "duration_seconds": [45],
            "viral": [0],
        }
    )
    result = known_channel_history_features(
        target, history, allow_retrospective_outcome_labels=True
    )
    # Formato short: 1 viral em 2. Faixa <=60s: 1 viral em 1.
    assert result.loc[0, "known_channel_format_viral_rate"] == pytest.approx((1 + 2) / 22)
    assert result.loc[0, "known_channel_duration_viral_rate"] == pytest.approx((1 + 2) / 21)


def test_safe_history_does_not_use_retroactive_viral_labels() -> None:
    """Sem snapshot disponível antes do alvo, o rótulo futuro não entra."""
    history = pd.DataFrame(
        {
            "video_id": ["old"],
            "channel_id": ["a"],
            "published_at": ["2026-01-01"],
            "target_observed_at": ["2026-04-01"],
            "duration_seconds": [30],
            "viral": [1],
        }
    )
    target = pd.DataFrame(
        {
            "video_id": ["target"],
            "channel_id": ["a"],
            "published_at": ["2026-03-01"],
            "duration_seconds": [30],
            "viral": [0],
        }
    )
    result = known_channel_history_features(
        target, history, outcome_observed_at_column="target_observed_at"
    )
    assert result.loc[0, "known_channel_history_available"] == 0.0
    assert result.loc[0, "known_channel_format_viral_rate"] == pytest.approx(0.1)


def test_temporal_labels_do_not_change_when_future_test_views_change() -> None:
    """O limiar do treino não pode ser recalibrado pelas views do período teste."""
    train = pd.DataFrame(
        {
            "video_id": ["a1", "a2", "a3", "a4"],
            "channel_id": ["a"] * 4,
            "sector": ["moda"] * 4,
            "target_view_count": [10, 20, 30, 40],
            "target_age_days": [30, 30, 30, 30],
        }
    )
    test = pd.DataFrame(
        {
            "video_id": ["future"],
            "channel_id": ["a"],
            "sector": ["moda"],
            "target_view_count": [50],
            "target_age_days": [30],
        }
    )
    labelled_train, labelled_test = temporal_as_of_labels(train, test)
    changed_test = test.assign(target_view_count=50_000)
    changed_train, changed_labelled_test = temporal_as_of_labels(train, changed_test)
    assert labelled_train["viral"].tolist() == changed_train["viral"].tolist()
    assert labelled_train["adjusted_performance"].tolist() == pytest.approx(
        changed_train["adjusted_performance"].tolist()
    )
    assert labelled_test["viral"].iloc[0] == changed_labelled_test["viral"].iloc[0]
