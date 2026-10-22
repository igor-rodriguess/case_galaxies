from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from galaxies.modeling.frozen_evaluation import causal_relative_targets, target_diagnostics


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ["a", "b", "c", "future"],
            "sector": ["moda"] * 4,
            "channel_id": ["channel"] * 4,
            "published_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-12-01"],
            "target_age_days": [10.0] * 4,
            "target_view_count": [100, 200, 300, 9_999_999],
        }
    )


def test_causal_target_does_not_use_future_rows() -> None:
    rows = _rows()
    first = causal_relative_targets(rows.iloc[:3], rows.iloc[:3])
    with_future = causal_relative_targets(rows, rows.iloc[:3])

    assert_series_equal(first["residual_age_channel"], with_future["residual_age_channel"])
    assert_series_equal(first["smoothed_ratio"], with_future["smoothed_ratio"])


def test_target_diagnostics_prefers_low_age_correlation() -> None:
    frame = pd.DataFrame(
        {
            "target_age_days": [1, 2, 3, 4],
            "residual_age_channel": [0.1, 0.0, -0.1, 0.0],
            "smoothed_ratio": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = target_diagnostics(frame)
    assert result.iloc[0]["definition"] == "residual_age_channel"
