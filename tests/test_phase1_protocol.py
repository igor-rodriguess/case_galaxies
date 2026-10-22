from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxies.agent.evaluation import calendar_split
from galaxies.modeling import frozen_evaluation
from galaxies.modeling.frozen_evaluation import causal_relative_targets, evaluate_outer_split
from galaxies.prospective import (
    PROSPECTIVE_START,
    assert_development_only,
    exclude_prospective,
)


def _ranking_rows() -> pd.DataFrame:
    rows = []
    for channel, offset in (("a", 0), ("b", 40)):
        for day in range(20):
            rows.append(
                {
                    "video_id": f"{channel}{day}",
                    "channel_id": channel,
                    "sector": "moda",
                    "published_at": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=offset + day),
                }
            )
    return pd.DataFrame(rows)


def test_calendar_split_never_trains_after_the_sector_cutoff() -> None:
    train, test, cutoffs = calendar_split(_ranking_rows())
    cutoff = pd.Timestamp(cutoffs["moda"])
    assert train["published_at"].max() < cutoff
    assert test["published_at"].min() >= cutoff
    assert set(test["channel_id"]) <= set(train["channel_id"])


def test_prospective_rows_are_excluded_and_guarded() -> None:
    frame = pd.DataFrame(
        {"published_at": [PROSPECTIVE_START - pd.Timedelta(days=1), PROSPECTIVE_START]}
    )
    kept = exclude_prospective(frame)
    assert len(kept) == 1
    assert_development_only(kept)
    with pytest.raises(ValueError, match="prospectivo"):
        assert_development_only(frame)


def _target_rows() -> pd.DataFrame:
    rows = []
    for channel, views in (("train", 100), ("reserved", 1_000_000)):
        for day in range(8):
            rows.append(
                {
                    "video_id": f"{channel}{day}",
                    "sector": "moda",
                    "channel_id": channel,
                    "published_at": f"2026-01-{day + 1:02d}",
                    "target_age_days": 30.0 + day,
                    "target_view_count": views + day,
                }
            )
    return pd.DataFrame(rows)


def test_reserved_channels_do_not_shape_other_channels_targets() -> None:
    rows = _target_rows()
    train_rows = rows.loc[rows["channel_id"] == "train"]
    alone = causal_relative_targets(train_rows, train_rows)
    excluded = causal_relative_targets(rows, train_rows, {"reserved"})
    included = causal_relative_targets(rows, train_rows)
    pd.testing.assert_series_equal(
        alone["residual_age_channel"].reset_index(drop=True),
        excluded["residual_age_channel"].reset_index(drop=True),
    )
    both = excluded["residual_age_channel"].notna() & included["residual_age_channel"].notna()
    assert both.any()
    assert not np.allclose(
        excluded.loc[both, "residual_age_channel"], included.loc[both, "residual_age_channel"]
    )


def test_reported_winner_is_the_internal_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    rows = 90
    frame = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(rows)],
            "sector": "moda",
            "channel_id": [f"c{i % 6}" for i in range(rows)],
            "published_at": pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC"),
            "duration_seconds": rng.uniform(10, 600, rows),
        }
    )
    target = rng.normal(size=rows)
    recipes = frozen_evaluation._recipes(2026)
    forced = next(recipe for recipe in recipes if recipe.name == "ridge_no_selection")
    monkeypatch.setattr(
        frozen_evaluation,
        "prepare_feature_frame",
        lambda frame, group, train: (frame[["duration_seconds"]], ["duration_seconds"], []),
    )
    monkeypatch.setattr(
        frozen_evaluation,
        "_select_recipe",
        lambda *args, **kwargs: (forced, [{"recipe": forced.name, "status": "ok"}]),
    )
    train, test = np.arange(70), np.arange(70, rows)
    result = evaluate_outer_split(frame, target, train, test, "metadata")
    assert result["winner"] == "ridge_no_selection"
    assert result["conformal"]["recipe"] == "ridge_no_selection"
    flagged = [row["model"] for row in result["candidates"] if row.get("selected_internally")]
    assert flagged == ["ridge_no_selection"]
    assert not any(row["used_for_selection"] for row in result["candidates"])
