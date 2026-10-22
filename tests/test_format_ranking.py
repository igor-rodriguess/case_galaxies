from __future__ import annotations

import numpy as np
import pandas as pd

from galaxies.agent.format_ranking import (
    channel_level,
    format_as_of_labels,
    inner_calendar_folds,
    ranking_metrics,
    shorts_duration_limit,
    shorts_population,
    unit_counts,
    with_format_units,
)


def _rows(days: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sector in ("moda", "beleza"):
        for channel in ("a", "b"):
            for day in range(days):
                rows.append(
                    {
                        "video_id": f"{sector}-{channel}-{day}",
                        "sector": sector,
                        "channel_id": f"{sector}-{channel}",
                        "published_at": pd.Timestamp("2026-01-01", tz="UTC")
                        + pd.Timedelta(days=day),
                        "duration_seconds": 30.0 if day % 2 else 600.0,
                        "target_age_days": 300.0 - day,
                        "target_view_count": float(rng.integers(100, 10_000)),
                    }
                )
    return pd.DataFrame(rows)


def test_shorts_population_drops_long_and_unknown_duration() -> None:
    rows = _rows()
    rows.loc[0, "duration_seconds"] = np.nan
    shorts = shorts_population(rows)
    assert shorts["duration_seconds"].le(180).all()
    assert shorts["duration_seconds"].notna().all()
    assert len(shorts) == int((rows["duration_seconds"] <= 180).sum())


def test_units_keep_the_real_channel() -> None:
    units = with_format_units(shorts_population(_rows()))
    assert units["channel_id"].str.endswith("::curto").all()
    assert (units["channel_id"] == units["source_channel_id"] + "::curto").all()


def test_inner_folds_validate_after_their_own_training_window() -> None:
    units = with_format_units(shorts_population(_rows()))
    folds = inner_calendar_folds(units)
    assert len(folds) == 3
    for inner_train, inner_valid in folds:
        for sector in units["sector"].unique():
            train_dates = inner_train.loc[inner_train["sector"] == sector, "published_at"]
            valid_dates = inner_valid.loc[inner_valid["sector"] == sector, "published_at"]
            assert train_dates.max() < valid_dates.min()
    covered = pd.concat([valid for _, valid in folds]).index
    assert covered.is_unique


def test_labels_use_only_units_with_enough_training_rows() -> None:
    units = with_format_units(shorts_population(_rows()))
    early = units["published_at"] < pd.Timestamp("2026-01-31", tz="UTC")
    train, test = units.loc[early], units.loc[~early]
    train_small = train.loc[~((train["channel_id"] == "moda-a::curto") & (train.index % 4 != 1))]
    labelled_train, labelled_test = format_as_of_labels(train_small, test, min_unit_train_rows=10)
    assert "moda-a::curto" not in set(labelled_train["channel_id"])
    assert "moda-a::curto" not in set(labelled_test["channel_id"])
    assert set(labelled_test["viral"].unique()) <= {0, 1}


def test_channel_level_sums_units_before_metrics() -> None:
    frame = pd.DataFrame(
        {
            "channel_id": ["c::curto"] * 4 + ["c::longo"] * 2,
            "source_channel_id": ["c"] * 6,
            "sector": ["moda"] * 6,
            "viral": [1, 0, 0, 0, 1, 0],
            "score": [0.9, 0.1, 0.2, 0.3, 0.1, 0.8],
        }
    )
    channels = channel_level(unit_counts(frame, "score"))
    assert len(channels) == 1
    metrics = ranking_metrics(channels)
    assert metrics["pairs"] == 4
    assert metrics["auc_same_format"] == 0.75


def test_shorts_limit_is_the_project_wide_180_second_rule() -> None:
    published = pd.Series(pd.to_datetime(["2015-05-01", "2023-03-01", "2025-01-10"], utc=True))
    assert shorts_duration_limit(published).tolist() == [180.0, 180.0, 180.0]
    rows = pd.DataFrame({"published_at": published, "duration_seconds": [30.0, 120.0, 120.0]})
    assert shorts_population(rows).index.tolist() == [0, 1, 2]
