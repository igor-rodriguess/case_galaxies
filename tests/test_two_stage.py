import numpy as np
import pandas as pd
import pytest

from galaxies.agent.protocol import build_dataset
from galaxies.agent.two_stage import (
    base_spec,
    cross_fitted_base_scores,
    fit_fusion_candidate,
    fusion_candidates,
)


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for sector in ("automotivo", "moda", "beleza"):
        for channel_number in range(4):
            channel = f"{sector}-{channel_number}"
            for index in range(24):
                quality = rng.normal()
                age = float(rng.uniform(35, 300))
                rows.append(
                    {
                        "video_id": f"{channel}-{index}",
                        "channel_id": channel,
                        "sector": sector,
                        "published_at": pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(days=index),
                        "target_age_days": age,
                        "target_view_count": int(np.exp(8 + quality + 0.2 * np.log1p(age))),
                        "duration_seconds": float(rng.choice([45, 90, 600])),
                        "title_character_count": int(rng.integers(10, 80)),
                        "description_character_count": int(rng.integers(0, 300)),
                        "tag_count": int(rng.integers(0, 10)),
                        "publish_hour_utc": int(rng.integers(0, 24)),
                        "publish_weekday_utc": int(rng.integers(0, 7)),
                        "category_id": "22",
                        "days_since_previous_upload": float(rng.uniform(1, 7)),
                        "prior_upload_interval_median_days": 3.0,
                        "vlm_available": index % 2 == 0,
                        "vlm_interval_end_seconds": 3.0,
                        "people_present": index % 3 == 0,
                        "speech_present": index % 2 == 0,
                        "scene_change_count": index % 4,
                    }
                )
    return build_dataset(pd.DataFrame(rows), tmp_path / "synthetic.jsonl", "synthetic")


def test_cross_fitted_base_scores_cover_every_target_without_nan(dataset) -> None:
    full = dataset.evaluable("full")
    vlm = dataset.evaluable("vlm")
    scores = cross_fitted_base_scores(full, vlm, spec=base_spec(), seed=2026, folds=3)
    assert scores.index.equals(vlm.index)
    assert np.isfinite(scores).all()


def test_fusion_candidates_fit_on_cross_fitted_score(dataset) -> None:
    rows = dataset.evaluable("vlm")
    train = rows.iloc[:-15]
    test = rows.iloc[-15:]
    train_score = cross_fitted_base_scores(
        dataset.evaluable("full"), train, spec=base_spec(), seed=2026, folds=3
    )
    test_score = train_score.iloc[: len(test)].copy()
    test_score.index = test.index
    for candidate in fusion_candidates():
        values = fit_fusion_candidate(
            candidate,
            train,
            train_score,
            test,
            test_score,
            seed=2026,
        )
        assert len(values) == len(test)
        assert np.isfinite(values).all()
