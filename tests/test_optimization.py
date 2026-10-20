from __future__ import annotations

import numpy as np
import pandas as pd

from galaxies.modeling.optimization import (
    _pipeline,
    expanding_window_splits,
    leakage_safe_config,
    learn_ensemble_weights,
    search_space,
)


def test_expanding_window_never_trains_on_the_future() -> None:
    folds = expanding_window_splits(30, 3)
    assert len(folds) == 3
    for train, validation in folds:
        assert train.max() < validation.min()


def test_leakage_safe_config_removes_channel_history() -> None:
    config = {
        "features": {
            "numeric": [
                "duration_seconds",
                "prior_views_median_observed",
                "channel_subscriber_count_at_prediction",
                "vlm_mean_confidence",
            ],
            "categorical": ["opening_type"],
        }
    }
    safe, excluded = leakage_safe_config(config)
    assert safe["features"]["numeric"] == [
        "duration_seconds",
        "vlm_mean_confidence",
    ]
    assert excluded == [
        "channel_subscriber_count_at_prediction",
        "prior_views_median_observed",
    ]


def test_ensemble_weights_are_non_negative_and_normalized() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.column_stack([target, target[::-1], target * 0.8])
    weights = learn_ensemble_weights(predictions, target)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_every_candidate_has_feature_and_hyperparameter_search() -> None:
    for candidate in (
        "linear_regression",
        "ridge",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "mlp",
    ):
        spaces = search_space(candidate)
        assert len(spaces) == 3
        assert all("preprocess__numeric__scaler" in space for space in spaces)
        assert any("select__percentile" in space for space in spaces)
        assert any("select__threshold" in space for space in spaces)
        assert all(
            any(key.startswith("model__") for key in space) for space in spaces
        )


def test_pipeline_removes_constant_encoded_columns() -> None:
    config = {
        "features": {
            "numeric": ["duration_seconds"],
            "categorical": ["opening_type"],
        }
    }
    features = pd.DataFrame(
        {
            "duration_seconds": [10.0, 20.0, 30.0, 40.0],
            "opening_type": ["same", "same", "same", "same"],
        }
    )
    pipeline = _pipeline(config, "ridge", 42)
    pipeline.fit(features, np.log1p([100, 200, 300, 400]))
    transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    retained = pipeline.named_steps["variance"].get_support()
    assert len(transformed_names) == 2
    assert retained.tolist() == [True, False]
