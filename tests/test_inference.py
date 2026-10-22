from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from galaxies.modeling.inference import FrozenBaselinePredictor, prepare_prepublication_features
from galaxies.modeling.preprocessing import make_pipeline


@pytest.fixture
def frozen_run(tmp_path):
    frame = pd.DataFrame(
        {
            "duration_seconds": [15.0, 60.0, 240.0, 600.0],
            "category_id": ["1", "1", "22", "22"],
            "sector": ["moda", "moda", "beleza", "beleza"],
        }
    )
    pipeline = make_pipeline(
        {
            "features": {
                "numeric": ["duration_seconds"],
                "categorical": ["category_id", "sector"],
            }
        },
        LogisticRegression(max_iter=1000),
    )
    pipeline.fit(frame, [0, 0, 1, 1])
    calibrator = LogisticRegression(max_iter=1000).fit([[0.1], [0.2], [0.8], [0.9]], [0, 0, 1, 1])
    joblib.dump(pipeline, tmp_path / "baseline_pipeline.joblib")
    joblib.dump(calibrator, tmp_path / "probability_calibrator.joblib")
    (tmp_path / "confidence_profile.json").write_text(
        json.dumps(
            {
                "numeric_training_range": {
                    "duration_seconds": {"p01": 15.0, "p99": 600.0, "missing_rate": 0.0}
                },
                "categorical_training_values": {
                    "category_id": ["1", "22"],
                    "sector": ["moda", "beleza"],
                },
                "low_confidence_rule": "test",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_prepare_derives_prepublication_fields() -> None:
    prepared = prepare_prepublication_features(
        {
            "duration_seconds": 30,
            "title": "Teste",
            "description": "Descrição",
            "tags": ["a", "b"],
            "published_at": "2026-10-20T13:00:00Z",
        }
    )
    assert prepared["log_duration"] == pytest.approx(np.log1p(30))
    assert prepared["is_short"] == 1.0
    assert prepared["title_character_count"] == 5
    assert prepared["description_character_count"] == 9
    assert prepared["tag_count"] == 2
    assert prepared["publish_hour_utc"] == 13


def test_frozen_predictor_returns_calibrated_prediction_and_confidence(frozen_run) -> None:
    result = FrozenBaselinePredictor.from_run(frozen_run).predict(
        {"duration_seconds": 60.0, "category_id": "1", "sector": "moda", "unused": "ignored"}
    )
    assert 0 <= result["ranking_score_raw"] <= 1
    assert 0 <= result["viral_probability_calibrated"] <= 1
    assert result["confidence"]["level"] == "standard"
    assert result["ignored_payload_fields"] == ["unused"]
    assert result["top_local_contributions"]


def test_frozen_predictor_marks_out_of_distribution_values(frozen_run) -> None:
    result = FrozenBaselinePredictor.from_run(frozen_run).predict(
        {"duration_seconds": 1000.0, "category_id": "99", "sector": "games"}
    )
    assert result["confidence"]["level"] == "low"
    assert {alert["type"] for alert in result["confidence"]["alerts"]} >= {
        "outside_training_range",
        "unseen_category",
    }


def test_frozen_predictor_compares_scenarios_without_claiming_causality(frozen_run) -> None:
    predictor = FrozenBaselinePredictor.from_run(frozen_run)
    result = predictor.compare(
        {"duration_seconds": 60.0, "category_id": "1", "sector": "moda"},
        {"duration_seconds": 240.0, "category_id": "22", "sector": "beleza"},
    )
    assert {item["feature"] for item in result["changed_model_inputs"]} == {
        "duration_seconds",
        "category_id",
        "sector",
    }
    assert "não é evidência causal" in result["comparison_scope"]
    assert np.isfinite(result["delta"]["ranking_score_raw"])
