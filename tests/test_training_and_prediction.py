from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from galaxies.dataset import write_jsonl
from galaxies.modeling import train_sector_models
from galaxies.prediction import PredictionService
from galaxies.schemas import PredictionRequest, Sector


def make_config() -> dict:
    return {
        "schema_version": "0.1.0",
        "sector": "automotivo",
        "target": {
            "column": "target_view_count",
            "metric_name": "view_count",
            "transform": "log1p",
            "horizon_days": 30,
            "requires_fixed_horizon": True,
        },
        "minimum_data": {"rows": 40, "channels": 3, "vlm_coverage": 0.5},
        "validation": {
            "temporal_test_fraction": 0.2,
            "channel_test_fraction": 0.25,
            "minimum_test_rows": 8,
            "random_state": 42,
        },
        "features": {
            "numeric": ["duration_seconds", "prior_views_median_observed"],
            "categorical": ["opening_type"],
        },
        "regression_models": [
            "dummy_median",
            "linear_regression",
            "gradient_boosting",
        ],
        "optional_regression_models": [],
        "classification": {
            "enabled": False,
            "label_column": "target_performance_band",
            "thresholds": None,
            "models": ["logistic_regression"],
        },
        "prediction_interval": {
            "enabled": True,
            "lower_quantile": 0.1,
            "upper_quantile": 0.9,
        },
    }


def test_training_promotes_valid_model_and_service_predicts(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    config_path = models_dir / "automotivo" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(make_config()), encoding="utf-8")

    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    openings = ["question", "demonstration", "person_speaking"]
    for index in range(80):
        duration = 10 + index % 45
        prior = 1000 + (index % 20) * 50
        views = int(500 + duration * 80 + prior * 1.5)
        rows.append(
            {
                "video_id": f"v-{index}",
                "sector": "automotivo",
                "channel_id": f"channel-{index % 4}",
                "published_at": (start + timedelta(days=index)).isoformat(),
                "target_view_count": views,
                "target_is_fixed_horizon": True,
                "vlm_available": True,
                "prior_video_count": index // 4,
                "channel_history_temporally_valid": True,
                "duration_seconds": duration,
                "prior_views_median_observed": prior,
                "opening_type": openings[index % len(openings)],
            }
        )
    dataset = tmp_path / "feature_map.jsonl"
    write_jsonl(dataset, rows)

    result = train_sector_models(dataset, "automotivo", models_dir=models_dir)

    assert result["manifest"]["promotable"] is True
    pointer_path = models_dir / "automotivo" / "artifacts" / "latest.json"
    assert pointer_path.exists()
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert not Path(pointer["bundle"]).is_absolute()

    service = PredictionService(models_dir)
    response = service.predict(
        PredictionRequest(
            sector=Sector.AUTOMOTIVO,
            video_id="new-video",
            features={
                "duration_seconds": 30,
                "prior_views_median_observed": 1500,
                "opening_type": "demonstration",
            },
        )
    )
    assert response.status == "ok"
    assert response.predicted_metric is not None
    assert response.predicted_metric > 0
    assert response.interval_low is not None
    assert response.interval_high is not None

    invalid = service.predict(
        PredictionRequest(
            sector=Sector.AUTOMOTIVO,
            features={"feature_that_does_not_exist": 1},
        )
    )
    assert invalid.status == "invalid_input"


def test_logistic_classification_pipeline_is_ready_for_future_bands(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    config = make_config()
    config["classification"]["enabled"] = True
    config["classification"]["thresholds"] = {"normal_max": 30}
    config_path = models_dir / "automotivo" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(80):
        duration = 10 + index % 45
        rows.append(
            {
                "video_id": f"class-{index}",
                "sector": "automotivo",
                "channel_id": f"channel-{index % 4}",
                "published_at": (start + timedelta(days=index)).isoformat(),
                "target_performance_band": "viral" if duration > 30 else "normal",
                "duration_seconds": duration,
                "prior_views_median_observed": 1000 + index,
                "opening_type": "demonstration",
            }
        )
    dataset = tmp_path / "classification_map.jsonl"
    write_jsonl(dataset, rows)

    result = train_sector_models(
        dataset,
        "automotivo",
        task="classification",
        models_dir=models_dir,
    )

    assert result["manifest"]["promotable"] is True
    assert result["manifest"]["winner"] == "logistic_regression"
