from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from galaxies.config import load_sector_config
from galaxies.prediction import ModelRepository
from galaxies.schemas import PredictionRequest, Sector


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_dir = tmp_path / "models" / "automotivo"
    config_dir.mkdir(parents=True)
    config = {
        "schema_version": "0.1.0",
        "sector": "automotivo",
        "target": {
            "column": "target_view_count",
            "metric_name": "view_count",
            "transform": "log1p",
            "horizon_days": 30,
            "requires_fixed_horizon": True,
        },
        "minimum_data": {"rows": 10, "channels": 2, "vlm_coverage": 0.5},
        "validation": {
            "temporal_test_fraction": 0.2,
            "channel_test_fraction": 0.2,
            "minimum_test_rows": 2,
            "random_state": 42,
        },
        "features": {"numeric": ["duration_seconds"], "categorical": []},
        "regression_models": ["linear_regression"],
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
        "unexpected": "typo",
    }
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_sector_config("automotivo", tmp_path / "models")


def test_prediction_request_rejects_empty_features() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(sector=Sector.BELEZA, features={})


def test_model_repository_rejects_pointer_outside_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "models" / "automotivo" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "latest.json").write_text(
        json.dumps(
            {
                "run_id": "unsafe",
                "bundle": "../../outside.joblib",
                "manifest": "../../manifest.json",
                "metrics": "../../metrics.json",
                "promotable": True,
            }
        ),
        encoding="utf-8",
    )

    status = ModelRepository(tmp_path / "models").status("automotivo")

    assert status.ready is False
    assert "fora de artifacts" in status.blocking_reasons[0]
