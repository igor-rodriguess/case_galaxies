"""Leitura e validação das configurações versionadas de cada setor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paths import MODELS_DIR

SECTORS = ("automotivo", "moda", "beleza")


class _ConfigModel(BaseModel):
    """Base estrita para detectar erros de digitação nos JSONs de configuração."""

    model_config = ConfigDict(extra="forbid")


class TargetConfig(_ConfigModel):
    column: str
    metric_name: str
    transform: Literal["log1p", "none"]
    horizon_days: int = Field(gt=0)
    requires_fixed_horizon: bool


class MinimumDataConfig(_ConfigModel):
    rows: int = Field(gt=1)
    channels: int = Field(gt=1)
    vlm_coverage: float = Field(ge=0, le=1)


class ValidationConfig(_ConfigModel):
    temporal_test_fraction: float = Field(gt=0, lt=1)
    channel_test_fraction: float = Field(gt=0, lt=1)
    minimum_test_rows: int = Field(gt=0)
    random_state: int


class FeaturesConfig(_ConfigModel):
    numeric: list[str]
    categorical: list[str]

    @model_validator(mode="after")
    def validate_feature_names(self) -> FeaturesConfig:
        if len(self.numeric) != len(set(self.numeric)):
            raise ValueError("A lista de features numéricas contém duplicatas.")
        if len(self.categorical) != len(set(self.categorical)):
            raise ValueError("A lista de features categóricas contém duplicatas.")
        duplicated = set(self.numeric) & set(self.categorical)
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(f"Features numéricas e categóricas sobrepostas: {names}")
        if not self.numeric and not self.categorical:
            raise ValueError("Ao menos uma feature deve ser configurada.")
        return self


class ClassificationConfig(_ConfigModel):
    enabled: bool
    label_column: str
    thresholds: dict[str, float] | None
    models: list[str]


class PredictionIntervalConfig(_ConfigModel):
    enabled: bool
    lower_quantile: float = Field(gt=0, lt=1)
    upper_quantile: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_order(self) -> PredictionIntervalConfig:
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile deve ser menor que upper_quantile.")
        return self


class PromotionConfig(_ConfigModel):
    require_temporal_baseline_win: bool = True
    require_channel_baseline_win: bool = True


class ServingConfig(_ConfigModel):
    max_missing_fraction: float = Field(default=0.5, ge=0, lt=1)


class SectorConfig(_ConfigModel):
    schema_version: str
    sector: str
    target: TargetConfig
    minimum_data: MinimumDataConfig
    validation: ValidationConfig
    features: FeaturesConfig
    regression_models: list[str] = Field(min_length=1)
    optional_regression_models: list[str] = Field(default_factory=list)
    classification: ClassificationConfig
    prediction_interval: PredictionIntervalConfig
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


def validate_sector(sector: str) -> str:
    if sector not in SECTORS:
        allowed = ", ".join(SECTORS)
        raise ValueError(f"Setor inválido: {sector!r}. Use um de: {allowed}.")
    return sector


def sector_config_path(sector: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / validate_sector(sector) / "config.json"


def load_sector_config(
    sector: str,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    path = sector_config_path(sector, models_dir)
    if not path.exists():
        raise FileNotFoundError(f"Configuração do setor não encontrada: {path}")
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido na configuração {path}: {exc.msg}.") from exc
    config = SectorConfig.model_validate(raw_config)
    if config.sector != sector:
        raise ValueError(f"A configuração {path} declara o setor {config.sector!r}.")
    return config.model_dump(mode="python")
