"""Contratos usados pela função de predição e pelo servidor MCP."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FeatureValue = str | int | float | bool | None


class ApiModel(BaseModel):
    """Contrato estrito e serializável usado na fronteira do MCP."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class Sector(StrEnum):
    AUTOMOTIVO = "automotivo"
    MODA = "moda"
    BELEZA = "beleza"


class PredictionRequest(ApiModel):
    sector: Sector
    features: dict[str, FeatureValue] = Field(min_length=1)
    video_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("features")
    @classmethod
    def validate_feature_names(cls, features: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
        if any(not name.strip() for name in features):
            raise ValueError("Nomes de features não podem ser vazios.")
        return features


class CreativePayload(ApiModel):
    video_id: str | None = Field(default=None, min_length=1, max_length=128)
    features: dict[str, FeatureValue] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def validate_feature_names(cls, features: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
        if any(not name.strip() for name in features):
            raise ValueError("Nomes de features não podem ser vazios.")
        return features


class Driver(ApiModel):
    feature: str
    importance: float
    value: Any = None
    direction: Literal["positive", "negative", "unknown"] = "unknown"
    scope: Literal["local", "global"] = "global"


class PredictionResponse(ApiModel):
    status: Literal["ok", "model_not_ready", "invalid_input"]
    sector: Sector
    video_id: str | None = None
    predicted_metric: float | None = None
    metric_name: str = "view_count"
    interval_low: float | None = None
    interval_high: float | None = None
    confidence_flag: Literal["normal", "low", "unavailable"] = "unavailable"
    relative_to_channel: float | None = None
    top_drivers: list[Driver] = Field(default_factory=list)
    explanation: str
    model_run_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ModelStatus(ApiModel):
    sector: Sector
    ready: bool
    run_id: str | None = None
    task: str | None = None
    model_name: str | None = None
    target: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    blocking_reasons: list[str] = Field(default_factory=list)


class RankedCreative(ApiModel):
    rank: int
    request_index: int
    prediction: PredictionResponse


class RankingResponse(ApiModel):
    sector: Sector
    status: Literal["ok", "model_not_ready", "partial", "invalid_input"]
    items: list[RankedCreative] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
