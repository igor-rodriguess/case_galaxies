"""Função Python que será consumida diretamente e pelas tools MCP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import validate_sector
from .dataset import latest_feature_map, load_jsonl
from .paths import MODELING_RUNS_DIR, MODELS_DIR
from .schemas import (
    CreativePayload,
    Driver,
    ModelStatus,
    PredictionRequest,
    PredictionResponse,
    RankedCreative,
    RankingResponse,
    Sector,
)

logger = logging.getLogger(__name__)


class ModelArtifactError(RuntimeError):
    """Indica que um artefato promovido está ausente, corrompido ou fora do diretório."""


class ModelRepository:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

    def _pointer_path(self, sector: str) -> Path:
        return self.models_dir / validate_sector(sector) / "artifacts" / "latest.json"

    def _exploratory_pointer_path(self, sector: str) -> Path:
        return self.models_dir / validate_sector(sector) / "artifacts" / "latest_exploratory.json"

    def _artifacts_root(self, sector: str) -> Path:
        return self.models_dir / validate_sector(sector) / "artifacts"

    def _resolve_reference(self, sector: str, reference: Any) -> Path:
        if not isinstance(reference, str) or not reference.strip():
            raise ModelArtifactError("O ponteiro de modelo contém uma referência vazia.")
        artifacts_root = self._artifacts_root(sector).resolve()
        supplied = Path(reference)
        candidate = supplied if supplied.is_absolute() else artifacts_root / supplied
        resolved = candidate.resolve()
        if resolved != artifacts_root and artifacts_root not in resolved.parents:
            raise ModelArtifactError("O ponteiro tenta acessar um arquivo fora de artifacts/.")
        if not resolved.is_file():
            raise ModelArtifactError(f"Artefato não encontrado: {resolved.name}.")
        return resolved

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Não foi possível ler {path.name}.") from exc
        if not isinstance(value, dict):
            raise ModelArtifactError(f"{path.name} não contém um objeto JSON.")
        return value

    def pointer(self, sector: str) -> dict[str, Any] | None:
        path = self._pointer_path(sector)
        return self._read_json_object(path) if path.exists() else None

    def load_bundle(self, sector: str) -> dict[str, Any]:
        pointer = self.pointer(sector)
        if pointer is None:
            raise FileNotFoundError(f"Nenhum modelo promovido para {sector}.")
        bundle_path = self._resolve_reference(sector, pointer.get("bundle"))
        try:
            bundle = joblib.load(bundle_path)
        except Exception as exc:
            raise ModelArtifactError("Não foi possível carregar o bundle promovido.") from exc
        if not isinstance(bundle, dict):
            raise ModelArtifactError("O bundle promovido possui formato inválido.")
        if bundle.get("sector") != sector or not bundle.get("promotable"):
            raise ModelArtifactError("O bundle não corresponde ao setor ou não é promovível.")
        return bundle

    def status(self, sector: str) -> ModelStatus:
        sector_enum = Sector(validate_sector(sector))
        try:
            pointer = self.pointer(sector)
        except ModelArtifactError as exc:
            return ModelStatus(
                sector=sector_enum,
                ready=False,
                blocking_reasons=[str(exc)],
            )
        if pointer is None:
            reasons = ["Nenhum modelo final foi promovido para este setor."]
            if self._exploratory_pointer_path(sector).exists():
                reasons.append(
                    "Existe um treino exploratório, mas ele não passou pelos gates finais."
                )
            return ModelStatus(
                sector=sector_enum,
                ready=False,
                blocking_reasons=reasons,
            )
        try:
            manifest = self._read_json_object(
                self._resolve_reference(sector, pointer.get("manifest"))
            )
            metrics = self._read_json_object(
                self._resolve_reference(sector, pointer.get("metrics"))
            )
            bundle = self.load_bundle(sector)
        except ModelArtifactError as exc:
            return ModelStatus(
                sector=sector_enum,
                ready=False,
                run_id=pointer.get("run_id"),
                blocking_reasons=[str(exc)],
            )
        return ModelStatus(
            sector=sector_enum,
            ready=bool(pointer.get("promotable")),
            run_id=pointer.get("run_id"),
            task=manifest.get("task"),
            model_name=manifest.get("winner"),
            target=bundle.get("target", {}).get("metric_name"),
            metrics=metrics.get("winner", {}).get("temporal", {}),
            blocking_reasons=[] if pointer.get("promotable") else ["Modelo não promovível."],
        )


class PredictionService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.repository = ModelRepository(models_dir)

    def model_status(self, sector: str) -> ModelStatus:
        return self.repository.status(sector)

    def validate_features(self, sector: str, features: dict[str, Any]) -> dict[str, Any]:
        status = self.model_status(sector)
        if not status.ready:
            return {
                "valid": False,
                "model_ready": False,
                "missing_features": [],
                "warnings": status.blocking_reasons,
            }
        bundle = self.repository.load_bundle(sector)
        return self._feature_validation(bundle, features)

    @classmethod
    def _feature_validation(
        cls, bundle: dict[str, Any], features: dict[str, Any]
    ) -> dict[str, Any]:
        _, missing, unknown = cls._prepare_feature_row(bundle, features)
        expected = bundle["feature_columns"]
        missing_fraction = len(missing) / len(expected) if expected else 1.0
        maximum = float(bundle.get("serving", {}).get("max_missing_fraction", 0.5))
        warnings: list[str] = []
        if unknown:
            warnings.append(f"{len(unknown)} features desconhecidas serão ignoradas.")
        if missing_fraction > maximum:
            warnings.append(
                f"Ausência de features em {missing_fraction:.1%}; máximo aceito: {maximum:.1%}."
            )
        return {
            "valid": missing_fraction <= maximum,
            "model_ready": True,
            "expected_feature_count": len(expected),
            "missing_features": missing,
            "unknown_features": unknown,
            "missing_fraction": missing_fraction,
            "max_missing_fraction": maximum,
            "warnings": warnings,
        }

    @staticmethod
    def _prepare_feature_row(
        bundle: dict[str, Any], features: dict[str, Any]
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        columns = list(bundle["feature_columns"])
        unknown = sorted(set(features) - set(columns))
        row = pd.DataFrame([features]).reindex(columns=columns)
        for column in bundle.get("numeric_features", []):
            row[column] = pd.to_numeric(row[column], errors="coerce")
            row[column] = row[column].replace([np.inf, -np.inf], np.nan)
        for column in bundle.get("categorical_features", []):
            row[column] = row[column].map(
                lambda value: (
                    None
                    if pd.isna(value) or (isinstance(value, str) and not value.strip())
                    else str(value)
                )
            )
        missing = [column for column in columns if pd.isna(row.iloc[0][column])]
        return row, missing, unknown

    @staticmethod
    def _inverse(values: Any, transform: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if transform == "log1p":
            return np.clip(np.expm1(array), 0, None)
        return np.clip(array, 0, None)

    @staticmethod
    def _global_drivers(bundle: dict[str, Any], features: dict[str, Any]) -> list[Driver]:
        return [
            Driver(
                feature=item["feature"],
                importance=max(0.0, float(item["importance_mean"])),
                value=features.get(item["feature"]),
                direction="unknown",
                scope="global",
            )
            for item in bundle.get("global_feature_importance", [])[:8]
        ]

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        sector = request.sector.value
        status = self.model_status(sector)
        if not status.ready:
            return PredictionResponse(
                status="model_not_ready",
                sector=request.sector,
                video_id=request.video_id,
                explanation=(
                    "O pipeline de integração está disponível, mas ainda não existe "
                    "um modelo que tenha passado pelos gates de dados e avaliação."
                ),
                warnings=status.blocking_reasons,
            )

        bundle = self.repository.load_bundle(sector)
        if bundle.get("task") != "regression":
            return PredictionResponse(
                status="invalid_input",
                sector=request.sector,
                video_id=request.video_id,
                explanation="O artefato promovido não é um regressor de visualizações.",
            )

        validation = self._feature_validation(bundle, request.features)
        if not validation["valid"]:
            return PredictionResponse(
                status="invalid_input",
                sector=request.sector,
                video_id=request.video_id,
                explanation="O payload não possui features suficientes para uma previsão responsável.",
                warnings=validation["warnings"],
            )

        row, missing, unknown = self._prepare_feature_row(bundle, request.features)
        target = bundle["target"]
        predicted = float(
            self._inverse(bundle["point_model"].predict(row), target.get("transform", "none"))[0]
        )
        if not np.isfinite(predicted):
            raise ModelArtifactError("O modelo produziu uma previsão não finita.")
        low = high = None
        if bundle.get("lower_model") is not None and bundle.get("upper_model") is not None:
            low = float(
                self._inverse(
                    bundle["lower_model"].predict(row),
                    target.get("transform", "none"),
                )[0]
            )
            high = float(
                self._inverse(
                    bundle["upper_model"].predict(row),
                    target.get("transform", "none"),
                )[0]
            )
            low, high = min(low, high), max(low, high)

        columns = bundle["feature_columns"]
        missing_fraction = len(missing) / len(columns) if columns else 1.0
        warnings: list[str] = []
        confidence = "low" if missing_fraction > 0.3 else "normal"
        if missing:
            warnings.append(
                f"{len(missing)} de {len(columns)} features estão ausentes e foram imputadas."
            )
        if unknown:
            warnings.append(f"{len(unknown)} features desconhecidas foram ignoradas.")

        relative = None
        prior_median = request.features.get("prior_views_median_observed")
        try:
            prior_median_value = float(prior_median) if prior_median is not None else None
        except (TypeError, ValueError):
            prior_median_value = None
        if (
            prior_median_value is not None
            and np.isfinite(prior_median_value)
            and prior_median_value > 0
        ):
            relative = predicted / prior_median_value

        drivers: list[Driver] = []
        explanation_mode = "global"
        shap_unavailable = False
        try:
            from .explain import local_shap_drivers

            drivers = [Driver(**item) for item in local_shap_drivers(bundle, row)]
            explanation_mode = "local_shap"
        except RuntimeError as exc:
            logger.info("Explicação SHAP indisponível: %s", exc)
            shap_unavailable = True
        except Exception:
            logger.exception("Falha ao calcular explicação SHAP; usando importância global.")
            shap_unavailable = True
        if shap_unavailable:
            drivers = self._global_drivers(bundle, request.features)
            warnings.append(
                "SHAP local indisponível; top_drivers usa importância global sem direção."
            )

        return PredictionResponse(
            status="ok",
            sector=request.sector,
            video_id=request.video_id,
            predicted_metric=predicted,
            metric_name=target.get("metric_name", "view_count"),
            interval_low=low,
            interval_high=high,
            confidence_flag=confidence,
            relative_to_channel=relative,
            top_drivers=drivers,
            explanation=(
                f"Previsão produzida pelo modelo {bundle['model_name']}. "
                f"Explicação disponível no modo {explanation_mode}."
            ),
            model_run_id=bundle["run_id"],
            warnings=warnings,
        )

    def rank(
        self,
        sector: Sector,
        creatives: list[CreativePayload],
    ) -> RankingResponse:
        if not creatives:
            return RankingResponse(
                sector=sector,
                status="invalid_input",
                warnings=["Envie ao menos um criativo para ordenar."],
            )
        predictions = [
            self.predict(
                PredictionRequest(
                    sector=sector,
                    video_id=item.video_id,
                    features=item.features,
                )
            )
            for item in creatives
        ]
        if predictions and all(item.status == "model_not_ready" for item in predictions):
            return RankingResponse(
                sector=sector,
                status="model_not_ready",
                warnings=predictions[0].warnings,
            )
        valid = [
            (index, prediction)
            for index, prediction in enumerate(predictions)
            if prediction.status == "ok" and prediction.predicted_metric is not None
        ]
        valid.sort(key=lambda item: item[1].predicted_metric or 0, reverse=True)
        items = [
            RankedCreative(rank=rank, request_index=index, prediction=prediction)
            for rank, (index, prediction) in enumerate(valid, start=1)
        ]
        status_value = "ok" if len(items) == len(creatives) else "partial"
        return RankingResponse(sector=sector, status=status_value, items=items)


class FeatureMapRepository:
    """Consulta o último mapa materializado sem reconstruir toda a coleta."""

    def __init__(self, runs_dir: Path = MODELING_RUNS_DIR):
        self.runs_dir = runs_dir

    def get(self, video_id: str) -> dict[str, Any] | None:
        feature_map = latest_feature_map(self.runs_dir)
        return next(
            (row for row in load_jsonl(feature_map) if row.get("video_id") == video_id),
            None,
        )


def feature_map_for_video(video_id: str) -> dict[str, Any] | None:
    return FeatureMapRepository().get(video_id)
