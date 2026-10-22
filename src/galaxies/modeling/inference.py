"""Inferência rastreável para a baseline congelada do Galaxies.

Este módulo não treina, seleciona ou recalibra modelos. Ele aplica exatamente os
artefatos congelados e retorna uma resposta própria para produto: nota de
ranking, probabilidade calibrada, alertas de cobertura e as contribuições
lineares locais. As contribuições descrevem associações do modelo; não são
recomendações causais de edição do criativo.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from galaxies.agent.protocol import SHORT_MAX_SECONDS


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def prepare_prepublication_features(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Completa somente derivações determinísticas disponíveis antes de publicar.

    Campos brutos opcionais ``title``, ``description`` e ``tags`` são usados
    apenas para calcular as contagens que a baseline já utiliza. Caso as
    contagens sejam enviadas diretamente, elas sempre prevalecem.
    """
    features = dict(payload)
    duration = pd.to_numeric(features.get("duration_seconds"), errors="coerce")
    if pd.notna(duration):
        duration = float(duration)
        if duration < 0:
            raise ValueError("duration_seconds não pode ser negativa.")
        features.setdefault("log_duration", float(np.log1p(duration)))
        features.setdefault("is_short", float(duration <= SHORT_MAX_SECONDS))
    if "title_character_count" not in features and features.get("title") is not None:
        features["title_character_count"] = len(str(features["title"]))
    if "description_character_count" not in features and features.get("description") is not None:
        features["description_character_count"] = len(str(features["description"]))
    if "tag_count" not in features and features.get("tags") is not None:
        tags = features["tags"]
        features["tag_count"] = len(tags) if isinstance(tags, (list, tuple, set)) else 1
    if features.get("published_at") is not None:
        published_at = pd.to_datetime(features["published_at"], utc=True, errors="coerce")
        if pd.notna(published_at):
            features.setdefault("publish_hour_utc", int(published_at.hour))
            features.setdefault("publish_weekday_utc", int(published_at.weekday()))
    return features


@dataclass(frozen=True)
class FrozenBaselinePredictor:
    """Artefatos imutáveis do classificador logístico de referência."""

    pipeline: Any
    calibrator: Any
    confidence_profile: dict[str, Any]
    artifact_dir: Path

    @classmethod
    def from_run(cls, artifact_dir: str | Path) -> FrozenBaselinePredictor:
        directory = Path(artifact_dir)
        required = {
            "pipeline": directory / "baseline_pipeline.joblib",
            "calibrator": directory / "probability_calibrator.joblib",
            "confidence": directory / "confidence_profile.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Artefato congelado incompleto: {', '.join(missing)}")
        return cls(
            pipeline=joblib.load(required["pipeline"]),
            calibrator=joblib.load(required["calibrator"]),
            confidence_profile=json.loads(required["confidence"].read_text(encoding="utf-8")),
            artifact_dir=directory,
        )

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.named_steps["preprocess"].feature_names_in_)

    def _frame(self, payload: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        prepared = prepare_prepublication_features(payload)
        missing = [
            name for name in self.feature_names if name not in prepared or prepared[name] is None
        ]
        row = {name: prepared.get(name) for name in self.feature_names}
        return pd.DataFrame([row]), {"prepared": prepared, "missing": missing}

    def _confidence(self, frame: pd.DataFrame, missing: list[str]) -> dict[str, Any]:
        alerts: list[dict[str, Any]] = []
        numeric = self.confidence_profile.get("numeric_training_range", {})
        categorical = self.confidence_profile.get("categorical_training_values", {})
        for column, bounds in numeric.items():
            value = pd.to_numeric(frame.iloc[0].get(column), errors="coerce")
            if pd.isna(value):
                alerts.append({"feature": column, "type": "missing_numeric"})
            elif value < bounds["p01"] or value > bounds["p99"]:
                alerts.append(
                    {
                        "feature": column,
                        "type": "outside_training_range",
                        "value": float(value),
                        "p01": bounds["p01"],
                        "p99": bounds["p99"],
                    }
                )
        for column, observed in categorical.items():
            value = frame.iloc[0].get(column)
            if value is None or pd.isna(value):
                alerts.append({"feature": column, "type": "missing_category"})
            elif str(value) not in observed:
                alerts.append(
                    {
                        "feature": column,
                        "type": "unseen_category",
                        "value": str(value),
                    }
                )
        return {
            "level": "low" if alerts else "standard",
            "rule": self.confidence_profile.get("low_confidence_rule"),
            "alerts": alerts,
            "missing_model_inputs": missing,
        }

    def _contributions(self, frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
        preprocessor = self.pipeline.named_steps["preprocess"]
        model = self.pipeline.named_steps["model"]
        transformed = np.asarray(preprocessor.transform(frame))[0]
        names = preprocessor.get_feature_names_out()
        contributions = transformed * np.asarray(model.coef_).ravel()
        order = np.argsort(np.abs(contributions))[::-1][:limit]
        return [
            {
                "feature": str(names[index]),
                "log_odds_contribution": float(contributions[index]),
                "direction": "raises_score" if contributions[index] >= 0 else "lowers_score",
            }
            for index in order
            if contributions[index] != 0
        ]

    def predict(self, payload: Mapping[str, Any], *, top_contributions: int = 8) -> dict[str, Any]:
        """Retorna previsão pronta para auditoria, nunca uma promessa de viralização."""
        if top_contributions < 1:
            raise ValueError("top_contributions deve ser ao menos 1.")
        frame, details = self._frame(payload)
        raw_score = float(self.pipeline.predict_proba(frame)[0, 1])
        probability = float(self.calibrator.predict_proba([[raw_score]])[0, 1])
        return {
            "schema_version": "0.1.0",
            "artifact_dir": str(self.artifact_dir),
            "ranking_score_raw": raw_score,
            "viral_probability_calibrated": probability,
            "prediction_scope": (
                "Probabilidade calibrada para o rótulo histórico top-10% intra-canal; "
                "não prevê views absolutas e não garante viralização."
            ),
            "confidence": self._confidence(frame, details["missing"]),
            "top_local_contributions": self._contributions(frame, top_contributions),
            "ignored_payload_fields": sorted(
                str(key)
                for key in payload
                if key not in self.feature_names
                and key not in {"title", "description", "tags", "published_at"}
            ),
        }

    def compare(
        self,
        baseline_payload: Mapping[str, Any],
        candidate_payload: Mapping[str, Any],
        *,
        top_contributions: int = 8,
    ) -> dict[str, Any]:
        """Compara dois cenários de publicação sob o mesmo modelo congelado.

        A comparação é contrafactual apenas no sentido de *modelo*: ela informa
        como a nota associada muda quando as entradas mudam. Não identifica o
        efeito causal de editar duração, texto ou horário no mundo real.
        """
        baseline = self.predict(baseline_payload, top_contributions=top_contributions)
        candidate = self.predict(candidate_payload, top_contributions=top_contributions)
        baseline_frame, _ = self._frame(baseline_payload)
        candidate_frame, _ = self._frame(candidate_payload)
        changed = []
        for column in self.feature_names:
            before, after = baseline_frame.iloc[0][column], candidate_frame.iloc[0][column]
            same = (pd.isna(before) and pd.isna(after)) or before == after
            if not same:
                changed.append(
                    {
                        "feature": column,
                        "baseline": _json_value(before),
                        "candidate": _json_value(after),
                    }
                )
        return {
            "schema_version": "0.1.0",
            "comparison_scope": (
                "Cenários associados pelo modelo congelado. A diferença não é evidência "
                "causal nem garantia de mudança de desempenho."
            ),
            "changed_model_inputs": changed,
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                "ranking_score_raw": candidate["ranking_score_raw"] - baseline["ranking_score_raw"],
                "viral_probability_calibrated": candidate["viral_probability_calibrated"]
                - baseline["viral_probability_calibrated"],
            },
        }


def prediction_to_json(prediction: Mapping[str, Any]) -> str:
    """Serializa uma previsão sem vazar tipos NumPy para consumidores do MCP/UI."""
    return json.dumps(prediction, ensure_ascii=False, indent=2, default=_json_value) + "\n"
