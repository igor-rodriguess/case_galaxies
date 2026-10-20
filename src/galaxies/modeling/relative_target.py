"""Alvo relativo ajustado por exposição e escala histórica do canal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler


@dataclass
class AgeChannelBaseline:
    """Estima views esperadas usando somente idade e canais do treino.

    A curva robusta modela ``log1p(views)`` em função de ``log1p(idade)``.
    Em seguida, cada canal recebe um efeito residual suavizado. Canais não
    vistos usam efeito zero, isto é, a curva global aprendida no treino.
    """

    channel_shrinkage: float = 5.0
    minimum_age_days: float = 1.0 / 24.0
    age_model: Pipeline = field(init=False)
    channel_effects: dict[str, float] = field(init=False, default_factory=dict)

    def _age_feature(self, frame: pd.DataFrame) -> np.ndarray:
        age = pd.to_numeric(frame["target_age_days"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.all(np.isfinite(age)) or np.any(age <= 0):
            raise ValueError("target_age_days deve ser finito e positivo.")
        return np.log1p(np.maximum(age, self.minimum_age_days)).reshape(-1, 1)

    def fit(self, frame: pd.DataFrame, log_views: np.ndarray) -> AgeChannelBaseline:
        self.age_model = Pipeline(
            steps=[
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
                ("regressor", HuberRegressor(alpha=0.01, max_iter=1_000)),
            ]
        )
        self.age_model.fit(self._age_feature(frame), log_views)
        age_prediction = self.age_model.predict(self._age_feature(frame))
        residual = np.asarray(log_views, dtype=float) - age_prediction
        channels = frame["channel_id"].astype(str).to_numpy()
        self.channel_effects = {}
        for channel in np.unique(channels):
            values = residual[channels == channel]
            self.channel_effects[channel] = float(
                values.sum() / (len(values) + self.channel_shrinkage)
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        age_prediction = self.age_model.predict(self._age_feature(frame))
        channel_adjustment = np.asarray(
            [self.channel_effects.get(str(value), 0.0) for value in frame["channel_id"]],
            dtype=float,
        )
        return age_prediction + channel_adjustment

    def residual_target(
        self, frame: pd.DataFrame, view_count: np.ndarray
    ) -> np.ndarray:
        return np.log1p(np.asarray(view_count, dtype=float)) - self.predict(frame)


def fit_relative_target(
    frame: pd.DataFrame,
    train_indices: list[int],
    view_count: np.ndarray,
) -> tuple[AgeChannelBaseline, np.ndarray]:
    """Ajusta a referência apenas no treino e calcula o índice para todas as linhas."""
    baseline = AgeChannelBaseline().fit(
        frame.loc[train_indices], np.log1p(view_count[train_indices])
    )
    return baseline, baseline.residual_target(frame, view_count)


def baseline_summary(baseline: AgeChannelBaseline) -> dict[str, Any]:
    effects = np.asarray(list(baseline.channel_effects.values()), dtype=float)
    return {
        "method": "Huber quadrático em log1p(idade) + efeito de canal suavizado",
        "channel_shrinkage": baseline.channel_shrinkage,
        "known_channels": len(baseline.channel_effects),
        "channel_effect_minimum": float(effects.min()) if effects.size else None,
        "channel_effect_median": float(np.median(effects)) if effects.size else None,
        "channel_effect_maximum": float(effects.max()) if effects.size else None,
    }
