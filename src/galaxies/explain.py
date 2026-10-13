"""Explicações locais opcionais, sem transformar SHAP em requisito de coleta."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def local_shap_drivers(
    bundle: dict[str, Any],
    row: pd.DataFrame,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Calcula SHAP quando o extra ``explain`` estiver instalado.

    A função não inventa uma explicação alternativa. Se o SHAP ou o modelo não
    forem compatíveis, o chamador deve declarar que só possui importância global.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Instale o extra [explain] para habilitar SHAP.") from exc

    pipeline = bundle["point_model"]
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(row)
    background = bundle.get("explanation_background")
    if background is None:
        raise RuntimeError("O bundle não contém uma amostra de fundo para SHAP.")
    explainer = shap.Explainer(estimator, np.asarray(background))
    explanation = explainer(np.asarray(transformed))
    values = np.asarray(explanation.values)
    if values.ndim > 2:
        values = values[..., 0]
    contribution = values[0]
    names = preprocessor.get_feature_names_out()
    order = np.argsort(np.abs(contribution))[::-1][:limit]
    return [
        {
            "feature": str(names[index]),
            "importance": float(abs(contribution[index])),
            "direction": "positive" if contribution[index] >= 0 else "negative",
            "scope": "local",
        }
        for index in order
    ]
