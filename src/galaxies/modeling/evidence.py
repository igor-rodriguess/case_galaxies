"""Evidência estatística pós-seleção sem escolher modelos pelo holdout."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_bootstrap_mae_difference(
    actual: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    iterations: int = 5_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compara erros absolutos pareados do candidato e do baseline.

    Diferença negativa favorece o candidato. O pareamento mantém cada vídeo
    alinhado entre os dois modelos e reduz variação irrelevante da comparação.
    """
    actual = np.asarray(actual, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if not (len(actual) == len(candidate) == len(baseline)):
        raise ValueError("actual, candidate e baseline precisam ter o mesmo tamanho.")
    if len(actual) < 2:
        raise ValueError("O bootstrap pareado exige pelo menos duas observações.")
    differences = np.abs(actual - candidate) - np.abs(actual - baseline)
    rng = np.random.default_rng(random_state)
    sampled = rng.integers(0, len(differences), size=(iterations, len(differences)))
    bootstrap_means = differences[sampled].mean(axis=1)
    candidate_mae = float(np.mean(np.abs(actual - candidate)))
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "rows": len(actual),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "mae_difference": float(differences.mean()),
        "relative_improvement": (
            float((baseline_mae - candidate_mae) / baseline_mae)
            if baseline_mae > 0
            else None
        ),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_candidate_better": float(np.mean(bootstrap_means < 0)),
        "statistically_better": bool(upper < 0),
    }
