"""Métricas de erro, ranking e classificação."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def pairwise_ranking_accuracy(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    groups: Iterable[Any],
) -> float | None:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    group_values = np.asarray(list(groups), dtype=object)
    correct = 0.0
    comparisons = 0
    for group in np.unique(group_values):
        positions = np.flatnonzero(group_values == group)
        for left, right in itertools.combinations(positions, 2):
            true_delta = true[left] - true[right]
            if true_delta == 0:
                continue
            predicted_delta = pred[left] - pred[right]
            comparisons += 1
            if predicted_delta == 0:
                correct += 0.5
            elif np.sign(predicted_delta) == np.sign(true_delta):
                correct += 1.0
    return correct / comparisons if comparisons else None


def regression_metrics(
    y_true_raw: Iterable[float],
    y_pred_raw: Iterable[float],
    groups: Iterable[Any],
) -> dict[str, float | None]:
    true = np.asarray(list(y_true_raw), dtype=float)
    predicted = np.clip(np.asarray(list(y_pred_raw), dtype=float), 0, None)
    true_log = np.log1p(true)
    predicted_log = np.log1p(predicted)
    correlation = (
        spearmanr(true, predicted).statistic
        if len(true) > 1 and np.ptp(true) > 0 and np.ptp(predicted) > 0
        else None
    )
    if correlation is not None and not np.isfinite(correlation):
        correlation = None
    denominator = np.maximum(np.abs(true), 1.0)
    return {
        "mae_log": float(mean_absolute_error(true_log, predicted_log)),
        "rmse_log": float(mean_squared_error(true_log, predicted_log) ** 0.5),
        "mae_raw": float(mean_absolute_error(true, predicted)),
        "mape_safe": float(np.mean(np.abs(true - predicted) / denominator)),
        "r2_log": float(r2_score(true_log, predicted_log)) if len(true) > 1 else None,
        "spearman": float(correlation) if correlation is not None else None,
        "pairwise_accuracy_intra_channel": pairwise_ranking_accuracy(true, predicted, groups),
    }


def classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
) -> dict[str, float]:
    true = list(y_true)
    predicted = list(y_pred)
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "f1_macro": float(f1_score(true, predicted, average="macro")),
    }
