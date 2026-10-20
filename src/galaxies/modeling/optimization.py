"""Busca rastreável de features, hiperparâmetros e ensembles de regressão."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..config import load_sector_config, validate_sector
from ..dataset import load_jsonl
from .candidates import make_regressor
from .feature_groups import FEATURE_SET_NAMES, config_for_feature_set, feature_family
from .metrics import regression_metrics, relative_regression_metrics
from .preprocessing import feature_columns, make_preprocessor
from .relative_target import baseline_summary, fit_relative_target
from .splits import channel_holdout, temporal_holdout
from .trainer import _feature_frame, _inverse_target, _transform_target

DEFAULT_CANDIDATES = (
    "linear_regression",
    "ridge",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "mlp",
)

ENSEMBLE_MEMBERS = (
    "ridge",
    "random_forest",
    "gradient_boosting",
    "mlp",
)


class WeightedRegressorEnsemble(RegressorMixin, BaseEstimator):
    """Combina regressões já encapsuladas em pipelines completos."""

    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        weights: Iterable[float] | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights

    def fit(self, x: Any, y: Any) -> WeightedRegressorEnsemble:
        self.estimators_ = [
            (name, clone(estimator).fit(x, y)) for name, estimator in self.estimators
        ]
        return self

    def predict(self, x: Any) -> np.ndarray:
        predictions = np.column_stack(
            [estimator.predict(x) for _, estimator in self.estimators_]
        )
        weights = None if self.weights is None else np.asarray(self.weights, dtype=float)
        return np.average(predictions, axis=1, weights=weights)


def expanding_window_splits(n_rows: int, requested_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cria folds em que toda validação ocorre depois do respectivo treino."""
    if n_rows < 6:
        raise ValueError("A validação temporal interna exige pelo menos seis linhas.")
    n_splits = min(requested_splits, n_rows - 2)
    if n_splits < 2:
        raise ValueError("São necessários ao menos dois folds temporais internos.")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return [(train, validation) for train, validation in splitter.split(np.arange(n_rows))]


def leakage_safe_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove features cujo timestamp histórico ainda não foi comprovado."""
    safe = deepcopy(config)
    excluded: list[str] = []
    for kind in ("numeric", "categorical"):
        selected = []
        for feature in config["features"][kind]:
            if feature_family(feature) == "channel_history":
                excluded.append(feature)
            else:
                selected.append(feature)
        safe["features"][kind] = selected
    return safe, sorted(excluded)


def _pipeline(config: dict[str, Any], model_name: str, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(config)),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("select", SelectPercentile(score_func=f_regression, percentile=100)),
            ("model", make_regressor(model_name, random_state)),
        ]
    )


def search_space(
    model_name: str,
    random_state: int = 42,
) -> list[dict[str, list[Any]]]:
    """Compara escalonamento e seleção de features dentro dos folds."""
    common: dict[str, list[Any]] = {
        "preprocess__numeric__scaler": [
            StandardScaler(),
            RobustScaler(quantile_range=(10.0, 90.0)),
        ]
    }
    specific: dict[str, list[Any]] = {
        "linear_regression": {
            "model__fit_intercept": [True, False],
        },
        "ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0],
            "model__fit_intercept": [True, False],
        },
        "random_forest": {
            "model__n_estimators": [200, 400, 700],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__max_features": [0.4, 0.6, 0.8, 1.0, "sqrt"],
        },
        "extra_trees": {
            "model__n_estimators": [200, 400, 700],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__max_features": [0.4, 0.6, 0.8, 1.0, "sqrt"],
        },
        "gradient_boosting": {
            "model__loss": ["huber", "absolute_error", "squared_error"],
            "model__n_estimators": [100, 200, 400, 700],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [2, 4, 8],
            "model__subsample": [0.7, 0.85, 1.0],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_iter": [100, 200, 400, 700],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [5, 10, 20],
            "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        },
        "mlp": {
            "model__hidden_layer_sizes": [
                (16,),
                (32,),
                (64, 32),
                (64, 32, 16),
                (128, 64),
            ],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "model__learning_rate_init": [0.0003, 0.001, 0.003, 0.01],
            "model__max_iter": [300, 600],
            "model__tol": [0.0001, 0.001],
            "model__n_iter_no_change": [15, 30],
        },
    }
    if model_name not in specific:
        raise ValueError(f"Sem espaço de busca para {model_name!r}.")
    selectors: list[dict[str, list[Any]]] = [
        {
            "select": [SelectPercentile(score_func=f_regression)],
            "select__percentile": [40, 60, 80, 100],
        },
        {
            "select": [
                SelectPercentile(
                    score_func=partial(
                        mutual_info_regression,
                        random_state=random_state,
                    )
                )
            ],
            "select__percentile": [40, 60, 80, 100],
        },
        {
            "select": [
                SelectFromModel(
                    ExtraTreesRegressor(
                        n_estimators=200,
                        min_samples_leaf=3,
                        n_jobs=1,
                        random_state=random_state,
                    )
                )
            ],
            "select__threshold": ["median", "1.25*mean"],
        },
    ]
    return [
        {**common, **selector, **specific[model_name]} for selector in selectors
    ]


def _selected_feature_names(pipeline: Pipeline) -> list[str]:
    transformed = np.asarray(
        pipeline.named_steps["preprocess"].get_feature_names_out(), dtype=object
    )
    transformed = transformed[pipeline.named_steps["variance"].get_support()]
    selector = pipeline.named_steps["select"]
    return transformed[selector.get_support()].astype(str).tolist()


def _prediction_rows(
    frame: pd.DataFrame,
    indices: list[int],
    predicted_raw: np.ndarray,
    split: str,
    model: str,
    target_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, prediction in zip(indices, predicted_raw, strict=True):
        rows.append(
            {
                "sector": str(frame.loc[index, "sector"]),
                "model": model,
                "split": split,
                "video_id": str(frame.loc[index, "video_id"]),
                "channel_id": str(frame.loc[index, "channel_id"]),
                "published_at": str(frame.loc[index, "published_at"]),
                "actual": float(frame.loc[index, target_column]),
                "predicted": float(prediction),
                "residual": float(prediction - frame.loc[index, target_column]),
            }
        )
    return rows


def _evaluate_fitted(
    estimator: Any,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    y_raw: np.ndarray,
    indices: list[int],
    transform: str,
    split: str,
    model_name: str,
    target_column: str,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    predicted_model = estimator.predict(features.loc[indices])
    return _evaluate_predictions(
        predicted_model,
        frame,
        y_raw,
        indices,
        transform,
        split,
        model_name,
        target_column,
    )


def _evaluate_predictions(
    predicted_model: np.ndarray,
    frame: pd.DataFrame,
    y_raw: np.ndarray,
    indices: list[int],
    transform: str,
    split: str,
    model_name: str,
    target_column: str,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    predicted = _inverse_target(predicted_model, transform)
    metrics = regression_metrics(
        y_raw[indices], predicted, frame.loc[indices, "channel_id"]
    )
    return metrics, _prediction_rows(
        frame, indices, predicted, split, model_name, target_column
    )


def _evaluate_target_predictions(
    predicted_model: np.ndarray,
    frame: pd.DataFrame,
    target_values: np.ndarray,
    indices: list[int],
    transform: str,
    split: str,
    model_name: str,
    target_column: str,
    target_mode: str,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    """Avalia contagens brutas ou o índice residual sem misturar as escalas."""
    if target_mode == "observed_views":
        return _evaluate_predictions(
            predicted_model,
            frame,
            target_values,
            indices,
            transform,
            split,
            model_name,
            target_column,
        )

    predicted = np.asarray(predicted_model, dtype=float)
    actual = np.asarray(target_values, dtype=float)[indices]
    metrics = relative_regression_metrics(
        actual, predicted, frame.loc[indices, "channel_id"]
    )
    rows = []
    for index, truth, estimate in zip(indices, actual, predicted, strict=True):
        rows.append(
            {
                "sector": str(frame.loc[index, "sector"]),
                "model": model_name,
                "split": split,
                "video_id": str(frame.loc[index, "video_id"]),
                "channel_id": str(frame.loc[index, "channel_id"]),
                "published_at": str(frame.loc[index, "published_at"]),
                "actual": float(truth),
                "predicted": float(estimate),
                "residual": float(estimate - truth),
            }
        )
    return metrics, rows


def _evaluate_target_fitted(
    estimator: Any,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target_values: np.ndarray,
    indices: list[int],
    transform: str,
    split: str,
    model_name: str,
    target_column: str,
    target_mode: str,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    return _evaluate_target_predictions(
        estimator.predict(features.loc[indices]),
        frame,
        target_values,
        indices,
        transform,
        split,
        model_name,
        target_column,
        target_mode,
    )


def _fit_quietly(estimator: Any, x: Any, y: Any) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return estimator.fit(x, y)


def _oof_predictions(
    estimators: list[tuple[str, Any]],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(x_train), len(estimators)), np.nan, dtype=float)
    for train_positions, validation_positions in cv:
        for column, (_, estimator) in enumerate(estimators):
            fitted = _fit_quietly(
                clone(estimator),
                x_train.iloc[train_positions], y_train[train_positions]
            )
            matrix[validation_positions, column] = fitted.predict(
                x_train.iloc[validation_positions]
            )
    covered = np.all(np.isfinite(matrix), axis=1)
    if not covered.any():
        raise RuntimeError("Os folds internos não produziram previsões OOF completas.")
    return matrix[covered], y_train[covered]


def learn_ensemble_weights(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Aprende pesos não negativos e normalizados sem tocar no holdout externo."""
    weights, _ = nnls(predictions, target)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(predictions.shape[1], 1.0 / predictions.shape[1])
    return weights / total


def _permutation_rows(
    estimator: Any,
    features: pd.DataFrame,
    y_model: np.ndarray,
    test_indices: list[int],
    random_state: int,
) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`sklearn.utils.parallel.delayed` should be used*",
            category=UserWarning,
        )
        measured = permutation_importance(
            estimator,
            features.loc[test_indices],
            y_model[test_indices],
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=random_state,
            n_jobs=1,
        )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            features.columns,
            measured.importances_mean,
            measured.importances_std,
            strict=True,
        )
    ]
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)


def _plain_value(value: Any) -> Any:
    if isinstance(value, BaseEstimator):
        return value.__class__.__name__
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def optimize_sector(
    dataset_path: Any,
    sector: str,
    *,
    search_iterations: int = 12,
    inner_splits: int = 3,
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    n_jobs: int = -1,
    target_mode: str = "observed_views",
    feature_set: str = "combined",
) -> dict[str, Any]:
    """Otimiza um setor sem usar os holdouts externos durante a busca."""
    sector = validate_sector(sector)
    if target_mode not in {"observed_views", "relative_performance"}:
        raise ValueError("target_mode deve ser observed_views ou relative_performance.")
    if feature_set not in FEATURE_SET_NAMES:
        raise ValueError(f"feature_set inválido: {feature_set!r}.")
    config = config_for_feature_set(load_sector_config(sector), feature_set)
    safe_config, excluded_features = leakage_safe_config(config)
    rows = load_jsonl(dataset_path)
    frame = pd.DataFrame(rows)
    vlm_available = frame.get(
        "vlm_available", pd.Series(False, index=frame.index)
    ).fillna(False)
    frame = frame.loc[
        (frame["sector"] == sector)
        & frame[config["target"]["column"]].notna()
        & vlm_available.astype(bool)
    ].copy()
    if target_mode == "relative_performance":
        age = pd.to_numeric(frame.get("target_age_days"), errors="coerce")
        frame = frame.loc[age.notna() & (age > 0)].copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame.sort_values("published_at", kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if len(frame) < 30:
        raise ValueError(f"{sector} possui somente {len(frame)} linhas comuns com VLM.")

    validation = config["validation"]
    temporal = temporal_holdout(
        frame,
        validation["temporal_test_fraction"],
        min(validation["minimum_test_rows"], max(1, len(frame) // 4)),
    )
    channel = channel_holdout(
        frame,
        validation["channel_test_fraction"],
        validation["random_state"],
    )
    features = _feature_frame(frame, safe_config)
    target_column = config["target"]["column"]
    y_raw = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=float)
    target_transform = config["target"]["transform"]
    relative_baselines: dict[str, Any] = {}
    if target_mode == "relative_performance":
        temporal_baseline, y_temporal = fit_relative_target(
            frame, temporal.train_indices, y_raw
        )
        channel_baseline, y_channel = fit_relative_target(
            frame, channel.train_indices, y_raw
        )
        y_model = y_temporal
        target_transform = "none"
        target_column = "target_relative_performance"
        relative_baselines = {
            "temporal": baseline_summary(temporal_baseline),
            "channel_holdout": baseline_summary(channel_baseline),
        }
    else:
        y_temporal = y_channel = _transform_target(
            y_raw, config["target"]["transform"]
        )
        y_model = y_temporal

    temporal_train = temporal.train_indices
    x_inner = features.loc[temporal_train].reset_index(drop=True)
    y_inner = y_model[temporal_train]
    cv = expanding_window_splits(len(x_inner), inner_splits)
    channel_train = channel.train_indices
    x_channel_inner = features.loc[channel_train].reset_index(drop=True)
    y_channel_inner = y_channel[channel_train]
    channel_cv = expanding_window_splits(len(x_channel_inner), inner_splits)
    candidate_rows: list[dict[str, Any]] = []
    fitted_temporal: dict[str, Any] = {}
    fitted_channel: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []

    dummy = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(safe_config)),
            ("model", DummyRegressor(strategy="median")),
        ]
    )
    _fit_quietly(
        dummy, features.loc[temporal.train_indices], y_model[temporal.train_indices]
    )
    dummy_temporal, dummy_temporal_rows = _evaluate_target_fitted(
        dummy,
        frame,
        features,
        y_temporal if target_mode == "relative_performance" else y_raw,
        temporal.test_indices,
        target_transform,
        "temporal",
        "dummy_median",
        target_column,
        target_mode,
    )
    dummy_channel = _fit_quietly(
        clone(dummy), features.loc[channel.train_indices], y_channel[channel.train_indices]
    )
    dummy_channel_metrics, dummy_channel_rows = _evaluate_target_fitted(
        dummy_channel,
        frame,
        features,
        y_channel if target_mode == "relative_performance" else y_raw,
        channel.test_indices,
        target_transform,
        "channel_holdout",
        "dummy_median",
        target_column,
        target_mode,
    )
    predictions.extend(dummy_temporal_rows + dummy_channel_rows)
    candidate_rows.append(
        {
            "model": "dummy_median",
            "kind": "baseline",
            "status": "ok",
            "inner_cv_mae_log": None,
            "inner_cv_mae_log_std": None,
            "best_params": {},
            "selected_feature_count": 0,
            "selected_features": [],
            "temporal": dummy_temporal,
            "channel_holdout": dummy_channel_metrics,
        }
    )

    for model_name in candidates:
        estimator = _pipeline(safe_config, model_name, validation["random_state"])
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_space(
                model_name, validation["random_state"]
            ),
            n_iter=search_iterations,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
            cv=cv,
            refit=True,
            random_state=validation["random_state"],
            error_score=np.nan,
            return_train_score=False,
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                search.fit(x_inner, y_inner)
            if not np.isfinite(search.best_score_):
                raise RuntimeError("Todas as combinações da busca falharam.")
            temporal_estimator = search.best_estimator_
            temporal_metrics, temporal_rows = _evaluate_target_fitted(
                temporal_estimator,
                frame,
                features,
                y_temporal if target_mode == "relative_performance" else y_raw,
                temporal.test_indices,
                target_transform,
                "temporal",
                model_name,
                target_column,
                target_mode,
            )
            channel_search = RandomizedSearchCV(
                estimator=_pipeline(
                    safe_config, model_name, validation["random_state"]
                ),
                param_distributions=search_space(
                    model_name, validation["random_state"]
                ),
                n_iter=search_iterations,
                scoring="neg_mean_absolute_error",
                n_jobs=n_jobs,
                cv=channel_cv,
                refit=True,
                random_state=validation["random_state"],
                error_score=np.nan,
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                channel_search.fit(x_channel_inner, y_channel_inner)
            if not np.isfinite(channel_search.best_score_):
                raise RuntimeError("A busca interna do corte por canal falhou.")
            channel_estimator = channel_search.best_estimator_
            channel_metrics, channel_rows = _evaluate_target_fitted(
                channel_estimator,
                frame,
                features,
                y_channel if target_mode == "relative_performance" else y_raw,
                channel.test_indices,
                target_transform,
                "channel_holdout",
                model_name,
                target_column,
                target_mode,
            )
            fitted_temporal[model_name] = temporal_estimator
            fitted_channel[model_name] = channel_estimator
            predictions.extend(temporal_rows + channel_rows)
            candidate_rows.append(
                {
                    "model": model_name,
                    "kind": "base",
                    "status": "ok",
                    "inner_cv_mae_log": float(-search.best_score_),
                    "inner_cv_mae_log_std": float(
                        search.cv_results_["std_test_score"][search.best_index_]
                    ),
                    "best_params": _plain_value(search.best_params_),
                    "channel_inner_cv_mae_log": float(-channel_search.best_score_),
                    "channel_best_params": _plain_value(
                        channel_search.best_params_
                    ),
                    "selected_feature_count": len(
                        _selected_feature_names(temporal_estimator)
                    ),
                    "selected_features": _selected_feature_names(temporal_estimator),
                    "temporal": temporal_metrics,
                    "channel_holdout": channel_metrics,
                }
            )
        except Exception as exc:
            candidate_rows.append(
                {
                    "model": model_name,
                    "kind": "base",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    ensemble_estimators = [
        (name, fitted_temporal[name])
        for name in ENSEMBLE_MEMBERS
        if name in fitted_temporal
    ]
    channel_ensemble_estimators = [
        (name, fitted_channel[name])
        for name in ENSEMBLE_MEMBERS
        if name in fitted_channel
    ]
    ensemble_specs: dict[str, np.ndarray] = {}
    channel_ensemble_specs: dict[str, np.ndarray] = {}
    if len(ensemble_estimators) >= 2 and len(channel_ensemble_estimators) >= 2:
        oof_matrix, oof_target = _oof_predictions(
            ensemble_estimators, x_inner, y_inner, cv
        )
        learned_weights = learn_ensemble_weights(oof_matrix, oof_target)
        channel_oof_matrix, channel_oof_target = _oof_predictions(
            channel_ensemble_estimators,
            x_channel_inner,
            y_channel_inner,
            channel_cv,
        )
        channel_learned_weights = learn_ensemble_weights(
            channel_oof_matrix, channel_oof_target
        )
        ensemble_specs = {
            "ensemble_equal": np.full(
                len(ensemble_estimators), 1.0 / len(ensemble_estimators)
            ),
            "ensemble_weighted": learned_weights,
        }
        channel_ensemble_specs = {
            "ensemble_equal": np.full(
                len(channel_ensemble_estimators),
                1.0 / len(channel_ensemble_estimators),
            ),
            "ensemble_weighted": channel_learned_weights,
        }
        for ensemble_name, weights in ensemble_specs.items():
            channel_weights = channel_ensemble_specs[ensemble_name]
            temporal_prediction = np.average(
                np.column_stack(
                    [
                        fitted_temporal[name].predict(
                            features.loc[temporal.test_indices]
                        )
                        for name, _ in ensemble_estimators
                    ]
                ),
                axis=1,
                weights=weights,
            )
            temporal_metrics, temporal_rows = _evaluate_target_predictions(
                temporal_prediction,
                frame,
                y_temporal if target_mode == "relative_performance" else y_raw,
                temporal.test_indices,
                target_transform,
                "temporal",
                ensemble_name,
                target_column,
                target_mode,
            )
            channel_prediction = np.average(
                np.column_stack(
                    [
                        fitted_channel[name].predict(features.loc[channel.test_indices])
                        for name, _ in channel_ensemble_estimators
                    ]
                ),
                axis=1,
                weights=channel_weights,
            )
            channel_metrics, channel_rows = _evaluate_target_predictions(
                channel_prediction,
                frame,
                y_channel if target_mode == "relative_performance" else y_raw,
                channel.test_indices,
                target_transform,
                "channel_holdout",
                ensemble_name,
                target_column,
                target_mode,
            )
            oof_blend = np.average(oof_matrix, axis=1, weights=weights)
            inner_mae = float(np.mean(np.abs(oof_target - oof_blend)))
            predictions.extend(temporal_rows + channel_rows)
            candidate_rows.append(
                {
                    "model": ensemble_name,
                    "kind": "ensemble",
                    "status": "ok",
                    "inner_cv_mae_log": inner_mae,
                    "inner_cv_mae_log_std": None,
                    "best_params": {
                        "members": [name for name, _ in ensemble_estimators],
                        "temporal_weights": [float(weight) for weight in weights],
                        "channel_weights": [
                            float(weight) for weight in channel_weights
                        ],
                    },
                    "selected_feature_count": None,
                    "selected_features": [],
                    "temporal": temporal_metrics,
                    "channel_holdout": channel_metrics,
                }
            )

    successful = [
        row
        for row in candidate_rows
        if row["status"] == "ok" and row["kind"] != "baseline"
    ]
    if not successful:
        raise RuntimeError(f"Nenhum candidato foi treinado com sucesso para {sector}.")
    best_inner = min(successful, key=lambda row: row["inner_cv_mae_log"])
    best_temporal = min(successful, key=lambda row: row["temporal"]["mae_log"])
    best_channel = min(
        successful, key=lambda row: row["channel_holdout"]["mae_log"]
    )
    if best_inner["model"] in ensemble_specs:
        winner_estimator = _fit_quietly(
            WeightedRegressorEnsemble(
                ensemble_estimators,
                ensemble_specs[best_inner["model"]].tolist(),
            ),
            features.loc[temporal.train_indices],
            y_model[temporal.train_indices],
        )
    else:
        winner_estimator = fitted_temporal[best_inner["model"]]
    feature_importance = _permutation_rows(
        winner_estimator,
        features,
        y_model,
        temporal.test_indices,
        validation["random_state"],
    )
    target_fixed = frame.get(
        "target_is_fixed_horizon", pd.Series(False, index=frame.index)
    ).fillna(False)
    target_age = pd.to_numeric(frame.get("target_age_days"), errors="coerce").dropna()
    numeric, categorical = feature_columns(safe_config)
    return {
        "schema_version": "0.2.0",
        "sector": sector,
        "exploratory": True,
        "promotable": False,
        "warning": (
            "O índice relativo corrige idade e escala do canal, mas usa snapshots "
            "públicos heterogêneos; a otimização continua exploratória."
            if target_mode == "relative_performance"
            else "O alvo é uma contagem observada em idades diferentes; a otimização "
            "valida o pipeline, mas não promove modelo para produção."
        ),
        "target": {
            "mode": target_mode,
            "column": target_column,
            "transform": target_transform,
            "definition": (
                "resíduo de log1p(views) após curva robusta de idade e efeito "
                "histórico suavizado do canal, ambos ajustados somente no treino"
                if target_mode == "relative_performance"
                else "contagem acumulada observada no snapshot"
            ),
            "baselines": relative_baselines,
        },
        "feature_set": feature_set,
        "dataset": str(dataset_path),
        "sample": {
            "rows": len(frame),
            "channels": int(frame["channel_id"].nunique()),
            "fixed_horizon_rows": int(target_fixed.astype(bool).sum()),
            "target_age_days": {
                "minimum": float(target_age.min()) if not target_age.empty else None,
                "median": float(target_age.median()) if not target_age.empty else None,
                "maximum": float(target_age.max()) if not target_age.empty else None,
            },
        },
        "leakage_control": {
            "excluded_channel_history_features": excluded_features,
            "eligible_numeric_features": numeric,
            "eligible_categorical_features": categorical,
            "feature_selection": (
                "VarianceThreshold remove constantes dentro de cada fold; a busca compara "
                "F-test, informação mútua e SelectFromModel com Extra Trees."
            ),
            "numeric_scaling": (
                "A busca compara StandardScaler e RobustScaler(quantis 10%-90%) "
                "dentro de cada fold."
            ),
        },
        "validation": {
            "inner": f"{len(cv)} folds expanding-window no treino temporal externo.",
            "channel_inner": (
                f"{len(channel_cv)} folds expanding-window somente nos canais de "
                "treino do holdout por canal."
            ),
            "temporal": temporal.description,
            "channel_holdout": channel.description,
            "search_iterations_per_model": search_iterations,
        },
        "selection": {
            "criterion": "menor MAE log médio na validação temporal interna",
            "best_inner_cv": best_inner["model"],
            "best_temporal_observed": best_temporal["model"],
            "best_channel_holdout_observed": best_channel["model"],
            "holdouts_used_for_tuning": False,
        },
        "candidates": candidate_rows,
        "feature_importance": feature_importance,
        "predictions": predictions,
    }
