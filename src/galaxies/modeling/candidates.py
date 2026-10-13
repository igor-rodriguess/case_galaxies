"""Registro explícito dos algoritmos comparados."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor


def make_regressor(name: str, random_state: int) -> Any:
    factories = {
        "dummy_median": lambda: DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression,
        "ridge": lambda: Ridge(alpha=1.0),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            loss="huber",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=500,
            random_state=random_state,
        ),
    }
    if name in factories:
        return factories[name]()
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Instale o extra [boosting] para usar XGBoost.") from exc
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("Instale o extra [boosting] para usar CatBoost.") from exc
        return CatBoostRegressor(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Instale o extra [boosting] para usar LightGBM.") from exc
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Regressor desconhecido: {name}")


def make_classifier(name: str, random_state: int) -> Any:
    factories = {
        "dummy_most_frequent": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest_classifier": lambda: RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting_classifier": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "mlp_classifier": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=500,
            random_state=random_state,
        ),
    }
    if name not in factories:
        raise ValueError(f"Classificador desconhecido: {name}")
    return factories[name]()
