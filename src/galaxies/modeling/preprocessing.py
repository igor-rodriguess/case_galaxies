"""Pré-processamento aprendido apenas no conjunto de treino."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_columns(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    numeric = list(dict.fromkeys(config["features"]["numeric"]))
    categorical = list(dict.fromkeys(config["features"]["categorical"]))
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(f"Features numéricas e categóricas sobrepostas: {overlap}")
    return numeric, categorical


def make_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    numeric, categorical = feature_columns(config)
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(config: dict[str, Any], estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(config)),
            ("model", estimator),
        ]
    )
