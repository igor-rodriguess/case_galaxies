"""Fusão em dois estágios com cross-fitting do modelo-base.

O primeiro estágio aproveita toda a população madura sem VLM. O segundo recebe
somente a previsão fora da dobra do primeiro estágio e os sinais audiovisuais,
evitando que a base menor descarte o conhecimento dos milhares de metadados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from galaxies.modeling.preprocessing import make_pipeline, make_preprocessor

from .evaluation import feature_frame, fit_predict, make_estimator, training_rows_and_labels
from .spec import ExperimentSpec


@dataclass(frozen=True)
class FusionCandidate:
    name: str
    spec: ExperimentSpec


def base_spec(pooling: str = "pooled") -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis="Primeiro estágio com toda a população madura",
        track="full",
        feature_groups=["metadata", "cadence"],
        pooling=pooling,
        model="logistic",
        complexity="low",
        evaluation_mode="new_channel",
    )


def fusion_candidates() -> tuple[FusionCandidate, ...]:
    candidates = []
    for include_transcript in (False, True):
        suffix = "transcript" if include_transcript else "audiovisual"
        groups = ["audiovisual", "audiovisual_derived"]
        if include_transcript:
            groups.append("transcript_stats")
        for model in ("logistic", "catboost", "catboost_ranker", "mlp"):
            candidates.append(
                FusionCandidate(
                    name=f"fusion_{suffix}_{model}",
                    spec=ExperimentSpec(
                        hypothesis=f"Fusão tardia {suffix} com {model}",
                        track="vlm",
                        feature_groups=groups,
                        pooling="pooled",
                        model=model,
                        complexity="low",
                        evaluation_mode="new_channel",
                        vlm_taxonomy="normalized",
                    ),
                )
            )
    candidates.extend(
        [
            FusionCandidate(
                name="fusion_transcript_mlp_medium",
                spec=ExperimentSpec(
                    hypothesis="Fusão transcrita com MLP média e regularizada",
                    track="vlm",
                    feature_groups=[
                        "audiovisual",
                        "audiovisual_derived",
                        "transcript_stats",
                    ],
                    pooling="pooled",
                    model="mlp",
                    complexity="medium",
                    evaluation_mode="new_channel",
                    vlm_taxonomy="normalized",
                ),
            ),
            FusionCandidate(
                name="fusion_transcript_mlp_clean_contrast",
                spec=ExperimentSpec(
                    hypothesis="Fusão transcrita com MLP e contraste top20 vs bottom50",
                    track="vlm",
                    feature_groups=[
                        "audiovisual",
                        "audiovisual_derived",
                        "transcript_stats",
                    ],
                    pooling="pooled",
                    model="mlp",
                    complexity="low",
                    train_label="clean20vs50",
                    evaluation_mode="new_channel",
                    vlm_taxonomy="normalized",
                ),
            ),
            FusionCandidate(
                name="fusion_transcript_mlp_medium_per_sector",
                spec=ExperimentSpec(
                    hypothesis="Fusão transcrita com MLP média, um modelo por setor",
                    track="vlm",
                    feature_groups=[
                        "audiovisual",
                        "audiovisual_derived",
                        "transcript_stats",
                    ],
                    pooling="per_sector",
                    model="mlp",
                    complexity="medium",
                    evaluation_mode="new_channel",
                    vlm_taxonomy="normalized",
                ),
            ),
        ]
    )
    return tuple(candidates)


def cross_fitted_base_scores(
    full_train: pd.DataFrame,
    target_rows: pd.DataFrame,
    *,
    spec: ExperimentSpec,
    seed: int,
    folds: int = 5,
) -> pd.Series:
    """Prevê cada linha-alvo com um modelo que não viu o canal dessa linha."""
    channels = full_train["channel_id"].nunique()
    splits = min(folds, channels)
    if splits < 2:
        return pd.Series(0.5, index=target_rows.index, dtype=float)
    scores = pd.Series(np.nan, index=target_rows.index, dtype=float)
    splitter = GroupKFold(n_splits=splits)
    for train_index, validation_index in splitter.split(
        full_train, groups=full_train["channel_id"]
    ):
        fit_rows = full_train.iloc[train_index]
        validation_channels = set(full_train.iloc[validation_index]["channel_id"])
        selected = target_rows["channel_id"].isin(validation_channels)
        if selected.any():
            scores.loc[selected] = fit_predict(
                spec, fit_rows, target_rows.loc[selected], seed
            )
    if scores.isna().any():
        raise RuntimeError("Cross-fitting deixou linhas sem previsão-base.")
    return scores


def _fusion_frame(
    spec: ExperimentSpec, rows: pd.DataFrame, base_scores: pd.Series
) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric, categorical = spec.columns()
    frame = feature_frame(spec, rows)
    frame["base_metadata_score"] = base_scores.reindex(rows.index).to_numpy()
    return frame, [*numeric, "base_metadata_score"], categorical


def fit_fusion_candidate(
    candidate: FusionCandidate,
    train: pd.DataFrame,
    train_base_scores: pd.Series,
    test: pd.DataFrame,
    test_base_scores: pd.Series,
    *,
    seed: int,
) -> np.ndarray:
    if candidate.spec.pooling == "per_sector":
        scores = pd.Series(np.nan, index=test.index, dtype=float)
        for sector, sector_test in test.groupby("sector", sort=False):
            sector_train = train[train["sector"] == sector]
            if sector_train.empty:
                sector_train = train
            scores.loc[sector_test.index] = _fit_fusion_single(
                candidate,
                sector_train,
                train_base_scores.reindex(sector_train.index),
                sector_test,
                test_base_scores.reindex(sector_test.index),
                seed=seed,
            )
        return scores.to_numpy()
    return _fit_fusion_single(
        candidate, train, train_base_scores, test, test_base_scores, seed=seed
    )


def _fit_fusion_single(
    candidate: FusionCandidate,
    train: pd.DataFrame,
    train_base_scores: pd.Series,
    test: pd.DataFrame,
    test_base_scores: pd.Series,
    *,
    seed: int,
) -> np.ndarray:
    spec = candidate.spec
    test_frame, _, _ = _fusion_frame(spec, test, test_base_scores)
    if spec.model != "catboost_ranker":
        fit_rows, labels = training_rows_and_labels(spec, train)
        train_frame, numeric, categorical = _fusion_frame(
            spec, fit_rows, train_base_scores.reindex(fit_rows.index)
        )
        pipeline = make_pipeline(
            {"features": {"numeric": numeric, "categorical": categorical}},
            make_estimator(spec, seed),
        )
        fit_params = {}
        if spec.model == "mlp":
            fit_params["model__sample_weight"] = compute_sample_weight("balanced", labels)
        pipeline.fit(train_frame, labels, **fit_params)
        return pipeline.predict_proba(test_frame)[:, 1]

    train_frame, numeric, categorical = _fusion_frame(spec, train, train_base_scores)
    ranking_group = train["channel_id"].astype(str)
    if "vlm_protocol_cohort" in train:
        ranking_group += "::" + train["vlm_protocol_cohort"].fillna("unknown").astype(str)
    order = np.argsort(ranking_group.to_numpy(), kind="stable")
    ordered_rows = train.iloc[order]
    ordered_frame = train_frame.iloc[order]
    ordered_group = ranking_group.iloc[order]
    relevance = (
        ordered_rows.assign(_group=ordered_group.to_numpy())
        .groupby("_group")["adjusted_performance"]
        .rank(method="average", pct=True)
        .to_numpy()
    )
    preprocessor = make_preprocessor(
        {"features": {"numeric": numeric, "categorical": categorical}}
    )
    transformed_train = preprocessor.fit_transform(ordered_frame)
    transformed_test = preprocessor.transform(test_frame)
    ranker = CatBoostRanker(
        loss_function="YetiRankPairwise",
        iterations=260,
        depth=4,
        learning_rate=0.035,
        l2_leaf_reg=8.0,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    ranker.fit(
        transformed_train,
        relevance,
        group_id=ordered_group.to_numpy(),
    )
    return np.asarray(ranker.predict(transformed_test), dtype=float)

