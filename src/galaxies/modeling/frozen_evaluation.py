"""Avaliação final, causal e reproduzível da base VLM congelada.

Este módulo não reutiliza os rótulos exploratórios anteriores. Cada alvo relativo
é calculado com vídeos publicados antes do vídeo avaliado; views, likes e
comentários nunca entram como feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

SECTORS = ("automotivo", "moda", "beleza")
METADATA_NUMERIC = (
    "duration_seconds",
    "title_character_count",
    "description_character_count",
    "tag_count",
    "publish_hour_utc",
    "days_since_previous_upload",
    "prior_upload_interval_median_days",
)
METADATA_CATEGORICAL = ("category_id", "publish_weekday_utc")
VLM_NUMERIC = (
    "vlm_mean_confidence",
    "people_count_estimate",
    "on_screen_text_count",
    "scene_change_count",
    "speech_start_seconds",
    "transcript_excerpt_word_count",
    "transcript_character_count",
    "transcript_unique_word_ratio",
    "transcript_words_per_second",
    "transcript_question_mark_count",
    "transcript_exclamation_mark_count",
    "transcript_number_token_count",
    "sound_effect_count",
    "evidence_count",
    "uncertainty_count",
)
VLM_CATEGORICAL = (
    "people_present",
    "faces_visible",
    "product_in_focus",
    "brand_legible",
    "framing",
    "camera_movement",
    "motion_intensity",
    "speech_present",
    "music_present",
    "opening_type",
    "product_prominence",
    "call_to_action_present",
    "audio_language",
    "transcript_second_person_cue",
    "transcript_call_to_action_cue",
)
COUNT_COLUMNS = {
    "duration_seconds",
    "title_character_count",
    "description_character_count",
    "tag_count",
    "on_screen_text_count",
    "scene_change_count",
    "transcript_excerpt_word_count",
    "transcript_character_count",
    "transcript_question_mark_count",
    "transcript_exclamation_mark_count",
    "transcript_number_token_count",
    "sound_effect_count",
    "evidence_count",
    "uncertainty_count",
}


@dataclass(frozen=True)
class Recipe:
    name: str
    estimator: Any
    scaler: str
    selector: str


def causal_relative_targets(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    excluded_curve_channels: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Calcula duas definições de alvo usando estritamente o passado publicado.

    ``residual_age_channel`` usa uma curva logarítmica por setor e um efeito de
    canal suavizado. ``smoothed_ratio`` compara views brutas com a mediana
    suavizada do histórico do canal, sem corrigir idade. Nenhuma observação
    posterior é consultada.

    ``excluded_curve_channels`` tira canais reservados da curva setorial e da
    mediana do setor. O efeito do próprio canal continua vindo só dos vídeos
    anteriores dele, então um canal reservado nunca molda o alvo de outro canal.
    """
    required = {"sector", "channel_id", "published_at", "target_age_days", "target_view_count"}
    missing = required - set(history.columns) | required - set(targets.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o alvo causal: {sorted(missing)}")
    source = history.copy()
    source["published_at"] = pd.to_datetime(source["published_at"], utc=True)
    source["_age_log"] = np.log1p(pd.to_numeric(source["target_age_days"], errors="coerce"))
    source["_view_log"] = np.log1p(pd.to_numeric(source["target_view_count"], errors="coerce"))
    source = source.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["published_at", "_age_log", "_view_log"]
    )
    out = targets[["video_id", "sector", "channel_id", "published_at", "target_age_days"]].copy()
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    out["_age_log"] = np.log1p(pd.to_numeric(out["target_age_days"], errors="coerce"))
    out["residual_age_channel"] = np.nan
    out["smoothed_ratio"] = np.nan
    out["history_rows"] = 0
    out["channel_history_rows"] = 0

    for index, row in out.sort_values("published_at", kind="stable").iterrows():
        prior = source.loc[
            (source["sector"] == row["sector"]) & (source["published_at"] < row["published_at"])
        ]
        own_channel = prior["channel_id"].astype(str) == str(row["channel_id"])
        excluded = prior["channel_id"].astype(str).isin(excluded_curve_channels)
        earlier = prior.loc[~excluded | own_channel]
        curve = prior.loc[~excluded]
        if len(curve) < 5 or pd.isna(row["_age_log"]):
            continue
        x = curve["_age_log"].to_numpy(float)
        y = curve["_view_log"].to_numpy(float)
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)) if np.var(x) > 1e-10 else 0.0
        global_expected = float(np.median(y) + slope * (row["_age_log"] - np.median(x)))
        channel = earlier.loc[earlier["channel_id"].astype(str) == str(row["channel_id"])]
        channel_count = len(channel)
        observed_views = float(
            pd.to_numeric(targets.loc[index, "target_view_count"], errors="coerce")
        )
        if channel_count:
            channel_residual = channel["_view_log"] - (
                np.median(y) + slope * (channel["_age_log"] - np.median(x))
            )
            weight = channel_count / (channel_count + 5.0)
            expected_a = global_expected + weight * float(np.median(channel_residual))
            channel_median_views = float(
                np.median(pd.to_numeric(channel["target_view_count"], errors="coerce"))
            )
            sector_median_views = float(
                np.median(pd.to_numeric(curve["target_view_count"], errors="coerce"))
            )
            expected_b_views = weight * channel_median_views + (1.0 - weight) * sector_median_views
        else:
            expected_a = global_expected
            expected_b_views = float(
                np.median(pd.to_numeric(curve["target_view_count"], errors="coerce"))
            )
        observed = float(np.log1p(observed_views))
        if np.isfinite(observed) and np.isfinite(observed_views) and np.isfinite(expected_b_views):
            smoothing = max(10.0, 0.01 * expected_b_views)
            out.loc[index, "residual_age_channel"] = observed - expected_a
            out.loc[index, "smoothed_ratio"] = (observed_views + smoothing) / (
                expected_b_views + smoothing
            )
            out.loc[index, "history_rows"] = len(curve)
            out.loc[index, "channel_history_rows"] = channel_count
    return out.drop(columns="_age_log")


def target_diagnostics(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    age = pd.to_numeric(targets["target_age_days"], errors="coerce")
    for column in ("residual_age_channel", "smoothed_ratio"):
        values = pd.to_numeric(targets[column], errors="coerce")
        valid = values.notna() & age.notna()
        rho = spearmanr(values[valid], age[valid]).statistic if valid.sum() > 2 else np.nan
        rows.append(
            {
                "definition": column,
                "rows": int(valid.sum()),
                "age_spearman": float(rho) if np.isfinite(rho) else None,
                "absolute_age_spearman": float(abs(rho)) if np.isfinite(rho) else None,
                "median": float(values[valid].median()) if valid.any() else None,
                "iqr": float(values[valid].quantile(0.75) - values[valid].quantile(0.25))
                if valid.any()
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["absolute_age_spearman", "definition"])


def prepare_feature_frame(
    frame: pd.DataFrame, group: str, train_indices: np.ndarray
) -> tuple[pd.DataFrame, list[str], list[str]]:
    group_columns = {
        "metadata": (METADATA_NUMERIC, METADATA_CATEGORICAL),
        "metadata_vlm": (METADATA_NUMERIC + VLM_NUMERIC, METADATA_CATEGORICAL + VLM_CATEGORICAL),
        "metadata_thumbnail": (METADATA_NUMERIC, METADATA_CATEGORICAL),
        "all": (METADATA_NUMERIC + VLM_NUMERIC, METADATA_CATEGORICAL + VLM_CATEGORICAL),
    }
    numeric, categorical = group_columns[group]
    train_rows = frame.iloc[train_indices]
    usable_numeric = [name for name in numeric if name in frame and train_rows[name].notna().any()]
    usable_categorical = [
        name for name in categorical if name in frame and train_rows[name].notna().any()
    ]
    data = pd.DataFrame(index=frame.index)
    for name in usable_numeric:
        values = pd.to_numeric(frame[name], errors="coerce")
        data[name] = np.log1p(values.clip(lower=0)) if name in COUNT_COLUMNS else values
    for name in usable_categorical:
        values = frame[name]
        if name == "brand_legible":
            values = values.fillna(False).astype(bool).map({True: "true", False: "false"})
        data[name] = values.astype("object")
    if group in {"metadata_thumbnail", "all"}:
        vectors = np.vstack(frame["thumbnail_embedding"].to_numpy())
        components = min(16, vectors.shape[0] - 1, vectors.shape[1])
        if components < 2:
            raise ValueError("Embeddings de thumbnail insuficientes para PCA.")
        pca = PCA(n_components=components, random_state=2026)
        pca.fit(vectors[train_indices])
        transformed = pca.transform(vectors)
        names = [f"thumbnail_pca_{position:02d}" for position in range(components)]
        data[names] = transformed
        usable_numeric.extend(names)
    return data, usable_numeric, usable_categorical


def _recipes(seed: int) -> list[Recipe]:
    return [
        Recipe("ridge_no_selection", Ridge(alpha=10.0), "standard", "none"),
        Recipe("ridge_f_test", Ridge(alpha=10.0), "robust", "f_test"),
        Recipe(
            "elastic_net_mi",
            ElasticNet(alpha=0.03, l1_ratio=0.35, max_iter=4_000),
            "robust",
            "mutual_info",
        ),
        # Execução serial: a versão atual do scikit-learn emite milhares de
        # avisos ao aninhar paralelismo durante permutation importance.
        Recipe(
            "random_forest",
            RandomForestRegressor(
                n_estimators=350, min_samples_leaf=3, max_features=0.7, n_jobs=1, random_state=seed
            ),
            "none",
            "extra_trees",
        ),
        Recipe(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=seed,
            ),
            "none",
            "none",
        ),
        Recipe(
            "mlp_regularized",
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                alpha=0.03,
                early_stopping=True,
                validation_fraction=0.18,
                max_iter=600,
                random_state=seed,
            ),
            "robust",
            "f_test",
        ),
    ]


def _pipeline(numeric: list[str], categorical: list[str], recipe: Recipe, seed: int) -> Pipeline:
    scaler: Any = "passthrough"
    if recipe.scaler == "standard":
        scaler = StandardScaler()
    elif recipe.scaler == "robust":
        scaler = RobustScaler(quantile_range=(10, 90))
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", scaler),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    if recipe.selector == "f_test":
        selector: Any = SelectPercentile(f_regression, percentile=65)
    elif recipe.selector == "mutual_info":
        selector = SelectPercentile(
            lambda x, y: mutual_info_regression(x, y, random_state=seed), percentile=65
        )
    elif recipe.selector == "extra_trees":
        selector = SelectFromModel(
            ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, n_jobs=1, random_state=seed),
            threshold="median",
        )
    else:
        selector = "passthrough"
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("variance", VarianceThreshold()),
            ("select", selector),
            ("model", clone(recipe.estimator)),
        ]
    )


def _metrics(
    y: np.ndarray, prediction: np.ndarray, train_y: np.ndarray, *, seed: int = 2026
) -> dict[str, Any]:
    error = y - prediction
    rho = spearmanr(y, prediction).statistic if len(y) > 2 and np.ptp(prediction) else np.nan
    q_low, q_high = np.quantile(train_y, [0.33, 0.67])
    bands = np.where(y <= q_low, "baixo", np.where(y >= q_high, "alto", "tipico"))
    per_band = {
        band: float(np.mean(np.abs(error[bands == band])))
        for band in ("baixo", "tipico", "alto")
        if (bands == band).any()
    }
    k = max(3, int(np.ceil(len(y) * 0.2)))
    actual_top = set(np.argsort(y)[-k:])
    predicted_top = set(np.argsort(prediction)[-k:])
    hits = len(actual_top & predicted_top)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(y), size=(1_000, len(y)))
    boot_mae = np.mean(np.abs(error[samples]), axis=1)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else None,
        "spearman": float(rho) if np.isfinite(rho) else None,
        "mae_ci95": [float(np.quantile(boot_mae, 0.025)), float(np.quantile(boot_mae, 0.975))],
        "mae_by_band": per_band,
        "precision_at_k": hits / k,
        "recall_at_k": hits / k,
        "lift_at_k": (hits / k) / (k / len(y)),
        "k": k,
    }


def _inner_splits(
    frame: pd.DataFrame, train_indices: np.ndarray, split_kind: str = "temporal"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dobras internas do treino que imitam a pergunta do holdout externo.

    ``temporal`` usa janelas crescentes no tempo; ``channel`` separa canais
    inteiros, para que a escolha da receita meça generalização a canal novo.
    """
    if split_kind == "channel":
        groups = frame.loc[train_indices, "channel_id"].astype(str).to_numpy()
        folds = min(3, len(np.unique(groups)))
        if folds < 2:
            raise ValueError("Seleção por canal exige pelo menos dois canais no treino.")
        return [
            (train_indices[left], train_indices[right])
            for left, right in GroupKFold(n_splits=folds).split(train_indices, groups=groups)
        ]
    ordered = train_indices[
        np.argsort(pd.to_datetime(frame.loc[train_indices, "published_at"], utc=True).to_numpy())
    ]
    split_count = min(3, max(2, len(ordered) // 25))
    return [
        (ordered[left], ordered[right])
        for left, right in TimeSeriesSplit(n_splits=split_count).split(ordered)
    ]


def _select_recipe(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    split_kind: str = "temporal",
) -> tuple[Recipe, list[dict[str, Any]]]:
    scores = []
    for recipe in _recipes(seed):
        fold_errors = []
        try:
            for left, right in _inner_splits(frame, train_indices, split_kind):
                fitted = _pipeline(numeric, categorical, recipe, seed).fit(x.loc[left], y[left])
                fold_errors.append(mean_absolute_error(y[right], fitted.predict(x.loc[right])))
            scores.append(
                {"recipe": recipe.name, "inner_mae": float(np.mean(fold_errors)), "status": "ok"}
            )
        except Exception as exc:
            scores.append(
                {
                    "recipe": recipe.name,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    successful = [item for item in scores if item["status"] == "ok"]
    if not successful:
        raise RuntimeError("Nenhuma receita concluiu a validação interna.")
    winner = min(successful, key=lambda item: item["inner_mae"])["recipe"]
    return next(recipe for recipe in _recipes(seed) if recipe.name == winner), scores


def _classification(y: np.ndarray, prediction: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    low, high = np.quantile(train_y, [0.33, 0.67])
    truth = np.where(y <= low, "baixo", np.where(y >= high, "alto", "tipico"))
    predicted = np.where(prediction <= low, "baixo", np.where(prediction >= high, "alto", "tipico"))
    high_truth = (truth == "alto").astype(int)
    order = ["baixo", "tipico", "alto"]
    bins = pd.qcut(
        pd.Series(prediction).rank(method="first"), q=min(5, len(prediction)), duplicates="drop"
    )
    calibration = (
        pd.DataFrame({"bin": bins, "pred": prediction, "high": high_truth})
        .groupby("bin", observed=True)
        .agg(predicted_mean=("pred", "mean"), observed_high=("high", "mean"), rows=("high", "size"))
        .reset_index(drop=True)
    )
    return {
        "thresholds_from_train": {"low": float(low), "high": float(high)},
        "macro_f1": float(
            f1_score(truth, predicted, labels=order, average="macro", zero_division=0)
        ),
        "pr_auc_high": float(average_precision_score(high_truth, prediction))
        if high_truth.sum()
        else None,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=order).tolist(),
        "labels": order,
        "calibration": calibration.to_dict(orient="records"),
    }


def _conformal(
    recipe: Recipe,
    x: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    frame: pd.DataFrame | None = None,
    split_kind: str = "temporal",
) -> dict[str, Any]:
    """Split conformal com o mesmo modelo cuja previsão é reportada.

    A calibração imita o holdout: no temporal, os 20% mais recentes do treino;
    no de canal, canais inteiros do treino que o ajuste próprio não vê.
    """
    if frame is not None and split_kind == "channel":
        groups = frame.loc[train, "channel_id"].astype(str).to_numpy()
        if len(np.unique(groups)) < 3:
            return {"available": False, "reason": "canais insuficientes para calibrar"}
        left, right = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
                train, groups=groups
            )
        )
        proper, calibration = train[left], train[right]
    else:
        if frame is not None:
            train = train[
                np.argsort(
                    pd.to_datetime(frame.loc[train, "published_at"], utc=True).to_numpy(),
                    kind="stable",
                )
            ]
        cutoff = max(10, int(len(train) * 0.8))
        proper, calibration = train[:cutoff], train[cutoff:]
    if len(calibration) < 8:
        return {"available": False, "reason": "calibração insuficiente"}
    fitted = _pipeline(numeric, categorical, recipe, seed).fit(x.loc[proper], y[proper])
    residuals = np.abs(y[calibration] - fitted.predict(x.loc[calibration]))
    quantile = float(np.quantile(residuals, 0.9, method="higher"))
    prediction = fitted.predict(x.loc[test])
    return {
        "available": True,
        "recipe": recipe.name,
        "calibration": split_kind,
        "calibration_rows": len(calibration),
        "absolute_residual_quantile_90": quantile,
        "coverage_test": float(
            ((y[test] >= prediction - quantile) & (y[test] <= prediction + quantile)).mean()
        ),
        "mean_interval_width": float(2 * quantile),
    }


def _ood_scores(
    pipeline: Pipeline,
    x: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    """Distância ao 5º vizinho do treino; ``exclude_self`` ignora o próprio ponto."""
    transformed_train = pipeline[:-1].transform(x.loc[train])
    transformed_test = pipeline[:-1].transform(x.loc[test])
    k = min(5 + int(exclude_self), len(train))
    neighbors = NearestNeighbors(n_neighbors=k).fit(transformed_train)
    return neighbors.kneighbors(transformed_test)[0][:, -1]


def evaluate_outer_split(
    frame: pd.DataFrame,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    group: str,
    seed: int = 2026,
    full_comparison: bool = True,
    split_kind: str = "temporal",
) -> dict[str, Any]:
    """Avalia um holdout externo com seleção estritamente interna.

    O modelo reportado é a receita de menor MAE nas dobras internas do treino.
    As métricas dos outros candidatos no holdout são só diagnóstico e nunca
    escolhem o vencedor, o intervalo, o OOD ou a classificação.
    """
    x, numeric, categorical = prepare_feature_frame(frame, group, train)
    selected, inner = _select_recipe(
        frame, x, target, train, numeric, categorical, seed, split_kind
    )
    recipes = _recipes(seed) if full_comparison else [selected]
    candidates: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for recipe in recipes:
        try:
            model = _pipeline(numeric, categorical, recipe, seed).fit(x.loc[train], target[train])
            prediction = model.predict(x.loc[test])
            candidates.append(
                {
                    "model": recipe.name,
                    "kind": "base",
                    "metrics": _metrics(target[test], prediction, target[train], seed=seed),
                    "prediction": prediction.tolist(),
                }
            )
            fitted[recipe.name] = model
        except Exception as exc:
            candidates.append(
                {
                    "model": recipe.name,
                    "kind": "base",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    baseline_prediction = np.zeros(len(test))
    candidates.insert(
        0,
        {
            "model": "historical_baseline",
            "kind": "baseline",
            "metrics": _metrics(target[test], baseline_prediction, target[train], seed=seed),
            "prediction": baseline_prediction.tolist(),
        },
    )
    members = [
        name
        for name in ("ridge_f_test", "random_forest", "hist_gradient_boosting", "mlp_regularized")
        if name in fitted
    ]
    if len(members) >= 2:
        matrix, truth = [], []
        for left, right in _inner_splits(frame, train, split_kind):
            rows = []
            for name in members:
                recipe = next(item for item in _recipes(seed) if item.name == name)
                rows.append(
                    _pipeline(numeric, categorical, recipe, seed)
                    .fit(x.loc[left], target[left])
                    .predict(x.loc[right])
                )
            matrix.append(np.column_stack(rows))
            truth.append(target[right])
        oof, oof_y = np.vstack(matrix), np.concatenate(truth)
        weights, _ = nnls(oof, oof_y)
        weights = (
            weights / weights.sum() if weights.sum() else np.full(len(members), 1 / len(members))
        )
        test_matrix = np.column_stack([fitted[name].predict(x.loc[test]) for name in members])
        blend = test_matrix @ weights
        candidates.append(
            {
                "model": "ensemble_oof",
                "kind": "ensemble",
                "members": members,
                "weights": weights.tolist(),
                "metrics": _metrics(target[test], blend, target[train], seed=seed),
                "prediction": blend.tolist(),
            }
        )
    for row in candidates:
        row["selected_internally"] = row["model"] == selected.name
        row["used_for_selection"] = False
    winner = next(row for row in candidates if row["model"] == selected.name)
    if "metrics" not in winner:
        raise RuntimeError(f"A receita escolhida internamente falhou no ajuste: {selected.name}")
    winner_prediction = np.asarray(winner["prediction"], dtype=float)
    pipeline = fitted[selected.name]
    ood = _ood_scores(pipeline, x, train, test)
    threshold = float(np.quantile(_ood_scores(pipeline, x, train, train, exclude_self=True), 0.95))
    predictions = frame.loc[test, ["video_id", "sector", "channel_id", "published_at"]].copy()
    predictions["actual"] = target[test]
    predictions["predicted"] = winner_prediction
    predictions["residual"] = winner_prediction - target[test]
    predictions["ood_distance"] = ood
    predictions["low_confidence_ood"] = ood > threshold
    return {
        "group": group,
        "selected_recipe": selected.name,
        "inner_selection": inner,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "candidates": candidates,
        "winner": winner["model"],
        "winner_metrics": winner["metrics"],
        "classification": _classification(target[test], winner_prediction, target[train]),
        "split_kind": split_kind,
        "conformal": _conformal(
            selected, x, target, train, test, numeric, categorical, seed, frame, split_kind
        ),
        "predictions": predictions,
        "pipeline": pipeline,
        "x": x,
    }


def make_splits(frame: pd.DataFrame, seed: int = 2026) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ordered = np.argsort(pd.to_datetime(frame["published_at"], utc=True).to_numpy())
    test_size = max(20, int(np.ceil(len(frame) * 0.2)))
    temporal = (ordered[:-test_size], ordered[-test_size:])
    groups = frame["channel_id"].astype(str)
    if groups.nunique() < 4:
        return {"temporal": temporal}
    left, right = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups)
    )
    return {"temporal": temporal, "channel_holdout": (left, right)}
