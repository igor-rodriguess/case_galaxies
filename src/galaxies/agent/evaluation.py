"""Executor determinístico: treina e avalia um ``ExperimentSpec``.

Avaliação principal: leave-one-channel-out (LOCO) sobre todos os canais fora do
cofre, o que mede a generalização para canal novo com 27 canais em vez de 2.
Checagem secundária: corte temporal (vídeos mais recentes da população madura).

Métrica principal: AUC intra-canal — entre pares (viral, não viral) do mesmo
canal, a fração em que o modelo dá nota maior ao viral. Baseline = 0,5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from galaxies.modeling.preprocessing import make_pipeline

from .channel_history import known_channel_history_features
from .protocol import (
    AgentDataset,
    apply_vlm_taxonomy,
    is_forbidden,
    within_channel_age_correlation,
)
from .spec import ExperimentSpec

TEMPORAL_TEST_QUANTILE = 0.75
MIN_SEGMENT_TRAIN_ROWS = 40
TOP_FRACTION = 0.10


def make_estimator(spec: ExperimentSpec, seed: int) -> Any:
    low = spec.complexity == "low"
    if spec.model == "logistic":
        return LogisticRegression(
            C=0.3 if low else 1.0,
            class_weight="balanced",
            max_iter=3000,
        )
    if spec.model == "hist_gb":
        return HistGradientBoostingClassifier(
            max_depth=2 if low else 3,
            learning_rate=0.05,
            max_iter=150 if low else 300,
            min_samples_leaf=30 if low else 15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    forest = ExtraTreesClassifier if spec.model == "extra_trees" else RandomForestClassifier
    return forest(
        n_estimators=300,
        min_samples_leaf=10 if low else 4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


HISTORY_NUMERIC = [
    "known_channel_format_viral_rate",
    "known_channel_duration_viral_rate",
    "known_channel_history_available",
]


def feature_frame(
    spec: ExperimentSpec, rows: pd.DataFrame, history_features: pd.DataFrame | None = None
) -> pd.DataFrame:
    numeric, categorical = spec.columns()
    columns = numeric + categorical
    if any(is_forbidden(column) for column in columns):
        raise ValueError("Spec com feature proibida chegou ao executor.")
    frame = rows[columns].copy()
    if "audiovisual" in spec.feature_groups and spec.vlm_taxonomy == "normalized":
        frame = apply_vlm_taxonomy(frame)
    for column in categorical:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    if history_features is not None:
        history = history_features.set_index("video_id")[HISTORY_NUMERIC]
        frame = frame.join(history, on=rows["video_id"])
    return frame


def _fit_predict(
    spec: ExperimentSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    *,
    history_pool: pd.DataFrame | None = None,
) -> np.ndarray:
    train, labels = training_rows_and_labels(spec, train)
    if len(train) == 0 or len(np.unique(labels)) < 2:
        fallback = float(labels.mean()) if len(labels) else 0.1
        return np.full(len(test), fallback)
    numeric, categorical = spec.columns()
    train_history = test_history = None
    if history_pool is not None:
        train_history = known_channel_history_features(train, history_pool)
        test_history = known_channel_history_features(test, history_pool)
        numeric += HISTORY_NUMERIC
    pipeline = make_pipeline(
        {"features": {"numeric": numeric, "categorical": categorical}},
        make_estimator(spec, seed),
    )
    pipeline.fit(feature_frame(spec, train, train_history), labels)
    return pipeline.predict_proba(feature_frame(spec, test, test_history))[:, 1]


def training_rows_and_labels(
    spec: ExperimentSpec, train: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Constrói o rótulo de treino sem consultar vídeos da dobra de teste."""
    if spec.train_label == "top10":
        return train, train["viral"].to_numpy()

    pieces: list[pd.DataFrame] = []
    for _, channel_rows in train.groupby("channel_id", sort=False):
        performance = channel_rows["adjusted_performance"]
        high = performance >= performance.quantile(0.80)
        low = performance <= performance.quantile(0.50)
        selected = channel_rows.loc[high | low].copy()
        selected["_training_label"] = high.loc[selected.index].astype(int)
        pieces.append(selected)
    selected_train = pd.concat(pieces) if pieces else train.iloc[0:0].copy()
    return selected_train, selected_train.get("_training_label", pd.Series(dtype=int)).to_numpy()


def fit_predict(
    spec: ExperimentSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    *,
    known_channel: bool = False,
) -> np.ndarray:
    """Aplica as escolhas de treino do spec (maturidade, setor, formato)."""
    train = train[train["target_age_days"] >= spec.train_min_age_days]
    scores = np.zeros(len(test))
    position = pd.Series(np.arange(len(test)), index=test.index)

    sector_groups = (
        [(sector, test[test["sector"] == sector]) for sector in test["sector"].unique()]
        if spec.pooling == "per_sector"
        else [(None, test)]
    )
    for sector, sector_test in sector_groups:
        sector_train = train if sector is None else train[train["sector"] == sector]
        if spec.format_strategy == "segmented":
            for segment in sector_test["segment"].unique():
                segment_test = sector_test[sector_test["segment"] == segment]
                segment_train = sector_train[sector_train["segment"] == segment]
                enough = (
                    len(segment_train) >= MIN_SEGMENT_TRAIN_ROWS
                    and len(np.unique(training_rows_and_labels(spec, segment_train)[1])) == 2
                )
                source = segment_train if enough else sector_train
                if known_channel:
                    scores[position.loc[segment_test.index]] = _fit_predict(
                        spec, source, segment_test, seed, history_pool=sector_train
                    )
                else:
                    scores[position.loc[segment_test.index]] = _fit_predict(
                        spec, source, segment_test, seed
                    )
        else:
            if known_channel:
                scores[position.loc[sector_test.index]] = _fit_predict(
                    spec, sector_train, sector_test, seed, history_pool=sector_train
                )
            else:
                scores[position.loc[sector_test.index]] = _fit_predict(
                    spec, sector_train, sector_test, seed
                )
    return scores


# --- Métricas -----------------------------------------------------------------


def channel_pair_counts(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Por canal: pares (viral, não viral) e quantos o modelo ordena certo."""
    rows = []
    for channel, group in frame.groupby("channel_id", sort=True):
        positives = group.loc[group["viral"] == 1, score_column].to_numpy()
        negatives = group.loc[group["viral"] == 0, score_column].to_numpy()
        total = len(positives) * len(negatives)
        if total == 0:
            correct = 0.0
        else:
            diff = positives[:, None] - negatives[None, :]
            correct = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        rows.append({"channel_id": channel, "pairs": total, "correct": correct})
    return pd.DataFrame(rows).set_index("channel_id")


def auc_intra_channel(frame: pd.DataFrame, score_column: str) -> float:
    counts = channel_pair_counts(frame, score_column)
    total = counts["pairs"].sum()
    return float(counts["correct"].sum() / total) if total else float("nan")


def top_fraction_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    """Marca o top 10% de cada canal pela nota e mede recall/precisão nos virais."""
    flagged_hits = flagged = positives = 0
    for _, group in frame.groupby("channel_id"):
        k = max(1, math.ceil(TOP_FRACTION * len(group)))
        top = group.nlargest(k, score_column)
        flagged_hits += int(top["viral"].sum())
        flagged += k
        positives += int(group["viral"].sum())
    base_rate = positives / len(frame) if len(frame) else float("nan")
    precision = flagged_hits / flagged if flagged else float("nan")
    return {
        "recall_top10": flagged_hits / positives if positives else float("nan"),
        "precision_top10": precision,
        "lift_top10": precision / base_rate if base_rate else float("nan"),
    }


def summarize_scores(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    labels = frame["viral"].to_numpy()
    summary = {
        "auc_intra": auc_intra_channel(frame, score_column),
        "auc_global": float(roc_auc_score(labels, frame[score_column]))
        if len(np.unique(labels)) == 2
        else float("nan"),
        "rows": len(frame),
        "positives": int(labels.sum()),
        "channels": int(frame["channel_id"].nunique()),
    }
    summary.update(top_fraction_metrics(frame, score_column))
    if "target_age_days" in frame.columns:
        summary["age_rho"] = within_channel_age_correlation(frame, score_column)
    return summary


def bootstrap_auc(
    counts: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """IC95% da AUC intra-canal reamostrando canais (não vídeos)."""
    rng = np.random.default_rng(seed)
    pairs = counts["pairs"].to_numpy()
    correct = counts["correct"].to_numpy()
    values = []
    for _ in range(iterations):
        pick = rng.integers(0, len(counts), len(counts))
        total = pairs[pick].sum()
        if total:
            values.append(correct[pick].sum() / total)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_comparison(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Diferença de AUC intra-canal com os MESMOS canais reamostrados nos dois."""
    joined = candidate.join(reference, lsuffix="_c", rsuffix="_r", how="inner")
    joined = joined[joined["pairs_c"] > 0]
    if joined.empty:
        return {
            "diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_better": 0.0,
        }
    pairs = joined["pairs_c"].to_numpy()
    correct_c = joined["correct_c"].to_numpy()
    correct_r = joined["correct_r"].to_numpy()
    observed = (correct_c.sum() - correct_r.sum()) / pairs.sum()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iterations):
        pick = rng.integers(0, len(joined), len(joined))
        total = pairs[pick].sum()
        diffs.append((correct_c[pick].sum() - correct_r[pick].sum()) / total)
    diffs_array = np.asarray(diffs)
    return {
        "diff": float(observed),
        "ci_low": float(np.quantile(diffs_array, 0.025)),
        "ci_high": float(np.quantile(diffs_array, 0.975)),
        "p_better": float((diffs_array > 0).mean()),
    }


# --- Execução -----------------------------------------------------------------


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    loco: dict[str, float]
    temporal: dict[str, float]
    predictions: pd.DataFrame  # video_id, channel_id, sector, viral, score_loco
    temporal_predictions: pd.DataFrame
    loco_counts: pd.DataFrame
    temporal_counts: pd.DataFrame


def temporal_split(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserva os 25% mais recentes de CADA canal para o modo conhecido.

    Um corte global pode deixar canais inteiros apenas no treino ou apenas no
    teste. Isso mede mudança de composição entre canais, e não a capacidade de
    prever o próximo criativo de um canal que já possui histórico.
    """
    test_indices: list[object] = []
    for _, channel_rows in rows.groupby("channel_id", sort=False):
        ordered = channel_rows.sort_values("published_at", kind="stable")
        test_size = max(1, int(np.ceil(len(ordered) * (1 - TEMPORAL_TEST_QUANTILE))))
        if len(ordered) <= test_size:
            continue
        test_indices.extend(ordered.index[-test_size:])
    test = rows.loc[test_indices].copy()
    train = rows.drop(index=test.index).copy()
    if test.empty or train.empty:
        raise ValueError("Dados insuficientes para divisão temporal por canal.")
    return train, test


def run_experiment(
    spec: ExperimentSpec,
    dataset: AgentDataset,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 1000,
) -> ExperimentResult:
    rows = dataset.evaluable(spec.track)
    if rows["channel_id"].isin(dataset.vault_channels).any():
        raise RuntimeError("Linha do cofre entrou na avaliação.")

    loco_scores = pd.Series(np.nan, index=rows.index)
    for channel in sorted(rows["channel_id"].unique()):
        test = rows[rows["channel_id"] == channel]
        train = rows[rows["channel_id"] != channel]
        loco_scores.loc[test.index] = fit_predict(spec, train, test, seed)

    predictions = rows[["video_id", "channel_id", "sector", "viral", "target_age_days"]].copy()
    predictions["score_loco"] = loco_scores
    loco = summarize_scores(predictions, "score_loco")
    loco_counts = channel_pair_counts(predictions, "score_loco")
    loco["auc_intra_ci_low"], loco["auc_intra_ci_high"] = bootstrap_auc(
        loco_counts, bootstrap_iterations, seed
    )

    train, test = temporal_split(rows)
    temporal_predictions = test[
        ["video_id", "channel_id", "sector", "viral", "target_age_days"]
    ].copy()
    if spec.evaluation_mode == "known_channel":
        temporal_predictions["score_temporal"] = fit_predict(
            spec, train, test, seed, known_channel=True
        )
    else:
        temporal_predictions["score_temporal"] = fit_predict(spec, train, test, seed)
    temporal = summarize_scores(temporal_predictions, "score_temporal")
    temporal_counts = channel_pair_counts(temporal_predictions, "score_temporal")
    temporal["auc_intra_ci_low"], temporal["auc_intra_ci_high"] = bootstrap_auc(
        temporal_counts, bootstrap_iterations, seed
    )
    return ExperimentResult(
        spec,
        loco,
        temporal,
        predictions,
        temporal_predictions,
        loco_counts,
        temporal_counts,
    )


def evaluate_on_vault(
    spec: ExperimentSpec,
    dataset: AgentDataset,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 1000,
) -> dict[str, float]:
    """Treina com tudo fora do cofre e mede uma única vez nos canais trancados."""
    train = dataset.evaluable(spec.track)
    test = dataset.vault(spec.track)
    if test.empty:
        return {"rows": 0}
    frame = test[["video_id", "channel_id", "sector", "viral", "target_age_days"]].copy()
    frame["score_vault"] = fit_predict(spec, train, test, seed)
    summary = summarize_scores(frame, "score_vault")
    counts = channel_pair_counts(frame, "score_vault")
    summary["auc_intra_ci_low"], summary["auc_intra_ci_high"] = bootstrap_auc(
        counts, bootstrap_iterations, seed
    )
    return summary
