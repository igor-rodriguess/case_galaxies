"""Avaliação pré-registrada de embeddings de thumbnail e texto.

Este módulo não altera o protocolo ``viral-intra-canal-v1``. Ele existe fora
do catálogo do agente até que a ablação seja medida: assim, PCA e vetores não
viram uma feature promovida antes de demonstrarem ganho em canais novos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from galaxies.modeling.preprocessing import make_preprocessor

from .evaluation import (
    bootstrap_auc,
    channel_pair_counts,
    summarize_scores,
    training_rows_and_labels,
)
from .protocol import CADENCE_NUMERIC, METADATA_CATEGORICAL, METADATA_NUMERIC, load_dataset
from .temporal_knn import temporal_other_channel_knn

TEXT_STAT_COLUMNS = [
    "title_word_count",
    "title_question_count",
    "title_exclamation_count",
    "title_number_token_count",
    "title_uppercase_ratio",
    "description_word_count",
]
BASE_NUMERIC = list(dict.fromkeys(METADATA_NUMERIC + CADENCE_NUMERIC + TEXT_STAT_COLUMNS))
BASE_CATEGORICAL = METADATA_CATEGORICAL


@dataclass(frozen=True)
class EmbeddingConfig:
    """Uma das configurações explicitamente previstas no roteiro do projeto."""

    name: str
    embeddings: tuple[str, ...] = ()
    pca_components: int | None = None
    knn_embedding: str | None = None
    cluster_embedding: str | None = None
    cluster_count: int | None = None
    estimator: str = "hist_gb"
    train_label: str = "top10"


PRE_REGISTERED_CONFIGS = (
    EmbeddingConfig("reference"),
    EmbeddingConfig("reference_plus_thumb_pca16", ("thumb",), 16),
    EmbeddingConfig("reference_plus_thumb_pca32", ("thumb",), 32),
    EmbeddingConfig("reference_plus_title_pca16", ("title",), 16),
    EmbeddingConfig("reference_plus_title_pca32", ("title",), 32),
    EmbeddingConfig("reference_plus_thumb_title_pca16", ("thumb", "title"), 16),
    EmbeddingConfig("reference_plus_thumb_title_pca32", ("thumb", "title"), 32),
    EmbeddingConfig("reference_plus_temporal_knn_thumb", knn_embedding="thumb"),
    # Extensões pequenas e pré-definidas: contrastam a referência limpa com
    # CatBoost e clusters sem usar rótulo, teste ou canal do cofre para aprender
    # qualquer transformação.
    EmbeddingConfig("clean20vs50_reference", train_label="clean20vs50"),
    EmbeddingConfig(
        "clean20vs50_plus_temporal_knn_thumb",
        knn_embedding="thumb",
        train_label="clean20vs50",
    ),
    EmbeddingConfig(
        "clean20vs50_thumb_cluster8_histgb",
        cluster_embedding="thumb",
        cluster_count=8,
        train_label="clean20vs50",
    ),
    EmbeddingConfig(
        "clean20vs50_thumb_pca16_catboost",
        embeddings=("thumb",),
        pca_components=16,
        estimator="catboost",
        train_label="clean20vs50",
    ),
    EmbeddingConfig(
        "clean20vs50_thumb_pca16_knn_catboost",
        embeddings=("thumb",),
        pca_components=16,
        knn_embedding="thumb",
        estimator="catboost",
        train_label="clean20vs50",
    ),
    EmbeddingConfig(
        "clean20vs50_thumb_cluster8_catboost",
        cluster_embedding="thumb",
        cluster_count=8,
        estimator="catboost",
        train_label="clean20vs50",
    ),
    EmbeddingConfig(
        "clean20vs50_thumb_cluster8_knn_catboost",
        knn_embedding="thumb",
        cluster_embedding="thumb",
        cluster_count=8,
        estimator="catboost",
        train_label="clean20vs50",
    ),
)


def latest_normalized_records(project_root: Path) -> Path:
    candidates: list[tuple[Path, int]] = []
    for run in (project_root / "data" / "youtube" / "runs").iterdir():
        path = run / "normalized_records.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates.append((path, len(payload.get("records", []))))
    if not candidates:
        raise FileNotFoundError("Nenhum normalized_records.json foi encontrado.")
    return max(candidates, key=lambda item: (item[1], item[0].stat().st_mtime_ns))[0]


def _words(value: object) -> list[str]:
    return str(value or "").replace("\n", " ").split()


def text_statistics(records: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas determinísticas de texto, disponíveis antes da publicação."""
    frame = records[["video_id", "title", "description"]].copy()
    title = frame["title"].fillna("").astype(str)
    description = frame["description"].fillna("").astype(str)
    title_words = title.map(_words)
    frame["title_word_count"] = title_words.map(len)
    frame["title_question_count"] = title.str.count(r"\?")
    frame["title_exclamation_count"] = title.str.count("!")
    frame["title_number_token_count"] = title_words.map(
        lambda words: sum(token.strip(".,:;!?%()").isdigit() for token in words)
    )
    letters = title.str.count(r"[A-Za-zÀ-ÿ]")
    uppercase = title.str.count(r"[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ]")
    frame["title_uppercase_ratio"] = uppercase / letters.replace(0, np.nan)
    frame["description_word_count"] = description.map(_words).map(len)
    return frame.drop(columns=["title", "description"])


def _load_embedding(path: Path, prefix: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding ausente: {path}")
    frame = pd.read_parquet(path, columns=["video_id", "embedding"])
    frame = frame.drop_duplicates("video_id", keep="last").copy()
    frame[prefix] = frame.pop("embedding").map(lambda value: np.asarray(value, dtype=float))
    return frame


def build_embedding_frame(
    feature_map: Path,
    embeddings_dir: Path,
    normalized_records: Path,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Une apenas insumos prévios à publicação e restringe à cobertura comum."""
    dataset = load_dataset(feature_map)
    frame = dataset.evaluable("full").copy()
    source_payload = json.loads(normalized_records.read_text(encoding="utf-8"))
    records = pd.DataFrame(source_payload["records"])
    stats = text_statistics(records)
    frame = frame.merge(stats, on="video_id", how="left", validate="one_to_one")

    for name in ("thumb", "title"):
        vectors = _load_embedding(embeddings_dir / f"{name}.parquet", f"_{name}_embedding")
        frame = frame.merge(vectors, on="video_id", how="left", validate="one_to_one")

    # A referência também usa esta mesma população: diferenças não podem vir
    # de vídeos que uma configuração perdeu por falta de embedding.
    needed = ["_thumb_embedding", "_title_embedding"]
    covered = frame[needed].notna().all(axis=1)
    frame = frame.loc[covered].copy()
    if frame.empty:
        raise ValueError("Não há interseção entre o mapa de features e os embeddings.")
    return frame, dataset.vault_channels


def _embedding_matrix(rows: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    matrices = [np.stack(rows[f"_{name}_embedding"].to_numpy()) for name in names]
    return np.hstack(matrices)


def _fit_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: EmbeddingConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    numeric_columns = list(BASE_NUMERIC)
    train_features = train[BASE_NUMERIC + BASE_CATEGORICAL].copy()
    test_features = test[BASE_NUMERIC + BASE_CATEGORICAL].copy()
    if config.knn_embedding:
        embedding_column = f"_{config.knn_embedding}_embedding"
        train_knn = temporal_other_channel_knn(
            train,
            train,
            embedding_column=embedding_column,
            outcome_column="viral",
        )
        test_knn = temporal_other_channel_knn(
            test,
            train,
            embedding_column=embedding_column,
            outcome_column="viral",
        )
        for source, features in ((train_knn, train_features), (test_knn, test_features)):
            features[["neighbor_outcome", "neighbor_count"]] = source[
                ["neighbor_outcome", "neighbor_count"]
            ].to_numpy()
        numeric_columns += ["neighbor_outcome", "neighbor_count"]
    preprocessor = make_preprocessor(
        {"features": {"numeric": numeric_columns, "categorical": BASE_CATEGORICAL}}
    )
    base_train = preprocessor.fit_transform(train_features[numeric_columns + BASE_CATEGORICAL])
    base_test = preprocessor.transform(test_features[numeric_columns + BASE_CATEGORICAL])
    if config.cluster_embedding:
        cluster_vectors_train = _embedding_matrix(train, (config.cluster_embedding,))
        cluster_vectors_test = _embedding_matrix(test, (config.cluster_embedding,))
        components_for_cluster = min(32, len(train) - 1, cluster_vectors_train.shape[1])
        if components_for_cluster < 1:
            raise ValueError("Treino insuficiente para clusters de criativo.")
        cluster_pca = PCA(n_components=components_for_cluster, random_state=seed)
        cluster_train = cluster_pca.fit_transform(cluster_vectors_train)
        cluster_test = cluster_pca.transform(cluster_vectors_test)
        requested_clusters = config.cluster_count or 8
        clusters = min(requested_clusters, len(train))
        if clusters < 2:
            raise ValueError("Treino insuficiente para K-Means.")
        kmeans = KMeans(n_clusters=clusters, n_init=20, random_state=seed)
        train_labels = kmeans.fit_predict(cluster_train)
        test_labels = kmeans.predict(cluster_test)
        base_train = np.hstack([base_train, np.eye(clusters)[train_labels]])
        base_test = np.hstack([base_test, np.eye(clusters)[test_labels]])

    if not config.embeddings:
        return base_train, base_test, 0

    vectors_train = _embedding_matrix(train, config.embeddings)
    vectors_test = _embedding_matrix(test, config.embeddings)
    requested = config.pca_components or vectors_train.shape[1]
    components = min(requested, vectors_train.shape[0] - 1, vectors_train.shape[1])
    if components < 1:
        raise ValueError("Treino insuficiente para PCA.")
    pca = PCA(n_components=components, random_state=42)
    return (
        np.hstack([base_train, pca.fit_transform(vectors_train)]),
        np.hstack([base_test, pca.transform(vectors_test)]),
        components,
    )


def _estimator(config: EmbeddingConfig, seed: int) -> object:
    if config.estimator == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - mensagem para ambiente sem extra.
            raise RuntimeError("CatBoost não está instalado no ambiente.") from exc
        return CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    return HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=150,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def score_loco(frame: pd.DataFrame, config: EmbeddingConfig, seed: int) -> tuple[pd.DataFrame, int]:
    """LOCO por canal, treinando cada setor somente com seu próprio setor."""
    pieces: list[pd.DataFrame] = []
    component_counts: list[int] = []
    for channel in sorted(frame["channel_id"].unique()):
        channel_scores, components = score_channel(frame, channel, config, seed)
        pieces.append(channel_scores)
        component_counts.append(components)
    scored = pd.concat(pieces).sort_index()
    if scored["score"].isna().any():
        raise RuntimeError("A avaliação LOCO deixou vídeos sem nota.")
    return scored, max(component_counts, default=0)


def score_channel(
    frame: pd.DataFrame,
    channel: str,
    config: EmbeddingConfig,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Avalia uma dobra LOCO isolada, para execuções retomáveis por canal."""
    test = frame[frame["channel_id"] == channel]
    if test.empty:
        raise ValueError(f"Canal fora da população de avaliação: {channel}")
    scored = test[
        ["video_id", "channel_id", "sector", "viral", "target_age_days", "adjusted_performance"]
    ].copy()
    scored["score"] = np.nan
    component_counts: list[int] = []
    for sector, sector_test in test.groupby("sector", sort=True):
        train = frame[(frame["channel_id"] != channel) & (frame["sector"] == sector)]
        if train["viral"].nunique() < 2:
            scored.loc[sector_test.index, "score"] = float(train["viral"].mean())
            continue
        selected_train, labels = training_rows_and_labels(
            SimpleNamespace(train_label=config.train_label), train
        )
        if len(selected_train) == 0 or len(np.unique(labels)) < 2:
            scored.loc[sector_test.index, "score"] = float(labels.mean()) if len(labels) else 0.1
            continue
        x_train, x_test, components = _fit_transform(selected_train, sector_test, config, seed)
        model = _estimator(config, seed)
        model.fit(x_train, labels)
        scored.loc[sector_test.index, "score"] = model.predict_proba(x_test)[:, 1]
        component_counts.append(components)
    if scored["score"].isna().any():
        raise RuntimeError(f"A dobra LOCO deixou vídeos sem nota: {channel}")
    return scored, max(component_counts, default=0)


def extreme_auc(scored: pd.DataFrame) -> float:
    """Diagnóstico: top 10% contra os 50% inferiores em cada canal."""
    keep = pd.Series(False, index=scored.index)
    for _, group in scored.groupby("channel_id"):
        lower = group["adjusted_performance"] <= group["adjusted_performance"].quantile(0.5)
        keep.loc[group.index] = lower | (group["viral"] == 1)
    subset = scored.loc[keep]
    positives = subset["viral"].to_numpy()
    if len(np.unique(positives)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(positives, subset["score"]))


def evaluate_configuration(
    frame: pd.DataFrame,
    config: EmbeddingConfig,
    *,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    scored, components = score_loco(frame, config, seed)
    summary = summarize_scores(scored, "score")
    counts = channel_pair_counts(scored, "score")
    summary["auc_intra_ci_low"], summary["auc_intra_ci_high"] = bootstrap_auc(
        counts, bootstrap_iterations, seed
    )
    summary["auc_top10_vs_bottom50"] = extreme_auc(scored)
    summary["pca_components_used"] = components
    return summary, scored, counts


def paired_embedding_diagnostic(
    frame: pd.DataFrame,
    *,
    seed: int,
    pca_components: int = 32,
) -> pd.DataFrame:
    """Diagnóstico Hessel/Ding: logística em diferenças de pares intra-canal.

    É deliberadamente somente um diagnóstico: não é combinado ao modelo de
    produção sem vencer a ablação multimodal principal.
    """
    scored = frame[["video_id", "channel_id", "sector", "viral", "target_age_days"]].copy()
    scored["score"] = np.nan
    for channel in sorted(frame["channel_id"].unique()):
        test = frame[frame["channel_id"] == channel]
        for sector, sector_test in test.groupby("sector", sort=True):
            train = frame[(frame["channel_id"] != channel) & (frame["sector"] == sector)]
            vectors_train = _embedding_matrix(train, ("thumb", "title"))
            vectors_test = _embedding_matrix(sector_test, ("thumb", "title"))
            components = min(pca_components, len(train) - 1, vectors_train.shape[1])
            pca = PCA(n_components=components, random_state=seed)
            transformed_train = pca.fit_transform(vectors_train)
            transformed_test = pca.transform(vectors_test)
            pairs: list[np.ndarray] = []
            for _, group in train.assign(_vector=list(transformed_train)).groupby("channel_id"):
                positive = np.stack(group.loc[group["viral"] == 1, "_vector"].to_numpy())
                negative = np.stack(group.loc[group["viral"] == 0, "_vector"].to_numpy())
                if len(positive) and len(negative):
                    delta = (positive[:, None, :] - negative[None, :, :]).reshape(-1, components)
                    pairs.extend([delta, -delta])
            if not pairs:
                scored.loc[sector_test.index, "score"] = 0.0
                continue
            x_pairs = np.vstack(pairs)
            y_pairs = np.concatenate(
                [
                    np.ones(len(block), dtype=int)
                    if index % 2 == 0
                    else np.zeros(len(block), dtype=int)
                    for index, block in enumerate(pairs)
                ]
            )
            model = LogisticRegression(C=0.3, max_iter=3000, random_state=seed)
            model.fit(x_pairs, y_pairs)
            scored.loc[sector_test.index, "score"] = model.decision_function(transformed_test)
    return scored
