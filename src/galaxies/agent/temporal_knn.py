"""Features de vizinhança temporal, sem vazamento entre canais.

O kNN é calculado para um vídeo-alvo usando somente candidatos de outros
canais, presentes no treino e publicados com antecedência mínima. O valor de
desempenho do vizinho é um rótulo histórico: ele nunca vem do vídeo-alvo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_other_channel_knn(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    embedding_column: str,
    outcome_column: str,
    k: int = 10,
    min_lag_days: int = 30,
) -> pd.DataFrame:
    """Retorna média ponderada de resultados de vizinhos históricos elegíveis.

    ``targets`` e ``candidates`` precisam conter ``video_id``, ``channel_id``,
    ``published_at`` e o vetor indicado. Candidatos devem ser exclusivamente
    linhas de treino da dobra atual. Ausência de vizinho é mantida como NaN e
    acompanhada por ``neighbor_count``; o imputador aprende esse caso no treino.
    """
    if k < 1 or min_lag_days < 0:
        raise ValueError("k deve ser >= 1 e min_lag_days deve ser >= 0.")
    required_target = {"video_id", "channel_id", "published_at", embedding_column}
    required_candidate = required_target | {outcome_column}
    missing = (required_target - set(targets.columns)) | (required_candidate - set(candidates.columns))
    if missing:
        raise ValueError(f"Colunas ausentes para kNN temporal: {sorted(missing)}")

    history = candidates.copy()
    history["published_at"] = pd.to_datetime(history["published_at"], utc=True)
    history = history.dropna(subset=[embedding_column, outcome_column])
    if history.empty:
        return pd.DataFrame(
            {"video_id": targets["video_id"], "neighbor_outcome": np.nan, "neighbor_count": 0}
        )
    vectors = np.stack(history[embedding_column].map(lambda value: np.asarray(value, dtype=float)))
    norms = np.linalg.norm(vectors, axis=1)
    results: list[dict[str, object]] = []

    for _, row in targets.iterrows():
        target_time = pd.Timestamp(row["published_at"])
        if target_time.tzinfo is None:
            target_time = target_time.tz_localize("UTC")
        else:
            target_time = target_time.tz_convert("UTC")
        cutoff = target_time - pd.Timedelta(days=min_lag_days)
        eligible = (history["channel_id"].to_numpy() != row["channel_id"]) & (
            history["published_at"].to_numpy() <= cutoff
        )
        count = int(eligible.sum())
        if not count:
            results.append(
                {"video_id": row["video_id"], "neighbor_outcome": np.nan, "neighbor_count": 0}
            )
            continue
        target_vector = np.asarray(row[embedding_column], dtype=float)
        target_norm = np.linalg.norm(target_vector)
        if not target_norm:
            results.append(
                {"video_id": row["video_id"], "neighbor_outcome": np.nan, "neighbor_count": 0}
            )
            continue
        similarity = vectors[eligible] @ target_vector / (norms[eligible] * target_norm)
        order = np.argsort(similarity)[::-1][:k]
        chosen_similarity = similarity[order]
        chosen_outcome = history.loc[eligible, outcome_column].to_numpy(dtype=float)[order]
        weights = np.maximum(chosen_similarity, 0) + 1e-6
        results.append(
            {
                "video_id": row["video_id"],
                "neighbor_outcome": float(np.average(chosen_outcome, weights=weights)),
                "neighbor_count": len(order),
            }
        )
    return pd.DataFrame(results)
