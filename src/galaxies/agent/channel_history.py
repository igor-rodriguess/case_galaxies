"""Features de histórico para o modo *canal conhecido*.

Estas features só existem quando o cliente já publicou antes.  Para cada
criativo-alvo, a função enxerga somente vídeos do mesmo canal publicados pelo
menos ``min_lag_days`` antes dele.  Assim, o cálculo não usa contagens de
vídeos (proxy de tempo), views, likes ou informações do futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _duration_bucket(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "missing"
    if seconds <= 60:
        return "<=60s"
    if seconds <= 180:
        return "61-180s"
    if seconds <= 600:
        return "181-600s"
    return ">600s"


def _smoothed_rate(successes: int, total: int, global_rate: float, strength: float) -> float:
    """Suavização beta-binomial; retorna a taxa global quando não há passado."""
    return float((successes + strength * global_rate) / (total + strength))


def _cosine_mean(target: np.ndarray, vectors: list[np.ndarray]) -> float | None:
    if not vectors:
        return None
    matrix = np.vstack(vectors)
    target_norm = np.linalg.norm(target)
    norms = np.linalg.norm(matrix, axis=1) * target_norm
    valid = norms > 0
    if not valid.any():
        return None
    return float(np.mean((matrix[valid] @ target) / norms[valid]))


def known_channel_history_features(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    *,
    outcome_column: str = "viral",
    embedding_column: str | None = None,
    min_lag_days: int = 30,
    smoothing_strength: float = 20.0,
    prior_rate: float = 0.10,
) -> pd.DataFrame:
    """Calcula taxas passadas e similaridade viral-não viral, sem futuro.

    ``history`` pode ser o treino temporal e ``targets`` o teste. Para evitar
    acidentalmente usar um alvo como seu próprio antecedente, a elegibilidade é
    sempre definida por ``published_at <= alvo - min_lag_days``.
    """
    required = {"video_id", "channel_id", "published_at", "duration_seconds", outcome_column}
    missing = required - set(targets.columns) - set(history.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para histórico de canal: {sorted(missing)}")
    if embedding_column and embedding_column not in targets.columns.union(history.columns):
        raise ValueError(f"Embedding ausente: {embedding_column}")

    target_rows = targets.copy()
    past_rows = history.copy()
    for frame in (target_rows, past_rows):
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
        frame["_duration_bucket"] = frame["duration_seconds"].map(_duration_bucket)
        frame[outcome_column] = pd.to_numeric(frame[outcome_column], errors="coerce")
    # Um prior fixo evita que o rótulo do próprio alvo contamine uma feature
    # durante a construção de linhas de treino.  Ele reflete a prevalência
    # definida pelo protocolo (top 10%), não a distribuição observada inteira.
    global_rate = float(prior_rate)
    if not 0.0 <= global_rate <= 1.0:
        raise ValueError("prior_rate deve estar entre 0 e 1.")

    rows: list[dict[str, object]] = []
    lag = pd.Timedelta(days=min_lag_days)
    for _, target in target_rows.iterrows():
        eligible = past_rows[
            (past_rows["channel_id"] == target["channel_id"])
            & (past_rows["published_at"] <= target["published_at"] - lag)
        ].dropna(subset=[outcome_column])
        bucket = target["_duration_bucket"]
        same_bucket = eligible[eligible["_duration_bucket"] == bucket]
        format_rate = _smoothed_rate(
            int(same_bucket[outcome_column].sum()), len(same_bucket), global_rate, smoothing_strength
        )
        duration_rate = _smoothed_rate(
            int(same_bucket[outcome_column].sum()), len(same_bucket), global_rate, smoothing_strength
        )
        row: dict[str, object] = {
            "video_id": target["video_id"],
            "known_channel_format_viral_rate": format_rate,
            "known_channel_duration_viral_rate": duration_rate,
            "known_channel_history_available": float(len(eligible) > 0),
        }
        if embedding_column:
            target_embedding = target.get(embedding_column)
            if target_embedding is None or (isinstance(target_embedding, float) and np.isnan(target_embedding)):
                row["known_channel_similarity_margin"] = np.nan
            else:
                target_vector = np.asarray(target_embedding, dtype=float)
                viral_vectors = [
                    np.asarray(value, dtype=float)
                    for value in eligible.loc[eligible[outcome_column] == 1, embedding_column].dropna()
                ]
                nonviral_vectors = [
                    np.asarray(value, dtype=float)
                    for value in eligible.loc[eligible[outcome_column] == 0, embedding_column].dropna()
                ]
                viral_mean = _cosine_mean(target_vector, viral_vectors)
                nonviral_mean = _cosine_mean(target_vector, nonviral_vectors)
                row["known_channel_similarity_margin"] = (
                    np.nan if viral_mean is None or nonviral_mean is None else viral_mean - nonviral_mean
                )
        rows.append(row)
    return pd.DataFrame(rows)
