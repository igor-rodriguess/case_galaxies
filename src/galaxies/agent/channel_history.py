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


def _format_bucket(seconds: float | int | None) -> str:
    """Formato amplo, distinto das faixas finas de duração.

    A taxa por formato responde à pergunta "Shorts funcionam neste canal?";
    a taxa por duração responde "esta faixa de duração funciona?". Mantê-las
    separadas evita duplicar a mesma feature no modo de canal conhecido.
    """
    if seconds is None or pd.isna(seconds):
        return "missing"
    return "short" if seconds <= 180 else "long"


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


def _prior_outcome_counts(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    group_columns: list[str],
    outcome_column: str,
    lag: pd.Timedelta,
) -> pd.DataFrame:
    """Conta sucessos anteriores por grupo com busca temporal vetorizada.

    Em vez de varrer todo o histórico para cada criativo-alvo, a função realiza
    uma busca ``asof`` por canal/formato ou canal/faixa. A condição continua
    exatamente a mesma: publicação no máximo em ``target - lag``.
    """
    columns = ["total", "successes"]
    result = pd.DataFrame(0.0, index=targets.index, columns=columns)
    if targets.empty or history.empty:
        return result
    target_rows = targets.copy()
    target_rows["_history_row"] = np.arange(len(target_rows))
    target_rows["_cutoff"] = target_rows["published_at"] - lag
    valid_targets = target_rows.dropna(subset=["_cutoff"])
    valid_history = history.dropna(subset=["published_at", outcome_column]).copy()
    if valid_targets.empty or valid_history.empty:
        return result

    lookup = pd.DataFrame(0.0, index=np.arange(len(target_rows)), columns=columns)
    for key, target_group in valid_targets.groupby(group_columns, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        history_group = valid_history
        for column, value in zip(group_columns, key_values, strict=True):
            history_group = history_group[history_group[column] == value]
        if history_group.empty:
            continue
        history_group = history_group.sort_values("published_at", kind="stable").copy()
        history_group["_total"] = np.arange(1, len(history_group) + 1, dtype=float)
        history_group["_successes"] = history_group[outcome_column].cumsum().astype(float)
        left = target_group.sort_values("_cutoff", kind="stable")
        matched = pd.merge_asof(
            left,
            history_group[["published_at", "_total", "_successes"]],
            left_on="_cutoff",
            right_on="published_at",
            direction="backward",
        )
        matched = matched.set_index("_history_row")
        lookup.loc[matched.index, "total"] = matched["_total"].fillna(0.0)
        lookup.loc[matched.index, "successes"] = matched["_successes"].fillna(0.0)
    lookup.index = targets.index
    return lookup


def _safe_prior_outcome_counts(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    group_columns: list[str],
    outcome_column: str,
    outcome_observed_at_column: str | None,
    lag: pd.Timedelta,
) -> pd.DataFrame:
    """Conta somente resultados que já eram observáveis na data do alvo.

    Uma publicação anterior não é, por si só, um resultado disponível antes de
    um novo criativo. Para uma taxa histórica entrar em produção, o registro
    precisa trazer quando o resultado foi observado (por exemplo, um snapshot
    de views aos 30 dias) e essa data precisa anteceder a publicação do alvo.

    O mapa atual foi coletado depois de quase todos os vídeos. Portanto, sem
    uma coluna explícita de disponibilidade, esta função retorna zero histórico
    e o chamador usa o prior. Isso é intencional: evita transformar uma
    avaliação retrospectiva em uma feature de produção.
    """
    result = pd.DataFrame(0.0, index=targets.index, columns=["total", "successes"])
    if not outcome_observed_at_column or outcome_observed_at_column not in history.columns:
        return result
    if targets.empty or history.empty:
        return result

    target_rows = targets.copy()
    target_rows["published_at"] = pd.to_datetime(
        target_rows["published_at"], utc=True, errors="coerce"
    )
    past_rows = history.copy()
    past_rows["published_at"] = pd.to_datetime(past_rows["published_at"], utc=True, errors="coerce")
    past_rows["_outcome_observed_at"] = pd.to_datetime(
        past_rows[outcome_observed_at_column], utc=True, errors="coerce"
    )
    past_rows[outcome_column] = pd.to_numeric(past_rows[outcome_column], errors="coerce")
    past_rows = past_rows.dropna(subset=["published_at", "_outcome_observed_at", outcome_column])
    if past_rows.empty:
        return result

    for index, target in target_rows.iterrows():
        published_at = target["published_at"]
        if pd.isna(published_at):
            continue
        eligible = past_rows[
            (past_rows["published_at"] <= published_at - lag)
            & (past_rows["_outcome_observed_at"] <= published_at)
        ]
        for column in group_columns:
            eligible = eligible[eligible[column] == target[column]]
        if not eligible.empty:
            result.loc[index, "total"] = float(len(eligible))
            result.loc[index, "successes"] = float(eligible[outcome_column].sum())
    return result


def known_channel_history_features(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    *,
    outcome_column: str = "viral",
    embedding_column: str | None = None,
    min_lag_days: int = 30,
    smoothing_strength: float = 20.0,
    prior_rate: float = 0.10,
    outcome_observed_at_column: str | None = None,
    allow_retrospective_outcome_labels: bool = False,
) -> pd.DataFrame:
    """Calcula histórico de canal somente quando o resultado era disponível.

    A condição de publicação (``published_at <= alvo - min_lag_days``) não é
    suficiente. A taxa também exige que o resultado histórico tenha sido
    *observado* até a publicação do alvo. Sem ``outcome_observed_at_column``,
    a saída segura usa apenas o prior e marca ``history_available=0``.

    ``allow_retrospective_outcome_labels=True`` preserva a antiga leitura para
    auditoria exploratória, mas nunca deve alimentar um modelo implantável: o
    rótulo ``viral`` atual é definido com a série completa do canal e pode
    incorporar vídeos posteriores ao alvo.
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
        frame["_format_bucket"] = frame["duration_seconds"].map(_format_bucket)
        frame[outcome_column] = pd.to_numeric(frame[outcome_column], errors="coerce")
    # Um prior fixo evita que o rótulo do próprio alvo contamine uma feature
    # durante a construção de linhas de treino.  Ele reflete a prevalência
    # definida pelo protocolo (top 10%), não a distribuição observada inteira.
    global_rate = float(prior_rate)
    if not 0.0 <= global_rate <= 1.0:
        raise ValueError("prior_rate deve estar entre 0 e 1.")

    lag = pd.Timedelta(days=min_lag_days)
    groupings = {
        "format": ["channel_id", "_format_bucket"],
        "duration": ["channel_id", "_duration_bucket"],
        "channel": ["channel_id"],
    }
    if allow_retrospective_outcome_labels:
        counts = {
            name: _prior_outcome_counts(target_rows, past_rows, grouping, outcome_column, lag)
            for name, grouping in groupings.items()
        }
    else:
        counts = {
            name: _safe_prior_outcome_counts(
                target_rows,
                past_rows,
                grouping,
                outcome_column,
                outcome_observed_at_column,
                lag,
            )
            for name, grouping in groupings.items()
        }
    format_counts = counts["format"]
    duration_counts = counts["duration"]
    channel_counts = counts["channel"]

    rows: list[dict[str, object]] = []
    for row_position, (_, target) in enumerate(target_rows.iterrows()):
        format_count = format_counts.iloc[row_position]
        duration_count = duration_counts.iloc[row_position]
        format_rate = _smoothed_rate(
            int(format_count["successes"]),
            int(format_count["total"]),
            global_rate,
            smoothing_strength,
        )
        duration_rate = _smoothed_rate(
            int(duration_count["successes"]),
            int(duration_count["total"]),
            global_rate,
            smoothing_strength,
        )
        row: dict[str, object] = {
            "video_id": target["video_id"],
            "known_channel_format_viral_rate": format_rate,
            "known_channel_duration_viral_rate": duration_rate,
            "known_channel_history_available": float(
                channel_counts.iloc[row_position]["total"] > 0
            ),
        }
        if embedding_column and allow_retrospective_outcome_labels:
            eligible = past_rows[
                (past_rows["channel_id"] == target["channel_id"])
                & (past_rows["published_at"] <= target["published_at"] - lag)
            ].dropna(subset=[outcome_column])
            target_embedding = target.get(embedding_column)
            if target_embedding is None or (
                isinstance(target_embedding, float) and np.isnan(target_embedding)
            ):
                row["known_channel_similarity_margin"] = np.nan
            else:
                target_vector = np.asarray(target_embedding, dtype=float)
                viral_vectors = [
                    np.asarray(value, dtype=float)
                    for value in eligible.loc[
                        eligible[outcome_column] == 1, embedding_column
                    ].dropna()
                ]
                nonviral_vectors = [
                    np.asarray(value, dtype=float)
                    for value in eligible.loc[
                        eligible[outcome_column] == 0, embedding_column
                    ].dropna()
                ]
                viral_mean = _cosine_mean(target_vector, viral_vectors)
                nonviral_mean = _cosine_mean(target_vector, nonviral_vectors)
                row["known_channel_similarity_margin"] = (
                    np.nan
                    if viral_mean is None or nonviral_mean is None
                    else viral_mean - nonviral_mean
                )
        elif embedding_column:
            row["known_channel_similarity_margin"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
