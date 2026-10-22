"""Base v2 de Shorts: limpeza por regras explícitas e rótulo de outlier por canal.

Camadas:

1. **Bruto**: ``normalized_records.json`` da coleta, nunca editado.
2. **Limpa**: um Short válido por linha. Cada exclusão tem motivo registrado.
3. **Rótulo**: só vídeos maduros. ``outlier_score`` = views do vídeo dividido
   pela mediana das views dos ``HISTORY`` Shorts anteriores do mesmo canal.
   ``viral`` = ``outlier_score >= VIRAL_K``.

As regras e limiares abaixo foram fixados antes de qualquer modelo ser
treinado nesta base. Mudá-los depois de olhar AUC ou precisão seria escolher o
alvo pelo resultado.

Views, likes e comentários só entram no rótulo, nunca como feature.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

SHORTS_GLOBAL_LAUNCH = pd.Timestamp("2021-07-12", tz="UTC")
SHORTS_180S_FROM = pd.Timestamp("2024-10-15", tz="UTC")
PROSPECTIVE_START = pd.Timestamp("2026-10-21", tz="UTC")

MIN_AGE_DAYS = 60.0  # HIPÓTESE: views de um Short estabilizam em ~2 meses
HISTORY = 30  # Shorts anteriores do canal usados como linha de base
MIN_HISTORY = 10  # sem 10 anteriores, a mediana do canal não é confiável
VIRAL_K = 3.0  # HIPÓTESE: "3x a mediana do canal", convenção de outlier

# Base v3: auditoria do rótulo com likes (nunca feature). Fixado antes de
# avaliar a v3, a partir do diagnóstico no período de treino da v2: Shorts
# impulsionados por anúncio têm views sem engajamento.
BOOSTED_FRACTION = 0.10  # likes/view abaixo de 10% da mediana dos anteriores
HIDDEN_LIKES_LIMIT = 0.5  # canal com likes ocultos em mais da metade sai do rótulo


def shorts_limit_seconds(published: pd.Series) -> pd.Series:
    """Duração máxima de um Short na data de publicação (NaN antes do lançamento)."""
    limit = pd.Series(np.nan, index=published.index, dtype=float)
    limit[(published >= SHORTS_GLOBAL_LAUNCH) & (published < SHORTS_180S_FROM)] = 60.0
    limit[published >= SHORTS_180S_FROM] = 180.0
    return limit


def normalized_title(title: object) -> str:
    """Título sem acento, caixa, pontuação, emoji e espaços repetidos."""
    text = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def clean_shorts(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica as regras da camada 2 em ordem e devolve (limpa, funil).

    Ordem: público; dentro do período de coleta (antes do prospectivo); depois
    do lançamento global dos Shorts; duração dentro do limite da época; ID
    único; reupload (mesmo canal, título normalizado e duração) fica só o
    primeiro publicado.
    """
    frame = raw.copy()
    frame["published"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["collected"] = pd.to_datetime(frame["collected_at"], utc=True)
    frame["duration_seconds"] = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    frame["views"] = pd.to_numeric(frame["view_count"], errors="coerce")
    frame["likes"] = pd.to_numeric(frame.get("like_count"), errors="coerce")
    limit = shorts_limit_seconds(frame["published"])

    steps: list[tuple[str, pd.Series]] = [
        ("não público", frame["privacy_status"].fillna("") != "public"),
        ("período prospectivo (validação final)", frame["published"] >= PROSPECTIVE_START),
        ("antes do lançamento global dos Shorts", frame["published"] < SHORTS_GLOBAL_LAUNCH),
        ("sem duração ou views", frame["duration_seconds"].isna() | frame["views"].isna()),
        ("duração acima do limite de Short da época", frame["duration_seconds"] > limit),
    ]
    funnel = [{"etapa": "coleta bruta", "removidos": 0, "restantes": len(frame)}]
    keep = pd.Series(True, index=frame.index)
    for reason, drop in steps:
        removed = keep & drop.fillna(False)
        keep &= ~removed
        funnel.append({"etapa": reason, "removidos": int(removed.sum()), "restantes": int(keep.sum())})
    frame = frame.loc[keep]

    before = len(frame)
    frame = frame.sort_values("collected").drop_duplicates("video_id", keep="last")
    funnel.append({"etapa": "ID duplicado", "removidos": before - len(frame), "restantes": len(frame)})

    frame["title_key"] = frame["title"].map(normalized_title)
    before = len(frame)
    frame = frame.sort_values("published").drop_duplicates(
        ["channel_id", "title_key", "duration_seconds"], keep="first"
    )
    funnel.append({"etapa": "reupload (canal, título e duração iguais)", "removidos": before - len(frame), "restantes": len(frame)})

    frame["age_days"] = (frame["collected"] - frame["published"]).dt.total_seconds() / 86400.0
    return frame.sort_values(["channel_id", "published"]).reset_index(drop=True), pd.DataFrame(funnel)


def outlier_labels(
    clean: pd.DataFrame,
    *,
    min_age_days: float = MIN_AGE_DAYS,
    history: int = HISTORY,
    min_history: int = MIN_HISTORY,
    viral_k: float = VIRAL_K,
) -> pd.DataFrame:
    """Camada 3: rótulo de outlier contra a mediana dos Shorts anteriores do canal.

    A linha de base usa só vídeos publicados **antes** do vídeo rotulado, do
    mesmo canal e da base limpa. Como são mais antigos, também são maduros.
    Vídeos jovens demais ou sem histórico suficiente ficam sem rótulo
    (``label_status`` diz o motivo), mas continuam servindo de linha de base
    para os seguintes quando forem maduros.
    """
    frame = clean.sort_values(["channel_id", "published"]).copy()
    baseline = np.full(len(frame), np.nan)
    prior_count = np.zeros(len(frame), dtype=int)
    for _, positions in frame.groupby("channel_id", sort=False).indices.items():
        views = frame["views"].to_numpy()[positions]
        for offset, position in enumerate(positions):
            window = views[max(0, offset - history) : offset]
            prior_count[position] = len(window)
            if len(window):
                baseline[position] = float(np.median(window))
    frame["baseline_median_views"] = baseline
    frame["prior_shorts"] = prior_count
    frame["outlier_score"] = frame["views"] / frame["baseline_median_views"].where(lambda s: s > 0)

    status = np.where(
        frame["age_days"] < min_age_days,
        "jovem demais",
        np.where(frame["prior_shorts"] < min_history, "histórico curto", "rotulado"),
    )
    status = np.where((status == "rotulado") & frame["outlier_score"].isna(), "linha de base zero", status)
    frame["label_status"] = status
    labelled = frame["label_status"] == "rotulado"
    frame["viral"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[labelled, "viral"] = (frame.loc[labelled, "outlier_score"] >= viral_k).astype(int)
    frame["log_outlier_score"] = np.log10(frame["outlier_score"].where(labelled))
    return frame.reset_index(drop=True)


def audit_boosted(
    labels: pd.DataFrame,
    *,
    history: int = HISTORY,
    boosted_fraction: float = BOOSTED_FRACTION,
    hidden_limit: float = HIDDEN_LIKES_LIMIT,
) -> pd.DataFrame:
    """Camada 3 da v3: tira do rótulo views prováveis de anúncio.

    ``like_rate`` de cada Short é comparada com a mediana dos ``history``
    Shorts anteriores do mesmo canal (só o passado, como a linha de base das
    views). Abaixo de ``boosted_fraction`` dessa mediana, o vídeo vira
    ``provável impulsionado``. Canais com likes ocultos em mais de
    ``hidden_limit`` dos vídeos saem do rótulo (``likes ocultos``), porque não
    dá para auditar. Likes nunca viram feature.
    """
    frame = labels.sort_values(["channel_id", "published"]).copy()
    frame["likes"] = pd.to_numeric(frame["likes"], errors="coerce")
    frame["like_rate"] = frame["likes"] / frame["views"].where(frame["views"] > 0)
    prior = np.full(len(frame), np.nan)
    rates = frame["like_rate"].to_numpy(dtype=float)
    for _, positions in frame.groupby("channel_id", sort=False).indices.items():
        channel_rates = rates[positions]
        for offset, position in enumerate(positions):
            window = channel_rates[max(0, offset - history) : offset]
            window = window[~np.isnan(window)]
            if len(window):
                prior[position] = float(np.median(window))
    frame["prior_median_like_rate"] = prior
    hidden_share = frame.groupby("channel_id")["likes"].transform(lambda s: s.isna().mean())
    labelled = frame["label_status"] == "rotulado"
    hidden = labelled & (hidden_share > hidden_limit)
    boosted = labelled & ~hidden & (frame["like_rate"] < boosted_fraction * frame["prior_median_like_rate"])
    frame.loc[hidden, "label_status"] = "likes ocultos"
    frame.loc[boosted, "label_status"] = "provável impulsionado"
    frame.loc[hidden | boosted, ["viral", "log_outlier_score"]] = np.nan
    frame["viral"] = frame["viral"].astype("Int64")
    return frame.reset_index(drop=True)
