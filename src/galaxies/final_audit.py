"""Auditoria da base da avaliação final (Parte 1).

Aplica o escopo de `docs/escopo_avaliacao_final.md` sobre a coleta da Data API,
as duas leituras de métricas e as análises VLM. Devolve tabelas para o relatório
e a lista congelada da base principal. Nada aqui ajusta modelo nem olha
desempenho preditivo: a única regra que usa views é a da idade mínima, que lê
apenas o crescimento entre as duas leituras do mesmo vídeo.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from galaxies.agent.protocol import is_forbidden, normalize_vlm_value

AUDIT_VERSION = "final-audit-v1"
SECTORS = ("automotivo", "moda", "beleza")
FORMATS = ("ate_60s", "61_180s", "acima_180s")


@dataclass(frozen=True)
class FinalScope:
    """Parâmetros do escopo; mudar qualquer um cria uma nova versão da base."""

    youtube_run: str = "data/youtube/runs/expansion-1000-per-sector-balanced-2026-10-16"
    second_snapshot: str = "data/youtube/full_snapshots/full-2026-10-20"
    case_control_plan: str = "data/features/plans/case-control-opening-v1/manifest.json"
    # Congela a base VLM logo após a reextração dos 58 vídeos (último resultado às
    # 01:44:36Z). Lotes posteriores não mudam esta auditoria.
    vlm_completed_before: str = "2026-10-20T01:45:00Z"
    vlm_model: str = "gemini-3.5-flash-lite"
    vlm_prompt_version: str = "galaxies-opening-v0.1.0"
    vlm_start_seconds: float = 0.0
    vlm_end_seconds: float = 3.0
    vlm_fps: float = 4.0
    age_candidates_days: tuple[int, ...] = (7, 14, 30)
    age_window_days: float = 7.0
    growth_threshold_per_day: float = 0.01
    fallback_min_age_days: int = 30
    sensitivity_min_age_days: int = 30
    short_max_seconds: float = 180.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Papéis das colunas -----------------------------------------------------

TARGET_SOURCE_COLUMNS = {"target_view_count", "like_count_snapshot", "comment_count_snapshot"}
SNAPSHOT_CONTAMINATED = {
    "channel_subscriber_count_snapshot",
    "channel_video_count_snapshot",
}
PROVENANCE_COLUMNS = {
    "schema_version",
    "video_id",
    "channel_id",
    "sector",
    "published_at",
    "collected_at",
    "youtube_record_path",
    "vlm_result_path",
    "channel_history_temporally_valid",
}
EXTRACTION_PROTOCOL_COLUMNS = {
    "vlm_available",
    "vlm_prompt_version",
    "vlm_model_version",
    "vlm_interval_start_seconds",
    "vlm_interval_end_seconds",
    "vlm_requested_fps",
    "transcript_available",
}


def column_role(column: str) -> str:
    """Classifica uma coluna do mapa de features pelo uso permitido.

    ``candidate_feature`` é a única categoria que pode entrar em um modelo, e
    ainda passa pela auditoria de ausência e constância.
    """
    if column in PROVENANCE_COLUMNS:
        return "provenance"
    if column in EXTRACTION_PROTOCOL_COLUMNS:
        return "extraction_protocol"
    if column in TARGET_SOURCE_COLUMNS or column.startswith("target_"):
        return "target_only"
    if column in SNAPSHOT_CONTAMINATED:
        return "snapshot_contaminated"
    if column.startswith(("prior_views_", "prior_likes_", "prior_comments_")):
        return "post_publication_history"
    if column == "prior_video_count":
        return "time_index"
    if is_forbidden(column):
        return "forbidden"
    return "candidate_feature"


# --- Formato, papel e protocolo ---------------------------------------------


def duration_format(seconds: object, short_max_seconds: float = 180.0) -> str:
    value = pd.to_numeric(seconds, errors="coerce")
    if pd.isna(value) or value <= 0:
        return "desconhecido"
    if value <= 60:
        return "ate_60s"
    if value <= short_max_seconds:
        return "61_180s"
    return "acima_180s"


def case_control_roles(plan: dict[str, Any]) -> dict[str, str]:
    return {item["video_id"]: item["role"] for item in plan.get("queue", [])}


def vlm_protocol_ok(frame: pd.DataFrame, scope: FinalScope) -> pd.Series:
    """True quando a análise segue exatamente o protocolo principal."""
    model = frame["vlm_model_requested"].fillna("")
    return (
        frame["vlm_available"].fillna(False).astype(bool)
        & (frame["vlm_prompt_version"] == scope.vlm_prompt_version)
        & (model == scope.vlm_model)
        & np.isclose(frame["vlm_interval_start_seconds"].astype(float), scope.vlm_start_seconds)
        & np.isclose(frame["vlm_interval_end_seconds"].astype(float), scope.vlm_end_seconds)
        & np.isclose(frame["vlm_requested_fps"].astype(float), scope.vlm_fps)
    )


# --- Segunda leitura e idade mínima -----------------------------------------


def snapshot_growth(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Une as duas leituras e mede o crescimento diário em log1p(views).

    ``first`` e ``second`` precisam de ``video_id``, ``observed_at`` e
    ``view_count``; ``first`` também traz ``published_at``.
    """
    merged = first[["video_id", "published_at", "observed_at", "view_count"]].merge(
        second[["video_id", "observed_at", "view_count"]],
        on="video_id",
        suffixes=("_1", "_2"),
        validate="one_to_one",
    )
    published = pd.to_datetime(merged["published_at"], utc=True)
    observed_1 = pd.to_datetime(merged["observed_at_1"], utc=True)
    observed_2 = pd.to_datetime(merged["observed_at_2"], utc=True)
    merged["age_days_1"] = (observed_1 - published).dt.total_seconds() / 86_400
    merged["days_between"] = (observed_2 - observed_1).dt.total_seconds() / 86_400
    if (merged["days_between"] <= 0).any():
        raise ValueError("A segunda leitura precisa ser posterior à primeira.")
    views_1 = pd.to_numeric(merged["view_count_1"], errors="coerce").astype(float)
    views_2 = pd.to_numeric(merged["view_count_2"], errors="coerce").astype(float)
    merged["log_growth"] = np.log1p(views_2) - np.log1p(views_1)
    merged["daily_log_growth"] = merged["log_growth"] / merged["days_between"]
    merged["views_decreased"] = views_2 < views_1
    return merged


def minimum_age_rule(
    growth: pd.DataFrame,
    candidates: tuple[int, ...] = (7, 14, 30),
    window_days: float = 7.0,
    threshold_per_day: float = 0.01,
    fallback_days: int = 30,
    bootstrap_resamples: int = 2_000,
    seed: int = 2026,
) -> tuple[int, list[dict[str, Any]]]:
    """Menor idade candidata cuja mediana de crescimento diário fica no limite.

    O intervalo bootstrap da mediana é só informativo: a decisão usa a mediana
    observada, como fixado em `docs/escopo_avaliacao_final.md`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    chosen: int | None = None
    for candidate in sorted(candidates):
        selected = growth.loc[
            (growth["age_days_1"] >= candidate) & (growth["age_days_1"] < candidate + window_days),
            "daily_log_growth",
        ].dropna()
        median = float(selected.median()) if len(selected) else None
        passes = median is not None and median <= threshold_per_day
        interval = None
        if len(selected) > 1:
            values = selected.to_numpy()
            medians = np.median(
                rng.choice(values, size=(bootstrap_resamples, len(values)), replace=True), axis=1
            )
            interval = [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]
        rows.append(
            {
                "candidate_days": candidate,
                "window": f"[{candidate}, {candidate + window_days:g}) dias",
                "videos": len(selected),
                "median_daily_log_growth": median,
                "median_ci95_bootstrap": interval,
                "passes": passes,
            }
        )
        if passes and chosen is None:
            chosen = candidate
    return (chosen if chosen is not None else fallback_days), rows


def growth_by_age(growth: pd.DataFrame, edges: tuple[float, ...]) -> pd.DataFrame:
    labels = [f"{left:g}-{right:g}" for left, right in itertools.pairwise(edges)]
    bucket = pd.cut(growth["age_days_1"], bins=list(edges), labels=labels, right=False)
    table = growth.groupby(bucket, observed=True)["daily_log_growth"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    table["interval_log_growth_sd"] = growth.groupby(bucket, observed=True)["log_growth"].std()
    return table.reset_index(names="age_bucket_days")


# --- Qualidade das colunas --------------------------------------------------


def missingness(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for sector, group in frame.groupby("sector"):
        for column in columns:
            rows.append(
                {
                    "sector": sector,
                    "column": column,
                    "missing_share": float(group[column].isna().mean()),
                    "distinct_values": int(group[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def unusable_columns(quality: pd.DataFrame) -> pd.DataFrame:
    """Colunas sem informação dentro de cada setor.

    ``100% ausente`` e ``constante`` saem do modelo. ``constante nos observados``
    tem um único valor mais ausências: a informação está só na ausência, e a
    coluna precisa ser recodificada antes de entrar (por exemplo, ausente = False).
    """
    flagged = quality.loc[
        (quality["missing_share"] >= 1.0) | (quality["distinct_values"] <= 1)
    ].copy()
    flagged["reason"] = np.select(
        [flagged["missing_share"] >= 1.0, flagged["missing_share"] > 0.0],
        ["100% ausente", "constante nos observados"],
        default="constante",
    )
    flagged["action"] = np.where(
        flagged["reason"] == "constante nos observados", "recodificar", "remover"
    )
    return flagged


def vlm_cardinality(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Valores distintos antes e depois da taxonomia fechada já existente."""
    rows = []
    for column in columns:
        raw = frame[column].dropna().astype(str)
        normalized = frame[column].map(lambda value, col=column: normalize_vlm_value(col, value))
        normalized = normalized.dropna().astype(str)
        rows.append(
            {
                "column": column,
                "distinct_raw": int(raw.nunique()),
                "distinct_normalized": int(normalized.nunique()),
                "other_share_normalized": float((normalized == "other").mean())
                if len(normalized)
                else None,
                "top_raw": raw.value_counts().head(3).to_dict(),
            }
        )
    return pd.DataFrame(rows)


# --- Duplicatas --------------------------------------------------------------


def normalized_title(title: object) -> str:
    text = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def duplicate_titles(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.assign(title_key=frame["title"].map(normalized_title))
    keyed = keyed.loc[keyed["title_key"] != ""]
    repeated = keyed.groupby(["channel_id", "title_key"])["video_id"].transform("size") > 1
    return keyed.loc[repeated, ["sector", "channel_id", "video_id", "title_key", "published_at"]]


def near_duplicate_pairs(
    embeddings: pd.DataFrame, video_ids: list[str], threshold: float = 0.97
) -> pd.DataFrame:
    """Pares de vídeos com embedding de thumbnail quase idêntico."""
    subset = embeddings.loc[embeddings["video_id"].isin(video_ids)].reset_index(drop=True)
    if len(subset) < 2:
        return pd.DataFrame(columns=["video_id_a", "video_id_b", "cosine"])
    matrix = np.vstack(subset["embedding"].to_numpy()).astype(float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = matrix @ matrix.T
    left, right = np.where(np.triu(similarity, k=1) >= threshold)
    return pd.DataFrame(
        {
            "video_id_a": subset["video_id"].to_numpy()[left],
            "video_id_b": subset["video_id"].to_numpy()[right],
            "cosine": similarity[left, right],
        }
    )


def analysis_hash(analysis: object) -> str:
    return hashlib.sha256(
        json.dumps(analysis, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


# --- Viés de seleção da amostra VLM -----------------------------------------


def within_channel_percentile(frame: pd.DataFrame, score: pd.Series) -> pd.Series:
    return score.groupby(frame["channel_id"]).rank(pct=True)


def selection_check(frame: pd.DataFrame, selected: pd.Series, score: pd.Series) -> pd.DataFrame:
    """Compara a posição intra-canal dos vídeos analisados com a dos demais.

    Se a amostra VLM não depende do resultado, o percentil médio fica perto de
    0,5 e a fração no top 10% perto de 0,10 nos dois grupos.
    """
    percentile = within_channel_percentile(frame, score)
    rows = []
    for sector, group in frame.groupby("sector"):
        for label, mask in (("analisados", selected), ("nao_analisados", ~selected)):
            values = percentile.loc[group.index][mask.loc[group.index]].dropna()
            margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else None
            rows.append(
                {
                    "sector": sector,
                    "group": label,
                    "videos": len(values),
                    "mean_percentile": float(values.mean()) if len(values) else None,
                    "mean_percentile_ci95": [
                        float(values.mean() - margin),
                        float(values.mean() + margin),
                    ]
                    if margin is not None
                    else None,
                    "top10_share": float((values > 0.9).mean()) if len(values) else None,
                }
            )
    return pd.DataFrame(rows)
