"""Amostragem caso-controle para anotar aberturas de vídeos.

O módulo separa a decisão de *quais vídeos anotar* da modelagem.  Ele usa o
rótulo congelado do protocolo somente para montar uma amostra eficiente: cada
caso viral recebe controles não virais do mesmo canal, formato e época.  Essa
informação não é passada ao VLM e jamais vira uma feature de produção.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from typing import Any

import pandas as pd

from galaxies.agent.protocol import MIN_EVAL_AGE_DAYS, age_adjusted_performance, viral_label

SECTOR_ORDER = ("automotivo", "moda", "beleza")


def parse_duration_seconds(value: object) -> float | None:
    """Converte a duração ISO-8601 da Data API em segundos."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"P(?:\d+D)?T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?",
        value,
    )
    if not match:
        return None
    return (
        int(match.group("h") or 0) * 3600
        + int(match.group("m") or 0) * 60
        + float(match.group("s") or 0)
    )


def _as_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepara apenas os campos necessários para seleção e pareamento."""
    rows: list[dict[str, Any]] = []
    for record in records:
        published = _as_utc(record.get("published_at"))
        collected = _as_utc(record.get("collected_at"))
        view_count = pd.to_numeric(record.get("view_count"), errors="coerce")
        age_days = (
            float((collected - published).total_seconds() / 86_400)
            if pd.notna(published) and pd.notna(collected)
            else math.nan
        )
        rows.append(
            {
                "video_id": record.get("video_id"),
                "sector": record.get("sector"),
                "channel_id": record.get("channel_id"),
                "channel_title": record.get("channel_title"),
                "published_at": published,
                "target_age_days": age_days,
                "target_view_count": view_count,
                "duration_seconds": parse_duration_seconds(record.get("duration")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["video_id"].isna().any() or frame["video_id"].duplicated().any():
        raise ValueError("A coleta precisa ter video_ids únicos e presentes.")
    frame["is_short"] = frame["duration_seconds"].le(180).where(
        frame["duration_seconds"].notna(), other=pd.NA
    )
    frame["in_population"] = (
        frame["target_age_days"].ge(MIN_EVAL_AGE_DAYS)
        & frame["target_view_count"].notna()
        & frame["published_at"].notna()
        & frame["channel_id"].notna()
    )
    frame["adjusted_performance"] = math.nan
    frame["viral"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    population = frame.loc[frame["in_population"]].copy()
    if not population.empty:
        performance = age_adjusted_performance(population)
        frame.loc[population.index, "adjusted_performance"] = performance
        frame.loc[population.index, "viral"] = viral_label(population, performance).astype("Int64")
    return frame


def _match_cost(case: pd.Series, control: pd.Series) -> tuple[float, float, float, str]:
    """Ordena controles por proximidade de época e duração, deterministicamente."""
    days_apart = abs((control["published_at"] - case["published_at"]).total_seconds()) / 86_400
    case_duration = max(float(case["duration_seconds"] or 0), 1.0)
    control_duration = max(float(control["duration_seconds"] or 0), 1.0)
    duration_gap = abs(math.log(control_duration) - math.log(case_duration))
    return (days_apart + 14 * duration_gap, days_apart, duration_gap, str(control["video_id"]))


def build_case_control_queue(
    records: list[dict[str, Any]],
    *,
    analyzed_video_ids: set[str],
    controls_per_case: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cria grupos caso-controle sem reutilizar controles na mesma rodada.

    Casos e controles já analisados ficam fora da fila. Um caso com menos que
    ``controls_per_case`` controles ainda é preservado, com a insuficiência
    explícita, porque a cobertura dos virais é a prioridade.
    """
    if not 1 <= controls_per_case <= 3:
        raise ValueError("controls_per_case deve estar entre 1 e 3.")
    frame = records_frame(records)
    population = frame.loc[frame["in_population"]].copy()
    population["already_analyzed"] = population["video_id"].isin(analyzed_video_ids)
    cases = population.loc[(population["viral"] == 1) & ~population["already_analyzed"]].copy()
    cases = cases.sort_values(["sector", "channel_id", "published_at", "video_id"])
    available_controls = population.loc[
        (population["viral"] == 0) & ~population["already_analyzed"]
    ].copy()
    used_controls: set[str] = set()
    groups: list[dict[str, Any]] = []

    for case_number, (_, case) in enumerate(cases.iterrows(), start=1):
        same_format = available_controls.loc[
            (available_controls["channel_id"] == case["channel_id"])
            & (available_controls["is_short"] == case["is_short"])
            & ~available_controls["video_id"].isin(used_controls)
        ].copy()
        # Formato ausente é preferível a trocar de canal; não inventamos que é short/long.
        candidates = same_format
        if candidates.empty:
            selected = candidates
        else:
            candidates["_cost"] = candidates.apply(
                lambda row, case_row=case: _match_cost(case_row, row), axis=1
            )
            selected = candidates.sort_values("_cost").head(controls_per_case)
        selected_ids = [str(video_id) for video_id in selected["video_id"]]
        used_controls.update(selected_ids)
        groups.append(
            {
                "group_id": f"cc-{case_number:04d}",
                "sector": str(case["sector"]),
                "channel_id": str(case["channel_id"]),
                "channel_title": case["channel_title"],
                "format": "short" if bool(case["is_short"]) else "long",
                "case": _queue_item(case, role="case", group_id=f"cc-{case_number:04d}"),
                "controls": [
                    _queue_item(row, role="control", group_id=f"cc-{case_number:04d}")
                    for _, row in selected.iterrows()
                ],
                "controls_requested": controls_per_case,
                "controls_selected": len(selected),
                "matching_status": "complete" if len(selected) == controls_per_case else "partial",
            }
        )

    items = _interleave_groups(groups)
    summary = {
        "records_total": len(frame),
        "mature_population": int(frame["in_population"].sum()),
        "virals_in_population": int((frame["viral"] == 1).sum()),
        "analyzed_video_ids": len(analyzed_video_ids),
        "cases_pending": len(groups),
        "controls_pending": sum(len(group["controls"]) for group in groups),
        "items_pending": len(items),
        "complete_groups": sum(group["matching_status"] == "complete" for group in groups),
        "partial_groups": sum(group["matching_status"] == "partial" for group in groups),
        "by_sector": dict(Counter(item["sector"] for item in items)),
    }
    return items, {"groups": groups, "summary": summary}


def _queue_item(row: pd.Series, *, role: str, group_id: str) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "role": role,
        "video_id": str(row["video_id"]),
        "sector": str(row["sector"]),
        "channel_id": str(row["channel_id"]),
        "channel_title": row["channel_title"],
        "published_at": row["published_at"].isoformat(),
        "format": "short" if bool(row["is_short"]) else "long",
        "duration_seconds": float(row["duration_seconds"]),
        # Estes campos existem para auditoria do plano. O executor VLM deve
        # consumir somente video_id e contexto de origem, nunca o rótulo.
        "selection_audit": {
            "eligible_30_days": True,
            "observed_label": "viral" if role == "case" else "non_viral_control",
        },
    }


def _interleave_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Alterna setor e canal para que uma cota curta continue representativa."""
    queues: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(lambda: defaultdict(deque))
    for group in groups:
        ordered = [group["case"], *group["controls"]]
        queues[group["sector"]][group["channel_id"]].extend(ordered)
    ordered_sectors = [*filter(lambda sector: sector in queues, SECTOR_ORDER), *sorted(set(queues) - set(SECTOR_ORDER))]
    output: list[dict[str, Any]] = []
    while any(queue for channels in queues.values() for queue in channels.values()):
        for sector in ordered_sectors:
            for channel_id in sorted(queues[sector]):
                if queues[sector][channel_id]:
                    output.append(queues[sector][channel_id].popleft())
    for position, item in enumerate(output, start=1):
        item["position"] = position
    return output
