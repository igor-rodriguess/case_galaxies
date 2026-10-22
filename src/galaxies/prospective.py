"""Reserva do período prospectivo usado como validação final.

Todos os holdouts da base atual já foram consultados durante o desenvolvimento.
A validação final, portanto, usa vídeos publicados a partir de
``PROSPECTIVE_START``, rotulados pela leitura observada perto do dia 30.
Até a abertura única descrita em `docs/validacao_final_prospectiva.md`, nenhum
treino, seleção ou diagnóstico pode ler esses vídeos.
"""

from __future__ import annotations

import pandas as pd

PROSPECTIVE_START = pd.Timestamp("2026-10-21T00:00:00Z")
HORIZON_DAYS = 30
HORIZON_TOLERANCE_HOURS = 18.0
MIN_VIDEOS_PER_SECTOR = 100


def is_prospective(published_at: pd.Series) -> pd.Series:
    published = pd.to_datetime(published_at, utc=True, errors="coerce")
    return published >= PROSPECTIVE_START


def exclude_prospective(frame: pd.DataFrame, column: str = "published_at") -> pd.DataFrame:
    """Remove do desenvolvimento os vídeos reservados para a validação final."""
    return frame.loc[~is_prospective(frame[column])].copy()


def assert_development_only(frame: pd.DataFrame, column: str = "published_at") -> None:
    reserved = int(is_prospective(frame[column]).sum())
    if reserved:
        raise ValueError(
            f"{reserved} vídeos do período prospectivo chegaram ao desenvolvimento; "
            "eles só podem ser lidos pela avaliação final."
        )


OBSERVATION_AGES_DAYS = (1, 3, 7, 14, HORIZON_DAYS)
SHORTS_MAX_SECONDS = 180.0


def window_status(
    published_at: pd.Timestamp,
    now: pd.Timestamp,
    observed_ages: set[int],
    ages: tuple[int, ...] = OBSERVATION_AGES_DAYS,
    tolerance_hours: float = HORIZON_TOLERANCE_HOURS,
) -> dict[str, list[int]]:
    """Separa as idades-alvo em devidas agora, perdidas e futuras.

    Uma leitura só vale dentro de ``idade ± tolerance_hours``. Fora disso a
    janela é marcada como perdida: nunca se usa uma leitura de outra idade no
    lugar, nem se interpola.
    """
    age_hours = (now - published_at).total_seconds() / 3_600
    status: dict[str, list[int]] = {"due": [], "missed": [], "pending": []}
    for age in ages:
        if age in observed_ages:
            continue
        distance = age_hours - age * 24
        if abs(distance) <= tolerance_hours:
            status["due"].append(age)
        elif distance > tolerance_hours:
            status["missed"].append(age)
        else:
            status["pending"].append(age)
    return status


def is_tracked_short(
    published_at: pd.Timestamp,
    duration_seconds: float | None,
    max_seconds: float = SHORTS_MAX_SECONDS,
) -> bool:
    """Short do período reservado: publicado na reserva e com duração até o limite."""
    return (
        duration_seconds is not None
        and 0 < duration_seconds <= max_seconds
        and published_at >= PROSPECTIVE_START
    )
