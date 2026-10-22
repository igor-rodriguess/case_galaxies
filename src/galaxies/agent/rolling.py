"""Divisões cronológicas repetidas para validação de canal conhecido.

O módulo não escolhe nem promove modelos. Ele fornece janelas crescentes que
permitem medir estabilidade no tempo sem ajustar o alvo com observações futuras.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pandas as pd


@dataclass(frozen=True)
class RollingCalendarSplit:
    """Uma janela de treino anterior e teste posterior por setor."""

    fold: int
    train: pd.DataFrame
    test: pd.DataFrame
    sector_windows: dict[str, dict[str, str]]


def fold_aware_channels(frame: pd.DataFrame) -> pd.DataFrame:
    """Evita comparar scores ou rótulos de janelas temporais diferentes.

    Um mesmo canal aparece em várias dobras, mas cada dobra tem ajuste de idade,
    limiar viral e modelo treinado em uma data distinta. Por isso AUC e Top-K
    agregados devem usar a unidade ``fold x channel``, e não misturar pares de
    períodos distintos como se seus escores fossem diretamente comparáveis.
    """
    required = {"fold", "channel_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame exige colunas para agregação rolante: {sorted(missing)}")
    result = frame.copy()
    result["channel_id"] = (
        result["fold"].astype(str).str.cat(result["channel_id"].astype(str), sep="::")
    )
    return result


def rolling_calendar_splits(
    rows: pd.DataFrame,
    *,
    boundaries: tuple[float, ...] = (0.50, 0.625, 0.75, 1.0),
) -> list[RollingCalendarSplit]:
    """Cria janelas temporais consecutivas, preservando a ordem por setor.

    Para cada janela, o treino contém publicações anteriores à fronteira inicial
    e o teste ocupa o intervalo até a próxima fronteira. Canais que ainda não
    aparecem no treino não entram no teste, pois sua referência histórica não
    existia no momento da previsão.
    """
    if len(boundaries) < 2 or boundaries[-1] != 1.0:
        raise ValueError("boundaries deve terminar em 1.0 e conter ao menos dois valores.")
    if any(left <= 0 or right <= left or right > 1 for left, right in pairwise(boundaries)):
        raise ValueError("boundaries deve ser estritamente crescente entre 0 e 1.")
    if "published_at" not in rows or "sector" not in rows or "channel_id" not in rows:
        raise ValueError("rows exige published_at, sector e channel_id.")

    prepared = rows.copy()
    prepared["published_at"] = pd.to_datetime(prepared["published_at"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["published_at", "sector", "channel_id"])
    splits: list[RollingCalendarSplit] = []
    for fold, (train_fraction, test_fraction) in enumerate(pairwise(boundaries), start=1):
        train_mask = pd.Series(False, index=prepared.index)
        test_mask = pd.Series(False, index=prepared.index)
        windows: dict[str, dict[str, str]] = {}
        for sector, sector_rows in prepared.groupby("sector", sort=False):
            dates = sector_rows["published_at"]
            train_until = dates.quantile(train_fraction)
            test_until = dates.quantile(test_fraction)
            train_mask.loc[sector_rows.index] = dates < train_until
            if test_fraction == 1.0:
                test_mask.loc[sector_rows.index] = (dates >= train_until) & (dates <= test_until)
            else:
                test_mask.loc[sector_rows.index] = (dates >= train_until) & (dates < test_until)
            windows[str(sector)] = {
                "train_until_exclusive": train_until.isoformat(),
                "test_until": test_until.isoformat(),
            }
        train = prepared.loc[train_mask].copy()
        test = prepared.loc[test_mask].copy()
        test = test[test["channel_id"].isin(set(train["channel_id"]))].copy()
        if train.empty or test.empty:
            continue
        splits.append(
            RollingCalendarSplit(
                fold=fold,
                train=train,
                test=test,
                sector_windows=windows,
            )
        )
    if not splits:
        raise ValueError("Dados insuficientes para formar uma janela temporal rolante.")
    return splits
