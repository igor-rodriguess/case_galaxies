"""Splits que respeitam tempo e canal."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class Holdout:
    train_indices: list[int]
    test_indices: list[int]
    description: str


def temporal_holdout(
    frame: pd.DataFrame,
    test_fraction: float,
    minimum_test_rows: int,
) -> Holdout:
    timestamps = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    if timestamps.isna().any():
        missing = int(timestamps.isna().sum())
        raise ValueError(f"{missing} linhas não têm published_at válido.")
    ordered = timestamps.sort_values(kind="stable").index.tolist()
    test_size = max(minimum_test_rows, math.ceil(len(ordered) * test_fraction))
    if test_size >= len(ordered):
        raise ValueError(f"O holdout temporal pediria {test_size} de {len(ordered)} linhas.")
    train = ordered[:-test_size]
    test = ordered[-test_size:]
    cutoff = timestamps.loc[test].min().isoformat()
    return Holdout(train, test, f"Treino antes de {cutoff}; teste a partir do corte.")


def channel_holdout(
    frame: pd.DataFrame,
    test_fraction: float,
    random_state: int,
) -> Holdout:
    groups = frame["channel_id"].astype(str)
    if groups.nunique() < 2:
        raise ValueError("O split por canal exige pelo menos dois canais.")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_state,
    )
    train_position, test_position = next(splitter.split(frame, groups=groups))
    train = frame.index[train_position].tolist()
    test = frame.index[test_position].tolist()
    held_out = sorted(groups.loc[test].unique().tolist())
    return Holdout(train, test, f"Canais fora do treino: {held_out}")
