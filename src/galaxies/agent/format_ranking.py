"""Ranking de vídeos de até 180 s com seleção aninhada por calendário.

O critério operacional definido para o projeto é direto: qualquer vídeo público
com duração positiva de até 180 segundos entra na população de Shorts. A Data
API não expõe uma marca canônica de Shorts nem orientação do arquivo; por isso
não aplicamos regras adicionais de data, proporção ou superfície do YouTube.

A unidade de comparação é **canal por formato** (com só Shorts, uma por canal):

- o ajuste de idade e o limiar top 10% são estimados por unidade, só no treino;
- os pares positivo/negativo da AUC são do mesmo canal e do mesmo formato;
- os intervalos reamostram canais reais, somando as unidades de cada canal.

A escolha do modelo usa só dobras de calendário dentro do treino. O corte
externo é lido uma única vez com a configuração congelada por setor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from galaxies.agent.evaluation import (
    TOP_FRACTION,
    calendar_split,
    channel_pair_counts,
    fit_predict,
)
from galaxies.agent.protocol import temporal_as_of_labels
from galaxies.agent.spec import ExperimentSpec

FORMAT_MAX_SECONDS = 180.0
SHORTS_MAX_SECONDS = 180.0
MIN_UNIT_TRAIN_ROWS = 10
INNER_CUTOFF_QUANTILES = (0.55, 0.70, 0.85)
UNIT_SEPARATOR = "::"


def shorts_duration_limit(published_at: pd.Series) -> pd.Series:
    """Retorna o limite operacional único, preservando a API pública da função."""
    return pd.Series(SHORTS_MAX_SECONDS, index=published_at.index, dtype=float)


def shorts_population(rows: pd.DataFrame, max_seconds: float = SHORTS_MAX_SECONDS) -> pd.DataFrame:
    """Mantém vídeos públicos de duração positiva até o limite operacional."""
    duration = pd.to_numeric(rows["duration_seconds"], errors="coerce")
    return rows.loc[duration.notna() & (duration > 0) & (duration <= max_seconds)].copy()


def with_format_units(rows: pd.DataFrame) -> pd.DataFrame:
    """Troca ``channel_id`` pela unidade canal por formato e guarda o canal real.

    O canal nunca é feature; a troca só muda quem é comparado com quem. Vídeos
    sem duração ficam de fora, porque não têm formato definido.
    """
    duration = pd.to_numeric(rows["duration_seconds"], errors="coerce")
    known = duration.notna() & (duration > 0)
    result = rows.loc[known].copy()
    result["source_channel_id"] = result["channel_id"].astype(str)
    result["format"] = np.where(duration.loc[known] > FORMAT_MAX_SECONDS, "longo", "curto")
    result["channel_id"] = result["source_channel_id"] + UNIT_SEPARATOR + result["format"]
    return result


def format_as_of_labels(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_unit_train_rows: int = MIN_UNIT_TRAIN_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rótulo por unidade, com idade e limiar ajustados só no treino.

    Unidades com menos de ``min_unit_train_rows`` vídeos no treino saem dos dois
    lados: um limiar top 10% com menos linhas que isso é instável demais.
    """
    sizes = train.groupby("channel_id").size()
    kept = set(sizes.loc[sizes >= min_unit_train_rows].index)
    train = train.loc[train["channel_id"].isin(kept)]
    test = test.loc[test["channel_id"].isin(kept)]
    return temporal_as_of_labels(train, test)


def inner_calendar_folds(
    train: pd.DataFrame, quantiles: Sequence[float] = INNER_CUTOFF_QUANTILES
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Dobras crescentes de calendário dentro do treino, por setor.

    A dobra ``k`` treina com tudo antes do corte ``k`` do setor e valida na
    janela até o corte seguinte (a última vai até o fim do treino). Nenhuma
    dobra lê o corte externo.
    """
    published = pd.to_datetime(train["published_at"], utc=True)
    folds: list[tuple[list[object], list[object]]] = [([], []) for _ in quantiles]
    for _, sector_rows in train.groupby("sector", sort=False):
        dates = published.loc[sector_rows.index]
        cutoffs = [dates.quantile(q) for q in quantiles]
        for position, cutoff in enumerate(cutoffs):
            following = cutoffs[position + 1] if position + 1 < len(cutoffs) else None
            window = dates >= cutoff
            if following is not None:
                window &= dates < following
            folds[position][0].extend(dates.index[dates < cutoff])
            folds[position][1].extend(dates.index[window])
    return [(train.loc[left], train.loc[right]) for left, right in folds]


def spec_name(spec: ExperimentSpec) -> str:
    return f"{spec.model}_{spec.pooling}_{spec.train_label}"


def unit_counts(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Pares e seleção Top-10% por unidade, com o canal real e o setor."""
    pairs = channel_pair_counts(frame, score_column)
    rows = []
    for unit, group in frame.groupby("channel_id", sort=True):
        k = max(1, math.ceil(TOP_FRACTION * len(group)))
        rows.append(
            {
                "channel_id": unit,
                "source_channel_id": str(group["source_channel_id"].iloc[0]),
                "sector": str(group["sector"].iloc[0]),
                "rows": float(len(group)),
                "positives": float(group["viral"].sum()),
                "selected": float(k),
                "hits": float(group.nlargest(k, score_column)["viral"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("channel_id").join(pairs)


def channel_level(counts: pd.DataFrame) -> pd.DataFrame:
    """Soma as unidades de cada canal real; é a unidade do bootstrap."""
    columns = ["pairs", "correct", "rows", "positives", "selected", "hits"]
    return counts.groupby(["sector", "source_channel_id"])[columns].sum()


def macro_auc(counts: pd.DataFrame) -> float:
    """Média da AUC de cada canal: cada canal pesa igual, tenha 10 ou 5.000 Shorts."""
    valid = counts.loc[counts["pairs"] > 0]
    return float((valid["correct"] / valid["pairs"]).mean()) if len(valid) else float("nan")


def ranking_metrics(counts: pd.DataFrame) -> dict[str, float]:
    """Métricas de ranking; ``counts`` deve ter uma linha por canal real.

    ``auc_macro_channel`` é a métrica principal desde a base de 14 mil Shorts,
    em que um canal sozinho teria a maioria dos pares. ``auc_same_format``
    (pares somados) fica como diagnóstico.
    """
    total = counts.sum()
    base_rate = total["positives"] / total["rows"] if total["rows"] else float("nan")
    precision = total["hits"] / total["selected"] if total["selected"] else float("nan")
    return {
        "auc_macro_channel": macro_auc(counts),
        "channels_with_pairs": float((counts["pairs"] > 0).sum()),
        "auc_same_format": float(total["correct"] / total["pairs"])
        if total["pairs"]
        else float("nan"),
        "pairs": float(total["pairs"]),
        "precision_top10": float(precision),
        "recall_top10": float(total["hits"] / total["positives"])
        if total["positives"]
        else float("nan"),
        "lift_top10": float(precision / base_rate) if base_rate else float("nan"),
        "rows": float(total["rows"]),
        "positives": float(total["positives"]),
    }


def bootstrap_channels(
    candidate: pd.DataFrame,
    reference: pd.DataFrame | None,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    """IC95% reamostrando canais reais; com ``reference``, também o delta pareado."""
    rng = np.random.default_rng(seed)
    keys = candidate.index
    values: dict[str, list[float]] = {
        "auc_macro_channel": [],
        "delta_auc_macro": [],
        "auc_same_format": [],
        "precision_top10": [],
        "recall_top10": [],
        "lift_top10": [],
        "delta_auc": [],
    }
    for _ in range(iterations):
        pick = keys[rng.integers(0, len(keys), len(keys))]
        metrics = ranking_metrics(candidate.loc[pick])
        if not np.isfinite(metrics["auc_same_format"]):
            continue
        for name in (
            "auc_macro_channel",
            "auc_same_format",
            "precision_top10",
            "recall_top10",
            "lift_top10",
        ):
            values[name].append(metrics[name])
        if reference is not None:
            base = ranking_metrics(reference.loc[pick])
            values["delta_auc"].append(metrics["auc_same_format"] - base["auc_same_format"])
            values["delta_auc_macro"].append(
                metrics["auc_macro_channel"] - base["auc_macro_channel"]
            )
    return {
        name: [float(np.nanquantile(series, 0.025)), float(np.nanquantile(series, 0.975))]
        for name, series in values.items()
        if series
    }


def score_specs(
    specs: Sequence[ExperimentSpec],
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Ajusta cada spec no treino e devolve as contagens por unidade no teste."""
    results = {}
    for spec in specs:
        scored = test[["channel_id", "source_channel_id", "sector", "viral"]].copy()
        scored["score"] = fit_predict(spec, train, test, seed, known_channel=True)
        results[spec_name(spec)] = unit_counts(scored, "score")
    return results


def nested_selection(
    specs: Sequence[ExperimentSpec],
    train: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Escolhe, por setor, a spec de maior AUC média por canal nas dobras internas.

    Os pares de cada canal são somados nas três dobras; a AUC do canal sai
    desse total, e a do setor é a média entre canais.

    Empates ficam com a spec que aparece antes em ``specs`` (as mais simples
    vêm primeiro). Devolve a escolha e a tabela de cada dobra.
    """
    rows = []
    for fold, (inner_train, inner_valid) in enumerate(inner_calendar_folds(train)):
        labelled_train, labelled_valid = format_as_of_labels(inner_train, inner_valid)
        if labelled_valid.empty or labelled_train.empty:
            continue
        for name, counts in score_specs(specs, labelled_train, labelled_valid, seed).items():
            for (sector, channel), channel_counts in counts.groupby(
                ["sector", "source_channel_id"]
            ):
                rows.append(
                    {
                        "fold": fold,
                        "spec": name,
                        "sector": sector,
                        "source_channel_id": channel,
                        "pairs": float(channel_counts["pairs"].sum()),
                        "correct": float(channel_counts["correct"].sum()),
                    }
                )
    table = pd.DataFrame(rows)
    order = {
        spec_name(spec): position for spec, position in zip(specs, range(len(specs)), strict=True)
    }
    per_channel = table.groupby(["sector", "spec", "source_channel_id"])[["pairs", "correct"]].sum()
    per_channel = per_channel.loc[per_channel["pairs"] > 0]
    per_channel["auc"] = per_channel["correct"] / per_channel["pairs"]
    summary = per_channel.groupby(["sector", "spec"]).agg(
        pairs=("pairs", "sum"),
        correct=("correct", "sum"),
        inner_auc=("auc", "mean"),
        channels=("auc", "size"),
    )
    summary["inner_auc_pooled_pairs"] = summary["correct"] / summary["pairs"]
    summary = summary.reset_index()
    summary["order"] = summary["spec"].map(order)
    chosen = {
        sector: str(
            group.sort_values(["inner_auc", "order"], ascending=[False, True]).iloc[0]["spec"]
        )
        for sector, group in summary.groupby("sector")
    }
    fold_auc = (
        table.loc[table["pairs"] > 0]
        .assign(auc=lambda frame: frame["correct"] / frame["pairs"])
        .pivot_table(index=["sector", "spec"], columns="fold", values="auc", aggfunc="mean")
    )
    fold_auc.columns = [f"fold_{column}_auc" for column in fold_auc.columns]
    summary = summary.merge(fold_auc.reset_index(), on=["sector", "spec"], how="left")
    return chosen, summary.drop(columns="order")


def outer_split(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Corte externo de calendário por setor, já com unidades e rótulo por formato."""
    train, test, cutoffs = calendar_split(rows)
    train, test = format_as_of_labels(train, test)
    return train, test, cutoffs
