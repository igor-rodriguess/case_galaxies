"""Gráficos do ranking de Shorts a partir de uma execução de evaluate_shorts_ranking.

Gera, em ``reports/figures/<run_id>/``:
- ``topk_por_setor.png``: Precision@K, Recall@K e Lift@K (K = fração do canal);
- ``auc_por_canal.png``: AUC de cada canal no corte externo;
- ``selecao_interna.png``: AUC média por canal de cada candidato em cada dobra.

Uso:
    python scripts/plot_shorts_ranking.py shorts-ranking-final-14k
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.evaluation import channel_pair_counts  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

SECTORS = ("automotivo", "moda", "beleza")
SECTOR_COLORS = {"automotivo": "#2a78d6", "moda": "#eb6834", "beleza": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FRACTIONS = np.arange(0.05, 0.55, 0.05)


def style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_2, labelsize=9)


def topk(frame: pd.DataFrame, fraction: float) -> dict[str, float]:
    hits = selected = 0
    for _, group in frame.groupby("channel_id"):
        k = max(1, math.ceil(fraction * len(group)))
        hits += int(group.nlargest(k, "score")["viral"].sum())
        selected += k
    positives, rows = int(frame["viral"].sum()), len(frame)
    precision = hits / selected
    return {
        "precision": precision,
        "recall": hits / positives if positives else np.nan,
        "lift": precision / (positives / rows) if positives else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    run_dir = MODELING_RUNS_DIR / args.run_id
    figures = ROOT / "reports" / "figures" / args.run_id
    figures.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=SURFACE)
    titles = {"precision": "Precision@K", "recall": "Recall@K", "lift": "Lift@K"}
    for axis, metric in zip(axes, titles, strict=True):
        style(axis)
        for sector in SECTORS:
            group = predictions.loc[predictions["sector"] == sector]
            if group.empty:
                continue
            values = [topk(group, fraction)[metric] for fraction in FRACTIONS]
            axis.plot(
                FRACTIONS * 100,
                values,
                color=SECTOR_COLORS[sector],
                linewidth=2,
                marker="o",
                markersize=4,
                label=sector,
            )
        if metric == "lift":
            axis.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
            axis.annotate(
                "acaso",
                (FRACTIONS[-1] * 100, 1.0),
                xytext=(-30, 4),
                textcoords="offset points",
                color=INK_2,
                fontsize=8,
            )
        if metric == "recall":
            axis.plot(FRACTIONS * 100, FRACTIONS, color=INK_2, linewidth=1, linestyle="--")
        axis.set_title(titles[metric], color=INK, loc="left", fontsize=10)
        axis.set_xlabel("K: % de cada canal selecionado", color=INK_2)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Ranking de Shorts no corte externo, por setor", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(figures / "topk_por_setor.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)

    counts = channel_pair_counts(predictions, "score").join(
        predictions.groupby("channel_id")[["sector", "source_channel_id"]].first()
    )
    counts = counts.loc[counts["pairs"] > 0].assign(
        auc=lambda frame: frame["correct"] / frame["pairs"]
    )
    counts = counts.sort_values(["sector", "auc"])
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.32 * len(counts))), facecolor=SURFACE)
    style(axis)
    positions = np.arange(len(counts))
    axis.scatter(
        counts["auc"], positions, c=[SECTOR_COLORS[s] for s in counts["sector"]], s=60, zorder=3
    )
    for position, (_, row) in zip(positions, counts.iterrows(), strict=True):
        axis.annotate(
            f"{int(row['pairs'])} pares",
            (row["auc"], position),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=7,
            color=INK_2,
        )
    axis.axvline(0.5, color=INK_2, linewidth=1, linestyle="--")
    axis.set_yticks(
        positions,
        [
            f"{s} · {c[:10]}"
            for s, c in zip(counts["sector"], counts["source_channel_id"], strict=True)
        ],
        fontsize=7,
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("AUC intra-canal (0,5 = acaso)", color=INK_2)
    axis.set_title("AUC de cada canal no corte externo", color=INK, loc="left")
    figure.tight_layout()
    figure.savefig(figures / "auc_por_canal.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)

    inner = pd.read_csv(run_dir / "inner_selection.csv")
    fold_columns = [column for column in inner.columns if column.startswith("fold_")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, facecolor=SURFACE)
    for axis, sector in zip(axes, SECTORS, strict=True):
        style(axis)
        group = inner.loc[inner["sector"] == sector].sort_values("inner_auc")
        for position, (_, row) in enumerate(group.iterrows()):
            values = row[fold_columns].to_numpy(dtype=float)
            axis.scatter(values, [position] * len(values), color="#b8b7b1", s=18, zorder=2)
            axis.scatter(
                [row["inner_auc"]], [position], color=SECTOR_COLORS[sector], s=60, zorder=3
            )
        axis.set_yticks(range(len(group)), group["spec"].str.replace("_", " "), fontsize=8)
        axis.axvline(0.5, color=INK_2, linewidth=1, linestyle="--")
        axis.set_title(sector, color=INK, loc="left", fontsize=10)
        axis.set_xlabel("AUC média por canal", color=INK_2)
    figure.suptitle(
        "Seleção interna: média (cor) e cada dobra (cinza)", color=INK, x=0.01, ha="left"
    )
    figure.tight_layout()
    figure.savefig(figures / "selecao_interna.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"Figuras em {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
