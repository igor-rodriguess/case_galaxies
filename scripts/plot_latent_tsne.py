"""t-SNE de um espaço latente (thumbnail ou título) dos Shorts até 180 s.

Descritivo: mostra se setores e vídeos Top-10% formam regiões próprias no
embedding. O rótulo é o mesmo da Fase 2 (as-of por unidade canal×formato, idade
e limiar ajustados no treino do corte externo). Nada aqui alimenta modelo.

A taxa de Top-10% entre os 15 vizinhos no espaço original (PCA 50) é comparada
com a taxa base, para não ler estrutura que o t-SNE inventa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.format_ranking import outer_split, shorts_population, with_format_units  # noqa: E402
from galaxies.agent.protocol import load_dataset  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

SECTOR_COLORS = {"automotivo": "#0065FF", "beleza": "#F574FF", "moda": "#00A88F"}
NEIGHBORS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, default=ROOT / "data" / "features" / "embeddings" / "thumb_resnet18.parquet")
    parser.add_argument("--label", default="Thumbnail (ResNet-18)")
    parser.add_argument(
        "--feature-map",
        type=Path,
        default=MODELING_RUNS_DIR / "feature-map-shorts-operational-180s-20261021" / "feature_map.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "final" / "modeling" / "figures" / "tsne_thumbnail.png")
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse", action="store_true", help="Reaproveita as coordenadas salvas ao lado da figura.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.feature_map)
    train, test, _ = outer_split(with_format_units(shorts_population(dataset.evaluable("full"), 180.0)))
    rows = pd.concat([train.assign(split="treino"), test.assign(split="teste")], ignore_index=True)
    vectors = pd.read_parquet(args.embedding, columns=["video_id", "embedding"]).drop_duplicates("video_id", keep="last")
    rows = rows.merge(vectors, on="video_id", how="inner", validate="one_to_one")
    matrix = np.stack(rows["embedding"].to_numpy()).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    reduced = PCA(n_components=50, random_state=args.seed).fit_transform(matrix)
    cached = args.output.with_suffix(".parquet")
    if args.reuse and cached.exists():
        saved = pd.read_parquet(cached).set_index("video_id").reindex(rows["video_id"])
        if saved[["x", "y"]].isna().any().any():
            raise SystemExit("Coordenadas salvas não cobrem a população atual; rode sem --reuse.")
        points = saved[["x", "y"]].to_numpy()
    else:
        print(f"{len(rows)} vídeos; t-SNE em andamento...", flush=True)
        points = TSNE(
            n_components=2, perplexity=args.perplexity, init="pca", learning_rate="auto", random_state=args.seed, n_jobs=-1
        ).fit_transform(reduced)

    viral = rows["viral"].to_numpy().astype(bool)
    _, index = NearestNeighbors(n_neighbors=NEIGHBORS + 1).fit(reduced).kneighbors(reduced)
    neighbor_rate = viral[index[:, 1:]].mean(axis=1)
    base_rate = float(viral.mean())
    stats = {
        "videos": int(len(rows)),
        "channels": int(rows["source_channel_id"].nunique()),
        "base_rate_top10": base_rate,
        "neighbor_top10_rate_if_top10": float(neighbor_rate[viral].mean()),
        "neighbor_top10_rate_if_not": float(neighbor_rate[~viral].mean()),
        "same_sector_neighbors": float(
            (rows["sector"].to_numpy()[index[:, 1:]] == rows["sector"].to_numpy()[:, None]).mean()
        ),
        "same_channel_neighbors": float(
            (rows["source_channel_id"].to_numpy()[index[:, 1:]] == rows["source_channel_id"].to_numpy()[:, None]).mean()
        ),
    }

    coordinates = rows[["video_id", "source_channel_id", "sector", "viral", "split"]].assign(x=points[:, 0], y=points[:, 1])
    coordinates.to_parquet(args.output.with_suffix(".parquet"), index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    for sector, color in SECTOR_COLORS.items():
        mask = rows["sector"].to_numpy() == sector
        axes[0].scatter(points[mask, 0], points[mask, 1], s=2, alpha=0.45, color=color, label=f"{sector} ({mask.sum():,})".replace(",", "."), rasterized=True)
    axes[0].legend(markerscale=6, frameon=False, loc="upper right")
    axes[0].set_title(f"Por setor · {stats['same_sector_neighbors']:.0%} dos vizinhos são do mesmo setor".replace(".", ","))

    # Cor = setor; forma = rótulo. Os virais vão por cima, maiores e com borda.
    sectors = rows["sector"].to_numpy()
    for sector, color in SECTOR_COLORS.items():
        mask = (sectors == sector) & ~viral
        axes[1].scatter(points[mask, 0], points[mask, 1], s=3, marker="o", alpha=0.18, color=color, linewidths=0, rasterized=True)
    for sector, color in SECTOR_COLORS.items():
        mask = (sectors == sector) & viral
        axes[1].scatter(points[mask, 0], points[mask, 1], s=16, marker="^", alpha=0.9, color=color, edgecolors="#0F1729", linewidths=0.3, rasterized=True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#7A869E", markersize=5, alpha=0.6, label=f"não viral ({(~viral).sum():,})".replace(",", ".")),
        plt.Line2D([], [], marker="^", linestyle="", color="#7A869E", markeredgecolor="#0F1729", markersize=8, label=f"viral: Top-10% do canal ({viral.sum():,})".replace(",", ".")),
    ] + [plt.Line2D([], [], marker="s", linestyle="", color=color, markersize=7, label=sector) for sector, color in SECTOR_COLORS.items()]
    axes[1].legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    axes[1].set_title("Setor (cor) × viral (forma) · rótulo as-of da Fase 2")

    hexes = axes[2].hexbin(points[:, 0], points[:, 1], C=viral.astype(float), reduce_C_function=np.mean, gridsize=38, mincnt=15, cmap="RdBu_r", vmin=0, vmax=2 * base_rate)
    fig.colorbar(hexes, ax=axes[2], label=f"taxa Top-10% na célula (base {base_rate:.1%})".replace(".", ","))
    axes[2].set_title("Taxa de Top-10% por região (células com ≥ 15 vídeos)")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"t-SNE · {args.label} · {stats['videos']:,} Shorts de {stats['channels']} canais".replace(",", ".")
        + f"\nVizinhos (k={NEIGHBORS}, espaço original): Top-10% entre vizinhos de um Top-10% = {stats['neighbor_top10_rate_if_top10']:.1%}"
        f" vs {stats['neighbor_top10_rate_if_not']:.1%} de um não Top-10%; mesmo canal = {stats['same_channel_neighbors']:.0%}".replace(".", ","),
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    args.output.with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
