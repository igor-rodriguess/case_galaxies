"""Ablação visual para vídeos de até 180 segundos, sem vazamento temporal.

Compara metadados/cadência contra o mesmo conjunto acrescido de um embedding
ResNet-18 de thumbnail. A redução PCA, a imputação, o encoder categórico e o
modelo são ajustados exclusivamente no treino de cada dobra. A escolha por
setor ocorre nas três dobras internas de calendário; o corte externo é lido
uma vez ao final.

O script exige cobertura visual total da população avaliada. Assim uma variante
não vence apenas por ter descartado vídeos difíceis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.evaluation import make_estimator, training_rows_and_labels  # noqa: E402
from galaxies.agent.format_ranking import (  # noqa: E402
    bootstrap_channels,
    channel_level,
    format_as_of_labels,
    inner_calendar_folds,
    outer_split,
    ranking_metrics,
    shorts_population,
    unit_counts,
    with_format_units,
)
from galaxies.agent.protocol import (  # noqa: E402
    CADENCE_NUMERIC,
    METADATA_CATEGORICAL,
    METADATA_NUMERIC,
    load_dataset,
)
from galaxies.agent.spec import ExperimentSpec  # noqa: E402
from galaxies.modeling.preprocessing import make_preprocessor  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


@dataclass(frozen=True)
class VisualConfig:
    name: str
    model: str
    pooling: str
    train_label: str
    pca_components: int | None = None

    @property
    def uses_visual(self) -> bool:
        return self.pca_components is not None

    def spec(self) -> ExperimentSpec:
        return ExperimentSpec(
            hypothesis=self.name,
            track="full",
            feature_groups=["metadata", "cadence"],
            pooling=self.pooling,  # type: ignore[arg-type]
            format_strategy="joint",
            model=self.model,  # type: ignore[arg-type]
            complexity="low",
            train_min_age_days=30,
            train_label=self.train_label,  # type: ignore[arg-type]
            evaluation_mode="known_channel",
        )


# Ordem também é a regra de desempate da seleção interna: a configuração mais
# simples vence empates, evitando premiar complexidade sem evidência.
CONFIGS = (
    VisualConfig("reference_logistic_pooled_top10", "logistic", "pooled", "top10"),
    VisualConfig("metadata_histgb_pooled_clean20", "hist_gb", "pooled", "clean20vs50"),
    VisualConfig("resnet_pca16_histgb_pooled_clean20", "hist_gb", "pooled", "clean20vs50", 16),
    VisualConfig("resnet_pca32_histgb_pooled_clean20", "hist_gb", "pooled", "clean20vs50", 32),
    VisualConfig("resnet_pca16_mlp_pooled_clean20", "mlp", "pooled", "clean20vs50", 16),
    VisualConfig("resnet_pca32_mlp_pooled_clean20", "mlp", "pooled", "clean20vs50", 32),
    VisualConfig("resnet_pca16_catboost_sector_top10", "catboost", "per_sector", "top10", 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-map", type=Path, required=True)
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=ROOT / "data" / "features" / "embeddings" / "thumb_resnet18.parquet",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(feature_map: Path, embedding_path: Path, max_seconds: float) -> pd.DataFrame:
    dataset = load_dataset(feature_map)
    rows = with_format_units(shorts_population(dataset.evaluable("full"), max_seconds))
    vectors = pd.read_parquet(embedding_path, columns=["video_id", "embedding"])
    vectors = vectors.drop_duplicates("video_id", keep="last").copy()
    vectors["_visual_embedding"] = vectors.pop("embedding").map(
        lambda value: np.asarray(value, dtype=np.float32)
    )
    rows = rows.merge(vectors, on="video_id", how="left", validate="one_to_one")
    missing = int(rows["_visual_embedding"].isna().sum())
    if missing:
        raise ValueError(
            f"Cobertura visual incompleta: faltam {missing} de {len(rows)} vídeos. "
            "Não compare variantes em populações diferentes."
        )
    dimensions = {len(vector) for vector in rows["_visual_embedding"]}
    if len(dimensions) != 1:
        raise ValueError(f"Dimensões de embedding inconsistentes: {sorted(dimensions)}")
    return rows


def _features(
    config: VisualConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_train: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    numeric = list(METADATA_NUMERIC) + list(CADENCE_NUMERIC)
    categorical = list(METADATA_CATEGORICAL)
    if config.pooling == "pooled":
        categorical.append("sector")
    preprocessor = make_preprocessor({"features": {"numeric": numeric, "categorical": categorical}})
    base_train = preprocessor.fit_transform(selected_train[numeric + categorical])
    base_test = preprocessor.transform(test[numeric + categorical])
    if not config.uses_visual:
        return base_train, base_test, 0

    # PCA pode olhar todos os criativos do treino, mas nunca o teste. A
    # transformação não recebe o rótulo e preserva exemplos intermediários que
    # o contraste limpo teria removido apenas para a etapa supervisionada.
    train_vectors = np.stack(train["_visual_embedding"].to_numpy())
    test_vectors = np.stack(test["_visual_embedding"].to_numpy())
    components = min(int(config.pca_components or 1), len(train_vectors) - 1, train_vectors.shape[1])
    if components < 1:
        raise ValueError("Treino insuficiente para PCA visual.")
    pca = PCA(n_components=components, random_state=seed)
    pca.fit(train_vectors)
    selected_vectors = np.stack(selected_train["_visual_embedding"].to_numpy())
    return (
        np.hstack([base_train, pca.transform(selected_vectors)]),
        np.hstack([base_test, pca.transform(test_vectors)]),
        components,
    )


def score_config(config: VisualConfig, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    spec = config.spec()
    positions = pd.Series(np.arange(len(test)), index=test.index)
    scores = np.full(len(test), 0.5, dtype=float)
    groups = [(None, test)] if config.pooling == "pooled" else list(test.groupby("sector", sort=False))
    for sector, sector_test in groups:
        sector_train = train if sector is None else train.loc[train["sector"] == sector]
        selected, labels = training_rows_and_labels(spec, sector_train)
        if selected.empty or len(np.unique(labels)) < 2:
            scores[positions.loc[sector_test.index]] = float(labels.mean()) if len(labels) else 0.5
            continue
        x_train, x_test, _ = _features(config, sector_train, sector_test, selected, seed)
        estimator = make_estimator(spec, seed)
        fit_params = {}
        if config.model == "mlp":
            fit_params["sample_weight"] = compute_sample_weight("balanced", labels)
        estimator.fit(x_train, labels, **fit_params)
        scores[positions.loc[sector_test.index]] = estimator.predict_proba(x_test)[:, 1]
    return scores


def counts_for(scores: np.ndarray, test: pd.DataFrame) -> pd.DataFrame:
    scored = test[["video_id", "channel_id", "source_channel_id", "sector", "viral"]].copy()
    scored["score"] = scores
    return channel_level(unit_counts(scored, "score"))


def choose_internally(rows: pd.DataFrame, seed: int) -> tuple[dict[str, str], pd.DataFrame]:
    observations: list[dict[str, object]] = []
    for fold, (inner_train, inner_valid) in enumerate(inner_calendar_folds(rows)):
        labelled_train, labelled_valid = format_as_of_labels(inner_train, inner_valid)
        for config in CONFIGS:
            counts = counts_for(score_config(config, labelled_train, labelled_valid, seed), labelled_valid)
            for (sector, channel), row in counts.iterrows():
                if row["pairs"]:
                    observations.append(
                        {
                            "fold": fold,
                            "sector": sector,
                            "channel_id": channel,
                            "config": config.name,
                            "pairs": float(row["pairs"]),
                            "correct": float(row["correct"]),
                        }
                    )
    table = pd.DataFrame(observations)
    grouped = table.groupby(["sector", "config", "channel_id"])[["pairs", "correct"]].sum()
    grouped = grouped.loc[grouped["pairs"] > 0].copy()
    grouped["auc"] = grouped["correct"] / grouped["pairs"]
    summary = grouped.groupby(["sector", "config"]).agg(
        inner_auc=("auc", "mean"), channels=("auc", "size"), pairs=("pairs", "sum")
    ).reset_index()
    order = {config.name: position for position, config in enumerate(CONFIGS)}
    summary["order"] = summary["config"].map(order)
    chosen = {
        sector: str(group.sort_values(["inner_auc", "order"], ascending=[False, True]).iloc[0]["config"])
        for sector, group in summary.groupby("sector")
    }
    return chosen, summary.drop(columns="order")


def main() -> int:
    args = parse_args()
    output = MODELING_RUNS_DIR / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.feature_map, args.embedding_path, args.max_seconds)
    train, test, cutoffs = outer_split(rows)
    print("Seleção interna da ablação visual...", flush=True)
    chosen, inner = choose_internally(train, args.seed)
    inner.to_csv(output / "inner_selection.csv", index=False)
    frozen = {
        "chosen_by_sector": chosen,
        "configs": [config.__dict__ for config in CONFIGS],
        "feature_map": str(args.feature_map),
        "embedding_path": str(args.embedding_path),
        "max_seconds": args.max_seconds,
        "seed": args.seed,
        "sector_cutoffs": cutoffs,
    }
    frozen_text = json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True)
    (output / "frozen_selection.json").write_text(frozen_text + "\n", encoding="utf-8")
    frozen_hash = hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()

    by_name = {config.name: config for config in CONFIGS}
    rows_report: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for sector, name in sorted(chosen.items()):
        sector_test = test.loc[test["sector"] == sector]
        config = by_name[name]
        candidate_counts = counts_for(score_config(config, train, sector_test, args.seed), sector_test)
        reference = by_name[CONFIGS[0].name]
        reference_counts = counts_for(score_config(reference, train, sector_test, args.seed), sector_test)
        metrics = ranking_metrics(candidate_counts)
        reference_metrics = ranking_metrics(reference_counts)
        intervals = bootstrap_channels(candidate_counts, reference_counts, args.bootstrap, args.seed)
        scored = sector_test[["video_id", "sector", "source_channel_id", "viral"]].copy()
        scored["score"] = score_config(config, train, sector_test, args.seed)
        scored["config"] = name
        predictions.append(scored)
        rows_report.append(
            {
                "sector": sector,
                "chosen": name,
                "uses_visual": config.uses_visual,
                **metrics,
                "reference_auc": reference_metrics["auc_macro_channel"],
                "delta_auc_macro": metrics["auc_macro_channel"] - reference_metrics["auc_macro_channel"],
                "ci": intervals,
            }
        )
    pd.concat(predictions).to_parquet(output / "predictions.parquet", index=False)
    report = {
        "schema_version": "0.1.0",
        "protocol": "operational-180s-nested-visual-v1",
        "frozen_selection_sha256": frozen_hash,
        "population": {"train_rows": len(train), "test_rows": len(test), "test_channels": int(test["source_channel_id"].nunique())},
        "sectors": rows_report,
        "promotion_rule": "Não promover se IC95% do delta contra a referência cruzar zero.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    lines = [
        "# Ablação visual de Shorts (até 180 s)",
        "",
        f"Cobertura visual total; seleção interna congelada (`{frozen_hash[:12]}`) antes do corte externo.",
        "",
        "| Setor | Escolhido | Usa imagem | AUC macro por canal | Ref. AUC | Δ AUC (IC95%) | Lift@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows_report:
        ci = row["ci"]
        delta = ci.get("delta_auc_macro", [float("nan"), float("nan")])
        lines.append(
            f"| {row['sector']} | {row['chosen']} | {'sim' if row['uses_visual'] else 'não'} | "
            f"{row['auc_macro_channel']:.3f} | {row['reference_auc']:.3f} | "
            f"{row['delta_auc_macro']:+.3f} [{delta[0]:+.3f}; {delta[1]:+.3f}] | {row['lift_top10']:.2f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
