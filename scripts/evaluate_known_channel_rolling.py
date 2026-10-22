"""Backtest cronológico rolante para candidatos de ranking em canal conhecido.

É uma ferramenta de diagnóstico de estabilidade. Os resultados não substituem a
avaliação final prospectiva nem promovem artefatos automaticamente.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    bootstrap_top_fraction,
    channel_pair_counts,
    channel_top_counts,
    fit_predict,
    paired_comparison,
    summarize_scores,
)
from galaxies.agent.protocol import latest_feature_map, load_dataset, temporal_as_of_labels  # noqa: E402
from galaxies.agent.rolling import fold_aware_channels, rolling_calendar_splits  # noqa: E402
from galaxies.agent.spec import ExperimentSpec  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest temporal rolante de ranking Galaxies.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def candidates() -> tuple[ExperimentSpec, ...]:
    common = {
        "track": "full",
        "feature_groups": ["metadata", "cadence"],
        "format_strategy": "joint",
        "train_min_age_days": 30,
        "evaluation_mode": "known_channel",
    }
    return (
        ExperimentSpec(
            hypothesis="Referência linear regularizada.",
            model="logistic",
            complexity="low",
            train_label="top10",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Árvores rasas capturam interações estáveis.",
            model="hist_gb",
            complexity="low",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Árvores por setor absorvem diferenças de domínio.",
            model="hist_gb",
            complexity="low",
            train_label="clean20vs50",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Ranking pareado por canal explora ordem relativa.",
            model="catboost_ranker",
            complexity="low",
            train_label="top10",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Extra Trees por setor testa interações com aleatorização forte.",
            model="extra_trees",
            complexity="low",
            train_label="clean20vs50",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Random Forest por setor oferece ensemble de baixa variância.",
            model="random_forest",
            complexity="low",
            train_label="clean20vs50",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="CatBoost classificador por setor trata interações regularizadas.",
            model="catboost",
            complexity="low",
            train_label="clean20vs50",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="MLP pequena verifica não linearidade sem profundidade excessiva.",
            model="mlp",
            complexity="low",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
    )


def candidate_name(spec: ExperimentSpec) -> str:
    return f"{spec.model}_{spec.pooling}_{spec.train_label}"


def main() -> int:
    args = parse_args()
    if args.bootstrap < 50:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 50.")
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    rows = load_dataset(feature_map).evaluable("full")
    splits = rolling_calendar_splits(rows)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = MODELING_RUNS_DIR / (args.run_id or f"known-channel-rolling-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=False)

    all_predictions: dict[str, list[pd.DataFrame]] = {candidate_name(spec): [] for spec in candidates()}
    fold_rows: list[dict[str, object]] = []
    windows = []
    for split in splits:
        train, test = temporal_as_of_labels(split.train, split.test)
        windows.append({"fold": split.fold, "sectors": split.sector_windows, "train_rows": len(train), "test_rows": len(test)})
        for spec in candidates():
            name = candidate_name(spec)
            prediction = test[
                ["video_id", "channel_id", "sector", "viral", "target_age_days"]
            ].copy()
            prediction["fold"] = split.fold
            prediction["score"] = fit_predict(spec, train, test, args.seed, known_channel=True)
            metrics = summarize_scores(prediction, "score")
            fold_rows.append({"fold": split.fold, "model": name, **metrics})
            all_predictions[name].append(prediction)

    reference_name = "logistic_pooled_top10"
    results: list[dict[str, object]] = []
    reference_counts: pd.DataFrame | None = None
    for name, pieces in all_predictions.items():
        predictions = pd.concat(pieces, ignore_index=True)
        predictions.to_parquet(output_dir / f"predictions-{name}.parquet", index=False)
        aggregate = fold_aware_channels(predictions)
        counts = channel_pair_counts(aggregate, "score")
        metrics = summarize_scores(aggregate, "score")
        metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
            counts, args.bootstrap, args.seed
        )
        metrics.update(bootstrap_top_fraction(channel_top_counts(predictions, "score"), args.bootstrap, args.seed))
        if name == reference_name:
            reference_counts = counts
        results.append({"name": name, "metrics": metrics})
    if reference_counts is None:
        raise RuntimeError("A referência logística não foi avaliada.")
    for item in results:
        if item["name"] != reference_name:
            candidate_frame = fold_aware_channels(
                pd.concat(all_predictions[item["name"]], ignore_index=True)
            )
            candidate_counts = channel_pair_counts(candidate_frame, "score")
            item["paired_vs_reference"] = paired_comparison(
                candidate_counts, reference_counts, args.bootstrap, args.seed
            )

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    reference_by_fold = fold_frame.loc[
        fold_frame["model"] == reference_name, ["fold", "auc_intra"]
    ].set_index("fold")["auc_intra"]
    stability: dict[str, dict[str, object]] = {}
    for name, group in fold_frame.groupby("model", sort=False):
        deltas = group.set_index("fold")["auc_intra"].sub(reference_by_fold, fill_value=float("nan"))
        stability[str(name)] = {
            "folds": len(deltas),
            "folds_better_than_reference": int((deltas > 0).sum()),
            "worst_fold_auc_delta": float(deltas.min()),
            "per_fold_auc_delta": {str(fold): float(value) for fold, value in deltas.items()},
        }
    for item in results:
        item["stability"] = stability[str(item["name"])]
    report = {
        "schema_version": "0.1.0",
        "feature_map": str(feature_map),
        "protocol": (
            "Janelas cronológicas crescentes; em cada dobra, ajuste de idade e limiar viral "
            "são calculados só no treino. Resultado exploratório: não promove modelo."
        ),
        "windows": windows,
        "reference": reference_name,
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Backtest rolante — canal conhecido",
        "",
        "Cada dobra usa apenas publicações anteriores para calibrar idade e definir o limiar ",
        "de viralidade. Este relatório é exploratório e não promove um campeão.",
        "",
        "| modelo | AUC intra-canal [IC95%] | Precisão@10 | Recall@10 | Lift@10 | Δ AUC vs referência [IC95%] | Dobras acima da referência |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metrics = item["metrics"]
        paired = item.get("paired_vs_reference")
        delta = "—" if not paired else f"{paired['diff']:+.3f} [{paired['ci_low']:+.3f}; {paired['ci_high']:+.3f}]"
        stable = item["stability"]
        lines.append(
            f"| {item['name']} | {metrics['auc_intra']:.3f} [{metrics['auc_intra_ci_low']:.3f}; {metrics['auc_intra_ci_high']:.3f}] | "
            f"{metrics['precision_top10']:.3f} | {metrics['recall_top10']:.3f} | {metrics['lift_top10']:.2f} | {delta} |"
            f" {stable['folds_better_than_reference']}/{stable['folds']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
