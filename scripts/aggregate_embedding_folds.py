"""Agrega dobras LOCO concluídas da ablação de embeddings.

O script só aceita uma cobertura completa de canais para evitar publicar uma
métrica baseada em dobras parciais. A comparação pareada exige previsões da
referência sobre os mesmos vídeos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.embedding_evaluation import extreme_auc  # noqa: E402
from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    channel_pair_counts,
    paired_comparison,
    summarize_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agrega dobras LOCO de uma configuração.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--expected-channels", type=int, default=27)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "data/modeling/runs" / args.run_id
    fold_dir = run_dir / "folds" / args.config
    paths = sorted(fold_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Nenhuma dobra encontrada em {fold_dir}")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    channels = int(predictions["channel_id"].nunique())
    if channels != args.expected_channels:
        raise ValueError(
            f"Cobertura incompleta: {channels}/{args.expected_channels} canais. "
            "Conclua as dobras antes de agregar."
        )
    if predictions["video_id"].duplicated().any():
        raise ValueError("Há video_id duplicado entre as dobras.")

    counts = channel_pair_counts(predictions, "score")
    metrics = summarize_scores(predictions, "score")
    metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
        counts, args.bootstrap, args.seed
    )
    metrics["auc_top10_vs_bottom50"] = extreme_auc(predictions)

    reference = pd.read_parquet(args.reference_predictions)
    expected_ids = set(predictions["video_id"])
    reference = reference[reference["video_id"].isin(expected_ids)].copy()
    if set(reference["video_id"]) != expected_ids:
        raise ValueError("A referência não cobre exatamente os vídeos das dobras.")
    comparison = paired_comparison(
        counts,
        channel_pair_counts(reference, "score"),
        args.bootstrap,
        args.seed,
    )

    predictions.to_parquet(run_dir / f"predictions-{args.config}.parquet", index=False)
    report = {
        "schema_version": "0.1.0",
        "protocol": "viral-intra-canal-v1",
        "config": args.config,
        "coverage": {
            "rows": len(predictions),
            "channels": channels,
            "positives": int(predictions["viral"].sum()),
        },
        "metrics": metrics,
        "paired_vs_reference": comparison,
        "reference_predictions": str(args.reference_predictions),
    }
    (run_dir / f"report-{args.config}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / f"report-{args.config}.md").write_text(
        "# Ablação de embeddings — resultado por configuração\n\n"
        f"Configuração: `{args.config}`.\n\n"
        "| AUC intra-canal [IC95%] | AUC top10 vs bottom50 | recall top10 | lift | rho idade | Δ pareado vs referência |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"| {metrics['auc_intra']:.3f} [{metrics['auc_intra_ci_low']:.3f}; {metrics['auc_intra_ci_high']:.3f}] "
        f"| {metrics['auc_top10_vs_bottom50']:.3f} | {metrics['recall_top10']:.3f} "
        f"| {metrics['lift_top10']:.2f} | {metrics['age_rho']:+.3f} "
        f"| {comparison['diff']:+.3f} [{comparison['ci_low']:+.3f}; {comparison['ci_high']:+.3f}] |\n",
        encoding="utf-8",
    )
    print(run_dir / f"report-{args.config}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
