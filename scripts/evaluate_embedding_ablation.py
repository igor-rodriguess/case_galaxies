"""Roda a ablação pré-registrada de thumbnail e título.

Não abre o cofre, não busca hiperparâmetros e nunca lê métricas pós-publicação
como entrada do modelo. A saída compara todas as variantes nos mesmos canais e
na mesma população coberta por ambos os embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.embedding_evaluation import (  # noqa: E402
    PRE_REGISTERED_CONFIGS,
    build_embedding_frame,
    evaluate_configuration,
    latest_normalized_records,
    paired_embedding_diagnostic,
)
from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    channel_pair_counts,
    paired_comparison,
    summarize_scores,
)
from galaxies.agent.protocol import latest_feature_map  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia embeddings em LOCO por canal.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--embeddings-dir", type=Path, default=PROJECT_ROOT / "data/features/embeddings")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=[config.name for config in PRE_REGISTERED_CONFIGS],
        help=(
            "Avalia somente uma configuração pré-registrada. Útil para retomar "
            "execuções longas; não executa a comparação pareada nem o diagnóstico."
        ),
    )
    return parser.parse_args()


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| configuração | AUC intra-canal [IC95%] | AUC top10 vs bottom50 | recall top10 | lift | rho idade | Δ pareado |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metrics = item["metrics"]
        paired = item.get("paired", {})
        auc = f"{metrics['auc_intra']:.3f} [{metrics['auc_intra_ci_low']:.3f}; {metrics['auc_intra_ci_high']:.3f}]"
        delta = "—" if not paired else f"{paired['diff']:+.3f} [{paired['ci_low']:+.3f}; {paired['ci_high']:+.3f}]"
        lines.append(
            "| {name} | {auc} | {extreme:.3f} | {recall:.3f} | {lift:.2f} | {rho:+.3f} | {delta} |".format(
                name=item["name"],
                auc=auc,
                extreme=metrics["auc_top10_vs_bottom50"],
                recall=metrics["recall_top10"],
                lift=metrics["lift_top10"],
                rho=metrics["age_rho"],
                delta=delta,
            )
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 50:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 50.")
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    records = args.records or latest_normalized_records(PROJECT_ROOT)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = args.run_id or f"embedding-ablation-{timestamp}"
    output_dir = PROJECT_ROOT / "data" / "modeling" / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    frame, vault_channels = build_embedding_frame(feature_map, args.embeddings_dir, records)
    configs = (
        tuple(config for config in PRE_REGISTERED_CONFIGS if config.name == args.only)
        if args.only
        else PRE_REGISTERED_CONFIGS
    )
    results: list[dict] = []
    counts_by_name = {}
    for config in configs:
        print(f"Avaliando {config.name}...", flush=True)
        metrics, predictions, counts = evaluate_configuration(
            frame,
            config,
            seed=args.seed,
            bootstrap_iterations=args.bootstrap,
        )
        predictions.to_parquet(output_dir / f"predictions-{config.name}.parquet", index=False)
        counts.to_parquet(output_dir / f"pairs-{config.name}.parquet")
        item = {
            "name": config.name,
            "embeddings": list(config.embeddings),
            "pca_components_requested": config.pca_components,
            "metrics": metrics,
        }
        results.append(item)
        counts_by_name[config.name] = counts

    paired_metrics = None
    if not args.only:
        reference = counts_by_name["reference"]
        for item in results[1:]:
            item["paired"] = paired_comparison(
                counts_by_name[item["name"]], reference, args.bootstrap, args.seed
            )

        print("Avaliando diagnóstico pareado thumb+título...", flush=True)
        paired_scores = paired_embedding_diagnostic(frame, seed=args.seed)
        paired_metrics = summarize_scores(paired_scores, "score")
        paired_counts = channel_pair_counts(paired_scores, "score")
        paired_metrics["auc_intra_ci_low"], paired_metrics["auc_intra_ci_high"] = bootstrap_auc(
            paired_counts, args.bootstrap, args.seed
        )
        paired_scores.to_parquet(output_dir / "predictions-paired-logistic.parquet", index=False)
        paired_counts.to_parquet(output_dir / "pairs-paired-logistic.parquet")

    report = {
        "schema_version": "0.1.0",
        "protocol": "viral-intra-canal-v1",
        "feature_map": str(feature_map),
        "normalized_records": str(records),
        "embeddings_dir": str(args.embeddings_dir),
        "coverage": {
            "rows": len(frame),
            "channels": int(frame["channel_id"].nunique()),
            "positives": int(frame["viral"].sum()),
            "vault_channels_excluded": list(vault_channels),
        },
        "evaluation": "LOCO por canal; treino por setor; PCA ajustado só no treino da dobra.",
        "results": results,
        "paired_embedding_diagnostic": paired_metrics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        "# Ablação de embeddings\n\n"
        "A população é a interseção coberta por thumbnail e título; o cofre segue fechado. "
        "PCA é ajustado em cada treino LOCO.\n\n"
        + markdown_table(results)
        + (
            "\n\n## Diagnóstico logístico pareado\n\n"
            "Thumb+título, diferenças entre pares viral/não viral no mesmo canal: "
            f"AUC {paired_metrics['auc_intra']:.3f} "
            f"[{paired_metrics['auc_intra_ci_low']:.3f}; {paired_metrics['auc_intra_ci_high']:.3f}].\n"
            if paired_metrics
            else "\n\nExecução individual; comparação pareada pendente da execução completa.\n"
        ),
        encoding="utf-8",
    )
    print(f"Relatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
