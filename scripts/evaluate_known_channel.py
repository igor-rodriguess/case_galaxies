"""Compara candidatos pré-definidos para o produto de canal conhecido.

O cenário é temporal: os últimos 25% dos vídeos de cada canal são o teste. As
features históricas de resultado só podem entrar se havia um snapshot de
resultado disponível antes do item avaliado. No mapa atual isso não existe;
assim a execução segura não usa taxas de resultado retrospectivas. Não abre o
cofre e não mede canal novo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    bootstrap_top_fraction,
    calendar_split,
    channel_pair_counts,
    channel_top_counts,
    fit_predict,
    paired_comparison,
    summarize_scores,
    temporal_split,
)
from galaxies.agent.protocol import (  # noqa: E402
    latest_feature_map,
    load_dataset,
    temporal_as_of_labels,
)
from galaxies.agent.spec import ExperimentSpec  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia ranking no cenário de canal conhecido.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("calendar", "per_channel"),
        default="calendar",
        help="calendar: uma data de corte por setor (padrão); per_channel: corte antigo, só para comparação.",
    )
    return parser.parse_args()


def candidates() -> tuple[ExperimentSpec, ...]:
    """Bateria curta, fixada antes de olhar o resultado desta execução."""
    common = {
        "track": "full",
        "feature_groups": ["metadata", "cadence"],
        "format_strategy": "joint",
        "train_min_age_days": 30,
        "evaluation_mode": "known_channel",
    }
    return (
        ExperimentSpec(
            hypothesis="Referência temporal com histórico de formato e duração.",
            model="logistic",
            complexity="low",
            train_label="top10",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Contraste limpo reduz ambiguidade perto do limiar viral.",
            model="logistic",
            complexity="low",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Árvores rasas capturam interações sem rede profunda.",
            model="hist_gb",
            complexity="low",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="MLP pequena regularizada verifica ganho não linear limitado.",
            model="mlp",
            complexity="low",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis=(
                "Modelos setoriais verificam se a relação entre criativo e "
                "desempenho muda entre automobilismo, moda e beleza."
            ),
            model="hist_gb",
            complexity="low",
            train_label="clean20vs50",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis=(
                "Ranking pareado por canal testa a formulação relativa da "
                "literatura sem usar métricas pós-publicação como feature."
            ),
            model="catboost_ranker",
            complexity="low",
            train_label="top10",
            pooling="per_sector",
            **common,
        ),
    )


def main() -> int:
    args = parse_args()
    if args.bootstrap < 50:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 50.")
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    dataset = load_dataset(feature_map)
    rows = dataset.evaluable("full")
    if args.split == "calendar":
        train, test, cutoffs = calendar_split(rows)
    else:
        train, test = temporal_split(rows)
        cutoffs = {}
    train, test = temporal_as_of_labels(train, test)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = args.run_id or f"known-channel-{timestamp}"
    output_dir = MODELING_RUNS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    results: list[dict] = []
    counts_by_name = {}
    for spec in candidates():
        name = f"{spec.model}_{spec.pooling}_{spec.train_label}"
        print(f"Avaliando {name}...", flush=True)
        predictions = test[["video_id", "channel_id", "sector", "viral", "target_age_days"]].copy()
        predictions["score"] = fit_predict(spec, train, test, args.seed, known_channel=True)
        metrics = summarize_scores(predictions, "score")
        counts = channel_pair_counts(predictions, "score")
        metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
            counts, args.bootstrap, args.seed
        )
        metrics.update(
            bootstrap_top_fraction(
                channel_top_counts(predictions, "score"), args.bootstrap, args.seed
            )
        )
        predictions.to_parquet(output_dir / f"predictions-{name}.parquet", index=False)
        counts.to_parquet(output_dir / f"pairs-{name}.parquet")
        results.append({"name": name, "spec": spec.model_dump(mode="json"), "metrics": metrics})
        counts_by_name[name] = counts

    reference_name = "logistic_pooled_top10"
    reference = counts_by_name[reference_name]
    for item in results:
        if item["name"] != reference_name:
            item["paired_vs_reference"] = paired_comparison(
                counts_by_name[item["name"]], reference, args.bootstrap, args.seed
            )

    report = {
        "schema_version": "0.1.0",
        "feature_map": str(feature_map),
        "split": args.split,
        "sector_cutoffs": cutoffs,
        "protocol": (
            f"Split {args.split}; curva de idade e limiar top-10% "
            "são ajustados apenas no treino; nenhum rótulo de resultado "
            "retrospectivo entra como histórico; cofre não foi aberto."
        ),
        "coverage": {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_positives": int(test["viral"].sum()),
            "test_channels": int(test["channel_id"].nunique()),
        },
        "reference": reference_name,
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Canal conhecido — avaliação temporal\n",
        "O cofre não foi aberto. A curva de idade e o limiar de viralidade são "
        "ajustados apenas no treino; taxas de resultado histórico foram desabilitadas "
        "porque o mapa não possui snapshots observados antes da publicação de cada alvo.\n",
        "| modelo | AUC intra-canal [IC95%] | Precisão@10 [IC95%] | Recall@10 [IC95%] | Lift@10 [IC95%] | rho idade | Δ AUC vs referência [IC95%] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metrics = item["metrics"]
        paired = item.get("paired_vs_reference")
        delta = (
            "—"
            if not paired
            else (f"{paired['diff']:+.3f} [{paired['ci_low']:+.3f}; {paired['ci_high']:+.3f}]")
        )
        lines.append(
            f"| {item['name']} | {metrics['auc_intra']:.3f} "
            f"[{metrics['auc_intra_ci_low']:.3f}; {metrics['auc_intra_ci_high']:.3f}] | "
            f"{metrics['precision_top10']:.3f} [{metrics['precision_ci_low']:.3f}; {metrics['precision_ci_high']:.3f}] | "
            f"{metrics['recall_top10']:.3f} [{metrics['recall_ci_low']:.3f}; {metrics['recall_ci_high']:.3f}] | "
            f"{metrics['lift_top10']:.2f} [{metrics['lift_ci_low']:.2f}; {metrics['lift_ci_high']:.2f}] | "
            f"{metrics['age_rho']:+.3f} | {delta} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Relatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
