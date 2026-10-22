"""Benchmark VLM separado por protocolo, com LOCO, temporal e cofre isolado."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    channel_pair_counts,
    evaluate_on_vault,
    paired_comparison,
    run_experiment,
    summarize_scores,
)
from galaxies.agent.protocol import AGE_SHORTCUT_LIMIT, latest_feature_map, load_dataset  # noqa: E402
from galaxies.agent.protocol_search import (  # noqa: E402
    ENSEMBLE_COMPONENTS,
    PRIMARY_VLM_COHORTS,
    isolate_protocol,
    preregistered_protocol_specs,
)
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

PROMOTION_P_BETTER = 0.90
TEMPORAL_TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara modelos sem misturar protocolos VLM.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cohort",
        choices=[cohort.name for cohort in PRIMARY_VLM_COHORTS],
        action="append",
        help="Limita a uma coorte; pode repetir. Padrão: todas as primárias.",
    )
    parser.add_argument("--skip-vault", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        choices=[name for name, _ in preregistered_protocol_specs()],
        help="Executa só candidatos selecionados; inclua reference_metadata_logistic.",
    )
    return parser.parse_args()


def clean_number(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: clean_number(value) for key, value in metrics.items()}


def ensemble_result(results: dict[str, object], bootstrap: int, seed: int) -> dict[str, object]:
    loco_parts = []
    temporal_parts = []
    for name in ENSEMBLE_COMPONENTS:
        result = results[name]
        loco_parts.append(
            result.predictions.set_index("video_id")["score_loco"].rename(name)
        )
        temporal_parts.append(
            result.temporal_predictions.set_index("video_id")["score_temporal"].rename(name)
        )
    template = results[ENSEMBLE_COMPONENTS[0]].predictions.set_index("video_id").copy()
    template["score_loco"] = pd.concat(loco_parts, axis=1).mean(axis=1)
    temporal = (
        results[ENSEMBLE_COMPONENTS[0]].temporal_predictions.set_index("video_id").copy()
    )
    temporal["score_temporal"] = pd.concat(temporal_parts, axis=1).mean(axis=1)
    loco_metrics = summarize_scores(template, "score_loco")
    temporal_metrics = summarize_scores(temporal, "score_temporal")
    loco_counts = channel_pair_counts(template, "score_loco")
    temporal_counts = channel_pair_counts(temporal, "score_temporal")
    loco_metrics["auc_intra_ci_low"], loco_metrics["auc_intra_ci_high"] = bootstrap_auc(
        loco_counts, bootstrap, seed
    )
    temporal_metrics["auc_intra_ci_low"], temporal_metrics["auc_intra_ci_high"] = bootstrap_auc(
        temporal_counts, bootstrap, seed
    )
    return {
        "loco": loco_metrics,
        "temporal": temporal_metrics,
        "predictions": template.reset_index(),
        "temporal_predictions": temporal.reset_index(),
        "loco_counts": loco_counts,
        "temporal_counts": temporal_counts,
    }


def eligible(item: dict, reference: dict) -> bool:
    paired = item.get("paired") or {}
    rho = item["loco"].get("age_rho")
    temporal = item["temporal"].get("auc_intra")
    reference_temporal = reference["temporal"].get("auc_intra")
    return bool(
        rho is not None
        and abs(rho) <= AGE_SHORTCUT_LIMIT
        and temporal is not None
        and reference_temporal is not None
        and temporal >= reference_temporal - TEMPORAL_TOLERANCE
        and paired.get("p_better", 0) >= PROMOTION_P_BETTER
        and paired.get("ci_low", -1) > 0
    )


def plot_results(items: list[dict], output: Path, title: str) -> None:
    ordered = sorted(items, key=lambda item: item["loco"]["auc_intra"] or -1)
    labels = [item["name"] for item in ordered]
    values = [item["loco"]["auc_intra"] for item in ordered]
    lows = [
        value - item["loco"]["auc_intra_ci_low"]
        for value, item in zip(values, ordered, strict=True)
    ]
    highs = [
        item["loco"]["auc_intra_ci_high"] - value
        for value, item in zip(values, ordered, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(10, max(5, len(items) * 0.42)))
    colors = ["#167D6A" if item.get("eligible") else "#8795A1" for item in ordered]
    axis.barh(labels, values, xerr=np.asarray([lows, highs]), color=colors, alpha=0.9)
    axis.axvline(0.5, color="#B83A3A", linestyle="--", linewidth=1, label="aleatório")
    axis.set_xlim(0.25, 0.90)
    axis.set_xlabel("AUC intra-canal LOCO (IC95% por canal)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_report(report: dict) -> str:
    lines = [
        "# Busca de modelos separada por protocolo VLM",
        "",
        "Os protocolos não foram misturados. A seleção usa apenas LOCO e temporal fora do cofre; "
        "o cofre é aberto uma vez somente para o candidato congelado.",
        "",
    ]
    for cohort in report["cohorts"]:
        coverage = cohort["coverage"]
        lines += [
            f"## {cohort['name']}",
            "",
            cohort["description"],
            "",
            f"Cobertura madura: {coverage['rows']} vídeos, {coverage['positives']} positivos, "
            f"{coverage['channels']} canais; cofre: {coverage['vault_rows']} vídeos.",
            "",
            "| modelo | AUC LOCO [IC95%] | AUC temporal | Lift@10 | rho idade | Δ vs referência [IC95%] | decisão |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in cohort["results"]:
            loco = item["loco"]
            paired = item.get("paired")
            delta = "—"
            if paired:
                delta = (
                    f"{paired['diff']:+.3f} [{paired['ci_low']:+.3f}; "
                    f"{paired['ci_high']:+.3f}]"
                )
            lines.append(
                f"| {item['name']} | {loco['auc_intra']:.3f} "
                f"[{loco['auc_intra_ci_low']:.3f}; {loco['auc_intra_ci_high']:.3f}] | "
                f"{item['temporal']['auc_intra']:.3f} | {loco['lift_top10']:.2f} | "
                f"{loco['age_rho']:+.3f} | {delta} | "
                f"{'elegível' if item.get('eligible') else 'não promovido'} |"
            )
        lines += [
            "",
            f"Candidato congelado: **{cohort['selected']}**.",
            "",
            f"Cofre: `{json.dumps(cohort.get('vault'), ensure_ascii=False)}`",
            "",
        ]
    lines += [
        "## Regra de promoção",
        "",
        "O ganho precisa ter IC95% pareado inteiramente acima de zero, probabilidade bootstrap "
        "de melhora >= 0,90, |rho com idade| <= 0,20 e AUC temporal no máximo 0,02 abaixo da referência.",
        "",
        "AUCs de artigos em bases, alvos ou splits diferentes não são tratadas como comparação direta.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 100.")
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    dataset = load_dataset(feature_map)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = args.run_id or f"vlm-protocol-search-{timestamp}"
    output_dir = PROJECT_ROOT / "data/modeling/runs" / run_id
    figure_dir = PROJECT_ROOT / "reports/figures" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)

    requested = set(args.cohort or [cohort.name for cohort in PRIMARY_VLM_COHORTS])
    requested_specs = set(args.only or [name for name, _ in preregistered_protocol_specs()])
    if "reference_metadata_logistic" not in requested_specs:
        raise SystemExit("ERRO: --only precisa incluir reference_metadata_logistic.")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_map": str(feature_map),
        "dataset_fingerprint": dataset.fingerprint,
        "selection_protocol": {
            "primary": "AUC intra-canal LOCO",
            "paired_bootstrap_unit": "channel",
            "temporal_tolerance": TEMPORAL_TOLERANCE,
            "minimum_p_better": PROMOTION_P_BETTER,
            "age_shortcut_limit": AGE_SHORTCUT_LIMIT,
            "vault_used_for_selection": False,
        },
        "cohorts": [],
    }

    for cohort in PRIMARY_VLM_COHORTS:
        if cohort.name not in requested:
            continue
        print(f"\n=== {cohort.name} ===", flush=True)
        cohort_dataset = isolate_protocol(dataset, cohort)
        evaluation_rows = cohort_dataset.evaluable("vlm")
        vault_rows = cohort_dataset.vault("vlm")
        raw_results: dict[str, object] = {}
        serialized: list[dict] = []
        for name, spec in preregistered_protocol_specs():
            if name not in requested_specs:
                continue
            print(f"Avaliando {name}...", flush=True)
            result = run_experiment(
                spec,
                cohort_dataset,
                seed=args.seed,
                bootstrap_iterations=args.bootstrap,
            )
            raw_results[name] = result
            result.predictions.to_csv(output_dir / f"{cohort.name}-{name}-loco.csv", index=False)
            result.temporal_predictions.to_csv(
                output_dir / f"{cohort.name}-{name}-temporal.csv", index=False
            )
            serialized.append(
                {
                    "name": name,
                    "spec": spec.model_dump(mode="json"),
                    "loco": clean_metrics(result.loco),
                    "temporal": clean_metrics(result.temporal),
                }
            )

        ensemble = None
        if set(ENSEMBLE_COMPONENTS).issubset(raw_results):
            print("Avaliando ensemble OOF pré-definido...", flush=True)
            ensemble = ensemble_result(raw_results, args.bootstrap, args.seed)
            ensemble["predictions"].to_csv(
                output_dir / f"{cohort.name}-ensemble_vlm_oof-loco.csv", index=False
            )
            ensemble["temporal_predictions"].to_csv(
                output_dir / f"{cohort.name}-ensemble_vlm_oof-temporal.csv", index=False
            )
            serialized.append(
                {
                    "name": "ensemble_vlm_oof",
                    "components": list(ENSEMBLE_COMPONENTS),
                    "loco": clean_metrics(ensemble["loco"]),
                    "temporal": clean_metrics(ensemble["temporal"]),
                }
            )

        reference_result = raw_results["reference_metadata_logistic"]
        reference_item = next(
            item for item in serialized if item["name"] == "reference_metadata_logistic"
        )
        for item in serialized:
            if item["name"] == "reference_metadata_logistic":
                item["eligible"] = True
                continue
            counts = (
                ensemble["loco_counts"]
                if item["name"] == "ensemble_vlm_oof"
                else raw_results[item["name"]].loco_counts
            )
            item["paired"] = clean_metrics(
                paired_comparison(
                    counts, reference_result.loco_counts, args.bootstrap, args.seed
                )
            )
            item["eligible"] = eligible(item, reference_item)

        eligible_items = [item for item in serialized if item.get("eligible")]
        selected_item = max(eligible_items, key=lambda item: item["loco"]["auc_intra"])
        selected_name = selected_item["name"]
        vault = None
        if not args.skip_vault and not vault_rows.empty:
            if selected_name == "ensemble_vlm_oof":
                component_vaults = []
                # O cofre do ensemble é avaliado pelos componentes já congelados.
                for component in ENSEMBLE_COMPONENTS:
                    component_vaults.append(
                        evaluate_on_vault(
                            raw_results[component].spec,
                            cohort_dataset,
                            seed=args.seed,
                            bootstrap_iterations=args.bootstrap,
                        )
                    )
                vault = {"kind": "component_sensitivity", "components": component_vaults}
            else:
                vault = clean_metrics(
                    evaluate_on_vault(
                        raw_results[selected_name].spec,
                        cohort_dataset,
                        seed=args.seed,
                        bootstrap_iterations=args.bootstrap,
                    )
                )

        cohort_report = {
            "name": cohort.name,
            "description": cohort.description,
            "protocol": {
                "prompt_version": cohort.prompt_version,
                "model_version": cohort.model_version,
                "end_seconds": cohort.end_seconds,
            },
            "coverage": {
                "rows": len(evaluation_rows),
                "positives": int(evaluation_rows["viral"].sum()),
                "channels": int(evaluation_rows["channel_id"].nunique()),
                "vault_rows": len(vault_rows),
                "vault_positives": int(vault_rows["viral"].sum()),
            },
            "results": serialized,
            "selected": selected_name,
            "vault": vault,
        }
        report["cohorts"].append(cohort_report)
        plot_results(
            serialized,
            figure_dir / f"{cohort.name}-auc-loco.png",
            f"Galaxies — {cohort.name}",
        )

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"\nRelatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

