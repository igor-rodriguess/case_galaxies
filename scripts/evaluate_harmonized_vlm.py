"""Avalia VLMs harmonizados sem permitir atalho de protocolo.

O treino deixa um canal inteiro de fora. A métrica primária forma pares apenas
dentro de canal + protocolo, impedindo que uma diferença constante entre
Gemini/3s e Pegasus/15s seja confundida com capacidade de ordenar criativos.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    channel_pair_counts,
    fit_predict,
    paired_comparison,
    run_experiment,
    summarize_scores,
)
from galaxies.agent.protocol import AGE_SHORTCUT_LIMIT, latest_feature_map, load_dataset  # noqa: E402
from galaxies.agent.protocol_search import (  # noqa: E402
    PRIMARY_VLM_COHORTS,
    harmonize_primary_protocols,
    isolate_protocol,
    preregistered_protocol_specs,
)
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

CANDIDATES = (
    "reference_metadata_logistic",
    "vlm_logistic",
    "vlm_extra_trees",
    "vlm_catboost_per_sector",
    "vlm_catboost_ranker",
    "vlm_derived_catboost",
    "vlm_derived_ranker",
    "vlm_transcript_catboost",
    "vlm_transcript_ranker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark harmonizado de protocolos VLM.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: clean(value) for key, value in metrics.items()}


def protocol_stratified_metrics(
    predictions: pd.DataFrame,
    source_rows: pd.DataFrame,
    score_column: str,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    lookup = source_rows.set_index("video_id")["vlm_protocol_cohort"]
    frame = predictions.copy()
    frame["protocol"] = frame["video_id"].map(lookup)
    frame["original_channel_id"] = frame["channel_id"]
    frame["channel_id"] = frame["channel_id"].astype(str) + "::" + frame["protocol"].astype(str)
    metrics = summarize_scores(frame, score_column)
    counts = channel_pair_counts(frame, score_column)
    metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
        counts, bootstrap, seed
    )
    metrics["original_channels"] = int(frame["original_channel_id"].nunique())
    metrics["channel_protocol_strata"] = int(frame["channel_id"].nunique())
    return metrics, counts, frame


def cross_protocol_diagnostics(
    spec,
    base_dataset,
    *,
    seed: int,
    bootstrap: int,
) -> list[dict[str, object]]:
    cohorts = {
        cohort.name: isolate_protocol(base_dataset, cohort).evaluable("vlm")
        for cohort in PRIMARY_VLM_COHORTS
    }
    diagnostics = []
    directions = (
        (PRIMARY_VLM_COHORTS[0].name, PRIMARY_VLM_COHORTS[1].name),
        (PRIMARY_VLM_COHORTS[1].name, PRIMARY_VLM_COHORTS[0].name),
    )
    for train_name, test_name in directions:
        train = cohorts[train_name]
        test = cohorts[test_name]
        shared = sorted(set(train["channel_id"]) & set(test["channel_id"]))
        test_shared = test[test["channel_id"].isin(shared)].copy()
        if test_shared.empty:
            continue
        scored = test_shared[
            ["video_id", "channel_id", "sector", "viral", "target_age_days"]
        ].copy()
        scored["score"] = fit_predict(spec, train, test_shared, seed)
        metrics = summarize_scores(scored, "score")
        counts = channel_pair_counts(scored, "score")
        metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
            counts, bootstrap, seed
        )
        diagnostics.append(
            {
                "train_protocol": train_name,
                "test_protocol": test_name,
                "shared_channels_only": True,
                "metrics": clean_metrics(metrics),
            }
        )
    return diagnostics


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 100.")
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    base_dataset = load_dataset(feature_map)
    dataset = harmonize_primary_protocols(base_dataset)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = args.run_id or f"vlm-harmonized-{timestamp}"
    output_dir = PROJECT_ROOT / "data/modeling/runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    catalog = dict(preregistered_protocol_specs())
    rows = dataset.evaluable("vlm")
    items = []
    raw_results = {}
    counts_by_name = {}
    for name in CANDIDATES:
        print(f"Avaliando {name}...", flush=True)
        spec = catalog[name]
        result = run_experiment(
            spec, dataset, seed=args.seed, bootstrap_iterations=args.bootstrap
        )
        raw_results[name] = result
        loco, counts, loco_frame = protocol_stratified_metrics(
            result.predictions, rows, "score_loco", args.bootstrap, args.seed
        )
        temporal_source = rows[rows["video_id"].isin(result.temporal_predictions["video_id"])]
        temporal, _, temporal_frame = protocol_stratified_metrics(
            result.temporal_predictions,
            temporal_source,
            "score_temporal",
            args.bootstrap,
            args.seed,
        )
        loco_frame.to_csv(output_dir / f"{name}-loco.csv", index=False)
        temporal_frame.to_csv(output_dir / f"{name}-temporal.csv", index=False)
        counts_by_name[name] = counts
        items.append(
            {
                "name": name,
                "spec": spec.model_dump(mode="json"),
                "loco": clean_metrics(loco),
                "temporal": clean_metrics(temporal),
            }
        )

    reference = next(item for item in items if item["name"] == CANDIDATES[0])
    for item in items:
        if item["name"] == CANDIDATES[0]:
            item["eligible"] = True
            continue
        item["paired"] = clean_metrics(
            paired_comparison(
                counts_by_name[item["name"]],
                counts_by_name[CANDIDATES[0]],
                args.bootstrap,
                args.seed,
            )
        )
        paired = item["paired"]
        item["eligible"] = bool(
            abs(item["loco"]["age_rho"]) <= AGE_SHORTCUT_LIMIT
            and item["temporal"]["auc_intra"] >= reference["temporal"]["auc_intra"] - 0.02
            and paired["p_better"] >= 0.90
            and paired["ci_low"] > 0
        )

    selected = max(
        (item for item in items if item["eligible"]),
        key=lambda item: item["loco"]["auc_intra"],
    )
    selected_spec = raw_results[selected["name"]].spec
    cross_protocol = cross_protocol_diagnostics(
        selected_spec,
        base_dataset,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_map": str(feature_map),
        "protocol": "LOCO por canal; AUC e bootstrap estratificados por canal + protocolo",
        "coverage": {
            "rows": len(rows),
            "positives": int(rows["viral"].sum()),
            "channels": int(rows["channel_id"].nunique()),
            "protocol_counts": rows["vlm_protocol_cohort"].value_counts().to_dict(),
        },
        "results": items,
        "selected_without_vault": selected["name"],
        "cross_protocol_diagnostics": cross_protocol,
        "vault_opened": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Avaliação VLM harmonizada",
        "",
        "LOCO deixa o canal inteiro fora do treino. As métricas formam pares somente dentro de "
        "canal + protocolo; o identificador do protocolo não entra como feature.",
        "",
        "| modelo | AUC estratificada [IC95%] | AUC temporal | Lift@10 | Δ pareado [IC95%] | decisão |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in items:
        paired = item.get("paired")
        delta = "—" if paired is None else (
            f"{paired['diff']:+.3f} [{paired['ci_low']:+.3f}; {paired['ci_high']:+.3f}]"
        )
        lines.append(
            f"| {item['name']} | {item['loco']['auc_intra']:.3f} "
            f"[{item['loco']['auc_intra_ci_low']:.3f}; {item['loco']['auc_intra_ci_high']:.3f}] | "
            f"{item['temporal']['auc_intra']:.3f} | {item['loco']['lift_top10']:.2f} | "
            f"{delta} | {'elegível' if item['eligible'] else 'não promovido'} |"
        )
    lines += [
        "",
        f"Selecionado sem abrir o cofre: **{selected['name']}**.",
        "",
        "## Transferência entre protocolos",
        "",
        "Esta é uma checagem de mudança de domínio nos canais presentes em ambos os protocolos.",
        "",
    ]
    for diagnostic in cross_protocol:
        metrics = diagnostic["metrics"]
        lines.append(
            f"- {diagnostic['train_protocol']} → {diagnostic['test_protocol']}: "
            f"AUC {metrics['auc_intra']:.3f} "
            f"[{metrics['auc_intra_ci_low']:.3f}; {metrics['auc_intra_ci_high']:.3f}]."
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Relatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

