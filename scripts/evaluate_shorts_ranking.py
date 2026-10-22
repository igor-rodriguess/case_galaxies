"""Fase 2: ranking só de Shorts, com seleção aninhada e corte externo único.

1. População: vídeos maduros fora do cofre com duração até ``--max-seconds``.
   Vídeos longos não entram em treino, rótulo nem métrica.
2. Corte externo de calendário por setor. O treino termina antes do corte.
3. Seleção por setor só em dobras de calendário dentro do treino, com rótulo e
   limiar recalculados em cada dobra. A configuração escolhida é congelada em
   ``frozen_selection.json`` antes de o teste externo ser pontuado.
4. O teste externo é pontuado uma vez: a escolha de cada setor e a referência
   fixada a priori (logística pooled top-10). Nenhum outro candidato é lido lá.

Este corte externo já foi consultado em fases anteriores (com outro rótulo), e
por isso é desenvolvimento. A validação final é o período prospectivo.

Uso:
    python scripts/evaluate_shorts_ranking.py --run-id shorts-ranking-phase2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.evaluation import fit_predict  # noqa: E402
from galaxies.agent.format_ranking import (  # noqa: E402
    bootstrap_channels,
    channel_level,
    nested_selection,
    outer_split,
    ranking_metrics,
    shorts_population,
    spec_name,
    unit_counts,
    with_format_units,
)
from galaxies.agent.protocol import (  # noqa: E402
    latest_feature_map,
    load_dataset,
    within_channel_age_correlation,
)
from galaxies.agent.spec import ExperimentSpec  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

REFERENCE = "logistic_pooled_top10"


def candidates() -> tuple[ExperimentSpec, ...]:
    """Lista fixada antes da execução; as mais simples primeiro (desempate)."""
    common = {
        "track": "full",
        "feature_groups": ["metadata", "cadence"],
        "format_strategy": "joint",
        "train_min_age_days": 30,
        "evaluation_mode": "known_channel",
        "complexity": "low",
    }
    return (
        ExperimentSpec(
            hypothesis="Referência linear.",
            model="logistic",
            train_label="top10",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Contraste limpo linear.",
            model="logistic",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Árvores rasas pooled.",
            model="hist_gb",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Árvores rasas por setor.",
            model="hist_gb",
            train_label="clean20vs50",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="Ranking pareado por setor.",
            model="catboost_ranker",
            train_label="top10",
            pooling="per_sector",
            **common,
        ),
        ExperimentSpec(
            hypothesis="MLP pequena regularizada.",
            model="mlp",
            train_label="clean20vs50",
            pooling="pooled",
            **common,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def score(spec: ExperimentSpec, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    scored = test[
        ["video_id", "channel_id", "source_channel_id", "sector", "viral", "target_age_days"]
    ].copy()
    scored["score"] = fit_predict(spec, train, test, seed, known_channel=True)
    return scored


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    output = MODELING_RUNS_DIR / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    dataset = load_dataset(feature_map)
    rows = with_format_units(shorts_population(dataset.evaluable("full"), args.max_seconds))
    train, test, cutoffs = outer_split(rows)
    specs = {spec_name(spec): spec for spec in candidates()}

    print("Seleção interna...", flush=True)
    chosen, inner = nested_selection(list(specs.values()), train, args.seed)
    inner.to_csv(output / "inner_selection.csv", index=False)
    frozen = {
        "chosen_by_sector": chosen,
        "specs": {name: specs[name].model_dump(mode="json") for name in set(chosen.values())},
        "reference": REFERENCE,
        "max_seconds": args.max_seconds,
        "seed": args.seed,
        "feature_map": str(feature_map),
        "sector_cutoffs": cutoffs,
    }
    frozen_text = json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True)
    (output / "frozen_selection.json").write_text(frozen_text + "\n", encoding="utf-8")
    frozen_hash = hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()

    print("Corte externo (uma leitura)...", flush=True)
    sector_rows = []
    chosen_counts, reference_counts, predictions = [], [], []
    for sector, name in sorted(chosen.items()):
        sector_test = test.loc[test["sector"] == sector]
        candidate = score(specs[name], train, sector_test, args.seed)
        reference = score(specs[REFERENCE], train, sector_test, args.seed)
        candidate_units, reference_units = (
            unit_counts(candidate, "score"),
            unit_counts(reference, "score"),
        )
        chosen_counts.append(candidate_units)
        reference_counts.append(reference_units)
        predictions.append(candidate.assign(model=name))
        candidate_channels = channel_level(candidate_units)
        reference_channels = channel_level(reference_units)
        metrics = ranking_metrics(candidate_channels)
        base = ranking_metrics(reference_channels)
        intervals = bootstrap_channels(
            candidate_channels,
            None if name == REFERENCE else reference_channels,
            args.bootstrap,
            args.seed,
        )
        sector_rows.append(
            {
                "sector": sector,
                "chosen": name,
                "channels": len(candidate_channels),
                **metrics,
                "reference_auc": base["auc_macro_channel"],
                "reference_lift_top10": base["lift_top10"],
                "delta_auc_macro": metrics["auc_macro_channel"] - base["auc_macro_channel"],
                "age_rho": within_channel_age_correlation(candidate, "score"),
                "ci": intervals,
            }
        )
    all_chosen = channel_level(pd.concat(chosen_counts))
    all_reference = channel_level(pd.concat(reference_counts))
    overall = {
        **ranking_metrics(all_chosen),
        "reference": ranking_metrics(all_reference),
        "ci": bootstrap_channels(all_chosen, all_reference, args.bootstrap, args.seed),
    }
    pd.concat(predictions).to_parquet(output / "predictions.parquet", index=False)
    report = {
        "schema_version": "0.1.0",
        "frozen_selection_sha256": frozen_hash,
        "population": {
            "max_seconds": args.max_seconds,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_positives": int(test["viral"].sum()),
            "test_channels": int(test["source_channel_id"].nunique()),
        },
        "sectors": sector_rows,
        "overall": overall,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Ranking de Shorts (até {args.max_seconds:g} s) — Fase 2",
        "",
        f"Treino {len(train)} vídeos; teste {len(test)} vídeos, "
        f"{int(test['viral'].sum())} positivos, {test['source_channel_id'].nunique()} canais. "
        f"Seleção congelada antes do teste (sha256 `{frozen_hash[:12]}`).",
        "",
        "| Setor | Escolhido nas dobras internas | Canais | AUC média por canal (IC95%) | Lift@10 (IC95%) | Ref. AUC | Δ AUC (IC95%) | rho idade |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sector_rows:
        ci = row["ci"]
        if "auc_macro_channel" not in ci:
            lines.append(
                f"| {row['sector']} | {row['chosen']} | {row['channels']} | sem pares no teste | — | — | — | — |"
            )
            continue
        delta_ci = ci.get("delta_auc_macro")
        delta = fmt(row["delta_auc_macro"]) + (
            f" [{fmt(delta_ci[0])}; {fmt(delta_ci[1])}]" if delta_ci else " (é a referência)"
        )
        lines.append(
            f"| {row['sector']} | {row['chosen']} | {row['channels']} | "
            f"{fmt(row['auc_macro_channel'])} [{fmt(ci['auc_macro_channel'][0])}; {fmt(ci['auc_macro_channel'][1])}] | "
            f"{fmt(row['lift_top10'], 2)} [{fmt(ci['lift_top10'][0], 2)}; {fmt(ci['lift_top10'][1], 2)}] | "
            f"{fmt(row['reference_auc'])} | {delta} | {row['age_rho']:+.3f} |"
        )
    ci = overall["ci"]
    if "delta_auc_macro" not in ci or "auc_macro_channel" not in ci:
        ci = {
            "auc_macro_channel": [float("nan")] * 2,
            "lift_top10": [float("nan")] * 2,
            "delta_auc_macro": [float("nan")] * 2,
            **ci,
        }
    lines += [
        "",
        f"Três setores juntos: AUC {fmt(overall['auc_macro_channel'])} "
        f"[{fmt(ci['auc_macro_channel'][0])}; {fmt(ci['auc_macro_channel'][1])}], "
        f"Lift@10 {fmt(overall['lift_top10'], 2)} [{fmt(ci['lift_top10'][0], 2)}; {fmt(ci['lift_top10'][1], 2)}], "
        f"referência AUC {fmt(overall['reference']['auc_macro_channel'])}, "
        f"Δ AUC [{fmt(ci['delta_auc_macro'][0])}; {fmt(ci['delta_auc_macro'][1])}].",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
