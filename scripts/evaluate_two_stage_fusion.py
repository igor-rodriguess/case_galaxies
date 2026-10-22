"""Avalia fusão cross-fitted de população completa + conteúdo VLM."""

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
    summarize_scores,
    temporal_split,
)
from galaxies.agent.protocol import build_dataset, latest_feature_map, load_dataset  # noqa: E402
from galaxies.agent.protocol_search import harmonize_primary_protocols  # noqa: E402
from galaxies.agent.two_stage import (  # noqa: E402
    base_spec,
    cross_fitted_base_scores,
    fit_fusion_candidate,
    fusion_candidates,
)
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusão em dois estágios sem vazamento.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=30, choices=[7, 14, 21, 30, 60])
    parser.add_argument("--base-pooling", choices=["pooled", "per_sector"], default="pooled")
    parser.add_argument(
        "--candidate",
        action="append",
        choices=[candidate.name for candidate in fusion_candidates()],
        help="Limita corretores; pode repetir. A base e o blend 50/50 continuam incluídos.",
    )
    return parser.parse_args()


def clean(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def metrics_for(
    frame: pd.DataFrame, score_column: str, bootstrap: int, seed: int
) -> tuple[dict[str, object], pd.DataFrame]:
    stratified = frame.copy()
    stratified["original_channel_id"] = stratified["channel_id"]
    stratified["channel_id"] = (
        stratified["channel_id"].astype(str)
        + "::"
        + stratified["vlm_protocol_cohort"].astype(str)
    )
    metrics = summarize_scores(stratified, score_column)
    counts = channel_pair_counts(stratified, score_column)
    metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
        counts, bootstrap, seed
    )
    return {key: clean(value) for key, value in metrics.items()}, counts


def temporal_full_training(full_rows: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Remove do primeiro estágio vídeos contemporâneos/futuros do canal testado."""
    keep = pd.Series(True, index=full_rows.index)
    cutoffs = test.groupby("channel_id")["published_at"].min()
    for channel, cutoff in cutoffs.items():
        keep &= ~(
            full_rows["channel_id"].eq(channel) & full_rows["published_at"].ge(cutoff)
        )
    return full_rows.loc[keep]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 100.")
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    primary_dataset = load_dataset(feature_map)
    if args.min_age == 30:
        base_dataset = primary_dataset
    else:
        raw = pd.read_json(feature_map, lines=True)
        base_dataset = build_dataset(
            raw,
            feature_map,
            f"{primary_dataset.fingerprint}-age{args.min_age}",
            min_eval_age_days=args.min_age,
            vault_channels=primary_dataset.vault_channels,
        )
    vlm_dataset = harmonize_primary_protocols(base_dataset)
    full_rows = base_dataset.evaluable("full")
    vlm_rows = vlm_dataset.evaluable("vlm")
    requested = set(args.candidate or [candidate.name for candidate in fusion_candidates()])
    candidates = tuple(
        candidate for candidate in fusion_candidates() if candidate.name in requested
    )
    stage_one_spec = base_spec(args.base_pooling)
    scores = {"base_metadata_full": pd.Series(np.nan, index=vlm_rows.index)}
    scores.update({candidate.name: pd.Series(np.nan, index=vlm_rows.index) for candidate in candidates})

    for channel in sorted(vlm_rows["channel_id"].unique()):
        print(f"LOCO canal {channel}...", flush=True)
        test = vlm_rows[vlm_rows["channel_id"] == channel]
        fusion_train = vlm_rows[vlm_rows["channel_id"] != channel]
        full_train = full_rows[full_rows["channel_id"] != channel]
        base_train = cross_fitted_base_scores(
            full_train, fusion_train, spec=stage_one_spec, seed=args.seed
        )
        base_test_values = fit_predict(stage_one_spec, full_train, test, args.seed)
        base_test = pd.Series(base_test_values, index=test.index)
        scores["base_metadata_full"].loc[test.index] = base_test
        for candidate in candidates:
            scores[candidate.name].loc[test.index] = fit_fusion_candidate(
                candidate,
                fusion_train,
                base_train,
                test,
                base_test,
                seed=args.seed,
            )

    temporal_train, temporal_test = temporal_split(vlm_rows)
    full_temporal_train = temporal_full_training(full_rows, temporal_test)
    temporal_base_train = cross_fitted_base_scores(
        full_temporal_train, temporal_train, spec=stage_one_spec, seed=args.seed
    )
    temporal_base_test = pd.Series(
        fit_predict(stage_one_spec, full_temporal_train, temporal_test, args.seed),
        index=temporal_test.index,
    )
    temporal_scores = {"base_metadata_full": temporal_base_test}
    for candidate in candidates:
        print(f"Temporal {candidate.name}...", flush=True)
        temporal_scores[candidate.name] = pd.Series(
            fit_fusion_candidate(
                candidate,
                temporal_train,
                temporal_base_train,
                temporal_test,
                temporal_base_test,
                seed=args.seed,
            ),
            index=temporal_test.index,
        )

    mlp_name = "fusion_transcript_mlp"
    if mlp_name in scores:
        scores[f"{mlp_name}_blend50"] = 0.5 * scores["base_metadata_full"] + 0.5 * scores[
            mlp_name
        ]
        temporal_scores[f"{mlp_name}_blend50"] = (
            0.5 * temporal_scores["base_metadata_full"] + 0.5 * temporal_scores[mlp_name]
        )

    run_id = args.run_id or datetime.now(UTC).strftime("two-stage-fusion-%Y-%m-%dT%H-%M-%SZ")
    output_dir = PROJECT_ROOT / "data/modeling/runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    items = []
    counts_by_name = {}
    for name, score in scores.items():
        prediction = vlm_rows[
            [
                "video_id",
                "channel_id",
                "sector",
                "viral",
                "target_age_days",
                "vlm_protocol_cohort",
            ]
        ].copy()
        prediction["score"] = score
        temporal_prediction = temporal_test[
            [
                "video_id",
                "channel_id",
                "sector",
                "viral",
                "target_age_days",
                "vlm_protocol_cohort",
            ]
        ].copy()
        temporal_prediction["score"] = temporal_scores[name]
        loco_metrics, counts = metrics_for(prediction, "score", args.bootstrap, args.seed)
        temporal_metrics, _ = metrics_for(
            temporal_prediction, "score", args.bootstrap, args.seed
        )
        prediction.to_csv(output_dir / f"{name}-loco.csv", index=False)
        temporal_prediction.to_csv(output_dir / f"{name}-temporal.csv", index=False)
        counts_by_name[name] = counts
        items.append({"name": name, "loco": loco_metrics, "temporal": temporal_metrics})

    reference = items[0]
    for item in items:
        if item["name"] == "base_metadata_full":
            item["eligible"] = True
            continue
        item["paired"] = {
            key: clean(value)
            for key, value in paired_comparison(
                counts_by_name[item["name"]],
                counts_by_name["base_metadata_full"],
                args.bootstrap,
                args.seed,
            ).items()
        }
        paired = item["paired"]
        item["eligible"] = bool(
            abs(item["loco"]["age_rho"]) <= 0.20
            and item["temporal"]["auc_intra"] >= reference["temporal"]["auc_intra"] - 0.02
            and paired["p_better"] >= 0.90
            and paired["ci_low"] > 0
        )
    selected = max(
        (item for item in items if item["eligible"]),
        key=lambda item: item["loco"]["auc_intra"],
    )
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_map": str(feature_map),
        "protocol": "Nested group cross-fitting; outer LOCO; metric within channel+protocol",
        "stage_one_pooling": args.base_pooling,
        "minimum_age_days": args.min_age,
        "coverage": {
            "full_stage_rows": len(full_rows),
            "fusion_stage_rows": len(vlm_rows),
            "fusion_channels": int(vlm_rows["channel_id"].nunique()),
        },
        "results": items,
        "selected_without_vault": selected["name"],
        "vault_opened": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fusão em dois estágios",
        "",
        "O primeiro estágio usa toda a população madura; sua nota chega ao segundo estágio "
        "somente por cross-fitting de canais. O cofre permanece fechado.",
        "",
        "| modelo | AUC LOCO [IC95%] | AUC temporal | Lift@10 | Δ [IC95%] | decisão |",
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
    lines += ["", f"Selecionado sem abrir o cofre: **{selected['name']}**.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

