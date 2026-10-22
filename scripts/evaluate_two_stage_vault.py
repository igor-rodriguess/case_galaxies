"""Avaliação final, sem retuning, da política two-stage no cofre de canais."""

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
)
from galaxies.agent.protocol import latest_feature_map, load_dataset  # noqa: E402
from galaxies.agent.protocol_search import harmonize_primary_protocols  # noqa: E402
from galaxies.agent.two_stage import (  # noqa: E402
    base_spec,
    cross_fitted_base_scores,
    fit_fusion_candidate,
    fusion_candidates,
)
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

FROZEN_CANDIDATE = "fusion_transcript_mlp_medium"
FROZEN_MLP_SECTORS = frozenset({"moda", "beleza"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Abre o cofre uma vez para a política two-stage.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def metrics(frame: pd.DataFrame, score_column: str, bootstrap: int, seed: int) -> tuple[dict, pd.DataFrame]:
    scored = frame.copy()
    scored["channel_id"] = (
        scored["channel_id"].astype(str)
        + "::"
        + scored["vlm_protocol_cohort"].astype(str)
    )
    summary = summarize_scores(scored, score_column)
    counts = channel_pair_counts(scored, score_column)
    summary["auc_intra_ci_low"], summary["auc_intra_ci_high"] = bootstrap_auc(
        counts, bootstrap, seed
    )
    return {key: clean(value) for key, value in summary.items()}, counts


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    base_dataset = load_dataset(feature_map)
    vlm_dataset = harmonize_primary_protocols(base_dataset)
    full_train = base_dataset.evaluable("full")
    fusion_train = vlm_dataset.evaluable("vlm")
    vault = vlm_dataset.vault("vlm")
    if vault.empty:
        raise SystemExit("ERRO: não há VLMs no cofre.")

    candidate = next(
        item for item in fusion_candidates() if item.name == FROZEN_CANDIDATE
    )
    stage_one = base_spec("pooled")
    train_base = cross_fitted_base_scores(
        full_train, fusion_train, spec=stage_one, seed=args.seed
    )
    vault_base = pd.Series(
        fit_predict(stage_one, full_train, vault, args.seed), index=vault.index
    )
    vault_mlp = pd.Series(
        fit_fusion_candidate(
            candidate,
            fusion_train,
            train_base,
            vault,
            vault_base,
            seed=args.seed,
        ),
        index=vault.index,
    )
    scored = vault[
        [
            "video_id",
            "channel_id",
            "sector",
            "viral",
            "target_age_days",
            "vlm_protocol_cohort",
        ]
    ].copy()
    scored["base_metadata_full"] = vault_base
    scored["fusion_transcript_mlp_medium"] = vault_mlp
    scored["sector_routed_policy"] = np.where(
        scored["sector"].isin(FROZEN_MLP_SECTORS), vault_mlp, vault_base
    )

    results = []
    counts = {}
    for score_column in (
        "base_metadata_full",
        "fusion_transcript_mlp_medium",
        "sector_routed_policy",
    ):
        summary, pair_counts = metrics(scored, score_column, args.bootstrap, args.seed)
        results.append({"name": score_column, "metrics": summary})
        counts[score_column] = pair_counts
    for item in results[1:]:
        item["paired_vs_base"] = {
            key: clean(value)
            for key, value in paired_comparison(
                counts[item["name"]], counts["base_metadata_full"], args.bootstrap, args.seed
            ).items()
        }

    by_sector = {}
    for sector, sector_frame in scored.groupby("sector", sort=True):
        by_sector[sector] = {}
        for score_column in (
            "base_metadata_full",
            "fusion_transcript_mlp_medium",
            "sector_routed_policy",
        ):
            summary, _ = metrics(sector_frame, score_column, args.bootstrap, args.seed)
            by_sector[sector][score_column] = summary

    run_id = args.run_id or datetime.now(UTC).strftime("two-stage-vault-%Y-%m-%dT%H-%M-%SZ")
    output_dir = PROJECT_ROOT / "data/modeling/runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    scored.to_csv(output_dir / "vault_predictions.csv", index=False)
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_map": str(feature_map),
        "frozen_before_vault": {
            "base": "pooled metadata + cadence logistic",
            "multimodal": FROZEN_CANDIDATE,
            "route_mlp_sectors": sorted(FROZEN_MLP_SECTORS),
            "route_base_sectors": ["automotivo"],
        },
        "coverage": {
            "rows": len(scored),
            "positives": int(scored["viral"].sum()),
            "channels": int(scored["channel_id"].nunique()),
        },
        "results": results,
        "by_sector": by_sector,
        "limitation": "O cofre já foi consultado para baselines anteriores; trate este resultado como confirmação final condicionada a esse histórico.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cofre — política two-stage congelada",
        "",
        "A regra foi definida antes de abrir esta rodada: MLP multimodal em moda/beleza, "
        "base de metadados em automobilismo.",
        "",
        "| política | AUC [IC95%] | Lift@10 | Δ vs base [IC95%] |",
        "|---|---:|---:|---:|",
    ]
    for item in results:
        result = item["metrics"]
        paired = item.get("paired_vs_base")
        delta = "—" if paired is None else (
            f"{paired['diff']:+.3f} [{paired['ci_low']:+.3f}; {paired['ci_high']:+.3f}]"
        )
        lines.append(
            f"| {item['name']} | {result['auc_intra']:.3f} "
            f"[{result['auc_intra_ci_low']:.3f}; {result['auc_intra_ci_high']:.3f}] | "
            f"{result['lift_top10']:.2f} | {delta} |"
        )
    lines += ["", "## Por setor", ""]
    for sector, sector_results in by_sector.items():
        lines.append(f"### {sector}")
        for name, result in sector_results.items():
            lines.append(f"- {name}: AUC {result['auc_intra']:.3f}, Lift@10 {result['lift_top10']:.2f}.")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
