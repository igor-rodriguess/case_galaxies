"""Avalia blends fixos das previsões OOF do backtest temporal rolante.

Os pesos são receitas explícitas e o relatório é exploratório. Ele não promove
modelo, pois as mesmas janelas já foram usadas durante o desenvolvimento.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.evaluation import (  # noqa: E402
    bootstrap_auc,
    bootstrap_top_fraction,
    channel_pair_counts,
    channel_top_counts,
    paired_comparison,
    summarize_scores,
)
from galaxies.agent.rolling import fold_aware_channels  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

COMPONENT_FILES = {
    "logistic": "predictions-logistic_pooled_top10.parquet",
    "hgb_pooled": "predictions-hist_gb_pooled_clean20vs50.parquet",
    "hgb_sector": "predictions-hist_gb_per_sector_clean20vs50.parquet",
    "catboost_ranker": "predictions-catboost_ranker_per_sector_top10.parquet",
    "mlp": "predictions-mlp_pooled_clean20vs50.parquet",
}

RECIPES = {
    "hgb_sector_catboost_equal": {"hgb_sector": 0.5, "catboost_ranker": 0.5},
    "hgb_sector_75_catboost_25": {"hgb_sector": 0.75, "catboost_ranker": 0.25},
    "four_family_equal": {
        "hgb_sector": 0.25,
        "catboost_ranker": 0.25,
        "mlp": 0.25,
        "logistic": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia ensemble no backtest rolante.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(source_dir: Path) -> pd.DataFrame:
    keys = ["video_id", "fold", "channel_id", "sector", "viral", "target_age_days"]
    merged: pd.DataFrame | None = None
    for name, filename in COMPONENT_FILES.items():
        frame = pd.read_parquet(source_dir / filename)
        selected = [*keys, "score"] if merged is None else ["video_id", "fold", "score"]
        frame = frame[selected].rename(columns={"score": name})
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on=["video_id", "fold"], validate="one_to_one")
        )
    if merged is None:
        raise RuntimeError("Nenhuma previsão foi carregada.")
    for name in COMPONENT_FILES:
        merged[f"{name}_rank"] = merged.groupby(["fold", "channel_id"])[name].rank(
            pct=True, method="average"
        )
    return merged


def evaluate(frame: pd.DataFrame, column: str, bootstrap: int, seed: int) -> dict:
    aggregate = fold_aware_channels(frame)
    metrics = summarize_scores(aggregate, column)
    pair_counts = channel_pair_counts(aggregate, column)
    metrics["auc_intra_ci_low"], metrics["auc_intra_ci_high"] = bootstrap_auc(
        pair_counts, bootstrap, seed
    )
    metrics.update(
        bootstrap_top_fraction(channel_top_counts(aggregate, column), bootstrap, seed)
    )
    per_sector = {}
    for sector, group in frame.groupby("sector", sort=True):
        per_sector[str(sector)] = summarize_scores(fold_aware_channels(group), column)
    return {"metrics": metrics, "per_sector": per_sector, "pair_counts": pair_counts}


def main() -> int:
    args = parse_args()
    if args.bootstrap < 50:
        raise SystemExit("ERRO: --bootstrap deve ser pelo menos 50.")
    source_dir = MODELING_RUNS_DIR / args.source_run
    output_dir = MODELING_RUNS_DIR / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = load_predictions(source_dir)

    reference = evaluate(frame, "logistic", args.bootstrap, args.seed)
    hgb = evaluate(frame, "hgb_sector", args.bootstrap, args.seed)
    results = []
    for name, weights in RECIPES.items():
        frame[name] = sum(weight * frame[f"{component}_rank"] for component, weight in weights.items())
        evaluation = evaluate(frame, name, args.bootstrap, args.seed)
        results.append(
            {
                "name": name,
                "weights": weights,
                "metrics": evaluation["metrics"],
                "per_sector": evaluation["per_sector"],
                "paired_vs_logistic": paired_comparison(
                    evaluation["pair_counts"], reference["pair_counts"], args.bootstrap, args.seed
                ),
                "paired_vs_hgb_sector": paired_comparison(
                    evaluation["pair_counts"], hgb["pair_counts"], args.bootstrap, args.seed
                ),
            }
        )
    output_predictions = frame[
        ["video_id", "fold", "channel_id", "sector", "viral", *RECIPES]
    ]
    output_predictions.to_parquet(output_dir / "ensemble_predictions.parquet", index=False)
    report = {
        "schema_version": "0.1.0",
        "source_run": args.source_run,
        "protocol": (
            "Blends de ranks OOF normalizados dentro de fold x canal. Pesos explícitos; "
            "resultado exploratório e inelegível para promoção sem nova janela prospectiva."
        ),
        "reference": {"logistic": reference["metrics"], "hgb_sector": hgb["metrics"]},
        "results": results,
    }
    serializable = json.loads(json.dumps(report, default=lambda value: value.to_dict()))
    (output_dir / "report.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Ensemble no backtest temporal rolante",
        "",
        "Os scores são ranks OOF dentro de cada janela e canal. Os resultados são exploratórios.",
        "",
        "| configuração | AUC [IC95%] | Precisão@10 | Lift@10 | Δ vs logística [IC95%] | Δ vs HGB [IC95%] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metric = item["metrics"]
        logistic = item["paired_vs_logistic"]
        hgb_delta = item["paired_vs_hgb_sector"]
        lines.append(
            f"| {item['name']} | {metric['auc_intra']:.3f} "
            f"[{metric['auc_intra_ci_low']:.3f}; {metric['auc_intra_ci_high']:.3f}] | "
            f"{metric['precision_top10']:.3f} | {metric['lift_top10']:.2f} | "
            f"{logistic['diff']:+.3f} [{logistic['ci_low']:+.3f}; {logistic['ci_high']:+.3f}] | "
            f"{hgb_delta['diff']:+.3f} [{hgb_delta['ci_low']:+.3f}; {hgb_delta['ci_high']:+.3f}] |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
