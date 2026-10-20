"""Valida o modelo já escolhido contra o baseline com bootstrap pareado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.dataset import write_json  # noqa: E402
from galaxies.modeling.evidence import paired_bootstrap_mae_difference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara os vencedores pré-selecionados ao baseline nos holdouts."
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--minimum-r2", type=float, default=0.05)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    summary = json.loads(
        (evaluation_dir / "summary.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(evaluation_dir / "predictions.csv")
    metrics = pd.read_csv(evaluation_dir / "model_metrics.csv")
    results: dict[str, object] = {
        "run_id": summary["run_id"],
        "selection_policy": (
            "O modelo é fixado pela validação interna; os holdouts servem somente "
            "para confirmação pós-seleção."
        ),
        "sectors": {},
    }
    markdown = [
        "# Evidência pós-seleção dos modelos",
        "",
        f"Execução avaliada: `{summary['run_id']}`.",
        "",
        "O vencedor de cada setor foi definido antes de consultar os holdouts. "
        "A tabela compara seu erro absoluto ao baseline de mediana nos mesmos vídeos, "
        "com intervalo de confiança de 95% por bootstrap pareado.",
        "",
        "| Setor | Vencedor | Corte | N | MAE vencedor | MAE baseline | Ganho | "
        "IC95% da diferença | R² | Evidência positiva |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sector, selection in summary["sectors"].items():
        winner = selection["best_inner_cv"]
        sector_result: dict[str, object] = {"winner": winner, "splits": {}}
        all_passed = True
        for split in ("temporal", "channel_holdout"):
            selected = predictions.loc[
                (predictions["sector"] == sector)
                & (predictions["split"] == split)
                & (predictions["model"].isin([winner, "dummy_median"]))
            ]
            pivot = selected.pivot(
                index="video_id", columns="model", values=["actual", "predicted"]
            ).dropna()
            actual = pivot[("actual", winner)].to_numpy(dtype=float)
            candidate = pivot[("predicted", winner)].to_numpy(dtype=float)
            baseline = pivot[("predicted", "dummy_median")].to_numpy(dtype=float)
            evidence = paired_bootstrap_mae_difference(
                actual,
                candidate,
                baseline,
                iterations=args.iterations,
            )
            metric_row = metrics.loc[
                (metrics["sector"] == sector)
                & (metrics["model"] == winner)
                & (metrics["split"] == split)
            ].iloc[0]
            r2 = float(metric_row["r2_log"])
            passed = bool(
                evidence["statistically_better"]
                and r2 >= args.minimum_r2
                and evidence["relative_improvement"]
                >= args.minimum_relative_improvement
            )
            evidence["r2"] = r2
            evidence["passed"] = passed
            sector_result["splits"][split] = evidence
            all_passed = all_passed and passed
            improvement = evidence["relative_improvement"]
            markdown.append(
                f"| {sector.title()} | {winner} | {split} | {evidence['rows']} | "
                f"{evidence['candidate_mae']:.3f} | {evidence['baseline_mae']:.3f} | "
                f"{improvement:.1%} | [{evidence['ci95_lower']:.3f}, "
                f"{evidence['ci95_upper']:.3f}] | {r2:.3f} | "
                f"{'sim' if passed else 'não'} |"
            )
        sector_result["confirmed_for_both_holdouts"] = all_passed
        results["sectors"][sector] = sector_result
    markdown.extend(
        [
            "",
            "## Regra de confirmação",
            "",
            "Um setor só é confirmado quando o limite superior do IC95% da diferença "
            f"de MAE fica abaixo de zero, o ganho é de pelo menos "
            f"{args.minimum_relative_improvement:.0%} e o R² é de pelo menos "
            f"{args.minimum_r2:.2f} nos dois cortes. Falhar nessa "
            "regra não prova que o algoritmo é inútil; indica que a base atual ainda não "
            "sustenta a afirmação de generalização.",
            "",
        ]
    )
    output_json = evaluation_dir / "winner_evidence.json"
    output_markdown = PROJECT_ROOT / "reports" / f"{summary['run_id']}-evidence.md"
    write_json(output_json, results)
    output_markdown.write_text("\n".join(markdown), encoding="utf-8")
    print(output_json)
    print(output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
