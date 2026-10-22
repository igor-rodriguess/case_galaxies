"""Consolida artefatos de decisão da avaliação congelada já executada.

Não refaz treino nem altera dados. Converte JSON/CSV rastreáveis em relatórios,
figuras de explicabilidade e tabelas de erros para a apresentação final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="final-frozen-evaluation-v4")
    return parser.parse_args()


def _save_calibration(sector: str, calibration: list[dict[str, object]], figures: Path) -> None:
    if not calibration:
        return
    frame = pd.DataFrame(calibration)
    plt.figure(figsize=(5.2, 4.4))
    plt.plot(frame["predicted_mean"], frame["observed_high"], "o-", color="#1b9e77")
    plt.xlabel("Score médio previsto")
    plt.ylabel("Fração observada de alto desempenho")
    plt.title(f"Calibração — {sector}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / f"{sector}_calibracao.png", dpi=170)
    plt.close()


def _save_shap(sector: str, importance: list[dict[str, object]], figures: Path) -> None:
    if not importance:
        return
    top = pd.DataFrame(importance[:15]).iloc[::-1]
    plt.figure(figsize=(8.3, 5.4))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#d95f02")
    plt.xlabel("|SHAP| médio")
    plt.title(f"SHAP global no holdout temporal — {sector}")
    plt.tight_layout()
    plt.savefig(figures / f"{sector}_shap.png", dpi=170)
    plt.close()


def _save_residuals(sector: str, predictions: pd.DataFrame, figures: Path) -> None:
    frame = predictions.loc[
        (predictions["sector"] == sector) & (predictions["split"] == "temporal")
    ].copy()
    if frame.empty:
        return
    frame["residual"] = frame["actual"] - frame["predicted"]
    plt.figure(figsize=(5.6, 4.5))
    colors = frame["low_confidence_ood"].map({True: "#d95f02", False: "#1b9e77"})
    plt.scatter(frame["predicted"], frame["residual"], c=colors, alpha=0.8)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Desempenho relativo previsto")
    plt.ylabel("Resíduo (real - previsto)")
    plt.title(f"Resíduos no holdout temporal — {sector}")
    plt.tight_layout()
    plt.savefig(figures / f"{sector}_residuos.png", dpi=170)
    plt.close()


def _candidate_map(split: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["model"]): row["metrics"] for row in split["candidates"] if "metrics" in row}


def _summary_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for sector, info in report["sectors"].items():
        temporal = info["temporal"]
        channel = info["channel_holdout"]
        temporal_models = _candidate_map(temporal)
        channel_models = _candidate_map(channel)
        baseline_t = temporal_models["historical_baseline"]["mae"]
        baseline_c = channel_models["historical_baseline"]["mae"]
        temporal_winner = str(temporal["winner"])
        channel_winner = str(channel["winner"])
        same = temporal_winner == channel_winner
        temporal_evidence = temporal["baseline_evidence"]
        channel_evidence = channel["baseline_evidence"]
        promoted = bool(
            same
            and temporal_evidence["consistently_better"]
            and channel_evidence["consistently_better"]
        )
        if promoted:
            decision = f"promovido: {temporal_winner}"
        elif same:
            decision = f"não promovido: IC bootstrap não confirma ganho de {temporal_winner}"
        else:
            decision = "não promovido: vencedor muda entre holdouts"
        rows.append(
            {
                "sector": sector,
                "temporal_winner": temporal_winner,
                "temporal_mae": temporal_models[temporal_winner]["mae"],
                "temporal_baseline_mae": baseline_t,
                "channel_winner": channel_winner,
                "channel_mae": channel_models[channel_winner]["mae"],
                "channel_baseline_mae": baseline_c,
                "promoted": promoted,
                "decision": decision,
            }
        )
    return rows


def _markdown(metadata: dict[str, object], summary: pd.DataFrame) -> str:
    target = metadata["chosen_target"]
    diagnostics = pd.DataFrame(metadata["target_definitions"])
    lines = [
        "# Relatório final de avaliação — base congelada",
        "",
        "## Escopo e integridade",
        "",
        "- Base: 581 vídeos únicos (Automotivo 165, Moda 215, Beleza 201), congelada em 2026-10-20T01:45:00Z.",
        "- Não entram como entrada: views, likes, comentários, inscritos, IDs, rótulos de caso/controle ou qualquer métrica pós-publicação.",
        "- VLM: abertura de 0-3 s; thumbnail embedding disponível para 100%; transcrição completa ainda não está disponível e, portanto, não é promovida.",
        "",
        "## Alvo",
        "",
        f"Alvo selecionado antes dos holdouts: `{target}`. A escolha usou a menor correlação absoluta de Spearman com idade, e não desempenho de modelo.",
        "",
        "| Definição | Linhas | Spearman com idade | IQR |",
        "|---|---:|---:|---:|",
    ]
    for row in diagnostics.to_dict(orient="records"):
        lines.append(
            f"| {row['definition']} | {row['rows']} | {row['age_spearman']:.3f} | {row['iqr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`residual_age_channel` é log(views) menos uma expectativa estimada somente com vídeos anteriores do mesmo setor/canal. `smoothed_ratio` é a razão suavizada views/mediana histórica prévia; ela foi mantida como sensibilidade, não como alvo principal.",
            "",
            "## Decisão de promoção",
            "",
            "Um modelo só é promovido se for o mesmo vencedor no holdout temporal e no holdout de canais e se ambos os ICs bootstrap confirmarem MAE menor que o baseline. Essa regra evita escolher um método que apenas se adaptou a um corte específico.",
            "",
            "| Setor | Melhor temporal (MAE) | Baseline temporal | Melhor canal (MAE) | Baseline canal | Decisão |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['sector']} | {row['temporal_winner']} ({row['temporal_mae']:.3f}) | {row['temporal_baseline_mae']:.3f} | {row['channel_winner']} ({row['channel_mae']:.3f}) | {row['channel_baseline_mae']:.3f} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Leitura responsável",
            "",
            "- **Moda** traz o sinal mais consistente de ganho sobre o baseline, mas muda de HistGradientBoosting no futuro para ElasticNet no canal inédito. É o melhor setor para aprofundar em uma próxima coleta, não para congelar um único algoritmo ainda.",
            "- **Automotivo** melhora pontualmente, mas os ganhos não têm IC bootstrap inteiramente abaixo de zero; além disso, é o setor com menos canais maduros.",
            "- **Beleza** tem Random Forest forte no corte temporal, porém não generaliza melhor que o baseline para canal novo.",
            "- A MLP foi testada regularizada; não foi promovida. Em pelo menos um fold ela alcançou o máximo de épocas sem convergir, um sinal adicional para não tratá-la como campeã.",
            "",
            "## Arquivos",
            "",
            "- `model_metrics.csv`: métricas de regressão por candidato, setor, holdout e ablação.",
            "- `predictions.csv`: previsões, faixas conformais e marcação OOD por vídeo avaliado.",
            "- `manual_error_review.csv`: maiores erros para conferência humana.",
            "- `feature_stability_external_splits.csv`: repetição das top features de permutação entre os dois holdouts.",
            "- Figuras em `reports/figures/final-frozen-evaluation-v4/` incluem comparação de modelos, previsão vs. real, resíduos, calibração, matriz de confusão, SHAP quando aplicável e importância por permutação.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output = ROOT / "reports" / "final" / "modelagem" / args.run_id
    figures = ROOT / "reports" / "figures" / args.run_id
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output / "predictions.csv")
    summary = pd.DataFrame(_summary_rows(report))
    summary.to_csv(output / "model_summary.csv", index=False)

    stability_rows = []
    for sector, info in report["sectors"].items():
        temporal = {row["feature"] for row in info["temporal"]["permutation_importance"][:10]}
        channel = {row["feature"] for row in info["channel_holdout"]["permutation_importance"][:10]}
        union = temporal | channel
        stability_rows.append(
            {
                "sector": sector,
                "top10_intersection_count": len(temporal & channel),
                "top10_jaccard": len(temporal & channel) / len(union) if union else np.nan,
                "shared_features": "; ".join(sorted(temporal & channel)),
            }
        )
        _save_calibration(sector, info["temporal"]["classification"]["calibration"], figures)
        _save_shap(sector, info["temporal"]["shap"].get("importance", []), figures)
        _save_residuals(sector, predictions, figures)
    pd.DataFrame(stability_rows).to_csv(
        output / "feature_stability_external_splits.csv", index=False
    )

    errors = predictions.copy()
    errors["absolute_error"] = (errors["actual"] - errors["predicted"]).abs()
    keep = [
        name
        for name in [
            "sector",
            "split",
            "video_id",
            "channel_id",
            "actual",
            "predicted",
            "absolute_error",
            "low_confidence_ood",
            "winner_model",
        ]
        if name in errors
    ]
    errors.sort_values("absolute_error", ascending=False).head(60)[keep].to_csv(
        output / "manual_error_review.csv", index=False
    )
    (output / "relatorio_final.md").write_text(
        _markdown(report["metadata"], summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(output / "relatorio_final.md"),
                "promoted": int(summary["promoted"].sum()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
