"""Artefatos tabulares, gráficos e relatório legível da avaliação de modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..dataset import write_json


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Instale o extra [evaluation] para gerar os gráficos."
        ) from exc
    return plt


def _model_label(name: str) -> str:
    labels = {
        "dummy_median": "Baseline mediana",
        "linear_regression": "Regressão linear",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting",
        "hist_gradient_boosting": "Hist. Gradient Boosting",
        "mlp": "MLP",
        "ensemble_equal": "Ensemble média",
        "ensemble_weighted": "Ensemble ponderado",
    }
    return labels.get(name, name)


def metrics_frame(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sector, report in reports.items():
        for candidate in report["candidates"]:
            if candidate["status"] != "ok":
                continue
            for split in ("temporal", "channel_holdout"):
                metrics = candidate[split]
                rows.append(
                    {
                        "sector": sector,
                        "model": candidate["model"],
                        "model_label": _model_label(candidate["model"]),
                        "kind": candidate["kind"],
                        "split": split,
                        "inner_cv_mae_log": candidate.get("inner_cv_mae_log"),
                        "selected_feature_count": candidate.get(
                            "selected_feature_count"
                        ),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def predictions_frame(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [row for report in reports.values() for row in report["predictions"]]
    )


def _comparison_chart(report: dict[str, Any], path: Path) -> None:
    plt = _pyplot()
    candidates = [
        item for item in report["candidates"] if item.get("status") == "ok"
    ]
    labels = [_model_label(item["model"]) for item in candidates]
    positions = np.arange(len(candidates))
    width = 0.38
    temporal_mae = [item["temporal"]["mae_log"] for item in candidates]
    channel_mae = [item["channel_holdout"]["mae_log"] for item in candidates]
    temporal_r2 = [item["temporal"]["r2_log"] for item in candidates]
    channel_r2 = [item["channel_holdout"]["r2_log"] for item in candidates]

    figure, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].bar(positions - width / 2, temporal_mae, width, label="Temporal")
    axes[0].bar(positions + width / 2, channel_mae, width, label="Canal não visto")
    axes[0].set_ylabel("MAE log (menor é melhor)")
    axes[0].set_title(f"Comparação de modelos — {report['sector'].title()}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_xticks(positions, [])

    axes[1].bar(positions - width / 2, temporal_r2, width, label="Temporal")
    axes[1].bar(positions + width / 2, channel_r2, width, label="Canal não visto")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("R² log (maior é melhor)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(positions, labels, rotation=32, ha="right")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _prediction_chart(report: dict[str, Any], path: Path) -> None:
    plt = _pyplot()
    winner = report["selection"]["best_inner_cv"]
    frame = pd.DataFrame(report["predictions"])
    frame = frame.loc[frame["model"] == winner].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, split, title in zip(
        axes,
        ("temporal", "channel_holdout"),
        ("Holdout temporal", "Canais não vistos"),
        strict=True,
    ):
        selected = frame.loc[frame["split"] == split]
        relative = report.get("target", {}).get("mode") == "relative_performance"
        actual = selected["actual"].to_numpy(dtype=float)
        predicted = selected["predicted"].to_numpy(dtype=float)
        if not relative:
            actual = np.maximum(actual, 1.0)
            predicted = np.maximum(predicted, 1.0)
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.scatter(actual, predicted, alpha=0.75, edgecolor="none")
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
        if not relative:
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.set_xlabel(
            "Índice relativo observado" if relative else "Visualizações observadas"
        )
        axis.set_ylabel(
            "Índice relativo previsto" if relative else "Visualizações previstas"
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Real x previsto — {report['sector'].title()} — {_model_label(winner)}"
    )
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _importance_chart(report: dict[str, Any], path: Path) -> None:
    plt = _pyplot()
    rows = [
        row for row in report["feature_importance"] if row["importance_mean"] > 0
    ][:15]
    if not rows:
        rows = report["feature_importance"][:15]
    rows = list(reversed(rows))
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.barh(
        [row["feature"] for row in rows],
        [row["importance_mean"] for row in rows],
        xerr=[row["importance_std"] for row in rows],
        alpha=0.85,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Queda do desempenho ao permutar a feature")
    axis.set_title(
        f"Importância por permutação — {report['sector'].title()} — "
        f"{_model_label(report['selection']['best_inner_cv'])}"
    )
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _summary_chart(reports: dict[str, dict[str, Any]], path: Path) -> None:
    plt = _pyplot()
    rows = []
    for sector, report in reports.items():
        winner = report["selection"]["best_inner_cv"]
        candidate = next(
            item for item in report["candidates"] if item["model"] == winner
        )
        rows.append(
            {
                "sector": sector.title(),
                "winner": _model_label(winner),
                "temporal": candidate["temporal"]["mae_log"],
                "channel": candidate["channel_holdout"]["mae_log"],
            }
        )
    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        [row["temporal"] for row in rows],
        width,
        label="Temporal",
    )
    axis.bar(
        positions + width / 2,
        [row["channel"] for row in rows],
        width,
        label="Canal não visto",
    )
    labels = [f"{row['sector']}\n{row['winner']}" for row in rows]
    axis.set_xticks(positions, labels)
    axis.set_ylabel("MAE log (menor é melhor)")
    axis.set_title("Candidatos escolhidos sem consultar os holdouts")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def _markdown_report(
    reports: dict[str, dict[str, Any]],
    run_id: str,
    figures_dir: Path,
) -> str:
    relative = all(
        report.get("target", {}).get("mode") == "relative_performance"
        for report in reports.values()
    )
    result_note = (
        "> **Resultado exploratório.** O alvo é o desvio de desempenho após "
        "ajustar idade do vídeo e escala histórica do canal somente no treino. "
        "Isso permite comparar os snapshots atuais sem fingir que são views_30d."
        if relative
        else "> **Resultado exploratório.** O alvo disponível é uma contagem acumulada "
        "observada em idades diferentes. Nenhum modelo desta execução deve ser "
        "promovido para produção."
    )
    lines = [
        "# Avaliação otimizada dos modelos de regressão",
        "",
        f"Execução: `{run_id}`.",
        "",
        result_note,
        "",
        "## Método",
        "",
        "A busca usa somente o treino temporal externo. Dentro dele, folds "
        "expanding-window ajustam hiperparâmetros, escalonamento e seleção de features. "
        "Para o corte por canal, uma busca interna separada usa exclusivamente os "
        "canais de treino. Os dois holdouts ficam isolados até a avaliação.",
        "",
        "Colunas constantes são removidas com VarianceThreshold. A busca compara "
        "StandardScaler e RobustScaler e testa seleção por F-test, informação mútua "
        "e importância de Extra Trees.",
        "",
        "Os ensembles combinam Ridge, Random Forest, Gradient Boosting e MLP. "
        "Os pesos do ensemble ponderado são aprendidos em previsões out-of-fold "
        "com restrição de pesos não negativos.",
        "",
        f"![Resumo dos vencedores](figures/{figures_dir.name}/summary_mae_log.png)",
        "",
        "## Resultados por setor",
        "",
    ]
    for sector, report in reports.items():
        lines.extend(
            [
                f"### {sector.title()}",
                "",
                f"Amostra comum: {report['sample']['rows']} vídeos, "
                f"{report['sample']['channels']} canais e "
                f"{report['sample']['fixed_horizon_rows']} alvos em horizonte fixo.",
                "",
                f"Escolha pela validação interna: "
                f"**{_model_label(report['selection']['best_inner_cv'])}**. "
                f"Melhor observado no holdout temporal: "
                f"**{_model_label(report['selection']['best_temporal_observed'])}**. "
                f"Melhor observado em canais não vistos: "
                f"**{_model_label(report['selection']['best_channel_holdout_observed'])}**.",
                "",
                "| Modelo | Tipo | MAE log interno | MAE log temporal | R² temporal | "
                "MAE log canal | R² canal | Spearman temporal |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in report["candidates"]:
            if candidate["status"] != "ok":
                continue
            lines.append(
                f"| {_model_label(candidate['model'])} | {candidate['kind']} | "
                f"{_format_metric(candidate.get('inner_cv_mae_log'))} | "
                f"{_format_metric(candidate['temporal']['mae_log'])} | "
                f"{_format_metric(candidate['temporal']['r2_log'])} | "
                f"{_format_metric(candidate['channel_holdout']['mae_log'])} | "
                f"{_format_metric(candidate['channel_holdout']['r2_log'])} | "
                f"{_format_metric(candidate['temporal']['spearman'])} |"
            )
        lines.extend(
            [
                "",
                f"![Comparação de {sector}](figures/{figures_dir.name}/comparison_{sector}.png)",
                "",
                f"![Real e previsto de {sector}](figures/{figures_dir.name}/predictions_{sector}.png)",
                "",
                f"![Features de {sector}](figures/{figures_dir.name}/features_{sector}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Como interpretar",
            "",
            (
                "- A seleção principal usa MAE no índice relativo residual; menor é melhor."
                if relative
                else "- A seleção principal usa MAE no alvo `log1p(views)`; menor é melhor."
            ),
            "- R² negativo indica desempenho quadrático pior que prever a média.",
            "- Spearman mede a capacidade de ordenar criativos, não a precisão em views.",
            "- Diferenças entre validação interna e holdout indicam instabilidade ou mudança "
            "de distribuição.",
            "- A importância por permutação foi calculada apenas no holdout temporal e pode "
            "oscilar com a amostra pequena.",
            "",
            "## Condição para uma decisão final",
            "",
            (
                "Usar este índice para comparar e ranquear criativos. A conversão para uma "
                "quantidade absoluta de views continua condicionada a um horizonte escolhido "
                "e à curva histórica do canal."
                if relative
                else "Repetir o protocolo quando houver um alvo temporal comparável. Até lá, "
                "a execução serve para validar o método, não para produção."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_artifacts(
    reports: dict[str, dict[str, Any]],
    output_dir: Path,
    run_id: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    project_root = output_dir.parents[3]
    figures_dir = project_root / "reports" / "figures" / run_id
    figures_dir.mkdir(parents=True, exist_ok=False)
    for sector, report in reports.items():
        write_json(output_dir / f"{sector}.json", report)
        _comparison_chart(report, figures_dir / f"comparison_{sector}.png")
        _prediction_chart(report, figures_dir / f"predictions_{sector}.png")
        _importance_chart(report, figures_dir / f"features_{sector}.png")
    _summary_chart(reports, figures_dir / "summary_mae_log.png")

    metrics_path = output_dir / "model_metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    metrics_frame(reports).to_csv(metrics_path, index=False, encoding="utf-8")
    predictions_frame(reports).to_csv(predictions_path, index=False, encoding="utf-8")
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": "0.2.0",
            "run_id": run_id,
            "exploratory": True,
            "sectors": {
                sector: report["selection"] for sector, report in reports.items()
            },
        },
    )
    report_path = project_root / "reports" / f"{run_id}.md"
    report_path.write_text(
        _markdown_report(reports, run_id, figures_dir), encoding="utf-8"
    )
    return {
        "report": str(report_path),
        "summary": str(output_dir / "summary.json"),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "figures": str(figures_dir),
    }
