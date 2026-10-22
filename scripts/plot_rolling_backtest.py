"""Gera figuras rastreáveis a partir de um backtest temporal rolante pronto."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plota estabilidade do backtest rolante.")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = MODELING_RUNS_DIR / args.run_id
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(run_dir / "fold_metrics.csv")
    figure_dir = ROOT / "reports" / "figures" / args.run_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "logistic_pooled_top10": "Logística",
        "hist_gb_pooled_clean20vs50": "HGB conjunto",
        "hist_gb_per_sector_clean20vs50": "HGB por setor",
        "catboost_ranker_per_sector_top10": "CatBoost ranker",
        "mlp_pooled_clean20vs50": "MLP pequena",
    }
    fig, axis = plt.subplots(figsize=(9, 5))
    for model, group in folds.groupby("model", sort=False):
        ordered = group.sort_values("fold")
        axis.plot(
            ordered["fold"],
            ordered["auc_intra"],
            marker="o",
            linewidth=2,
            label=labels.get(model, model),
        )
    axis.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1, label="aleatório")
    axis.set(
        xlabel="Janela temporal",
        ylabel="AUC intra-canal",
        title="Estabilidade temporal por modelo",
        xticks=sorted(folds["fold"].unique()),
        ylim=(0.25, 0.85),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "auc-por-janela.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(report["results"])
    summary["auc"] = summary["metrics"].map(lambda item: item["auc_intra"])
    summary["low"] = summary["metrics"].map(lambda item: item["auc_intra_ci_low"])
    summary["high"] = summary["metrics"].map(lambda item: item["auc_intra_ci_high"])
    summary["label"] = summary["name"].map(labels).fillna(summary["name"])
    summary.sort_values("auc", inplace=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    errors = [summary["auc"] - summary["low"], summary["high"] - summary["auc"]]
    axis.barh(summary["label"], summary["auc"], xerr=errors, color="#176B87", capsize=4)
    axis.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1, label="aleatório")
    axis.set(
        xlabel="AUC intra-canal (IC 95% por bootstrap de fold x canal)",
        title="Comparação agregada sem misturar janelas temporais",
        xlim=(0.35, 0.90),
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / "auc-agregada-ic95.png", dpi=180)
    plt.close(fig)
    print(figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
