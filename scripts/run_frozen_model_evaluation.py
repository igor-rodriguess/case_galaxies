"""Executa a avaliação final da base congelada sem alterar a coleta.

Uso:
    python scripts/run_frozen_model_evaluation.py --run-id final-frozen-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge

ROOT = Path(__file__).resolve().parents[1]
TARGET_NAME = "residual_age_channel"
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.protocol import apply_vlm_taxonomy  # noqa: E402
from galaxies.dataset import build_feature_rows  # noqa: E402
from galaxies.final_audit import FinalScope  # noqa: E402
from galaxies.modeling.frozen_evaluation import (  # noqa: E402
    SECTORS,
    causal_relative_targets,
    evaluate_outer_split,
    make_splits,
    target_diagnostics,
)
from galaxies.prospective import exclude_prospective  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia modelos na base VLM congelada.")
    parser.add_argument("--run-id", default="final-frozen-evaluation-v1")
    return parser.parse_args()


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def load_frozen_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scope = FinalScope()
    cutoff = datetime.fromisoformat(scope.vlm_completed_before.replace("Z", "+00:00"))
    all_rows, _ = build_feature_rows(ROOT / scope.youtube_run, vlm_completed_before=cutoff)
    history = exclude_prospective(apply_vlm_taxonomy(pd.DataFrame(all_rows)))
    base = pd.read_csv(ROOT / "reports" / "final" / "auditoria" / "base_principal.csv")
    frame = history.loc[history["video_id"].isin(base["video_id"])].copy()
    thumb = pd.read_parquet(ROOT / "data" / "features" / "embeddings" / "thumb.parquet")
    embeddings = dict(zip(thumb["video_id"], thumb["embedding"], strict=True))
    frame["thumbnail_embedding"] = frame["video_id"].map(embeddings)
    if frame["thumbnail_embedding"].isna().any():
        raise ValueError("A base congelada precisa ter embedding de thumbnail em 100% dos vídeos.")
    targets = causal_relative_targets(history, frame)
    frame = frame.merge(
        targets[
            [
                "video_id",
                "residual_age_channel",
                "smoothed_ratio",
                "history_rows",
                "channel_history_rows",
            ]
        ],
        on="video_id",
        validate="one_to_one",
    )
    diagnostics = target_diagnostics(frame)
    # O alvo é fixado antes da avaliação. Os diagnósticos usam a base inteira,
    # inclusive os holdouts, então não podem escolher a definição.
    chosen = TARGET_NAME
    metadata = {
        "scope": scope.as_dict(),
        "base_rows": len(frame),
        "target_definitions": diagnostics.to_dict(orient="records"),
        "chosen_target": chosen,
        "choice_rule": (
            "fixado a priori (residual_age_channel); os diagnósticos de idade são "
            "só descritivos e não escolhem o alvo"
        ),
    }
    return frame, history, metadata


def _split_target(
    sector_frame: pd.DataFrame,
    history: pd.DataFrame,
    split_name: str,
    test: np.ndarray,
    target_name: str,
) -> np.ndarray:
    """Alvo do split; no de canal, a curva setorial não vê os canais reservados."""
    if split_name != "channel_holdout":
        return sector_frame[target_name].to_numpy(float)
    reserved = set(sector_frame.loc[test, "channel_id"].astype(str))
    recomputed = causal_relative_targets(history, sector_frame, reserved)
    return recomputed[target_name].to_numpy(float)


def _input_importance(
    result: dict[str, object], test: np.ndarray, target: np.ndarray
) -> list[dict[str, float | str]]:
    pipeline = result["pipeline"]
    x = result["x"]
    assert hasattr(pipeline, "predict")
    measured = permutation_importance(
        pipeline,
        x.loc[test],
        target[test],
        scoring="neg_mean_absolute_error",
        n_repeats=15,
        random_state=2026,
        n_jobs=1,
    )
    rows = [
        {"feature": str(name), "importance_mean": float(mean), "importance_std": float(std)}
        for name, mean, std in zip(
            x.columns, measured.importances_mean, measured.importances_std, strict=True
        )
    ]
    return sorted(rows, key=lambda item: item["importance_mean"], reverse=True)


def _shap_values(
    result: dict[str, object], train: np.ndarray, test: np.ndarray
) -> dict[str, object]:
    if result["winner"] == "ensemble_oof":
        return {"available": False, "reason": "Ensemble não tem explicação SHAP única."}
    try:
        import shap
    except ImportError:
        return {"available": False, "reason": "Pacote shap não está instalado."}
    try:
        pipeline = result["pipeline"]
        x = result["x"]
        transformed_train = pipeline[:-1].transform(x.loc[train])
        transformed_test = pipeline[:-1].transform(x.loc[test])
        names = np.asarray(pipeline.named_steps["preprocess"].get_feature_names_out(), dtype=object)
        names = names[pipeline.named_steps["variance"].get_support()]
        selector = pipeline.named_steps["select"]
        if selector != "passthrough":
            names = names[selector.get_support()]
        background = transformed_train[: min(40, len(transformed_train))]
        sample = transformed_test[: min(40, len(transformed_test))]
        model = pipeline.named_steps["model"]
        # O explicador automático cai em PermutationExplainer para MLP e pode
        # custar horas. O relatório preserva permutation importance no holdout
        # nesse caso; SHAP é usado quando há explicador exato/rápido.
        if isinstance(model, RandomForestRegressor):
            values = shap.TreeExplainer(model)(sample)
        elif isinstance(model, (Ridge, ElasticNet)):
            values = shap.LinearExplainer(model, background)(sample)
        elif isinstance(model, HistGradientBoostingRegressor):
            values = shap.Explainer(model, background)(sample)
        else:
            return {
                "available": False,
                "reason": "MLP não recebe SHAP aproximado nesta rodada; use importance por permutação no holdout.",
            }
        importance = np.abs(values.values).mean(axis=0)
        rows = [
            {"feature": str(name), "mean_abs_shap": float(value)}
            for name, value in zip(names, importance, strict=True)
        ]
        return {
            "available": True,
            "importance": sorted(rows, key=lambda row: row["mean_abs_shap"], reverse=True),
        }
    except Exception as exc:  # SHAP é opcional; não substituímos por dado inventado.
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _bootstrap_win(actual: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    baseline = np.zeros(len(actual))
    diff = np.abs(actual - candidate) - np.abs(actual - baseline)
    rng = np.random.default_rng(2026)
    samples = rng.integers(0, len(diff), size=(2_000, len(diff)))
    means = diff[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mae_difference_vs_baseline": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "consistently_better": bool(high < 0),
    }


def _plot_comparison(
    sector: str, split_results: dict[str, dict[str, object]], figures: Path
) -> None:
    temporal = split_results["temporal"]
    rows = [row for row in temporal["candidates"] if "metrics" in row]
    labels = [str(row["model"]) for row in rows]
    mae = [row["metrics"]["mae"] for row in rows]
    plt.figure(figsize=(11, 4.5))
    plt.bar(
        labels,
        mae,
        color=["#666" if name == "historical_baseline" else "#2a78d6" for name in labels],
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE do desempenho relativo")
    plt.title(f"Modelos e ensemble OOF — {sector}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / f"{sector}_modelos.png", dpi=170)
    plt.close()

    prediction = temporal["predictions"]
    plt.figure(figsize=(5.5, 5))
    plt.scatter(
        prediction["actual"],
        prediction["predicted"],
        c=prediction["low_confidence_ood"].map({True: "#d95f02", False: "#1b9e77"}),
        alpha=0.8,
    )
    low = min(prediction["actual"].min(), prediction["predicted"].min())
    high = max(prediction["actual"].max(), prediction["predicted"].max())
    plt.plot([low, high], [low, high], "--", color="black")
    plt.xlabel("Desempenho relativo observado")
    plt.ylabel("Desempenho relativo previsto")
    plt.title(f"Previsão vs. real — {sector}")
    plt.tight_layout()
    plt.savefig(figures / f"{sector}_previsao_real.png", dpi=170)
    plt.close()

    importance = temporal.get("permutation_importance", [])[:15]
    if importance:
        display = list(reversed(importance))
        plt.figure(figsize=(8, 5))
        plt.barh(
            [row["feature"] for row in display],
            [row["importance_mean"] for row in display],
            xerr=[row["importance_std"] for row in display],
            color="#7570b3",
        )
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"Importância por permutação no holdout — {sector}")
        plt.tight_layout()
        plt.savefig(figures / f"{sector}_permutation.png", dpi=170)
        plt.close()

    matrix = np.asarray(temporal["classification"]["confusion_matrix"])
    plt.figure(figsize=(4.4, 4))
    plt.imshow(matrix, cmap="Blues")
    for left in range(matrix.shape[0]):
        for right in range(matrix.shape[1]):
            plt.text(right, left, str(matrix[left, right]), ha="center", va="center")
    labels = temporal["classification"]["labels"]
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.xlabel("Classe prevista pela regressão")
    plt.ylabel("Classe observada")
    plt.title(f"Classificação secundária — {sector}")
    plt.tight_layout()
    plt.savefig(figures / f"{sector}_matriz_confusao.png", dpi=170)
    plt.close()


def _markdown(sector: str, result: dict[str, object], target_name: str) -> str:
    lines = [
        f"# Avaliação final — {sector.title()}",
        "",
        f"Alvo principal: `{target_name}`.",
        "",
        "## Comparação temporal",
        "",
        "As métricas de holdout dos demais candidatos são diagnóstico: o modelo reportado é o escolhido nas dobras internas do treino.",
        "",
        "| Modelo | MAE | RMSE | R² | Spearman | Lift@K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["temporal"]["candidates"]:
        if "metrics" not in row:
            continue
        metrics = row["metrics"]
        lines.append(
            f"| {row['model']}{' (escolhido nas dobras internas)' if row.get('selected_internally') else ''} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} | {metrics['r2']:.3f} | {metrics['spearman'] if metrics['spearman'] is not None else '—'} | {metrics['lift_at_k']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Vencedor selecionado internamente: **{result['temporal']['winner']}**.",
            f"Intervalo conformal: `{json.dumps(result['temporal']['conformal'], ensure_ascii=False)}`.",
            f"Classificação secundária: macro-F1 `{result['temporal']['classification']['macro_f1']:.3f}`, PR-AUC alto `{result['temporal']['classification']['pr_auc_high']}`.",
            "",
            "A promoção exige vencer o baseline em temporal e por canal com IC bootstrap favorável; caso contrário, o baseline permanece a referência.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output = ROOT / "reports" / "final" / "modelagem" / args.run_id
    figures = ROOT / "reports" / "figures" / args.run_id
    if output.exists() or figures.exists():
        raise FileExistsError("run-id já existe; escolha outro para preservar a rastreabilidade.")
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    frame, history, metadata = load_frozen_frame()
    target_name = str(metadata["chosen_target"])
    metrics_rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    report: dict[str, object] = {"metadata": metadata, "sectors": {}}
    for sector in SECTORS:
        sector_frame = (
            frame.loc[frame["sector"] == sector]
            .sort_values("published_at", kind="stable")
            .reset_index(drop=True)
        )
        sector_frame = sector_frame.loc[sector_frame[target_name].notna()].reset_index(drop=True)
        split_results: dict[str, dict[str, object]] = {}
        for split_name, (train, test) in make_splits(sector_frame).items():
            split_kind = "channel" if split_name == "channel_holdout" else "temporal"
            target = _split_target(sector_frame, history, split_name, test, target_name)
            valid = np.isfinite(target)
            train, test = train[valid[train]], test[valid[test]]
            combined = evaluate_outer_split(
                sector_frame,
                target,
                train,
                test,
                "all",
                full_comparison=True,
                split_kind=split_kind,
            )
            combined["permutation_importance"] = _input_importance(combined, test, target)
            combined["shap"] = _shap_values(combined, train, test)
            combined["baseline_evidence"] = _bootstrap_win(
                target[test], combined["predictions"]["predicted"].to_numpy(float)
            )
            ablations: dict[str, dict[str, object]] = {}
            for group in ("metadata", "metadata_vlm", "metadata_thumbnail", "all"):
                ablation = (
                    combined
                    if group == "all"
                    else evaluate_outer_split(
                        sector_frame,
                        target,
                        train,
                        test,
                        group,
                        full_comparison=False,
                        split_kind=split_kind,
                    )
                )
                ablations[group] = {
                    "winner": ablation["winner"],
                    "metrics": ablation["winner_metrics"],
                    "selected_recipe": ablation["selected_recipe"],
                }
            combined["ablations"] = ablations
            split_results[split_name] = combined
            prediction = combined["predictions"].copy()
            prediction["split"] = split_name
            prediction["model"] = combined["winner"]
            all_predictions.append(prediction)
            for candidate in combined["candidates"]:
                if "metrics" in candidate:
                    metrics_rows.append(
                        {
                            "sector": sector,
                            "split": split_name,
                            "feature_group": "all",
                            "model": candidate["model"],
                            **candidate["metrics"],
                        }
                    )
            for group, ablation in ablations.items():
                metrics_rows.append(
                    {
                        "sector": sector,
                        "split": split_name,
                        "feature_group": group,
                        "model": ablation["winner"],
                        **ablation["metrics"],
                    }
                )
        _plot_comparison(sector, split_results, figures)
        serializable = {}
        for split, result in split_results.items():
            serializable[split] = {
                key: value
                for key, value in result.items()
                if key not in {"pipeline", "x", "predictions"}
            }
            serializable[split]["predictions"] = result["predictions"].to_dict(orient="records")
        report["sectors"][sector] = serializable
        (output / f"{sector}.md").write_text(
            _markdown(sector, split_results, target_name), encoding="utf-8"
        )
    pd.DataFrame(metrics_rows).to_csv(output / "model_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    _json(output / "evaluation.json", report)
    (output / "README.md").write_text(
        "# Avaliação final\n\nVeja os relatórios por setor, `model_metrics.csv`, `predictions.csv` e os gráficos correspondentes.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "figures": str(figures), "target": target_name},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
