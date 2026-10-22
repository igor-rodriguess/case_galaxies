"""Congela a referência vencedora e produz explicabilidade/observabilidade."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.evaluation import (  # noqa: E402
    evaluate_on_vault,
    feature_frame,
    make_estimator,
    run_experiment,
)
from galaxies.agent.protocol import latest_feature_map, load_dataset  # noqa: E402
from galaxies.agent.two_stage import base_spec  # noqa: E402
from galaxies.modeling.preprocessing import make_pipeline  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Congela e explica a baseline Galaxies.")
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Não serializável: {type(value)!r}")


def main() -> int:
    args = parse_args()
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    dataset = load_dataset(feature_map)
    spec = base_spec("pooled")
    rows = dataset.evaluable("full")
    numeric, categorical = spec.columns()
    pipeline = make_pipeline(
        {"features": {"numeric": numeric, "categorical": categorical}},
        make_estimator(spec, args.seed),
    )
    train_frame = feature_frame(spec, rows)
    pipeline.fit(train_frame, rows["viral"].to_numpy())

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = args.run_id or f"baseline-freeze-{timestamp}"
    output_dir = PROJECT_ROOT / "data/modeling/runs" / run_id
    figure_dir = PROJECT_ROOT / "reports/figures" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, output_dir / "baseline_pipeline.joblib")

    print("Gerando previsões LOCO e temporal...", flush=True)
    evaluation = run_experiment(
        spec, dataset, seed=args.seed, bootstrap_iterations=args.bootstrap
    )
    vault = evaluate_on_vault(
        spec, dataset, seed=args.seed, bootstrap_iterations=args.bootstrap
    )
    oof = evaluation.predictions.copy()

    raw_score = oof["score_loco"].to_numpy().reshape(-1, 1)
    labels = oof["viral"].to_numpy()
    calibrated_oof = np.full(len(oof), np.nan)
    groups = oof["channel_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(5, oof["channel_id"].nunique()))
    for train_index, test_index in splitter.split(raw_score, labels, groups=groups):
        calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
        calibrator.fit(raw_score[train_index], labels[train_index])
        calibrated_oof[test_index] = calibrator.predict_proba(raw_score[test_index])[:, 1]
    final_calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
    final_calibrator.fit(raw_score, labels)
    joblib.dump(final_calibrator, output_dir / "probability_calibrator.joblib")
    oof["score_calibrated_loco"] = calibrated_oof
    probability_true_raw, probability_pred_raw = calibration_curve(
        labels, oof["score_loco"], n_bins=8, strategy="quantile"
    )
    probability_true_cal, probability_pred_cal = calibration_curve(
        labels, calibrated_oof, n_bins=8, strategy="quantile"
    )
    brier_raw = float(brier_score_loss(labels, oof["score_loco"]))
    brier_calibrated = float(brier_score_loss(labels, calibrated_oof))
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    evaluation.temporal_predictions.to_csv(output_dir / "temporal_predictions.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="#7f8c8d", label="calibração perfeita")
    axis.plot(
        probability_pred_raw,
        probability_true_raw,
        marker="o",
        color="#B85042",
        label="escore cru",
    )
    axis.plot(
        probability_pred_cal,
        probability_true_cal,
        marker="o",
        color="#176B87",
        label="Platt cross-calibrada",
    )
    axis.set(xlabel="probabilidade prevista", ylabel="fração viral observada", title="Calibração LOCO")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "calibration-loco.png", dpi=180)
    plt.close(figure)

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(train_frame)
    feature_names = preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_.ravel()}
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients.sort_values("abs_coefficient", ascending=False, inplace=True)
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)

    shap_method = "linear_coefficient_fallback"
    shap_values = np.abs(model.coef_.ravel())
    try:
        import shap

        explainer = shap.LinearExplainer(model, transformed)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        shap_values = np.mean(np.abs(np.asarray(values)), axis=0)
        shap_method = "shap.LinearExplainer"
    except Exception as exc:  # pragma: no cover - depende de binários do ambiente.
        (output_dir / "shap_fallback.txt").write_text(str(exc), encoding="utf-8")
    shap_frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_values})
    shap_frame.sort_values("mean_abs_shap", ascending=False, inplace=True)
    shap_frame.to_csv(output_dir / "shap_global.csv", index=False)
    top = shap_frame.head(15).sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature"], top["mean_abs_shap"], color="#159895")
    axis.set(xlabel="impacto médio absoluto", title="SHAP global — baseline logística")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "shap-global.png", dpi=180)
    plt.close(figure)

    numeric_profile = {}
    for column in numeric:
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        numeric_profile[column] = {
            "p01": float(values.quantile(0.01)) if len(values) else None,
            "p99": float(values.quantile(0.99)) if len(values) else None,
            "missing_rate": float(rows[column].isna().mean()),
        }
    categorical_profile = {
        column: sorted(str(value) for value in rows[column].dropna().unique())
        for column in categorical
    }
    confidence_profile = {
        "numeric_training_range": numeric_profile,
        "categorical_training_values": categorical_profile,
        "low_confidence_rule": "marcar quando qualquer numérico ficar fora de [p01, p99] ou categoria não tiver sido observada no treino",
    }
    (output_dir / "confidence_profile.json").write_text(
        json.dumps(confidence_profile, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_map": feature_map,
        "model": spec.model_dump(mode="json"),
        "training": {
            "rows": len(rows),
            "positives": int(rows["viral"].sum()),
            "channels": int(rows["channel_id"].nunique()),
            "post_publication_features": "explicitamente bloqueadas pelo protocolo",
        },
        "evaluation": {
            "loco": evaluation.loco,
            "temporal": evaluation.temporal,
            "vault": vault,
            "oof_brier_raw": brier_raw,
            "oof_brier_cross_calibrated": brier_calibrated,
        },
        "explainability": {
            "method": shap_method,
            "top_features": shap_frame.head(15).to_dict(orient="records"),
            "interpretation": "SHAP descreve associação no modelo; não prova causalidade nem que editar a feature produzirá viralização.",
        },
        "confidence": confidence_profile,
        "decision": "Baseline logística permanece referência implantável; modelos multimodais são mantidos como pesquisa, pois falharam no cofre.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "# Baseline congelada e explicável\n\n"
        "A referência implantável usa metadados pré-publicação e cadência. O cofre não aprovou a "
        "MLP multimodal; por isso ela não é promovida.\n\n"
        f"- LOCO: AUC intra-canal {evaluation.loco['auc_intra']:.3f}.\n"
        f"- Temporal: AUC intra-canal {evaluation.temporal['auc_intra']:.3f}.\n"
        f"- Cofre: AUC intra-canal {vault.get('auc_intra', float('nan')):.3f}.\n"
        f"- Brier OOF cru: {brier_raw:.3f}; calibrado por canal: {brier_calibrated:.3f}.\n\n"
        "Os artefatos incluem coeficientes, SHAP global, curva de calibração e perfil de confiança.\n",
        encoding="utf-8",
    )
    print(f"Relatório: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
