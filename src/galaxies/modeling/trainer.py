"""Compara candidatos sem esconder as regras de validação do experimento."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ..config import load_sector_config, validate_sector
from ..dataset import load_jsonl, write_json
from ..paths import MODELS_DIR
from .candidates import make_classifier, make_regressor
from .metrics import classification_metrics, regression_metrics
from .preprocessing import feature_columns, make_pipeline
from .splits import Holdout, channel_holdout, temporal_holdout


class DatasetGateError(ValueError):
    """O conjunto ainda não satisfaz as condições para treino final."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("Treino bloqueado:\n- " + "\n- ".join(reasons))


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run_id(prefix: str) -> str:
    return f"{prefix}-" + utc_now().replace(":", "-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_frame(
    dataset_path: Path,
    sector: str,
    task: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = load_jsonl(dataset_path)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise DatasetGateError(["O mapa de features está vazio."])
    if "sector" not in frame:
        raise DatasetGateError(["A coluna sector não existe no mapa de features."])
    frame = frame.loc[frame["sector"] == sector].copy()
    target_column = (
        config["target"]["column"]
        if task == "regression"
        else config["classification"]["label_column"]
    )
    if target_column not in frame:
        raise DatasetGateError([f"A coluna-alvo {target_column!r} não existe."])
    frame = frame.loc[frame[target_column].notna()].copy()
    frame.reset_index(drop=True, inplace=True)
    if frame.empty:
        raise DatasetGateError([f"Nenhuma linha do setor {sector} possui {target_column}."])
    return frame


def _gate_reasons(
    frame: pd.DataFrame,
    config: dict[str, Any],
    task: str,
) -> list[str]:
    reasons: list[str] = []
    minimum = config["minimum_data"]
    if len(frame) < minimum["rows"]:
        reasons.append(f"{len(frame)} linhas; mínimo configurado: {minimum['rows']}.")
    channels = frame["channel_id"].nunique() if "channel_id" in frame else 0
    if channels < minimum["channels"]:
        reasons.append(f"{channels} canais; mínimo configurado: {minimum['channels']}.")
    if task == "regression":
        if config["target"]["requires_fixed_horizon"]:
            valid = frame.get("target_is_fixed_horizon", pd.Series(False, index=frame.index))
            invalid_count = int((valid != True).sum())  # noqa: E712
            if invalid_count:
                reasons.append(f"{invalid_count} linhas não têm alvo em horizonte fixo.")
        vlm = frame.get("vlm_available", pd.Series(False, index=frame.index)).fillna(False)
        coverage = float(vlm.astype(bool).mean())
        if coverage < minimum["vlm_coverage"]:
            reasons.append(f"Cobertura VLM {coverage:.1%}; mínimo: {minimum['vlm_coverage']:.1%}.")
        prior_count = frame.get("prior_video_count", pd.Series(0, index=frame.index)).fillna(0)
        history_valid = frame.get(
            "channel_history_temporally_valid", pd.Series(False, index=frame.index)
        )
        invalid_history = (prior_count > 0) & (history_valid != True)  # noqa: E712
        if invalid_history.any():
            reasons.append(
                f"{int(invalid_history.sum())} linhas usam histórico observado depois da publicação."
            )
    else:
        classification = config["classification"]
        if not classification.get("enabled"):
            reasons.append(
                "A classificação está desabilitada até a definição dos limiares das faixas."
            )
        if classification.get("thresholds") is None:
            reasons.append("Os limiares das faixas de desempenho ainda não foram definidos.")
        label = classification["label_column"]
        if frame[label].nunique() < 2:
            reasons.append("A classificação exige pelo menos duas classes observadas.")
    return reasons


def _feature_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    numeric, categorical = feature_columns(config)
    columns = numeric + categorical
    features = frame.reindex(columns=columns).copy()
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical:
        features[column] = features[column].map(
            lambda value: None if pd.isna(value) else str(value)
        )
    return features


def _inverse_target(values: Any, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.clip(np.expm1(array), 0, None)
    if transform == "none":
        return np.clip(array, 0, None)
    raise ValueError(f"Transformação de alvo desconhecida: {transform}")


def _transform_target(values: Any, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (array < 0).any():
        raise ValueError("A variável-alvo não pode conter valores negativos.")
    if transform == "log1p":
        return np.log1p(array)
    if transform == "none":
        return array
    raise ValueError(f"Transformação de alvo desconhecida: {transform}")


def _evaluate_regressor(
    model_name: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    y_raw: np.ndarray,
    y_model: np.ndarray,
    config: dict[str, Any],
    temporal: Holdout,
    channel: Holdout,
) -> tuple[dict[str, Any], Any]:
    random_state = config["validation"]["random_state"]
    result: dict[str, Any] = {"model": model_name, "status": "ok"}
    temporal_pipeline = make_pipeline(config, make_regressor(model_name, random_state))
    temporal_pipeline.fit(features.loc[temporal.train_indices], y_model[temporal.train_indices])
    temporal_prediction = _inverse_target(
        temporal_pipeline.predict(features.loc[temporal.test_indices]),
        config["target"]["transform"],
    )
    result["temporal"] = regression_metrics(
        y_raw[temporal.test_indices],
        temporal_prediction,
        frame.loc[temporal.test_indices, "channel_id"],
    )

    channel_pipeline = make_pipeline(config, make_regressor(model_name, random_state))
    channel_pipeline.fit(features.loc[channel.train_indices], y_model[channel.train_indices])
    channel_prediction = _inverse_target(
        channel_pipeline.predict(features.loc[channel.test_indices]),
        config["target"]["transform"],
    )
    result["channel_holdout"] = regression_metrics(
        y_raw[channel.test_indices],
        channel_prediction,
        frame.loc[channel.test_indices, "channel_id"],
    )
    return result, temporal_pipeline


def _historical_baseline(
    frame: pd.DataFrame,
    y_raw: np.ndarray,
    holdout: Holdout,
) -> dict[str, float | None]:
    train_median = float(np.median(y_raw[holdout.train_indices]))
    values = pd.to_numeric(
        frame.loc[holdout.test_indices, "prior_views_median_observed"],
        errors="coerce",
    ).fillna(train_median)
    return regression_metrics(
        y_raw[holdout.test_indices],
        values.to_numpy(dtype=float),
        frame.loc[holdout.test_indices, "channel_id"],
    )


def _evaluate_classifier(
    model_name: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    config: dict[str, Any],
    temporal: Holdout,
    channel: Holdout,
) -> tuple[dict[str, Any], Any]:
    random_state = config["validation"]["random_state"]
    result: dict[str, Any] = {"model": model_name, "status": "ok"}
    temporal_pipeline = make_pipeline(config, make_classifier(model_name, random_state))
    temporal_pipeline.fit(features.loc[temporal.train_indices], labels[temporal.train_indices])
    temporal_prediction = temporal_pipeline.predict(features.loc[temporal.test_indices])
    result["temporal"] = classification_metrics(labels[temporal.test_indices], temporal_prediction)
    channel_pipeline = make_pipeline(config, make_classifier(model_name, random_state))
    channel_pipeline.fit(features.loc[channel.train_indices], labels[channel.train_indices])
    channel_prediction = channel_pipeline.predict(features.loc[channel.test_indices])
    result["channel_holdout"] = classification_metrics(
        labels[channel.test_indices], channel_prediction
    )
    return result, temporal_pipeline


def _feature_summary(features: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    numeric, categorical = feature_columns(config)
    numeric_summary: dict[str, Any] = {}
    for column in numeric:
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        numeric_summary[column] = (
            {
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
            }
            if not values.empty
            else None
        )
    categorical_summary = {
        column: sorted(features[column].dropna().astype(str).unique().tolist())[:100]
        for column in categorical
    }
    return {"numeric": numeric_summary, "categorical": categorical_summary}


def _permutation_importance(
    pipeline: Any,
    features: pd.DataFrame,
    target: np.ndarray,
    test_indices: list[int],
    random_state: int,
    scoring: str,
) -> list[dict[str, Any]]:
    measured = permutation_importance(
        pipeline,
        features.loc[test_indices],
        target[test_indices],
        scoring=scoring,
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(importance),
            "importance_std": float(deviation),
        }
        for feature, importance, deviation in zip(
            features.columns,
            measured.importances_mean,
            measured.importances_std,
            strict=True,
        )
    ]
    return sorted(rows, key=lambda item: item["importance_mean"], reverse=True)


def _train_intervals(
    config: dict[str, Any],
    features: pd.DataFrame,
    y_model: np.ndarray,
    y_raw: np.ndarray,
    temporal: Holdout,
) -> tuple[Any, Any, dict[str, float]]:
    interval = config["prediction_interval"]
    random_state = config["validation"]["random_state"]

    def estimator(alpha: float) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
        )

    lower_eval = make_pipeline(config, estimator(interval["lower_quantile"]))
    upper_eval = make_pipeline(config, estimator(interval["upper_quantile"]))
    train_x = features.loc[temporal.train_indices]
    train_y = y_model[temporal.train_indices]
    lower_eval.fit(train_x, train_y)
    upper_eval.fit(train_x, train_y)
    test_x = features.loc[temporal.test_indices]
    low = _inverse_target(lower_eval.predict(test_x), config["target"]["transform"])
    high = _inverse_target(upper_eval.predict(test_x), config["target"]["transform"])
    low, high = np.minimum(low, high), np.maximum(low, high)
    truth = y_raw[temporal.test_indices]
    metrics = {
        "coverage": float(np.mean((truth >= low) & (truth <= high))),
        "mean_width": float(np.mean(high - low)),
        "lower_quantile": float(interval["lower_quantile"]),
        "upper_quantile": float(interval["upper_quantile"]),
    }
    lower_final = make_pipeline(config, estimator(interval["lower_quantile"]))
    upper_final = make_pipeline(config, estimator(interval["upper_quantile"]))
    lower_final.fit(features, y_model)
    upper_final.fit(features, y_model)
    return lower_final, upper_final, metrics


def train_sector_models(
    dataset_path: Path,
    sector: str,
    *,
    task: str = "regression",
    allow_exploratory: bool = False,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    sector = validate_sector(sector)
    if task not in {"regression", "classification"}:
        raise ValueError("task deve ser regression ou classification.")
    config = load_sector_config(sector, models_dir)
    frame = _prepare_frame(dataset_path, sector, task, config)
    gate_reasons = _gate_reasons(frame, config, task)
    if gate_reasons and not allow_exploratory:
        raise DatasetGateError(gate_reasons)

    validation = config["validation"]
    temporal = temporal_holdout(
        frame,
        validation["temporal_test_fraction"],
        min(validation["minimum_test_rows"], max(1, len(frame) // 4)),
    )
    channel = channel_holdout(
        frame,
        validation["channel_test_fraction"],
        validation["random_state"],
    )
    features = _feature_frame(frame, config)
    evaluation_pipelines: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []

    if task == "regression":
        y_raw = pd.to_numeric(frame[config["target"]["column"]], errors="raise").to_numpy(
            dtype=float
        )
        y_model = _transform_target(y_raw, config["target"]["transform"])
        model_names = config["regression_models"] + config.get("optional_regression_models", [])
        for name in model_names:
            try:
                result, fitted = _evaluate_regressor(
                    name,
                    frame,
                    features,
                    y_raw,
                    y_model,
                    config,
                    temporal,
                    channel,
                )
                leaderboard.append(result)
                evaluation_pipelines[name] = fitted
            except ImportError as exc:
                leaderboard.append({"model": name, "status": "skipped", "reason": str(exc)})
            except Exception as exc:  # mantém os outros candidatos comparáveis
                leaderboard.append(
                    {
                        "model": name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        leaderboard.append(
            {
                "model": "historical_channel_median",
                "status": "baseline_only",
                "temporal": _historical_baseline(frame, y_raw, temporal),
                "channel_holdout": _historical_baseline(frame, y_raw, channel),
            }
        )
        successful = [
            item
            for item in leaderboard
            if item["status"] == "ok" and item["model"] != "dummy_median"
        ]
        if not successful:
            raise RuntimeError("Nenhum regressor candidato terminou com sucesso.")
        winner = min(
            successful,
            key=lambda item: (
                item["temporal"]["mae_log"],
                item["channel_holdout"]["mae_log"],
                -(item["temporal"]["spearman"] or -1.0),
            ),
        )
        promotion_checks: dict[str, Any] = {}
        for split_name in ("temporal", "channel_holdout"):
            baseline_scores = [
                item[split_name]["mae_log"]
                for item in leaderboard
                if item["model"] in {"dummy_median", "historical_channel_median"}
                and item.get(split_name)
            ]
            best_baseline = min(baseline_scores) if baseline_scores else None
            candidate_score = winner[split_name]["mae_log"]
            promotion_checks[split_name] = {
                "metric": "mae_log",
                "candidate": candidate_score,
                "best_baseline": best_baseline,
                "passed": best_baseline is not None and candidate_score < best_baseline,
            }
        promotion_policy = config.get("promotion", {})
        required_checks = []
        if promotion_policy.get("require_temporal_baseline_win", True):
            required_checks.append(promotion_checks["temporal"]["passed"])
        if promotion_policy.get("require_channel_baseline_win", True):
            required_checks.append(promotion_checks["channel_holdout"]["passed"])
        beats_baseline = bool(required_checks) and all(required_checks)
        final_model = make_pipeline(
            config,
            make_regressor(winner["model"], validation["random_state"]),
        )
        final_model.fit(features, y_model)
        lower_model = upper_model = None
        interval_metrics: dict[str, Any] = {"enabled": False}
        if config["prediction_interval"].get("enabled"):
            lower_model, upper_model, measured = _train_intervals(
                config, features, y_model, y_raw, temporal
            )
            interval_metrics = {"enabled": True, **measured}
        importance = _permutation_importance(
            evaluation_pipelines[winner["model"]],
            features,
            y_model,
            temporal.test_indices,
            validation["random_state"],
            "neg_mean_absolute_error",
        )
        target_metadata = config["target"]
    else:
        label_column = config["classification"]["label_column"]
        labels = frame[label_column].astype(str).to_numpy()
        model_names = ["dummy_most_frequent"] + config["classification"]["models"]
        for name in model_names:
            try:
                result, fitted = _evaluate_classifier(
                    name, frame, features, labels, config, temporal, channel
                )
                leaderboard.append(result)
                evaluation_pipelines[name] = fitted
            except Exception as exc:
                leaderboard.append(
                    {
                        "model": name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        successful = [
            item
            for item in leaderboard
            if item["status"] == "ok" and item["model"] != "dummy_most_frequent"
        ]
        if not successful:
            raise RuntimeError("Nenhum classificador candidato terminou com sucesso.")
        winner = max(
            successful,
            key=lambda item: (
                item["temporal"]["f1_macro"],
                item["channel_holdout"]["f1_macro"],
            ),
        )
        baseline = next(item for item in leaderboard if item["model"] == "dummy_most_frequent")
        promotion_checks = {
            split_name: {
                "metric": "f1_macro",
                "candidate": winner[split_name]["f1_macro"],
                "best_baseline": baseline[split_name]["f1_macro"],
                "passed": winner[split_name]["f1_macro"] > baseline[split_name]["f1_macro"],
            }
            for split_name in ("temporal", "channel_holdout")
        }
        promotion_policy = config.get("promotion", {})
        required_checks = []
        if promotion_policy.get("require_temporal_baseline_win", True):
            required_checks.append(promotion_checks["temporal"]["passed"])
        if promotion_policy.get("require_channel_baseline_win", True):
            required_checks.append(promotion_checks["channel_holdout"]["passed"])
        beats_baseline = bool(required_checks) and all(required_checks)
        final_model = make_pipeline(
            config,
            make_classifier(winner["model"], validation["random_state"]),
        )
        final_model.fit(features, labels)
        lower_model = upper_model = None
        interval_metrics = {"enabled": False}
        importance = _permutation_importance(
            evaluation_pipelines[winner["model"]],
            features,
            labels,
            temporal.test_indices,
            validation["random_state"],
            "accuracy",
        )
        target_metadata = {
            "column": label_column,
            "metric_name": "performance_band",
            "transform": "none",
        }

    run_id = _run_id(f"{sector}-{task}")
    artifacts_root = models_dir / sector / "artifacts"
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    data_gates_passed = not gate_reasons
    promotable = data_gates_passed and beats_baseline
    is_exploratory = not promotable
    non_promotion_reasons = list(gate_reasons)
    if not beats_baseline:
        non_promotion_reasons.append(
            "O candidato vencedor não superou os baselines em todos os cortes exigidos."
        )
    background_size = min(100, len(features))
    background_rows = features.sample(
        n=background_size,
        random_state=validation["random_state"],
    )
    explanation_background = final_model.named_steps["preprocess"].transform(background_rows)
    bundle = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "sector": sector,
        "task": task,
        "model_name": winner["model"],
        "target": target_metadata,
        "feature_columns": features.columns.tolist(),
        "numeric_features": feature_columns(config)[0],
        "categorical_features": feature_columns(config)[1],
        "point_model": final_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "global_feature_importance": importance,
        "explanation_background": explanation_background,
        "feature_summary": _feature_summary(features, config),
        "serving": config.get("serving", {"max_missing_fraction": 0.5}),
        "promotable": promotable,
        "exploratory": is_exploratory,
    }
    bundle_path = run_dir / "model_bundle.joblib"
    temporary_bundle = bundle_path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary_bundle)
    temporary_bundle.replace(bundle_path)

    metrics = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "sector": sector,
        "task": task,
        "winner": winner,
        "beats_baseline": beats_baseline,
        "promotion_checks": promotion_checks,
        "prediction_interval": interval_metrics,
        "leaderboard": leaderboard,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "feature_importance.json", importance)
    manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "created_at": utc_now(),
        "sector": sector,
        "task": task,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "rows_used": len(frame),
            "channels": int(frame["channel_id"].nunique()),
        },
        "validation": {
            "temporal": temporal.description,
            "channel_holdout": channel.description,
        },
        "winner": winner["model"],
        "beats_baseline": beats_baseline,
        "promotion_checks": promotion_checks,
        "exploratory": is_exploratory,
        "promotable": promotable,
        "non_promotion_reasons": non_promotion_reasons,
        "blocking_reasons_overridden": gate_reasons if gate_reasons and allow_exploratory else [],
        "warning": (
            "Resultado exploratório; não usar como evidência de desempenho."
            if is_exploratory
            else None
        ),
    }
    write_json(run_dir / "training_manifest.json", manifest)
    pointer = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "bundle": (Path(run_id) / "model_bundle.joblib").as_posix(),
        "manifest": (Path(run_id) / "training_manifest.json").as_posix(),
        "metrics": (Path(run_id) / "metrics.json").as_posix(),
        "promotable": promotable,
    }
    if promotable:
        write_json(artifacts_root / "latest.json", pointer)
    else:
        write_json(artifacts_root / "latest_exploratory.json", pointer)
    return {"run_dir": str(run_dir), "manifest": manifest, "metrics": metrics}
