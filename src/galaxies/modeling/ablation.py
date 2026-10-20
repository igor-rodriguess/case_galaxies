"""Avaliação exploratória por grupos de features em uma amostra comum."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_sector_config, validate_sector
from .feature_groups import FEATURE_SET_NAMES, config_for_feature_set, feature_family
from .splits import channel_holdout, temporal_holdout
from .trainer import (
    _evaluate_regressor,
    _feature_frame,
    _historical_baseline,
    _prepare_frame,
    _transform_target,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def evaluate_feature_groups(dataset_path: Path, sector: str) -> dict[str, Any]:
    """Compara grupos na mesma amostra de vídeos com VLM disponível."""
    sector = validate_sector(sector)
    base_config = load_sector_config(sector)
    frame = _prepare_frame(dataset_path, sector, "regression", base_config)
    vlm = frame.get("vlm_available", pd.Series(False, index=frame.index)).fillna(False)
    frame = frame.loc[vlm.astype(bool)].copy().reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError(
            f"A ablação exige pelo menos 30 vídeos com VLM; {sector} possui {len(frame)}."
        )

    validation = base_config["validation"]
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
    y_raw = pd.to_numeric(
        frame[base_config["target"]["column"]], errors="raise"
    ).to_numpy(dtype=float)
    y_model = _transform_target(y_raw, base_config["target"]["transform"])
    prior_count = pd.to_numeric(frame.get("prior_video_count"), errors="coerce").fillna(0)
    history_valid = frame.get(
        "channel_history_temporally_valid", pd.Series(False, index=frame.index)
    )
    invalid_history_rows = int(((prior_count > 0) & (history_valid != True)).sum())  # noqa: E712
    fixed_horizon = frame.get(
        "target_is_fixed_horizon", pd.Series(False, index=frame.index)
    ).fillna(False)
    ages = pd.to_numeric(frame.get("target_age_days"), errors="coerce").dropna()
    baseline = {
        "temporal": _historical_baseline(frame, y_raw, temporal),
        "channel_holdout": _historical_baseline(frame, y_raw, channel),
    }

    evaluations: dict[str, Any] = {}
    model_names = list(base_config["regression_models"])
    for feature_set in FEATURE_SET_NAMES:
        config = config_for_feature_set(base_config, feature_set)
        features = _feature_frame(frame, config)
        uses_channel_history = any(
            feature_family(column) == "channel_history" for column in features.columns
        )
        leakage_safe = not uses_channel_history or invalid_history_rows == 0
        leaderboard: list[dict[str, Any]] = []
        for model_name in model_names:
            try:
                result, _ = _evaluate_regressor(
                    model_name,
                    frame,
                    features,
                    y_raw,
                    y_model,
                    config,
                    temporal,
                    channel,
                )
                leaderboard.append(result)
            except Exception as exc:
                leaderboard.append(
                    {
                        "model": model_name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        successful = [
            item
            for item in leaderboard
            if item["status"] == "ok" and item["model"] != "dummy_median"
        ]
        if not successful:
            raise RuntimeError(f"Nenhum candidato terminou em {sector}/{feature_set}.")
        winner_temporal = min(
            successful,
            key=lambda item: (
                item["temporal"]["mae_log"],
                item["channel_holdout"]["mae_log"],
                -(item["temporal"]["spearman"] or -1.0),
            ),
        )
        winner_channel = min(
            successful,
            key=lambda item: (
                item["channel_holdout"]["mae_log"],
                item["temporal"]["mae_log"],
                -(item["channel_holdout"]["spearman"] or -1.0),
            ),
        )
        dummy = next(item for item in leaderboard if item["model"] == "dummy_median")
        evaluations[feature_set] = {
            "feature_count": len(features.columns),
            "uses_channel_history": uses_channel_history,
            "leakage_safe": leakage_safe,
            "numeric_features": config["features"]["numeric"],
            "categorical_features": config["features"]["categorical"],
            "winner_temporal": winner_temporal,
            "winner_channel_holdout": winner_channel,
            "skill_vs_dummy": {
                "temporal_mae_log": 1.0
                - winner_temporal["temporal"]["mae_log"]
                / dummy["temporal"]["mae_log"],
                "channel_holdout_mae_log": 1.0
                - winner_channel["channel_holdout"]["mae_log"]
                / dummy["channel_holdout"]["mae_log"],
            },
            "leaderboard": leaderboard,
        }

    best_temporal_unrestricted = min(
        evaluations,
        key=lambda name: evaluations[name]["winner_temporal"]["temporal"]["mae_log"],
    )
    best_channel_unrestricted = min(
        evaluations,
        key=lambda name: evaluations[name]["winner_channel_holdout"]["channel_holdout"][
            "mae_log"
        ],
    )
    safe_sets = [name for name in evaluations if evaluations[name]["leakage_safe"]]
    best_temporal = min(
        safe_sets,
        key=lambda name: evaluations[name]["winner_temporal"]["temporal"]["mae_log"],
    )
    best_channel = min(
        safe_sets,
        key=lambda name: evaluations[name]["winner_channel_holdout"]["channel_holdout"][
            "mae_log"
        ],
    )
    return {
        "schema_version": "0.1.0",
        "evaluated_at": utc_now(),
        "sector": sector,
        "exploratory": True,
        "warning": (
            "O alvo não está em horizonte fixo; resultados não promovem modelo."
        ),
        "dataset": str(dataset_path),
        "common_sample": {
            "rows": len(frame),
            "channels": int(frame["channel_id"].nunique()),
            "requires_vlm": True,
        },
        "data_quality": {
            "fixed_horizon_target_rows": int(fixed_horizon.astype(bool).sum()),
            "temporally_invalid_history_rows": invalid_history_rows,
            "target_age_days": {
                "minimum": float(ages.min()) if not ages.empty else None,
                "median": float(ages.median()) if not ages.empty else None,
                "maximum": float(ages.max()) if not ages.empty else None,
            },
        },
        "validation": {
            "temporal": temporal.description,
            "channel_holdout": channel.description,
        },
        "historical_channel_baseline": baseline,
        "best_feature_set_temporal": best_temporal,
        "best_feature_set_channel_holdout": best_channel,
        "best_feature_set_temporal_unrestricted": best_temporal_unrestricted,
        "best_feature_set_channel_holdout_unrestricted": best_channel_unrestricted,
        "feature_sets": evaluations,
    }
