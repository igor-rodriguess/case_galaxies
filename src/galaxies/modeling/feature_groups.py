"""Grupos de features para medir a contribuição de cada fonte."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

FEATURE_SET_NAMES = (
    "metadata",
    "channel_history",
    "audiovisual",
    "metadata_channel",
    "metadata_audiovisual",
    "combined",
)

_CHANNEL_PREFIXES = ("channel_", "prior_")
_CHANNEL_EXACT = {"days_since_previous_upload"}
_AUDIOVISUAL_PREFIXES = (
    "vlm_",
    "people_",
    "on_screen_",
    "scene_",
    "speech_",
    "transcript_",
    "sound_",
    "evidence_",
    "uncertainty_",
    "beauty_",
    "fashion_",
    "automotive_",
)
_AUDIOVISUAL_EXACT = {
    "faces_visible",
    "product_in_focus",
    "brand_legible",
    "framing",
    "camera_movement",
    "motion_intensity",
    "audio_language",
    "music_present",
    "opening_type",
    "product_prominence",
    "call_to_action_present",
}


def feature_family(column: str) -> str:
    """Classifica uma coluna como metadado, histórico ou audiovisual."""
    if column in _CHANNEL_EXACT or column.startswith(_CHANNEL_PREFIXES):
        return "channel_history"
    if column in _AUDIOVISUAL_EXACT or column.startswith(_AUDIOVISUAL_PREFIXES):
        return "audiovisual"
    return "metadata"


def feature_sets(config: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Gera subconjuntos sem alterar a configuração original do setor."""
    numeric = list(config["features"]["numeric"])
    categorical = list(config["features"]["categorical"])

    def select(families: set[str]) -> dict[str, list[str]]:
        return {
            "numeric": [name for name in numeric if feature_family(name) in families],
            "categorical": [
                name for name in categorical if feature_family(name) in families
            ],
        }

    return {
        "metadata": select({"metadata"}),
        "channel_history": select({"channel_history"}),
        "audiovisual": select({"audiovisual"}),
        "metadata_channel": select({"metadata", "channel_history"}),
        "metadata_audiovisual": select({"metadata", "audiovisual"}),
        "combined": select({"metadata", "channel_history", "audiovisual"}),
    }


def config_for_feature_set(
    config: dict[str, Any],
    feature_set: str,
) -> dict[str, Any]:
    """Copia a configuração e mantém apenas as features do conjunto pedido."""
    groups = feature_sets(config)
    if feature_set not in groups:
        raise ValueError(
            f"Conjunto de features desconhecido: {feature_set}. "
            f"Opções: {', '.join(FEATURE_SET_NAMES)}."
        )
    scoped = deepcopy(config)
    scoped["features"] = groups[feature_set]
    return scoped
