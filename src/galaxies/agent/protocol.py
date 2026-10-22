"""Protocolo congelado do agente: rótulo viral, cofre, populações e features.

Nada aqui pode ser alterado pelo agente. O gerador de hipóteses escolhe apenas
entre as opções do catálogo (``spec.py``); o rótulo, o cofre e a população de
avaliação ficam fixos por ``PROTOCOL_VERSION``. Mudar qualquer constante deste
arquivo exige subir a versão, porque invalida a comparação com o histórico.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.prospective import exclude_prospective

PROTOCOL_VERSION = "viral-intra-canal-v1"

# Só vídeos com pelo menos 30 dias entram na avaliação: abaixo disso as views
# ainda estão subindo e o rótulo vira ruído de idade (diagnóstico de 18/10/2026).
MIN_EVAL_AGE_DAYS = 30
# "Viral" = top 10% do próprio canal, depois do ajuste por idade.
VIRAL_QUANTILE = 0.90
# Um canal por setor fica trancado; só é avaliado com --abrir-cofre.
VAULT_CHANNELS_PER_SECTOR = 1
VAULT_SEED = 2026
# Limite atual de duração de Shorts.
SHORT_MAX_SECONDS = 180
# Meta pedida para o projeto, medida em AUC intra-canal.
GOAL_AUC = 0.85

SECTORS = ("automotivo", "moda", "beleza")

# Colunas que existem no mapa de features mas só são conhecidas depois da
# publicação (ou são o próprio alvo). Nenhuma delas pode virar feature.
FORBIDDEN_EXACT = frozenset(
    {
        "video_id",
        "channel_id",
        "collected_at",
        "published_at",
        "like_count_snapshot",
        "comment_count_snapshot",
        "channel_subscriber_count_snapshot",
        "channel_video_count_snapshot",
        "channel_history_temporally_valid",
        "vlm_result_path",
        "youtube_record_path",
        "viral",
        "adjusted_performance",
        "in_vault",
        "segment",
        # Posição do vídeo na série coletada é um índice de tempo: dentro do
        # canal ela reproduz a idade, e na hora da previsão todo vídeo novo é
        # o último da série. Primeira rodada mostrou o atalho (18/10/2026).
        "prior_video_count",
    }
)
FORBIDDEN_PREFIXES = (
    "target_",
    "prior_views_",
    "prior_likes_",
    "prior_comments_",
)

# Grupos de features disponíveis ANTES da publicação.
METADATA_NUMERIC = [
    "duration_seconds",
    "log_duration",
    "is_short",
    "title_character_count",
    "description_character_count",
    "tag_count",
    "publish_hour_utc",
]
METADATA_CATEGORICAL = ["category_id", "publish_weekday_utc"]

# Cadência vem só das datas de publicação dos vídeos anteriores, que são
# conhecidas no momento da previsão. Desempenho passado (views/likes) não entra.
CADENCE_NUMERIC = [
    "days_since_previous_upload",
    "prior_upload_interval_median_days",
]

# Limite para considerar que uma nota carrega a idade do vídeo (|rho| de
# Spearman dentro do canal). Acima disso o experimento não pode ser promovido.
AGE_SHORTCUT_LIMIT = 0.20

AUDIOVISUAL_NUMERIC = [
    "people_count_estimate",
    "on_screen_text_count",
    "scene_change_count",
    "speech_start_seconds",
    "transcript_excerpt_word_count",
    "sound_effect_count",
]
AUDIOVISUAL_CATEGORICAL = [
    "people_present",
    "faces_visible",
    "product_in_focus",
    "brand_legible",
    "framing",
    "camera_movement",
    "motion_intensity",
    "speech_present",
    "music_present",
    "opening_type",
    "product_prominence",
    "call_to_action_present",
    "audio_language",
    "beauty_swatch_present",
    "beauty_skin_application_present",
    "beauty_before_after_present",
    "fashion_try_on_present",
    "fashion_full_body_present",
    "fashion_garment_detail_present",
    "automotive_car_closeup_present",
    "automotive_engine_sound_present",
    "automotive_driving_pov_present",
    "automotive_interior_present",
]

AUDIOVISUAL_DERIVED_NUMERIC = [
    "early_speech_signal",
    "scene_changes_per_second",
    "screen_text_per_second",
    "sound_effects_per_second",
    "multimodal_signal_count",
]

TRANSCRIPT_NUMERIC = [
    "transcript_character_count",
    "transcript_unique_word_ratio",
    "transcript_words_per_second",
    "transcript_question_mark_count",
    "transcript_exclamation_mark_count",
    "transcript_number_token_count",
]
TRANSCRIPT_CATEGORICAL = [
    "transcript_available",
    "transcript_second_person_cue",
    "transcript_call_to_action_cue",
]

FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "metadata": {"numeric": METADATA_NUMERIC, "categorical": METADATA_CATEGORICAL},
    "cadence": {"numeric": CADENCE_NUMERIC, "categorical": []},
    "audiovisual": {"numeric": AUDIOVISUAL_NUMERIC, "categorical": AUDIOVISUAL_CATEGORICAL},
    "audiovisual_derived": {"numeric": AUDIOVISUAL_DERIVED_NUMERIC, "categorical": []},
    "transcript_stats": {"numeric": TRANSCRIPT_NUMERIC, "categorical": TRANSCRIPT_CATEGORICAL},
}


def is_forbidden(column: str) -> bool:
    return column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)


# --- Taxonomia fechada para as saídas livres do VLM -------------------------

_TAXONOMY_RULES: dict[str, list[tuple[str, str]]] = {
    # A ordem importa: a primeira regra que casar define a categoria.
    "framing": [
        (r"pov|first.?person|primeira pessoa", "pov"),
        (r"extreme|macro|detalhe|detail", "extreme_close_up"),
        (r"medium.?close|close.?medium|médio.?close", "medium_close_up"),
        (r"close|closeup|primeiro plano", "close_up"),
        (r"wide|long|full|aberto|geral|plano geral|corpo inteiro", "wide"),
        (r"medium|médio|medio|mid|american|americano", "medium"),
    ],
    "camera_movement": [
        (r"static|estátic|estatic|fixed|fixa|tripod|tripé|none|sem movimento", "static"),
        (r"hand.?held|handheld|na mão|selfie", "handheld"),
        (r"zoom", "zoom"),
        (r"pan|tilt|track|dolly|travel|orbit|follow|panorâm", "moving"),
        (r"cut|corte|jump|transition|transiç", "cut"),
        (r"dynamic|dinâmic|dinamic|fast|rápid|shak", "dynamic"),
    ],
    "motion_intensity": [
        (r"high|alta|alto|intens|fast", "high"),
        (r"medium|média|medio|médio|moderate|moderad", "medium"),
        (r"low|baixa|baixo|static|estátic|slow|lent|none", "low"),
    ],
    "audio_language": [
        (r"^pt|portug", "pt"),
        (r"^en|english|inglês", "en"),
    ],
}


def normalize_vlm_value(column: str, value: object) -> object:
    """Mapeia texto livre do VLM para uma categoria fechada (ou ``other``)."""
    if column not in _TAXONOMY_RULES or value is None:
        return value
    if isinstance(value, float) and np.isnan(value):
        return value
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    for pattern, label in _TAXONOMY_RULES[column]:
        if re.search(pattern, text):
            return label
    return "other"


def apply_vlm_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in _TAXONOMY_RULES:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(
                lambda value, col=column: normalize_vlm_value(col, value)
            )
    return normalized


# --- Carga e rótulo -----------------------------------------------------------


@dataclass(frozen=True)
class AgentDataset:
    """Base pronta para o agente: rótulo congelado e cofre marcados."""

    frame: pd.DataFrame
    source_path: Path
    fingerprint: str
    vault_channels: tuple[str, ...]
    notes: list[str] = field(default_factory=list)

    def evaluable(self, track: str) -> pd.DataFrame:
        """Linhas fora do cofre que entram na avaliação da trilha."""
        rows = self.frame[self.frame["in_population"] & ~self.frame["in_vault"]]
        if track == "vlm":
            rows = rows[rows["vlm_available"]]
        return rows

    def vault(self, track: str) -> pd.DataFrame:
        rows = self.frame[self.frame["in_population"] & self.frame["in_vault"]]
        if track == "vlm":
            rows = rows[rows["vlm_available"]]
        return rows


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


AGE_SLOPE_SHRINKAGE_ROWS = 30


def age_adjusted_performance(frame: pd.DataFrame) -> pd.Series:
    """Resíduo de log1p(views) após remover a curva de idade DENTRO de cada canal.

    Como o rótulo compara vídeos do mesmo canal, a idade precisa ser removida
    dentro do canal: uma curva única por setor deixava os vídeos mais novos de
    cada canal com cara de virais (primeira rodada, 18/10/2026). A inclinação
    de cada canal é encolhida para a do setor quando o canal tem poucas linhas.

    Usa somente views e idade. É uma definição de rótulo, não uma feature: é
    calculada uma única vez sobre a população avaliável e fica congelada.
    """
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    log_views = np.log1p(frame["target_view_count"].astype(float))
    log_age = np.log1p(frame["target_age_days"].astype(float))
    channel = frame["channel_id"]
    views_centered = log_views - log_views.groupby(channel).transform("mean")
    age_centered = log_age - log_age.groupby(channel).transform("mean")
    for _, group in frame.groupby("sector"):
        x_sector = age_centered.loc[group.index]
        y_sector = views_centered.loc[group.index]
        denominator = float((x_sector**2).sum())
        sector_slope = float((x_sector * y_sector).sum() / denominator) if denominator else 0.0
        for _, channel_rows in group.groupby("channel_id"):
            x = age_centered.loc[channel_rows.index]
            y = views_centered.loc[channel_rows.index]
            spread = float((x**2).sum())
            channel_slope = float((x * y).sum() / spread) if spread else sector_slope
            weight = len(channel_rows) / (len(channel_rows) + AGE_SLOPE_SHRINKAGE_ROWS)
            slope = weight * channel_slope + (1 - weight) * sector_slope
            result.loc[channel_rows.index] = y - slope * x
    return result


def fit_age_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajusta a normalização de idade usando apenas uma janela de treino.

    ``age_adjusted_performance`` é adequado para congelar o rótulo de uma
    população completa. Em um backtest temporal, porém, estimar suas médias e
    inclinações com o período de teste faria o passado depender do futuro.
    Esta função produz parâmetros por canal a partir de ``frame`` para que uma
    janela seguinte seja pontuada sem acessar suas views durante o ajuste.
    """
    required = {"channel_id", "sector", "target_view_count", "target_age_days"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para ajuste temporal: {sorted(missing)}")
    rows = frame.copy()
    rows["_log_views"] = np.log1p(pd.to_numeric(rows["target_view_count"], errors="coerce"))
    rows["_log_age"] = np.log1p(pd.to_numeric(rows["target_age_days"], errors="coerce"))
    rows = rows.dropna(subset=["channel_id", "sector", "_log_views", "_log_age"])
    if rows.empty:
        return pd.DataFrame(
            columns=["channel_id", "sector", "mean_log_views", "mean_log_age", "age_slope"]
        )

    rows["_mean_log_views"] = rows.groupby("channel_id")["_log_views"].transform("mean")
    rows["_mean_log_age"] = rows.groupby("channel_id")["_log_age"].transform("mean")
    rows["_views_centered"] = rows["_log_views"] - rows["_mean_log_views"]
    rows["_age_centered"] = rows["_log_age"] - rows["_mean_log_age"]
    sector_slopes: dict[object, float] = {}
    for sector, group in rows.groupby("sector", sort=False):
        denominator = float((group["_age_centered"] ** 2).sum())
        sector_slopes[sector] = (
            float((group["_age_centered"] * group["_views_centered"]).sum() / denominator)
            if denominator
            else 0.0
        )

    parameters: list[dict[str, object]] = []
    for (sector, channel), group in rows.groupby(["sector", "channel_id"], sort=False):
        denominator = float((group["_age_centered"] ** 2).sum())
        sector_slope = sector_slopes[sector]
        channel_slope = (
            float((group["_age_centered"] * group["_views_centered"]).sum() / denominator)
            if denominator
            else sector_slope
        )
        weight = len(group) / (len(group) + AGE_SLOPE_SHRINKAGE_ROWS)
        parameters.append(
            {
                "channel_id": channel,
                "sector": sector,
                "mean_log_views": float(group["_log_views"].mean()),
                "mean_log_age": float(group["_log_age"].mean()),
                "age_slope": weight * channel_slope + (1 - weight) * sector_slope,
            }
        )
    return pd.DataFrame(parameters)


def apply_age_adjustment(frame: pd.DataFrame, parameters: pd.DataFrame) -> pd.Series:
    """Aplica parâmetros de idade já ajustados, sem refit na janela alvo."""
    values = frame[["channel_id", "sector", "target_view_count", "target_age_days"]].merge(
        parameters,
        on=["channel_id", "sector"],
        how="left",
        validate="many_to_one",
    )
    log_views = np.log1p(pd.to_numeric(values["target_view_count"], errors="coerce"))
    log_age = np.log1p(pd.to_numeric(values["target_age_days"], errors="coerce"))
    adjusted = (log_views - values["mean_log_views"]) - values["age_slope"] * (
        log_age - values["mean_log_age"]
    )
    return pd.Series(adjusted.to_numpy(), index=frame.index, dtype=float)


def temporal_as_of_labels(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    quantile: float = VIRAL_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria rótulos temporais sem calibrar idade ou limiar com o futuro.

    Os parâmetros de idade e o limiar por canal são estimados no treino. O
    teste recebe o mesmo ajuste e o mesmo limiar. Isso representa a pergunta
    operacional: com tudo que se conhecia até a data de corte, quais próximos
    vídeos superariam o padrão histórico daquele canal?
    """
    parameters = fit_age_adjustment(train)
    train_result = train.copy()
    test_result = test.copy()
    train_result["adjusted_performance"] = apply_age_adjustment(train_result, parameters)
    test_result["adjusted_performance"] = apply_age_adjustment(test_result, parameters)
    thresholds = train_result.groupby("channel_id")["adjusted_performance"].quantile(quantile)
    train_result["viral"] = (
        train_result["adjusted_performance"] >= train_result["channel_id"].map(thresholds)
    ).astype(int)
    test_result["viral"] = (
        test_result["adjusted_performance"] >= test_result["channel_id"].map(thresholds)
    ).astype(int)
    return train_result, test_result


def within_channel_age_correlation(frame: pd.DataFrame, score_column: str) -> float:
    """Spearman médio (ponderado por linhas) entre a nota e a idade, dentro do canal.

    Perto de 0 é o esperado. Longe de 0 indica que a nota está "adivinhando" a
    idade do vídeo em vez da qualidade do criativo.
    """
    total = weighted = 0.0
    for _, group in frame.groupby("channel_id"):
        if len(group) < 5 or group[score_column].nunique() < 2:
            continue
        rho = group[score_column].rank().corr(group["target_age_days"].rank())
        if pd.notna(rho):
            weighted += rho * len(group)
            total += len(group)
    return float(weighted / total) if total else float("nan")


def viral_label(frame: pd.DataFrame, performance: pd.Series) -> pd.Series:
    """1 quando o vídeo está no top 10% do próprio canal (após ajuste de idade)."""
    thresholds = performance.groupby(frame["channel_id"]).transform(
        lambda values: values.quantile(VIRAL_QUANTILE)
    )
    return (performance >= thresholds).astype(int)


def choose_vault_channels(frame: pd.DataFrame) -> tuple[str, ...]:
    rng = np.random.default_rng(VAULT_SEED)
    chosen: list[str] = []
    for sector in sorted(frame["sector"].unique()):
        channels = sorted(frame.loc[frame["sector"] == sector, "channel_id"].unique())
        if len(channels) <= VAULT_CHANNELS_PER_SECTOR + 2:
            continue  # setor pequeno demais para trancar canal
        picks = rng.choice(channels, size=VAULT_CHANNELS_PER_SECTOR, replace=False)
        chosen.extend(str(channel) for channel in picks)
    return tuple(sorted(chosen))


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["duration_seconds"] = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    frame["log_duration"] = np.log1p(frame["duration_seconds"])
    frame["is_short"] = (frame["duration_seconds"] <= SHORT_MAX_SECONDS).astype(float)
    frame.loc[frame["duration_seconds"].isna(), "is_short"] = np.nan
    frame["segment"] = np.where(frame["is_short"] == 1.0, "short", "long")
    frame["vlm_available"] = frame.get("vlm_available", False).fillna(False).astype(bool)

    def numeric_column(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    interval = numeric_column("vlm_interval_end_seconds")
    safe_interval = interval.where(interval > 0)
    scene_changes = numeric_column("scene_change_count")
    screen_text = numeric_column("on_screen_text_count")
    sound_effects = numeric_column("sound_effect_count")
    speech_start = numeric_column("speech_start_seconds")

    def binary_signal(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        normalized = frame[column].map(
            lambda value: str(value).strip().lower() if pd.notna(value) else None
        )
        return normalized.map(
            lambda value: (
                1.0
                if value in {"true", "1", "yes", "sim"}
                else (0.0 if value in {"false", "0", "no", "não", "nao"} else np.nan)
            )
        )

    frame["early_speech_signal"] = np.where(
        binary_signal("speech_present").eq(1.0) & speech_start.le(1.0), 1.0, 0.0
    )
    frame.loc[~frame["vlm_available"], "early_speech_signal"] = np.nan
    frame["scene_changes_per_second"] = scene_changes / safe_interval
    frame["screen_text_per_second"] = screen_text / safe_interval
    frame["sound_effects_per_second"] = sound_effects / safe_interval
    signal_columns = [
        "people_present",
        "product_in_focus",
        "speech_present",
        "music_present",
        "call_to_action_present",
    ]
    signals = pd.concat([binary_signal(column) for column in signal_columns], axis=1)
    frame["multimodal_signal_count"] = signals.sum(axis=1, min_count=1)

    for column in METADATA_CATEGORICAL + AUDIOVISUAL_CATEGORICAL + TRANSCRIPT_CATEGORICAL:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: value if value is None or pd.isna(value) else str(value)
            )
        else:
            frame[column] = None
    for column in (
        METADATA_NUMERIC
        + CADENCE_NUMERIC
        + AUDIOVISUAL_NUMERIC
        + AUDIOVISUAL_DERIVED_NUMERIC
        + TRANSCRIPT_NUMERIC
    ):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_dataset(
    raw: pd.DataFrame,
    source_path: Path,
    fingerprint: str,
    *,
    min_eval_age_days: int = MIN_EVAL_AGE_DAYS,
    vault_channels: tuple[str, ...] | None = None,
) -> AgentDataset:
    frame = prepare_frame(raw)
    population = (frame["target_age_days"] >= min_eval_age_days) & frame[
        "target_view_count"
    ].notna()
    frame["in_population"] = population
    frame["adjusted_performance"] = np.nan
    frame["viral"] = -1
    labelled = frame[population]
    performance = age_adjusted_performance(labelled)
    frame.loc[labelled.index, "adjusted_performance"] = performance
    frame.loc[labelled.index, "viral"] = viral_label(labelled, performance)
    vault = vault_channels or choose_vault_channels(frame[population])
    frame["in_vault"] = frame["channel_id"].isin(vault)
    labelled = frame[population]
    label_age_rho = within_channel_age_correlation(
        labelled.assign(_label=labelled["viral"].astype(float)), "_label"
    )
    notes = [
        f"{int(population.sum())} de {len(frame)} vídeos têm {min_eval_age_days}+ dias e entram na avaliação.",
        f"Rótulo: top {round((1 - VIRAL_QUANTILE) * 100)}% do canal após ajuste de idade dentro do canal.",
        f"Correlação do rótulo com a idade dentro do canal: {label_age_rho:+.3f} (esperado perto de 0).",
        f"Cofre: {len(vault)} canal(is) trancado(s), um por setor.",
    ]
    return AgentDataset(frame, source_path, fingerprint, vault, notes)


def load_dataset(path: Path) -> AgentDataset:
    # Vídeos do período prospectivo são a validação final e nunca entram aqui.
    raw = exclude_prospective(pd.read_json(path, lines=True))
    return build_dataset(raw, path, file_fingerprint(path))


def latest_feature_map(modeling_runs_dir: Path) -> Path:
    candidates = sorted(
        modeling_runs_dir.glob("*/feature_map.jsonl"),
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum feature_map.jsonl em {modeling_runs_dir}. "
            "Rode scripts/build_modeling_dataset.py antes."
        )
    return candidates[-1]
