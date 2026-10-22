"""Coortes e catálogo pré-registrado para comparar protocolos VLM.

Os VLMs do projeto foram produzidos com janelas, prompts e provedores distintos.
Tratá-los como uma única população permitiria que o classificador aprendesse o
protocolo de coleta (que também mudou a prevalência de casos) em vez do conteúdo.
Este módulo mantém cada protocolo isolado e define a busca antes de olhar o cofre.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .protocol import AgentDataset
from .spec import ExperimentSpec


@dataclass(frozen=True)
class VLMProtocolCohort:
    name: str
    prompt_version: str
    model_version: str
    end_seconds: float
    description: str


PRIMARY_VLM_COHORTS = (
    VLMProtocolCohort(
        name="gemini_v010_3s",
        prompt_version="galaxies-opening-v0.1.0",
        model_version="gemini-3.5-flash-lite",
        end_seconds=3.0,
        description="Gemini Flash-Lite, taxonomia v0.1 e abertura de 3 segundos.",
    ),
    VLMProtocolCohort(
        name="pegasus_v020_15s_case_control",
        prompt_version="galaxies-opening-v0.2.0-closed-taxonomy",
        model_version="pegasus1.6",
        end_seconds=15.0,
        description="TwelveLabs Pegasus, taxonomia fechada v0.2 e janela de 15 segundos.",
    ),
)


def isolate_protocol(dataset: AgentDataset, cohort: VLMProtocolCohort) -> AgentDataset:
    """Mantém disponível para a trilha VLM somente o protocolo selecionado."""
    frame = dataset.frame.copy()
    interval = frame["vlm_interval_end_seconds"].astype(float)
    selected = (
        frame["vlm_available"]
        & frame["vlm_prompt_version"].eq(cohort.prompt_version)
        & frame["vlm_model_version"].eq(cohort.model_version)
        & np.isclose(interval, cohort.end_seconds, equal_nan=False)
    )
    frame["vlm_available"] = selected
    frame["vlm_protocol_cohort"] = None
    frame.loc[selected, "vlm_protocol_cohort"] = cohort.name
    digest = hashlib.sha256(f"{dataset.fingerprint}:{cohort.name}".encode()).hexdigest()[:16]
    rows = int((selected & frame["in_population"]).sum())
    notes = [*dataset.notes, f"Coorte {cohort.name}: {rows} VLMs maduros."]
    return AgentDataset(frame, dataset.source_path, digest, dataset.vault_channels, notes)


def harmonize_primary_protocols(dataset: AgentDataset) -> AgentDataset:
    """Une somente as coortes primárias e preserva o domínio para estratificação.

    A coluna ``vlm_protocol_cohort`` nunca entra como feature. Ela serve para
    definir pares de ranking e métricas dentro de canal + protocolo.
    """
    frame = dataset.frame.copy()
    frame["vlm_protocol_cohort"] = None
    selected = pd.Series(False, index=frame.index)
    for cohort in PRIMARY_VLM_COHORTS:
        interval = frame["vlm_interval_end_seconds"].astype(float)
        cohort_mask = (
            frame["vlm_available"]
            & frame["vlm_prompt_version"].eq(cohort.prompt_version)
            & frame["vlm_model_version"].eq(cohort.model_version)
            & np.isclose(interval, cohort.end_seconds, equal_nan=False)
        )
        frame.loc[cohort_mask, "vlm_protocol_cohort"] = cohort.name
        selected |= cohort_mask
    frame["vlm_available"] = selected
    digest = hashlib.sha256(
        f"{dataset.fingerprint}:harmonized-primary-protocols".encode()
    ).hexdigest()[:16]
    rows = int((selected & frame["in_population"]).sum())
    notes = [*dataset.notes, f"Coorte harmonizada: {rows} VLMs maduros em protocolos primários."]
    return AgentDataset(frame, dataset.source_path, digest, dataset.vault_channels, notes)


def _spec(
    name: str,
    *,
    model: str,
    audiovisual: bool,
    complexity: str = "low",
    pooling: str = "pooled",
    train_label: str = "top10",
    taxonomy: str = "normalized",
    derived: bool = False,
    transcript: bool = False,
) -> tuple[str, ExperimentSpec]:
    groups = ["metadata", "cadence"]
    if audiovisual:
        groups.append("audiovisual")
    if derived:
        groups.append("audiovisual_derived")
    if transcript:
        groups.append("transcript_stats")
    spec = ExperimentSpec(
        hypothesis=f"Benchmark pré-registrado {name}",
        rationale="Compara a contribuição audiovisual nos mesmos vídeos e dobras LOCO.",
        expected_effect="Aumentar ordenação intra-canal sem criar atalho de idade.",
        track="vlm",
        feature_groups=groups,
        pooling=pooling,
        format_strategy="joint",
        model=model,
        complexity=complexity,
        train_min_age_days=30,
        train_label=train_label,
        evaluation_mode="new_channel",
        vlm_taxonomy=taxonomy,
    )
    return name, spec


def preregistered_protocol_specs() -> tuple[tuple[str, ExperimentSpec], ...]:
    """Espaço pequeno, fixo e suficiente para ablação e famílias principais."""
    return (
        _spec("reference_metadata_logistic", model="logistic", audiovisual=False),
        _spec("metadata_hist_gb", model="hist_gb", audiovisual=False),
        _spec("metadata_extra_trees", model="extra_trees", audiovisual=False),
        _spec("metadata_catboost", model="catboost", audiovisual=False),
        _spec("metadata_catboost_ranker", model="catboost_ranker", audiovisual=False),
        _spec("metadata_mlp", model="mlp", audiovisual=False),
        _spec("vlm_logistic", model="logistic", audiovisual=True),
        _spec("vlm_hist_gb", model="hist_gb", audiovisual=True),
        _spec("vlm_extra_trees", model="extra_trees", audiovisual=True),
        _spec("vlm_catboost", model="catboost", audiovisual=True),
        _spec("vlm_catboost_ranker", model="catboost_ranker", audiovisual=True),
        _spec("vlm_mlp", model="mlp", audiovisual=True),
        _spec("vlm_catboost_raw", model="catboost", audiovisual=True, taxonomy="raw"),
        _spec(
            "vlm_catboost_per_sector",
            model="catboost",
            audiovisual=True,
            pooling="per_sector",
        ),
        _spec(
            "vlm_catboost_clean_contrast",
            model="catboost",
            audiovisual=True,
            train_label="clean20vs50",
        ),
        _spec(
            "vlm_derived_catboost",
            model="catboost",
            audiovisual=True,
            derived=True,
        ),
        _spec(
            "vlm_derived_ranker",
            model="catboost_ranker",
            audiovisual=True,
            derived=True,
        ),
        _spec(
            "vlm_transcript_catboost",
            model="catboost",
            audiovisual=True,
            derived=True,
            transcript=True,
        ),
        _spec(
            "vlm_transcript_ranker",
            model="catboost_ranker",
            audiovisual=True,
            derived=True,
            transcript=True,
        ),
    )


ENSEMBLE_COMPONENTS = ("vlm_logistic", "vlm_extra_trees", "vlm_catboost")

