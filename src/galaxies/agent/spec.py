"""Catálogo fechado de experimentos que o agente pode propor.

O LLM não escreve código nem escolhe colunas livremente: ele preenche um
``ExperimentSpec``. Cada campo é uma alavanca com poucas opções, o que mantém o
espaço de busca pequeno (menos risco de "vencedor por sorte") e impede que uma
hipótese use features pós-publicação.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .protocol import FEATURE_GROUPS, is_forbidden

Track = Literal["full", "vlm"]
FeatureGroup = Literal[
    "metadata",
    "cadence",
    "audiovisual",
    "audiovisual_derived",
    "transcript_stats",
    "text_stats",
]
Pooling = Literal["per_sector", "pooled"]
FormatStrategy = Literal["joint", "segmented"]
ModelFamily = Literal[
    "logistic",
    "hist_gb",
    "extra_trees",
    "random_forest",
    "catboost",
    "catboost_ranker",
    "mlp",
]
Complexity = Literal["low", "medium"]
Taxonomy = Literal["raw", "normalized"]
TrainLabel = Literal["top10", "clean20vs50"]
EvaluationMode = Literal["new_channel", "known_channel"]

CONFIG_FIELDS = (
    "track",
    "feature_groups",
    "pooling",
    "format_strategy",
    "model",
    "complexity",
    "train_min_age_days",
    "train_label",
    "evaluation_mode",
    "vlm_taxonomy",
)

CATALOG_DESCRIPTION = {
    "track": {
        "full": "todos os vídeos com 30+ dias fora do cofre (~2.600); sem features audiovisuais",
        "vlm": "só vídeos com análise audiovisual (~300); permite o grupo audiovisual",
    },
    "feature_groups": {
        "metadata": "duração, Shorts (<=180s), tamanho de título/descrição, tags, hora, dia, categoria",
        "cadence": "dias desde o upload anterior e intervalo mediano entre uploads do canal",
        "audiovisual": "atributos dos 3s iniciais extraídos pelo VLM (só na trilha vlm)",
        "audiovisual_derived": "ritmo, fala precoce e densidade multimodal derivados da janela VLM",
        "transcript_stats": "estatísticas determinísticas do trecho transcrito, sem texto cru",
        "text_stats": "contagens determinísticas do título e descrição; elegível apenas na ablação de embeddings",
    },
    "pooling": {
        "per_sector": "um modelo por setor",
        "pooled": "um modelo para os três setores, com o setor como feature",
    },
    "format_strategy": {
        "joint": "Shorts e longos no mesmo modelo",
        "segmented": "modelos separados para Shorts e para longos",
    },
    "model": {
        "logistic": "regressão logística regularizada",
        "hist_gb": "HistGradientBoosting raso",
        "extra_trees": "Extra Trees",
        "random_forest": "Random Forest",
        "catboost": "CatBoost raso e regularizado após o pré-processamento da dobra",
        "catboost_ranker": "CatBoost com perda de ranking e grupos definidos por canal",
        "mlp": "rede neural pequena, regularizada e com parada antecipada",
    },
    "complexity": {"low": "regularização forte", "medium": "regularização moderada"},
    "train_min_age_days": {
        "30": "treina com todos os vídeos rotulados",
        "60": "treina só com vídeos de 60+ dias (rótulo mais maduro, menos linhas)",
        "90": "treina só com vídeos de 90+ dias",
    },
    "train_label": {
        "top10": "positivo é o top 10% do canal no treino",
        "clean20vs50": "positivo é top 20%; negativo é a metade inferior; descarta o meio",
    },
    "evaluation_mode": {
        "new_channel": "canal inteiramente fora do treino (LOCO); sem histórico do próprio canal",
        "known_channel": "últimos 25% de cada canal; libera somente taxas históricas calculadas 30+ dias antes",
    },
    "vlm_taxonomy": {
        "raw": "categorias do VLM como vieram (texto livre, até ~100 valores)",
        "normalized": "categorias do VLM mapeadas para uma taxonomia fechada",
    },
}


class ExperimentSpec(BaseModel):
    """Uma hipótese testável, descrita só com opções do catálogo."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=5, max_length=400)
    rationale: str = Field(default="", max_length=800)
    expected_effect: str = Field(default="", max_length=300)
    track: Track = "full"
    feature_groups: list[FeatureGroup] = Field(default_factory=lambda: ["metadata"], min_length=1)
    pooling: Pooling = "per_sector"
    format_strategy: FormatStrategy = "joint"
    model: ModelFamily = "logistic"
    complexity: Complexity = "low"
    train_min_age_days: Literal[30, 60, 90] = 30
    train_label: TrainLabel = "top10"
    evaluation_mode: EvaluationMode = "new_channel"
    vlm_taxonomy: Taxonomy = "raw"

    @model_validator(mode="after")
    def check_rules(self) -> ExperimentSpec:
        groups = list(dict.fromkeys(self.feature_groups))
        self.feature_groups = sorted(groups)
        vlm_groups = {"audiovisual", "audiovisual_derived", "transcript_stats"}
        if vlm_groups.intersection(self.feature_groups) and self.track != "vlm":
            raise ValueError("Grupos audiovisuais só podem ser usados na trilha 'vlm'.")
        if "audiovisual" not in self.feature_groups:
            # A taxonomia não muda nada sem features audiovisuais; fixa para
            # evitar duplicatas que diferem só nesse campo.
            self.vlm_taxonomy = "raw"
        if "text_stats" in self.feature_groups:
            raise ValueError(
                "text_stats é avaliado no executor de embeddings para garantir a mesma cobertura; "
                "não é uma alavanca do agente genérico."
            )
        numeric, categorical = self.columns()
        leaked = [column for column in numeric + categorical if is_forbidden(column)]
        if leaked:
            raise ValueError(f"Features proibidas (pós-publicação): {leaked}")
        return self

    def columns(self) -> tuple[list[str], list[str]]:
        numeric: list[str] = []
        categorical: list[str] = []
        for group in self.feature_groups:
            numeric += FEATURE_GROUPS[group]["numeric"]
            categorical += FEATURE_GROUPS[group]["categorical"]
        if self.pooling == "pooled":
            categorical.append("sector")
        return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))

    def config(self) -> dict:
        return {name: getattr(self, name) for name in CONFIG_FIELDS}

    def key(self) -> str:
        """Identidade do experimento: só a configuração, nunca o texto."""
        payload = json.dumps(self.config(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def short_label(self) -> str:
        groups = "+".join(self.feature_groups)
        parts = [
            self.track,
            groups,
            self.model,
            self.complexity,
            self.pooling,
            self.format_strategy,
            f"treino>={self.train_min_age_days}d",
            self.evaluation_mode,
        ]
        if "audiovisual" in self.feature_groups:
            parts.append(f"vlm={self.vlm_taxonomy}")
        return " · ".join(parts)
