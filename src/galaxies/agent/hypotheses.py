"""Geradores de hipóteses: Gemini (padrão) e um modo offline determinístico.

O Gemini recebe o protocolo, o catálogo, o diagnóstico e o histórico de
experimentos, e devolve propostas em JSON que viram ``ExperimentSpec``. Também
pode sugerir ideias de backlog — hipóteses que exigem algo fora do catálogo
(por exemplo, thumbnails) e que ficam registradas para decisão humana.

O modo offline usa hipóteses semeadas pelo diagnóstico de 18/10/2026 e, quando
elas acabam, uma busca local: muda uma alavanca por vez a partir do campeão.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .protocol import GOAL_AUC, MIN_EVAL_AGE_DAYS, PROTOCOL_VERSION
from .spec import CATALOG_DESCRIPTION, CONFIG_FIELDS, ExperimentSpec

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SEED_HYPOTHESES: list[dict[str, Any]] = [
    {
        "hypothesis": "Referência: regressão logística só com metadados, um modelo por setor.",
        "rationale": "Estabelece o piso honesto da trilha completa no novo alvo.",
        "expected_effect": "AUC intra-canal pouco acima de 0,5.",
        "track": "full",
        "feature_groups": ["metadata"],
        "model": "logistic",
    },
    {
        "hypothesis": "Juntar os três setores num modelo só aumenta o sinal.",
        "rationale": "Cada setor tem poucas centenas de virais; padrões de duração e cadência devem se transferir.",
        "expected_effect": "Ganho de AUC em relação ao modelo por setor.",
        "track": "full",
        "feature_groups": ["metadata"],
        "model": "hist_gb",
        "pooling": "pooled",
    },
    {
        "hypothesis": "Cadência de publicação ajuda a separar virais.",
        "rationale": "Intervalo desde o upload anterior é conhecido antes de publicar e estava agrupado com histórico vazado.",
        "expected_effect": "Ganho sobre metadados puros.",
        "track": "full",
        "feature_groups": ["metadata", "cadence"],
        "model": "hist_gb",
        "pooling": "pooled",
    },
    {
        "hypothesis": "Shorts e longos se comportam diferente e merecem modelos separados.",
        "rationale": "Entre 46% e 61% dos vídeos têm até 3 minutos; a dinâmica de audiência muda com o formato.",
        "expected_effect": "Ganho, principalmente em beleza e moda.",
        "track": "full",
        "feature_groups": ["metadata", "cadence"],
        "model": "hist_gb",
        "pooling": "pooled",
        "format_strategy": "segmented",
    },
    {
        "hypothesis": "Treinar só com vídeos maduros (90+ dias) limpa o rótulo.",
        "rationale": "Vídeos de 30 a 90 dias ainda podem acumular views; o ajuste de idade é imperfeito.",
        "expected_effect": "Ganho se o ruído de idade dominar; perda se faltar dado.",
        "track": "full",
        "feature_groups": ["metadata", "cadence"],
        "model": "hist_gb",
        "pooling": "pooled",
        "train_min_age_days": 90,
    },
    {
        "hypothesis": "Referência da trilha audiovisual: só metadados e cadência nos vídeos com VLM.",
        "rationale": "Isola o efeito das features audiovisuais na mesma amostra.",
        "expected_effect": "Piso da trilha vlm.",
        "track": "vlm",
        "feature_groups": ["metadata", "cadence"],
        "model": "logistic",
        "pooling": "pooled",
    },
    {
        "hypothesis": "A abertura do vídeo (3s iniciais) adiciona sinal além dos metadados.",
        "rationale": "É a pergunta central do produto de criativos.",
        "expected_effect": "Ganho sobre a referência vlm.",
        "track": "vlm",
        "feature_groups": ["audiovisual", "cadence", "metadata"],
        "model": "extra_trees",
        "pooling": "pooled",
    },
    {
        "hypothesis": "Taxonomia fechada nas saídas do VLM reduz ruído de categorias livres.",
        "rationale": "Enquadramento tem ~80 valores e movimento de câmera ~100 por variação de escrita.",
        "expected_effect": "Ganho sobre a versão com categorias cruas.",
        "track": "vlm",
        "feature_groups": ["audiovisual", "cadence", "metadata"],
        "model": "extra_trees",
        "pooling": "pooled",
        "vlm_taxonomy": "normalized",
    },
]

DIAGNOSIS_CONTEXT = """\
Diagnóstico humano de 18/10/2026 (execução nested-665, regressão):
- O holdout temporal antigo testava vídeos com 0,5 a 14 dias de vida; o viés
  explicava 33% a 50% do erro. Removendo só o viés, o R² ficava positivo.
- O holdout por canal antigo tinha só 2 canais por setor.
- Shorts e longos estavam misturados (46% a 61% dos vídeos com até 3 min).
- As categorias do VLM vêm em texto livre (enquadramento ~80 valores).
- "Histórico de canal" juntava cadência (válida) com views passadas (vazadas).
- Na tarefa nova, acurácia é enganosa: chutar "não viral" acerta ~90%.
- Primeira rodada do agente: o rótulo com ajuste de idade por setor deixava os
  vídeos mais novos de cada canal com cara de virais, e a "posição na série"
  virou atalho de idade. O ajuste agora é feito dentro do canal, a posição foi
  proibida e todo experimento mede rho idade (correlação nota x idade).
"""


@dataclass
class Proposal:
    specs: list[ExperimentSpec] = field(default_factory=list)
    invalid: list[dict[str, Any]] = field(default_factory=list)
    backlog: list[dict[str, str]] = field(default_factory=list)
    source: str = "offline"
    note: str = ""


def parse_specs(items: Sequence[dict[str, Any]]) -> tuple[list[ExperimentSpec], list[dict]]:
    specs: list[ExperimentSpec] = []
    invalid: list[dict[str, Any]] = []
    for item in items:
        try:
            specs.append(ExperimentSpec.model_validate(item))
        except ValidationError as exc:
            errors = "; ".join(error["msg"] for error in exc.errors())
            invalid.append({"proposal": item, "reason": errors})
    return specs, invalid


def neighbours(spec: ExperimentSpec) -> list[ExperimentSpec]:
    """Specs que diferem do campeão em exatamente uma alavanca."""
    options: dict[str, list[Any]] = {
        "model": [
            "logistic",
            "hist_gb",
            "extra_trees",
            "random_forest",
            "catboost",
            "catboost_ranker",
            "mlp",
        ],
        "complexity": ["low", "medium"],
        "pooling": ["per_sector", "pooled"],
        "format_strategy": ["joint", "segmented"],
        "train_min_age_days": [30, 60, 90],
        "train_label": ["top10", "clean20vs50"],
    }
    groups_options = [["metadata"], ["cadence", "metadata"]]
    if spec.track == "vlm":
        groups_options += [
            ["audiovisual", "cadence", "metadata"],
            ["audiovisual", "metadata"],
            ["audiovisual", "audiovisual_derived", "cadence", "metadata"],
            ["audiovisual", "cadence", "metadata", "transcript_stats"],
        ]
        options["vlm_taxonomy"] = ["raw", "normalized"]
    result: list[ExperimentSpec] = []
    base = spec.config()
    for name, values in options.items():
        for value in values:
            if value == base[name]:
                continue
            candidate = {**base, name: value}
            result.append(
                _spec_from_config(candidate, f"Variar {name} para {value} a partir do campeão.")
            )
    for groups in groups_options:
        if sorted(groups) != base["feature_groups"]:
            candidate = {**base, "feature_groups": groups}
            result.append(
                _spec_from_config(candidate, f"Usar grupos {'+'.join(groups)} a partir do campeão.")
            )
    return result


def _spec_from_config(config: dict[str, Any], hypothesis: str) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            **config,
            "hypothesis": hypothesis,
            "rationale": "Busca local: uma alavanca por vez.",
            "expected_effect": "Desconhecido; mede sensibilidade.",
        }
    )


class OfflineGenerator:
    """Sementes do diagnóstico, depois busca local em volta dos campeões."""

    source = "offline"

    def propose(self, context: dict[str, Any], count: int) -> Proposal:
        tried = set(context.get("tried_keys", []))
        specs: list[ExperimentSpec] = []
        seeds, _ = parse_specs(SEED_HYPOTHESES)
        for spec in seeds:
            if spec.key() not in tried and len(specs) < count:
                specs.append(spec)
                tried.add(spec.key())
        for champion in context.get("champion_specs", []):
            for spec in neighbours(champion):
                if spec.key() not in tried and len(specs) < count:
                    specs.append(spec)
                    tried.add(spec.key())
        note = "" if specs else "Sem hipóteses novas no catálogo."
        return Proposal(specs=specs, source=self.source, note=note)


# --- Gemini -----------------------------------------------------------------


def _string_enum(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "STRING", "enum": list(values)}


def response_schema() -> dict[str, Any]:
    spec_properties = {
        "hypothesis": {"type": "STRING"},
        "rationale": {"type": "STRING"},
        "expected_effect": {"type": "STRING"},
        "track": _string_enum(["full", "vlm"]),
        "feature_groups": {
            "type": "ARRAY",
            "items": _string_enum(
                [
                    "metadata",
                    "cadence",
                    "audiovisual",
                    "audiovisual_derived",
                    "transcript_stats",
                    "text_stats",
                ]
            ),
        },
        "pooling": _string_enum(["per_sector", "pooled"]),
        "format_strategy": _string_enum(["joint", "segmented"]),
        "model": _string_enum(
            [
                "logistic",
                "hist_gb",
                "extra_trees",
                "random_forest",
                "catboost",
                "catboost_ranker",
                "mlp",
            ]
        ),
        "complexity": _string_enum(["low", "medium"]),
        "train_min_age_days": {"type": "INTEGER", "description": "Um de 30, 60 ou 90."},
        "train_label": _string_enum(["top10", "clean20vs50"]),
        "evaluation_mode": _string_enum(["new_channel", "known_channel"]),
        "vlm_taxonomy": _string_enum(["raw", "normalized"]),
    }
    return {
        "type": "OBJECT",
        "properties": {
            "proposals": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": spec_properties,
                    "required": ["hypothesis", "rationale", "expected_effect", *CONFIG_FIELDS],
                },
            },
            "backlog": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "idea": {"type": "STRING"},
                        "why": {"type": "STRING"},
                        "needs": {"type": "STRING"},
                    },
                    "required": ["idea", "why", "needs"],
                },
            },
        },
        "required": ["proposals", "backlog"],
    }


def build_prompt(context: dict[str, Any], count: int) -> str:
    history = context.get("history_lines") or ["(nenhum experimento ainda)"]
    champions = context.get("champion_lines") or ["(nenhum campeão ainda)"]
    dataset_lines = context.get("dataset_lines", [])
    return f"""\
Você é o gerador de hipóteses de um laboratório de ML que prevê criativos
virais no YouTube (setores automotivo, moda e beleza). Seu trabalho é propor os
próximos experimentos com maior chance de melhorar a métrica, sem repetir o que
já foi testado.

PROTOCOLO FIXO ({PROTOCOL_VERSION}) — você não pode mudar:
- Rótulo: viral = top 10% do próprio canal, após ajuste por idade, só vídeos
  com {MIN_EVAL_AGE_DAYS}+ dias.
- Métrica principal: AUC intra-canal em leave-one-channel-out (baseline 0,5).
  Checagem: corte temporal. Meta do projeto: AUC >= {GOAL_AUC}.
- Um canal por setor fica num cofre que você nunca vê.
- Só features disponíveis antes da publicação. Views, curtidas, comentários e
  inscritos atuais são proibidos.
- Experimento com |rho idade| > 0,20 vira "suspeito (idade)" e não é promovido.

BASE:
{chr(10).join(dataset_lines)}

CATÁLOGO (as únicas alavancas que você pode mexer):
{json.dumps(CATALOG_DESCRIPTION, ensure_ascii=False, indent=2)}
Regras: "audiovisual" só na trilha "vlm"; train_min_age_days em {{30, 60, 90}}.

{DIAGNOSIS_CONTEXT}
CAMPEÕES ATUAIS:
{chr(10).join(champions)}

HISTÓRICO (mais recentes por último; AUC LOCO [IC95%] | AUC temporal | status):
{chr(10).join(history)}

TAREFA:
1. Proponha exatamente {count} experimentos novos (configuração diferente de
   todas as do histórico). Prefira hipóteses que testem uma ideia clara, que
   expliquem resultados anteriores ou que explorem perto do campeão.
2. Para cada um, escreva em português a hipótese, o raciocínio e o efeito
   esperado.
3. Em "backlog", liste até 3 ideias promissoras que NÃO cabem no catálogo
   (por exemplo, novas fontes de dados ou modelos), com o porquê e o que seria
   necessário. Não repita ideias já listadas: {json.dumps(context.get("backlog_ideas", []), ensure_ascii=False)}
"""


def load_dotenv(path: Path) -> None:
    """Carrega variáveis de um .env simples sem sobrescrever o ambiente."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


Transport = Callable[[str, dict[str, Any], str], dict[str, Any]]


def http_transport(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


class GeminiGenerator:
    """Pede propostas ao Gemini com saída estruturada; cai no offline se falhar."""

    source = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        transport: Transport = http_transport,
        max_attempts: int = 3,
        fallback: OfflineGenerator | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.transport = transport
        self.max_attempts = max_attempts
        self.fallback = fallback or OfflineGenerator()

    def _request(self, prompt: str, *, with_schema: bool) -> dict[str, Any]:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        generation_config: dict[str, Any] = {
            "temperature": 0.7,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        }
        if with_schema:
            generation_config["responseSchema"] = response_schema()
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.transport(endpoint, payload, self.api_key)
            except HTTPError as exc:
                last_error = exc
                if exc.code == 400 and with_schema:
                    raise  # schema recusado: o chamador tenta sem schema
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_attempts:
                    time.sleep(5 * attempt)
                    continue
                raise
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(5 * attempt)
                    continue
                raise
        raise RuntimeError(f"Gemini falhou: {last_error}")

    @staticmethod
    def _extract_json(raw: dict[str, Any]) -> dict[str, Any]:
        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Resposta do Gemini não é um objeto JSON.")
        return parsed

    def propose(self, context: dict[str, Any], count: int) -> Proposal:
        prompt = build_prompt(context, count)
        try:
            try:
                raw = self._request(prompt, with_schema=True)
            except HTTPError as exc:
                if exc.code != 400:
                    raise
                raw = self._request(prompt, with_schema=False)
            parsed = self._extract_json(raw)
        except Exception as exc:  # qualquer falha do provedor cai no modo offline
            message = str(exc).replace(self.api_key, "<redacted>")
            proposal = self.fallback.propose(context, count)
            proposal.note = (
                f"Gemini indisponível ({type(exc).__name__}: {message[:200]}); usei o modo offline."
            )
            return proposal

        specs, invalid = parse_specs(parsed.get("proposals", []))
        tried = set(context.get("tried_keys", []))
        fresh: list[ExperimentSpec] = []
        for spec in specs:
            if spec.key() in tried:
                invalid.append({"proposal": spec.config(), "reason": "configuração já testada"})
                continue
            tried.add(spec.key())
            fresh.append(spec)
        backlog = [
            {key: str(item.get(key, "")) for key in ("idea", "why", "needs")}
            for item in parsed.get("backlog", [])
            if isinstance(item, dict)
        ]
        return Proposal(specs=fresh[:count], invalid=invalid, backlog=backlog, source=self.source)
