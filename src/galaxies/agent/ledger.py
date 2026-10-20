"""Histórico de experimentos, campeões e relatório do agente.

O histórico global (``data/agent/ledger.jsonl``) acumula todas as rodadas. Só
registros com o mesmo protocolo e a mesma impressão digital da base contam para
deduplicação e campeões: se o mapa de features mudar, o agente recomeça a
comparação do zero em vez de misturar números incomparáveis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation import ExperimentResult
from .protocol import AGE_SHORTCUT_LIMIT, GOAL_AUC, PROTOCOL_VERSION
from .spec import ExperimentSpec

PROMOTION_P_BETTER = 0.90
TEMPORAL_TOLERANCE = 0.02


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


@dataclass
class AgentPaths:
    root: Path  # data/agent
    run_id: str

    @property
    def ledger(self) -> Path:
        return self.root / "ledger.jsonl"

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    def ensure(self) -> None:
        if self.run_dir.exists():
            raise FileExistsError(f"A execução {self.run_id} já existe; use outro --run-id.")
        self.predictions_dir.mkdir(parents=True)


class Ledger:
    def __init__(self, paths: AgentPaths, fingerprint: str) -> None:
        self.paths = paths
        self.fingerprint = fingerprint
        self.records: list[dict[str, Any]] = []
        if paths.ledger.exists():
            for line in paths.ledger.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if (
                        record.get("protocol_version") == PROTOCOL_VERSION
                        and record.get("dataset_fingerprint") == fingerprint
                    ):
                        self.records.append(record)

    # --- consultas ---------------------------------------------------------
    def tried_keys(self) -> set[str]:
        return {record["key"] for record in self.records}

    def champion(self, track: str) -> dict[str, Any] | None:
        winners = [
            record
            for record in self.records
            if record["spec"]["track"] == track and record["status"] in {"referência", "promovido"}
        ]
        return winners[-1] if winners else None

    def champion_specs(self) -> list[ExperimentSpec]:
        specs = []
        for track in ("full", "vlm"):
            record = self.champion(track)
            if record:
                specs.append(ExperimentSpec.model_validate(record["spec"]))
        return specs

    def counts_for(self, record: dict[str, Any], split: str) -> pd.DataFrame:
        path = self.paths.root / record[f"{split}_counts_path"]
        return pd.read_csv(path, index_col="channel_id")

    # --- escrita -----------------------------------------------------------
    def save_result(
        self,
        result: ExperimentResult,
        *,
        round_number: int,
        source: str,
        status: str,
        comparison: dict[str, Any] | None,
    ) -> dict[str, Any]:
        key = result.spec.key()
        base = self.paths.predictions_dir
        files = {
            "predictions_path": base / f"{key}_loco_predictions.csv",
            "temporal_predictions_path": base / f"{key}_temporal_predictions.csv",
            "loco_counts_path": base / f"{key}_loco_counts.csv",
            "temporal_counts_path": base / f"{key}_temporal_counts.csv",
        }
        result.predictions.to_csv(files["predictions_path"], index=False)
        result.temporal_predictions.to_csv(files["temporal_predictions_path"], index=False)
        result.loco_counts.to_csv(files["loco_counts_path"])
        result.temporal_counts.to_csv(files["temporal_counts_path"])
        record = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_fingerprint": self.fingerprint,
            "run_id": self.paths.run_id,
            "round": round_number,
            "source": source,
            "key": key,
            "label": result.spec.short_label(),
            "spec": result.spec.model_dump(mode="json"),
            "loco": result.loco,
            "temporal": result.temporal,
            "comparison": comparison,
            "status": status,
            "created_at": utc_now(),
            **{name: path.relative_to(self.paths.root).as_posix() for name, path in files.items()},
        }
        record = _clean(record)
        self.records.append(record)
        with self.paths.ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


# --- texto para o gerador e para o relatório -------------------------------------


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value)


def history_lines(records: list[dict[str, Any]], limit: int = 40) -> list[str]:
    lines = []
    for record in records[-limit:]:
        loco = record["loco"]
        lines.append(
            f"- [{record['key']}] {record['label']} :: {fmt(loco.get('auc_intra'))} "
            f"[{fmt(loco.get('auc_intra_ci_low'))}; {fmt(loco.get('auc_intra_ci_high'))}] | "
            f"temporal {fmt(record['temporal'].get('auc_intra'))} | rho idade {fmt(loco.get('age_rho'), 2)} | {record['status']} "
            f"| hipótese: {record['spec']['hypothesis']}"
        )
    return lines


def champion_lines(ledger: Ledger) -> list[str]:
    lines = []
    for track in ("full", "vlm"):
        record = ledger.champion(track)
        if record:
            lines.append(
                f"- trilha {track}: {record['label']} :: AUC LOCO {fmt(record['loco']['auc_intra'])} "
                f"| temporal {fmt(record['temporal']['auc_intra'])} | recall top10% "
                f"{fmt(record['loco'].get('recall_top10'))}"
            )
    return lines


def _metric_cell(block: dict[str, Any]) -> str:
    return (
        f"{fmt(block.get('auc_intra'))} [{fmt(block.get('auc_intra_ci_low'))}; "
        f"{fmt(block.get('auc_intra_ci_high'))}]"
    )


def write_report(
    *,
    paths: AgentPaths,
    ledger: Ledger,
    dataset_lines: list[str],
    rounds: list[dict[str, Any]],
    backlog: list[dict[str, str]],
    vault: dict[str, dict[str, Any]] | None,
    generator_source: str,
) -> Path:
    run_records = [record for record in ledger.records if record["run_id"] == paths.run_id]
    lines = [
        f"# Agente de hipóteses — {paths.run_id}",
        "",
        f"Protocolo `{PROTOCOL_VERSION}` · base `{ledger.fingerprint}` · gerador `{generator_source}`",
        "",
        "## Protocolo",
        "",
        "- **Alvo:** viral = top 10% do próprio canal, após ajuste por idade, só vídeos com 30+ dias.",
        "- **Métrica principal:** AUC intra-canal em leave-one-channel-out (0,5 = acaso).",
        "- **Checagem:** mesmo cálculo num corte temporal (25% mais recentes da população madura).",
        f"- **Meta:** AUC ≥ {fmt(GOAL_AUC, 2)}.",
        f"- **Promoção:** P(melhor que o campeão) ≥ {fmt(PROMOTION_P_BETTER, 2)} no LOCO, "
        f"sem perder mais de {fmt(TEMPORAL_TOLERANCE, 2)} no temporal e com |rho idade| ≤ {fmt(AGE_SHORTCUT_LIMIT, 2)}.",
        "",
        *[f"- {line}" for line in dataset_lines],
        "",
        "## Campeões",
        "",
        "| Trilha | Configuração | AUC LOCO [IC95%] | AUC temporal | Recall top 10% | Lift top 10% | rho idade | Falta para a meta |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for track in ("full", "vlm"):
        record = ledger.champion(track)
        if not record:
            continue
        gap = GOAL_AUC - (record["loco"].get("auc_intra") or 0)
        lines.append(
            f"| {track} | {record['label']} | {_metric_cell(record['loco'])} | "
            f"{fmt(record['temporal'].get('auc_intra'))} | {fmt(record['loco'].get('recall_top10'))} | "
            f"{fmt(record['loco'].get('lift_top10'), 2)} | {fmt(record['loco'].get('age_rho'), 2)} | {fmt(gap)} |"
        )
    ranking = sorted(
        ledger.records, key=lambda record: record["loco"].get("auc_intra") or 0, reverse=True
    )
    lines += [
        "",
        f"## Ranking ({len(ledger.records)} experimentos nesta base)",
        "",
        "| # | Trilha | Configuração | AUC LOCO [IC95%] | AUC temporal | rho idade | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, record in enumerate(ranking[:15], start=1):
        lines.append(
            f"| {index} | {record['spec']['track']} | {record['label']} | {_metric_cell(record['loco'])} | "
            f"{fmt(record['temporal'].get('auc_intra'))} | {fmt(record['loco'].get('age_rho'), 2)} | {record['status']} |"
        )
    lines += ["", "## Rodadas desta execução", ""]
    for round_info in rounds:
        lines.append(f"### Rodada {round_info['round']} — gerador `{round_info['source']}`")
        if round_info.get("note"):
            lines.append(f"\n> {round_info['note']}\n")
        for record in [r for r in run_records if r["round"] == round_info["round"]]:
            comparison = record.get("comparison") or {}
            versus = (
                f" Diferença vs. campeão: {fmt(comparison.get('diff'))} "
                f"[{fmt(comparison.get('ci_low'))}; {fmt(comparison.get('ci_high'))}], "
                f"P(melhor) {fmt(comparison.get('p_better'), 2)}."
                if comparison
                else ""
            )
            lines += [
                f"- **{record['spec']['hypothesis']}**",
                f"  `{record['label']}` → AUC LOCO {_metric_cell(record['loco'])}, "
                f"temporal {fmt(record['temporal'].get('auc_intra'))} — **{record['status']}**.{versus}",
            ]
        for item in round_info.get("invalid", []):
            lines.append(f"- Rejeitada pelo validador: {item['reason']}")
        lines.append("")
    if backlog:
        lines += ["## Backlog (ideias fora do catálogo, para decisão humana)", ""]
        for item in backlog:
            lines.append(f"- **{item['idea']}** — {item['why']} *Precisa de:* {item['needs']}")
        lines.append("")
    if vault:
        lines += [
            "## Cofre (avaliado uma única vez)",
            "",
            "| Trilha | Linhas | Virais | AUC intra-canal [IC95%] | Recall top 10% |",
            "|---|---|---|---|---|",
        ]
        for track, metrics in vault.items():
            lines.append(
                f"| {track} | {metrics.get('rows')} | {metrics.get('positives', '—')} | "
                f"{_metric_cell(metrics)} | {fmt(metrics.get('recall_top10'))} |"
            )
        lines.append("")
    lines += [
        "## Como ler",
        "",
        f"- Foram testadas {len(ledger.records)} configurações nesta base. Quanto mais testes, "
        "maior a chance de um campeão vencer por sorte; o cofre existe para conferir isso.",
        "- O IC95% reamostra canais, não vídeos, porque a generalização que importa é para canal novo.",
        f"- rho idade é a correlação, dentro do canal, entre a nota do modelo e a idade do vídeo. Acima de "
        f"{fmt(AGE_SHORTCUT_LIMIT, 2)} em módulo o experimento vira 'suspeito (idade)' e não pode ser promovido: "
        "a nota estaria adivinhando quando o vídeo saiu, não a qualidade do criativo.",
        "- AUC 0,85 significa que, num par (viral, não viral) do mesmo canal, o modelo acerta 85% das vezes.",
        "",
    ]
    path = paths.run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
