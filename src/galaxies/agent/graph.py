"""Grafo LangGraph do agente de hipóteses.

    diagnosticar → propor → validar → executar → julgar ─┬─> propor (próxima rodada)
                                 │                       └─> relatorio
                                 └─(nada novo)───────────────> relatorio

O estado do grafo guarda só dados leves (specs, resumos). A base, o histórico e
os resultados pesados ficam no ``AgentContext``, fora do estado.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from pydantic import ValidationError

from .evaluation import ExperimentResult, evaluate_on_vault, paired_comparison, run_experiment
from .hypotheses import Proposal
from .ledger import (
    PROMOTION_P_BETTER,
    TEMPORAL_TOLERANCE,
    AgentPaths,
    Ledger,
    champion_lines,
    history_lines,
    utc_now,
    write_report,
)
from .protocol import AGE_SHORTCUT_LIMIT, GOAL_AUC, PROTOCOL_VERSION, AgentDataset
from .spec import ExperimentSpec


class Generator(Protocol):
    source: str

    def propose(self, context: dict[str, Any], count: int) -> Proposal: ...


class AgentState(TypedDict, total=False):
    round: int
    max_rounds: int
    per_round: int
    proposals: list[dict[str, Any]]
    invalid: list[dict[str, Any]]
    source: str
    note: str
    rounds: list[dict[str, Any]]
    backlog: list[dict[str, str]]
    stop_reason: str
    report_path: str


@dataclass
class AgentContext:
    dataset: AgentDataset
    ledger: Ledger
    paths: AgentPaths
    generator: Generator
    seed: int = 42
    bootstrap_iterations: int = 1000
    open_vault: bool = False
    log: Callable[[str], None] = print
    dataset_lines: list[str] = field(default_factory=list)
    pending: dict[str, ExperimentResult] = field(default_factory=dict)
    vault_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def describe_dataset(dataset: AgentDataset) -> list[str]:
    lines = list(dataset.notes)
    for track in ("full", "vlm"):
        rows = dataset.evaluable(track)
        lines.append(
            f"Trilha {track}: {len(rows)} vídeos, {int(rows['viral'].sum())} virais, "
            f"{rows['channel_id'].nunique()} canais fora do cofre."
        )
    return lines


def build_graph(context: AgentContext):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError('LangGraph não está instalado. Rode: pip install -e ".[agent]"') from exc

    def diagnosticar(state: AgentState) -> AgentState:
        context.dataset_lines = describe_dataset(context.dataset)
        for line in context.dataset_lines:
            context.log(f"[diagnóstico] {line}")
        previous = len(context.ledger.records)
        if previous:
            context.log(
                f"[diagnóstico] {previous} experimentos anteriores nesta base serão considerados."
            )
        return {"round": 0, "rounds": [], "backlog": []}

    def propor(state: AgentState) -> AgentState:
        round_number = state["round"] + 1
        ideas = [item["idea"] for item in state.get("backlog", [])]
        proposal = context.generator.propose(
            {
                "tried_keys": sorted(context.ledger.tried_keys()),
                "champion_specs": context.ledger.champion_specs(),
                "history_lines": history_lines(context.ledger.records),
                "champion_lines": champion_lines(context.ledger),
                "dataset_lines": context.dataset_lines,
                "backlog_ideas": ideas,
            },
            state["per_round"],
        )
        backlog = list(state.get("backlog", []))
        for item in proposal.backlog:
            if item.get("idea") and item["idea"] not in ideas:
                backlog.append(item)
        context.log(
            f"[rodada {round_number}] {len(proposal.specs)} hipóteses de '{proposal.source}'"
            + (f" — {proposal.note}" if proposal.note else "")
        )
        return {
            "round": round_number,
            "proposals": [spec.model_dump(mode="json") for spec in proposal.specs],
            "invalid": proposal.invalid,
            "source": proposal.source,
            "note": proposal.note,
            "backlog": backlog,
        }

    def validar(state: AgentState) -> AgentState:
        tried = context.ledger.tried_keys()
        accepted: list[dict[str, Any]] = []
        invalid = list(state.get("invalid", []))
        for item in state.get("proposals", []):
            try:
                spec = ExperimentSpec.model_validate(item)
            except ValidationError as exc:
                invalid.append({"proposal": item, "reason": str(exc.errors()[0]["msg"])})
                continue
            if spec.key() in tried:
                invalid.append({"proposal": item, "reason": "configuração já testada"})
                continue
            if len(accepted) >= state["per_round"]:
                invalid.append({"proposal": item, "reason": "acima do orçamento da rodada"})
                continue
            tried.add(spec.key())
            accepted.append(spec.model_dump(mode="json"))
        for item in invalid:
            context.log(f"[validador] rejeitada: {item['reason']}")
        update: AgentState = {"proposals": accepted, "invalid": invalid}
        if not accepted:
            update["stop_reason"] = "nenhuma hipótese nova e válida na rodada"
        return update

    def executar(state: AgentState) -> AgentState:
        context.pending = {}
        for item in state["proposals"]:
            spec = ExperimentSpec.model_validate(item)
            context.log(f"[executor] {spec.short_label()}")
            result = run_experiment(
                spec,
                context.dataset,
                seed=context.seed,
                bootstrap_iterations=context.bootstrap_iterations,
            )
            context.pending[spec.key()] = result
        return {}

    def julgar(state: AgentState) -> AgentState:
        for item in state["proposals"]:
            spec = ExperimentSpec.model_validate(item)
            result = context.pending[spec.key()]
            champion = context.ledger.champion(spec.track)
            comparison = None
            age_rho = result.loco.get("age_rho")
            shortcut = age_rho is not None and abs(age_rho) > AGE_SHORTCUT_LIMIT
            if shortcut:
                status = "suspeito (idade)"
            elif champion is None:
                status = "referência"
            else:
                comparison = paired_comparison(
                    result.loco_counts,
                    context.ledger.counts_for(champion, "loco"),
                    context.bootstrap_iterations,
                    context.seed,
                )
                comparison["reference_key"] = champion["key"]
                temporal_ok = (result.temporal.get("auc_intra") or 0) >= (
                    (champion["temporal"].get("auc_intra") or 0) - TEMPORAL_TOLERANCE
                )
                promoted = (
                    comparison["diff"] > 0
                    and comparison["p_better"] >= PROMOTION_P_BETTER
                    and temporal_ok
                )
                status = "promovido" if promoted else "não promovido"
            context.ledger.save_result(
                result,
                round_number=state["round"],
                source=state.get("source", "?"),
                status=status,
                comparison=comparison,
            )
            context.log(
                f"[juiz] AUC LOCO {result.loco['auc_intra']:.3f} "
                f"[{result.loco['auc_intra_ci_low']:.3f}; {result.loco['auc_intra_ci_high']:.3f}] "
                f"temporal {result.temporal['auc_intra']:.3f} rho idade {result.loco.get('age_rho', float('nan')):+.2f} → {status}"
            )
        context.pending = {}
        rounds = list(state.get("rounds", []))
        rounds.append(
            {
                "round": state["round"],
                "source": state.get("source", "?"),
                "note": state.get("note", ""),
                "invalid": state.get("invalid", []),
            }
        )
        return {"rounds": rounds}

    def relatorio(state: AgentState) -> AgentState:
        if context.open_vault:
            for spec in context.ledger.champion_specs():
                context.vault_results[spec.track] = evaluate_on_vault(
                    spec,
                    context.dataset,
                    seed=context.seed,
                    bootstrap_iterations=context.bootstrap_iterations,
                )
                context.log(f"[cofre] trilha {spec.track}: {context.vault_results[spec.track]}")
        rounds = list(state.get("rounds", []))
        last_recorded = rounds[-1]["round"] if rounds else 0
        if state.get("round", 0) > last_recorded:
            rounds.append(
                {
                    "round": state.get("round", 0),
                    "source": state.get("source", "?"),
                    "note": state.get("note", ""),
                    "invalid": state.get("invalid", []),
                }
            )
        path = write_report(
            paths=context.paths,
            ledger=context.ledger,
            dataset_lines=context.dataset_lines,
            rounds=rounds,
            backlog=state.get("backlog", []),
            vault=context.vault_results or None,
            generator_source=getattr(context.generator, "source", "?"),
        )
        summary = {
            "run_id": context.paths.run_id,
            "finished_at": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "dataset_fingerprint": context.dataset.fingerprint,
            "feature_map": f"{context.dataset.source_path.parent.name}/{context.dataset.source_path.name}",
            "goal_auc_intra": GOAL_AUC,
            "experiments_in_run": sum(
                1 for record in context.ledger.records if record["run_id"] == context.paths.run_id
            ),
            "experiments_in_dataset": len(context.ledger.records),
            "stop_reason": state.get("stop_reason", "rodadas concluídas"),
            "champions": {
                track: {
                    "key": record["key"],
                    "label": record["label"],
                    "loco": record["loco"],
                    "temporal": record["temporal"],
                }
                for track in ("full", "vlm")
                if (record := context.ledger.champion(track))
            },
            "vault": context.vault_results or None,
        }
        (context.paths.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (context.paths.run_dir / "backlog.json").write_text(
            json.dumps(state.get("backlog", []), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        context.log(f"[relatório] {path}")
        return {"report_path": str(path)}

    def depois_de_validar(state: AgentState) -> str:
        return "executar" if state.get("proposals") else "relatorio"

    def depois_de_julgar(state: AgentState) -> str:
        return "propor" if state["round"] < state["max_rounds"] else "relatorio"

    graph = StateGraph(AgentState)
    graph.add_node("diagnosticar", diagnosticar)
    graph.add_node("propor", propor)
    graph.add_node("validar", validar)
    graph.add_node("executar", executar)
    graph.add_node("julgar", julgar)
    graph.add_node("relatorio", relatorio)
    graph.add_edge(START, "diagnosticar")
    graph.add_edge("diagnosticar", "propor")
    graph.add_edge("propor", "validar")
    graph.add_conditional_edges("validar", depois_de_validar, ["executar", "relatorio"])
    graph.add_edge("executar", "julgar")
    graph.add_conditional_edges("julgar", depois_de_julgar, ["propor", "relatorio"])
    graph.add_edge("relatorio", END)
    return graph.compile()


def run_agent(context: AgentContext, *, rounds: int, per_round: int) -> AgentState:
    app = build_graph(context)
    return app.invoke(
        {"max_rounds": rounds, "per_round": per_round},
        config={"recursion_limit": 10 + rounds * 6},
    )
