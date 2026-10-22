"""Roda o agente de hipóteses para o modelo de criativos virais.

O agente propõe experimentos (Gemini ou modo offline), roda cada um com o
protocolo congelado, compara com o campeão e escreve um relatório em
``data/agent/runs/<run_id>/report.md``.

Uso, a partir da raiz do projeto:
    pip install -e ".[agent]"
    python scripts/run_hypothesis_agent.py --rounds 3 --per-round 4
    python scripts/run_hypothesis_agent.py --offline          # sem chamar o Gemini
    python scripts/run_hypothesis_agent.py --open-vault       # confere os campeões no cofre

O histórico em ``data/agent/ledger.jsonl`` é cumulativo: execuções seguintes
não repetem configurações já testadas na mesma base.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.graph import AgentContext, run_agent  # noqa: E402
from galaxies.agent.hypotheses import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    GeminiGenerator,
    OfflineGenerator,
    load_dotenv,
)
from galaxies.agent.ledger import AgentPaths, Ledger  # noqa: E402
from galaxies.agent.protocol import latest_feature_map, load_dataset  # noqa: E402
from galaxies.paths import DATA_DIR, MODELING_RUNS_DIR  # noqa: E402

AGENT_DIR = DATA_DIR / "agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente de hipóteses do Galaxies.")
    parser.add_argument("--rounds", type=int, default=3, help="Número de rodadas (padrão: 3).")
    parser.add_argument(
        "--per-round", type=int, default=4, help="Experimentos por rodada (padrão: 4)."
    )
    parser.add_argument(
        "--offline", action="store_true", help="Não chama o Gemini; usa sementes e busca local."
    )
    parser.add_argument(
        "--feature-map", type=Path, help="feature_map.jsonl (padrão: o mais recente)."
    )
    parser.add_argument("--run-id", help="Identificador da execução (padrão: data e hora UTC).")
    parser.add_argument(
        "--open-vault",
        action="store_true",
        help="Ao final, avalia os campeões nos canais do cofre. Use raramente.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Reamostragens por canal.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", help="Modelo Gemini (padrão: GEMINI_AGENT_MODEL ou GEMINI_MODEL)."
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.rounds < 1 or args.per_round < 1:
        print("ERRO: --rounds e --per-round precisam ser >= 1.", file=sys.stderr)
        return 2

    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    print(f"Base: {feature_map}")
    dataset = load_dataset(feature_map)

    if args.offline:
        generator = OfflineGenerator()
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("AVISO: GEMINI_API_KEY ausente no .env; seguindo no modo offline.")
            generator = OfflineGenerator()
        else:
            model = args.model or os.getenv(
                "GEMINI_AGENT_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            )
            print(f"Gerador de hipóteses: Gemini ({model})")
            generator = GeminiGenerator(api_key, model)

    run_id = args.run_id or datetime.now(UTC).strftime("agent-%Y-%m-%dT%H-%M-%SZ")
    paths = AgentPaths(AGENT_DIR, run_id)
    paths.ensure()
    ledger = Ledger(paths, dataset.fingerprint)
    context = AgentContext(
        dataset=dataset,
        ledger=ledger,
        paths=paths,
        generator=generator,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap,
        open_vault=args.open_vault,
    )
    final_state = run_agent(context, rounds=args.rounds, per_round=args.per_round)
    print(f"\nRelatório: {final_state.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
