"""Otimiza regressões e gera o dossiê gráfico da comparação."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.config import SECTORS  # noqa: E402
from galaxies.dataset import latest_feature_map  # noqa: E402
from galaxies.modeling.feature_groups import FEATURE_SET_NAMES  # noqa: E402
from galaxies.modeling.optimization import optimize_sector  # noqa: E402
from galaxies.modeling.reporting import write_evaluation_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Busca features e hiperparâmetros sem tocar nos holdouts externos e "
            "gera JSON, CSV, gráficos e relatório Markdown."
        )
    )
    parser.add_argument("--sector", choices=(*SECTORS, "all"), default="all")
    parser.add_argument("--dataset", type=Path, help="feature_map.jsonl; usa o mais recente.")
    parser.add_argument("--run-id", help="Nome da execução e do relatório.")
    parser.add_argument("--search-iterations", type=int, default=12)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--target-mode",
        choices=("observed_views", "relative_performance"),
        default="observed_views",
        help=(
            "observed_views usa a contagem acumulada; relative_performance "
            "remove a curva de idade e a escala histórica do canal."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_NAMES,
        default="combined",
        help="Família de atributos usada na comparação.",
    )
    return parser.parse_args()


def default_run_id() -> str:
    date = datetime.now(UTC).date().isoformat()
    return f"optimized-model-evaluation-{date}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.search_iterations < 1:
        raise ValueError("--search-iterations deve ser positivo.")
    if args.inner_splits < 2:
        raise ValueError("--inner-splits deve ser pelo menos 2.")
    dataset = args.dataset or latest_feature_map()
    sectors = SECTORS if args.sector == "all" else (args.sector,)
    run_id = args.run_id or default_run_id()
    output_dir = PROJECT_ROOT / "data" / "modeling" / "evaluations" / run_id
    if output_dir.exists():
        raise FileExistsError(f"A avaliação já existe: {output_dir}")

    reports = {}
    for sector in sectors:
        print(f"Otimizando {sector}...", flush=True)
        reports[sector] = optimize_sector(
            dataset,
            sector,
            search_iterations=args.search_iterations,
            inner_splits=args.inner_splits,
            n_jobs=args.jobs,
            target_mode=args.target_mode,
            feature_set=args.feature_set,
        )
        selected = reports[sector]["selection"]["best_inner_cv"]
        print(f"{sector}: selecionado pela validação interna = {selected}", flush=True)

    artifacts = write_evaluation_artifacts(reports, output_dir, run_id)
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
