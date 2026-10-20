"""Compara metadados, histórico e audiovisual na mesma amostra por setor."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.config import SECTORS  # noqa: E402
from galaxies.dataset import latest_feature_map, write_json  # noqa: E402
from galaxies.modeling.ablation import evaluate_feature_groups  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa ablação de grupos de features em amostra comum com VLM."
    )
    parser.add_argument("--sector", choices=(*SECTORS, "all"), default="all")
    parser.add_argument("--dataset", type=Path, help="Mapa JSONL; usa o mais recente.")
    parser.add_argument("--run-id", help="Identificador opcional da avaliação.")
    return parser.parse_args()


def default_run_id() -> str:
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z").replace(":", "-")
    return f"feature-ablation-{timestamp}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    dataset = args.dataset or latest_feature_map()
    sectors = SECTORS if args.sector == "all" else (args.sector,)
    run_id = args.run_id or default_run_id()
    output_dir = PROJECT_ROOT / "data" / "modeling" / "evaluations" / run_id
    reports = {}
    for sector in sectors:
        print(f"Avaliando {sector}...", flush=True)
        report = evaluate_feature_groups(dataset, sector)
        reports[sector] = report
        write_json(output_dir / f"{sector}.json", report)
    summary = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "dataset": str(dataset),
        "sectors": {
            sector: {
                "rows": report["common_sample"]["rows"],
                "best_temporal": report["best_feature_set_temporal"],
                "best_channel_holdout": report["best_feature_set_channel_holdout"],
                "best_temporal_unrestricted": report[
                    "best_feature_set_temporal_unrestricted"
                ],
                "best_channel_holdout_unrestricted": report[
                    "best_feature_set_channel_holdout_unrestricted"
                ],
            }
            for sector, report in reports.items()
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Resultados: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
