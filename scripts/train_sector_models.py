"""Compara os modelos configurados para um setor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.config import SECTORS  # noqa: E402
from galaxies.dataset import latest_feature_map  # noqa: E402
from galaxies.modeling import DatasetGateError, train_sector_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina e compara modelos por setor.")
    parser.add_argument("--sector", required=True, choices=SECTORS)
    parser.add_argument("--task", choices=("regression", "classification"), default="regression")
    parser.add_argument("--dataset", type=Path, help="feature_map.jsonl; usa o mais recente.")
    parser.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="Testa o encanamento mesmo com gates falhando; nunca promove o artefato.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    dataset = args.dataset or latest_feature_map()
    try:
        result = train_sector_models(
            dataset,
            args.sector,
            task=args.task,
            allow_exploratory=args.allow_exploratory,
        )
    except DatasetGateError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Use --allow-exploratory somente para validar o código, não o modelo.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
