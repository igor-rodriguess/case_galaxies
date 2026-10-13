"""Gera o mapa de features e o relatório de prontidão sem treinar modelos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.dataset import create_modeling_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Junta YouTube + VLM por video_id e audita a prontidão dos dados."
    )
    parser.add_argument("--run-id", help="ID opcional e retomável para a saída.")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    run_dir, audit = create_modeling_run(run_id=args.run_id)
    print(f"Mapa de features: {run_dir / 'feature_map.jsonl'}")
    print(f"Auditoria: {run_dir / 'readiness_audit.json'}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
