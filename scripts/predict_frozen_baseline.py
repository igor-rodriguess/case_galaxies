"""Executa uma previsão local e rastreável da baseline congelada."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.modeling.inference import FrozenBaselinePredictor, prediction_to_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prevê com a baseline Galaxies congelada.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON de um criativo pré-publicação."
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="JSON de cenário alternativo; produz comparação associativa, não causal.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("O arquivo de entrada deve ser um objeto JSON.")
    predictor = FrozenBaselinePredictor.from_run(args.artifact_dir)
    if args.compare_with:
        alternative = json.loads(args.compare_with.read_text(encoding="utf-8"))
        if not isinstance(alternative, dict):
            raise ValueError("O arquivo alternativo deve ser um objeto JSON.")
        prediction = predictor.compare(payload, alternative)
    else:
        prediction = predictor.predict(payload)
    rendered = prediction_to_json(prediction)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
