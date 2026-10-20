"""Monta uma fila rastreável de abertura VLM no desenho caso-controle.

Não faz chamadas de IA. A saída pode ser auditada e aprovada antes de qualquer
uso de quota; o rótulo de seleção nunca é enviado ao extrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.case_control import build_case_control_queue  # noqa: E402
from galaxies.dataset import load_latest_vlm_results  # noqa: E402

YOUTUBE_RUNS = PROJECT_ROOT / "data" / "youtube" / "runs"
PLANS_DIR = PROJECT_ROOT / "data" / "features" / "plans"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def latest_records() -> tuple[Path, list[dict]]:
    candidates = [path for path in YOUTUBE_RUNS.iterdir() if (path / "normalized_records.json").exists()]
    if not candidates:
        raise FileNotFoundError("Nenhuma coleta normalizada foi encontrada.")
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    payload = json.loads((source / "normalized_records.json").read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("normalized_records.json não contém uma lista records.")
    return source, records


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta a fila caso-controle para análise VLM.")
    parser.add_argument("--run-id", default="case-control-opening-v1")
    parser.add_argument("--controls-per-case", type=int, default=3)
    args = parser.parse_args()
    source, records = latest_records()
    analyzed = set(load_latest_vlm_results())
    items, detail = build_case_control_queue(
        records, analyzed_video_ids=analyzed, controls_per_case=args.controls_per_case
    )
    plan = {
        "schema_version": "0.1.0",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocol": {
            "name": "viral-intra-canal-v1-case-control",
            "minimum_age_days": 30,
            "case_definition": "viral: top 10% intra-canal após ajuste de idade",
            "control_definition": "não viral, não analisado, mesmo canal, formato e época próxima",
            "controls_per_case_requested": args.controls_per_case,
            "selection_labels_sent_to_vlm": False,
            "selection_labels_used_as_model_features": False,
        },
        "source": {
            "youtube_run": source.relative_to(PROJECT_ROOT).as_posix(),
            "existing_vlm_results_count": len(analyzed),
        },
        "summary": detail["summary"],
        "queue": items,
        "groups": detail["groups"],
    }
    output = PLANS_DIR / args.run_id / "manifest.json"
    write_json(output, plan)
    print(json.dumps({"manifest": str(output), "summary": plan["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
