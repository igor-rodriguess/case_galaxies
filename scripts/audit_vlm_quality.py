"""Gera um relatório de qualidade das análises VLM existentes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.dataset import load_latest_vlm_results  # noqa: E402
from galaxies.vlm_quality import audit_results  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita qualidade de resultados VLM.")
    parser.add_argument("--run-id", default="vlm-quality-audit-2026-10-18")
    args = parser.parse_args()
    report = audit_results(load_latest_vlm_results())
    report.update(
        {
            "schema_version": "0.1.0",
            "audited_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "policy": "Relatório somente leitura; não modifica nem completa análises.",
        }
    )
    output = PROJECT_ROOT / "data" / "features" / "audits" / args.run_id / "quality_report.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), **{key: report[key] for key in ("results_audited", "results_with_issues", "issues_by_kind")}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
