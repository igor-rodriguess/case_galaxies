"""Mede quanto da AUC intra-canal do ranking vem só da diferença Short x longo.

Para cada execução de `evaluate_known_channel.py`, recalcula a AUC comparando
só pares do mesmo canal **e do mesmo formato** (até 180 s ou acima). Também mede
duração e comprimento do título sozinhos, nos dois recortes. Não treina nada.

Uso:
    python scripts/diagnose_format_confounding.py RUN_ID [RUN_ID ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.evaluation import auc_intra_channel  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

SHORT_MAX_SECONDS = 180


def within_format(frame: pd.DataFrame) -> pd.DataFrame:
    fmt = np.where(frame["duration_seconds"] > SHORT_MAX_SECONDS, "longo", "curto")
    return frame.assign(channel_id=frame["channel_id"].astype(str) + "::" + fmt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    rows = []
    for run_id in args.run_ids:
        run_dir = MODELING_RUNS_DIR / run_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        features = pd.read_json(report["feature_map"], lines=True)[
            ["video_id", "duration_seconds", "title_character_count"]
        ]
        for path in sorted(run_dir.glob("predictions-*.parquet")):
            data = pd.read_parquet(path).merge(features, on="video_id", validate="one_to_one")
            duration = data.assign(score=data["duration_seconds"])
            title = data.assign(score=data["title_character_count"].fillna(0))
            rows.append(
                {
                    "run_id": run_id,
                    "model": path.stem.removeprefix("predictions-"),
                    "auc_intra": auc_intra_channel(data, "score"),
                    "auc_intra_same_format": auc_intra_channel(within_format(data), "score"),
                    "duration_only": auc_intra_channel(duration, "score"),
                    "duration_only_same_format": auc_intra_channel(
                        within_format(duration), "score"
                    ),
                    "title_length_same_format": auc_intra_channel(within_format(title), "score"),
                    "short_positive_rate": float(
                        data.loc[data["duration_seconds"] <= SHORT_MAX_SECONDS, "viral"].mean()
                    ),
                    "long_positive_rate": float(
                        data.loc[data["duration_seconds"] > SHORT_MAX_SECONDS, "viral"].mean()
                    ),
                }
            )
    table = pd.DataFrame(rows)
    output = MODELING_RUNS_DIR / args.run_ids[0] / "format_confounding.csv"
    table.to_csv(output, index=False)
    print(table.round(3).to_string(index=False))
    print(f"Tabela: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
