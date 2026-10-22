"""Monta o mapa de features da base de Shorts a partir do histórico completo.

Entrada: ``data/youtube/shorts_history/<run_id>/`` (de
``collect_channel_shorts_history.py``). Saída: um ``feature_map.jsonl`` no
formato lido por ``galaxies.agent.protocol.load_dataset``.

- Só vídeos públicos de até 180 segundos, conforme critério operacional do
  projeto (``format_ranking.shorts_population``).
- Metadados: os mesmos grupos do ranking (``metadata`` e ``cadence``). Título,
  descrição e tags são os do snapshot da coleta; podem ter sido editados depois
  da publicação, o que fica registrado como limitação.
- Cadência usa as datas de publicação de **todos** os uploads anteriores do
  canal, curtos ou longos, lidas na playlist de uploads. Datas de publicação
  anteriores são conhecidas no momento da previsão.
- Views, likes e comentários entram só como ``target_*``/snapshot e são barrados
  como feature por ``is_forbidden``.

Uso:
    python scripts/build_shorts_feature_map.py --history shorts-history-2026-10-21
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

from galaxies.agent.format_ranking import shorts_population  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402
from galaxies.prospective import exclude_prospective  # noqa: E402

HISTORY_ROOT = ROOT / "data" / "youtube" / "shorts_history"


def upload_dates(run_dir: Path) -> pd.DataFrame:
    rows = []
    for page in sorted((run_dir / "raw").glob("*/playlistItems-page-*.json")):
        payload = json.loads(page.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            details = item.get("contentDetails", {})
            if details.get("videoId") and details.get("videoPublishedAt"):
                rows.append(
                    {
                        "channel_id": page.parent.name,
                        "video_id": details["videoId"],
                        "published_at": details["videoPublishedAt"],
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates("video_id")
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    return frame


def cadence(uploads: pd.DataFrame) -> pd.DataFrame:
    """Dias desde o upload anterior e mediana dos intervalos anteriores, por canal."""
    ordered = uploads.sort_values(["channel_id", "published_at", "video_id"]).copy()
    gaps = ordered.groupby("channel_id")["published_at"].diff().dt.total_seconds() / 86_400
    ordered["days_since_previous_upload"] = gaps
    ordered["prior_upload_interval_median_days"] = gaps.groupby(ordered["channel_id"]).transform(
        lambda values: values.shift(1).expanding().median()
    )
    return ordered[["video_id", "days_since_previous_upload", "prior_upload_interval_median_days"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_dir = HISTORY_ROOT / args.history
    records = pd.DataFrame(
        json.loads((run_dir / "normalized_records.json").read_text(encoding="utf-8"))["records"]
    )
    records["published_at"] = pd.to_datetime(records["published_at"], utc=True)
    collected = pd.to_datetime(records["collected_at"], utc=True)
    records = records.loc[records["privacy_status"].fillna("public") == "public"]
    records = shorts_population(exclude_prospective(records))

    frame = pd.DataFrame(
        {
            "schema_version": "shorts-0.1.0",
            "video_id": records["video_id"],
            "channel_id": records["channel_id"],
            "sector": records["sector"],
            "published_at": records["published_at"].map(lambda value: value.isoformat()),
            "collected_at": records["collected_at"],
            "duration_seconds": pd.to_numeric(records["duration_seconds"], errors="coerce"),
            "definition": records["definition"],
            "category_id": records["category_id"],
            "title_character_count": records["title"].fillna("").str.len(),
            "description_character_count": records["description"].fillna("").str.len(),
            "tag_count": records["tags"].map(
                lambda tags: len(tags) if isinstance(tags, list) else 0
            ),
            "publish_hour_utc": records["published_at"].dt.hour,
            "publish_weekday_utc": records["published_at"].dt.weekday,
            "like_count_snapshot": pd.to_numeric(records["like_count"], errors="coerce"),
            "comment_count_snapshot": pd.to_numeric(records["comment_count"], errors="coerce"),
            "target_view_count": pd.to_numeric(records["view_count"], errors="coerce"),
            "target_observed_at": records["collected_at"],
            "target_age_days": (
                collected.loc[records.index] - records["published_at"]
            ).dt.total_seconds()
            / 86_400,
            "target_is_fixed_horizon": False,
            "vlm_available": False,
        }
    )
    frame = frame.merge(
        cadence(upload_dates(run_dir)), on="video_id", how="left", validate="one_to_one"
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)

    run_id = args.run_id or f"feature-map-{args.history}"
    output = MODELING_RUNS_DIR / run_id
    output.mkdir(parents=True, exist_ok=False)
    frame.to_json(output / "feature_map.jsonl", orient="records", lines=True, force_ascii=False)
    manifest = {
        "source": str(run_dir.relative_to(ROOT)),
        "rows": len(frame),
        "by_sector": frame["sector"].value_counts().to_dict(),
        "channels": int(frame["channel_id"].nunique()),
        "eligibility": "Vídeo público, duração positiva de até 180 s; sem período prospectivo.",
        "metrics_snapshot": "collected_at de cada canal (21/10/2026); alvo, nunca feature.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"feature_map": str(output / "feature_map.jsonl"), **manifest}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
