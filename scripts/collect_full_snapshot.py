"""Relê as métricas atuais de todos os vídeos de uma coleta normalizada.

Uma segunda leitura do mesmo vídeo mostra quanto as views ainda crescem em cada
idade. Ela não reconstrói o passado e nunca vira feature: serve para auditar o
alvo e escolher a idade mínima. Cada lote de 50 IDs guarda a resposta bruta e o
horário da própria chamada. Rodar de novo com o mesmo --run-id só busca o que faltou.

Uso:
    python scripts/collect_full_snapshot.py --dry-run
    python scripts/collect_full_snapshot.py --run-id full-2026-10-19
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from collect_daily_snapshots import latest_records  # noqa: E402
from youtube_extract import load_dotenv, save_json, utc_now, youtube_get  # noqa: E402

SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "youtube" / "full_snapshots"
BATCH_SIZE = 50
PARTS = "snippet,contentDetails,statistics,status"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relê as métricas de todos os vídeos coletados.")
    parser.add_argument("--records", type=Path, help="normalized_records.json de origem.")
    parser.add_argument("--run-id", help="ID da leitura; padrão full-AAAA-MM-DD.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not args.dry_run and not os.getenv("YOUTUBE_API_KEY"):
        raise SystemExit("ERRO: preencha YOUTUBE_API_KEY no .env.")

    records_path = (args.records or latest_records()).resolve()
    records = json.loads(records_path.read_text(encoding="utf-8")).get("records", [])
    by_id = {record["video_id"]: record for record in records if record.get("video_id")}
    run_id = args.run_id or f"full-{datetime.now(UTC):%Y-%m-%d}"
    run_dir = SNAPSHOT_ROOT / run_id
    manifest_path = run_dir / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    items = {item["video_id"]: item for item in prior.get("items", [])}
    pending = [video_id for video_id in by_id if items.get(video_id, {}).get("status") != "ok"]
    batches = [pending[start : start + BATCH_SIZE] for start in range(0, len(pending), BATCH_SIZE)]
    if args.dry_run:
        print(
            json.dumps(
                {"records": str(records_path), "pending": len(pending), "calls": len(batches)}
            )
        )
        return 0

    first_batch = len(list((run_dir / "raw").glob("videos-*.json"))) + 1
    for offset, batch in enumerate(batches):
        requested_at = utc_now()
        response = youtube_get("videos", {"part": PARTS, "id": ",".join(batch)})
        received_at = utc_now()
        raw_path = run_dir / "raw" / f"videos-{first_batch + offset:04d}.json"
        save_json(raw_path, response)
        returned = {item.get("id"): item for item in response.get("items", [])}
        for video_id in batch:
            source = by_id[video_id]
            video = returned.get(video_id) or {}
            statistics = video.get("statistics", {})
            items[video_id] = {
                "video_id": video_id,
                "channel_id": source.get("channel_id"),
                "sector": source.get("sector"),
                "published_at": source.get("published_at"),
                "requested_at": requested_at,
                "observed_at": received_at,
                "status": "ok" if video else "not_returned",
                "raw_response": raw_path.relative_to(run_dir).as_posix(),
                "view_count": optional_int(statistics.get("viewCount")),
                "like_count": optional_int(statistics.get("likeCount")),
                "comment_count": optional_int(statistics.get("commentCount")),
                "duration_iso8601": video.get("contentDetails", {}).get("duration"),
                "title": video.get("snippet", {}).get("title"),
                "privacy_status": video.get("status", {}).get("privacyStatus"),
            }
        save_json(manifest_path, _manifest(run_id, records_path, items))

    manifest = _manifest(run_id, records_path, items)
    save_json(manifest_path, manifest)
    print(json.dumps({"snapshot": str(manifest_path), **manifest["summary"]}, ensure_ascii=False))
    return 0


def _manifest(run_id: str, records_path: Path, items: dict[str, dict]) -> dict:
    statuses = [item["status"] for item in items.values()]
    return {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "purpose": "Segunda leitura para medir crescimento de views por idade; nunca é feature.",
        "records_source": records_path.relative_to(PROJECT_ROOT).as_posix(),
        "source": {"api": "YouTube Data API v3", "endpoint": "videos.list", "parts": PARTS},
        "summary": {
            "videos": len(items),
            "ok": statuses.count("ok"),
            "not_returned": statuses.count("not_returned"),
        },
        "items": sorted(items.values(), key=lambda item: item["video_id"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
