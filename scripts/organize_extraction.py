"""Organiza a coleta validada em JSONs rastreáveis, sem alterar os dados brutos."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "data" / "youtube" / "runs"
SCHEMA_VERSION = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organiza uma coleta bruta do YouTube.")
    parser.add_argument(
        "--source-run-id",
        help="Nome da coleta em data/youtube/runs; padrão: a mais recente.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))


def iso_duration_seconds(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return None
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def slugify(value: str) -> str:
    """Converte um nome de canal em um nome de pasta estável e legível."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or "canal-sem-nome"


def main() -> int:
    args = parse_args()
    run_dirs = [
        path
        for path in RUNS_DIR.iterdir()
        if path.is_dir() and (path / "normalized_records.json").exists()
    ]
    if not run_dirs:
        raise RuntimeError("Nenhuma coleta encontrada")
    if args.source_run_id:
        run_dir = RUNS_DIR / args.source_run_id
        if run_dir not in run_dirs:
            raise FileNotFoundError(f"Coleta não encontrada: {run_dir}")
    else:
        run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime_ns)
    normalized = load_json(run_dir / "normalized_records.json")["records"]
    metadata = load_json(run_dir / "run_metadata.json")
    curated_dir = run_dir / "curated"

    channel_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    channel_index: dict[str, dict] = {}

    for record in normalized:
        channel_id = record["channel_id"]
        video_id = record["video_id"]
        sector = record["sector"]
        channel_title = (
            record.get("channel_title") or record.get("selected_channel_label") or channel_id
        )
        channel_folder = slugify(channel_title)
        duration_seconds = iso_duration_seconds(record.get("duration"))
        raw_base = f"raw/{channel_id}"
        raw_files = sorted(
            path.relative_to(run_dir).as_posix()
            for path in (run_dir / raw_base).glob("*.json")
        )

        video_document = {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "video_id": video_id,
                "youtube_watch_url": f"https://www.youtube.com/watch?v={video_id}",
                "channel_id": channel_id,
                "channel_folder": channel_folder,
                "uploads_playlist_id": record.get("uploads_playlist_id"),
            },
            "classification": {
                "sector": sector,
                "selected_channel_label": record.get("selected_channel_label"),
                "youtube_short_confirmed": None,
                "youtube_short_confirmation_note": "A Data API v3 não fornece neste registro um indicador direto e confiável de Short. Duração isolada não confirma o formato.",
            },
            "publication": {
                "published_at": record.get("published_at"),
                "title": record.get("title"),
                "description": record.get("description"),
                "tags": record.get("tags"),
                "category_id": record.get("category_id"),
            },
            "content_details": {
                "duration_iso8601": record.get("duration"),
                "duration_seconds": duration_seconds,
                "definition": record.get("definition"),
            },
            "metrics_snapshot": {
                "observed_at": record.get("collected_at"),
                "view_count": optional_int(record.get("view_count")),
                "like_count": optional_int(record.get("like_count")),
                "comment_count": optional_int(record.get("comment_count")),
                "interpretation_note": "Valores observados na coleta; não representam automaticamente desempenho no dia 30.",
            },
            "channel_snapshot": {
                "channel_title": record.get("channel_title"),
                "subscriber_count_observed_at_collection": optional_int(
                    record.get("channel_subscriber_count_at_collection")
                ),
                "video_count_observed_at_collection": optional_int(
                    record.get("channel_video_count_at_collection")
                ),
                "interpretation_note": "Inscritos atuais na coleta; não representam inscritos na data de publicação.",
            },
            "provenance": {
                "provider": "YouTube",
                "api": "YouTube Data API v3",
                "collected_at": record.get("collected_at"),
                "raw_path_base": "source_run_directory",
                "endpoints": ["channels.list", "playlistItems.list", "videos.list"],
                "raw_files": raw_files,
                "selection_file": "config/selected_channels.json",
                "selection_path_base": "project_root",
                "playlist_position_at_collection": record.get("playlist_position"),
            },
        }

        save_json(
            curated_dir / "videos" / sector / channel_folder / f"{video_id}.json", video_document
        )
        channel_counts[channel_id] += 1
        sector_counts[sector] += 1
        channel_index[channel_id] = {
            "channel_id": channel_id,
            "channel_title": channel_title,
            "sector": sector,
            "folder": f"curated/videos/{sector}/{channel_folder}",
        }
        for field in ("description", "tags", "view_count", "like_count", "comment_count"):
            if record.get(field) in (None, "", []):
                missing_counts[field] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "galaxies-youtube-extraction",
        "source_run": run_dir.name,
        "collected_at": metadata.get("collected_at"),
        "selection_version": metadata.get("selection_version"),
        "record_count": len(normalized),
        "unique_video_count": len({record["video_id"] for record in normalized}),
        "sector_counts": dict(sorted(sector_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "channel_index": sorted(
            channel_index.values(), key=lambda item: (item["sector"], item["channel_title"].lower())
        ),
        "missing_value_counts": dict(sorted(missing_counts.items())),
        "storage": {
            "raw": "raw/<channel_id>/{channels,playlistItems-page-*,videos-batch-*}.json",
            "curated": "curated/videos/<sector>/<channel-name>/<video_id>.json",
        },
        "known_limitations": [
            "A amostra contém uploads recentes; a Data API não confirma diretamente quais são Shorts.",
            "Métricas são snapshots no instante da coleta.",
            "Contagem de inscritos é atual na coleta, não histórica na publicação.",
        ],
    }
    save_json(curated_dir / "manifest.json", manifest)
    print(f"Organizados {len(normalized)} vídeos em {curated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
