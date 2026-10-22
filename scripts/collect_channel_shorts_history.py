"""Coleta o histórico completo de Shorts dos canais selecionados.

Para cada canal: enumera a playlist de uploads inteira, lê a duração de todos os
vídeos e busca os detalhes completos só dos que têm até ``--max-seconds``. Toda
resposta da API é guardada em ``raw``. O resultado fica em
``data/youtube/shorts_history/<run_id>/``, fora de ``data/youtube/runs``, para
não virar a "coleta mais recente" usada por outros scripts.

As métricas são o snapshot do instante da coleta (``collected_at`` por canal).
Elas servem para construir o alvo, nunca como feature. A coleta é retomável:
canais com ``done.json`` não são repetidos.

Uso:
    python scripts/collect_channel_shorts_history.py --run-id shorts-history-2026-10-21
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from youtube_extract import load_dotenv, save_json, utc_now, youtube_get

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "youtube" / "shorts_history"
SELECTION = PROJECT_ROOT / "config" / "selected_channels.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--channel", action="append", help="Limita a um ou mais channel_id.")
    return parser.parse_args()


def duration_seconds(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"P(?:(?P<d>\d+)D)?T?(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?", value
    )
    if not match or value in {"P", "PT"}:
        return None
    return (
        int(match.group("d") or 0) * 86_400
        + int(match.group("h") or 0) * 3_600
        + int(match.group("m") or 0) * 60
        + float(match.group("s") or 0)
    )


def chunks(values: list[str], size: int = 50) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def collect_channel(selected: dict, run_dir: Path, max_seconds: float) -> dict:
    channel_id = selected["channel_id"]
    channel_dir = run_dir / "raw" / channel_id
    collected_at = utc_now()
    channel_response = youtube_get(
        "channels", {"part": "snippet,contentDetails,statistics", "id": channel_id}
    )
    save_json(channel_dir / "channels.json", channel_response)
    if not channel_response.get("items"):
        return {"channel_id": channel_id, "status": "not_found"}
    channel = channel_response["items"][0]
    uploads_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]

    playlist_items: list[dict] = []
    page_token, page = None, 0
    while True:
        page += 1
        params = {"part": "contentDetails", "playlistId": uploads_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        response = youtube_get("playlistItems", params)
        save_json(channel_dir / f"playlistItems-page-{page:04d}.json", response)
        playlist_items.extend(response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token or not response.get("items"):
            break
    video_ids = list(
        dict.fromkeys(
            item["contentDetails"]["videoId"]
            for item in playlist_items
            if item.get("contentDetails", {}).get("videoId")
        )
    )

    durations: dict[str, float | None] = {}
    for number, batch in enumerate(chunks(video_ids), start=1):
        response = youtube_get("videos", {"part": "contentDetails", "id": ",".join(batch)})
        save_json(channel_dir / f"durations-batch-{number:04d}.json", response)
        for item in response.get("items", []):
            durations[item["id"]] = duration_seconds(item.get("contentDetails", {}).get("duration"))
    shorts = [
        vid for vid in video_ids if (durations.get(vid) or 0) > 0 and durations[vid] <= max_seconds
    ]

    records = []
    for number, batch in enumerate(chunks(shorts), start=1):
        response = youtube_get(
            "videos", {"part": "snippet,contentDetails,statistics,status", "id": ",".join(batch)}
        )
        save_json(channel_dir / f"shorts-details-batch-{number:04d}.json", response)
        for video in response.get("items", []):
            snippet, details, stats = (
                video.get("snippet", {}),
                video.get("contentDetails", {}),
                video.get("statistics", {}),
            )
            records.append(
                {
                    "video_id": video["id"],
                    "sector": selected["sector"],
                    "selected_channel_label": selected["label"],
                    "channel_id": channel_id,
                    "channel_title": channel.get("snippet", {}).get("title"),
                    "uploads_playlist_id": uploads_id,
                    "published_at": snippet.get("publishedAt"),
                    "collected_at": collected_at,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "tags": snippet.get("tags"),
                    "category_id": snippet.get("categoryId"),
                    "duration": details.get("duration"),
                    "duration_seconds": duration_seconds(details.get("duration")),
                    "definition": details.get("definition"),
                    "privacy_status": video.get("status", {}).get("privacyStatus"),
                    "view_count": stats.get("viewCount"),
                    "like_count": stats.get("likeCount"),
                    "comment_count": stats.get("commentCount"),
                    "source": {
                        "api": "YouTube Data API v3",
                        "endpoints": ["channels.list", "playlistItems.list", "videos.list"],
                        "raw_directory": f"raw/{channel_id}",
                        "metrics_note": "Snapshot em collected_at; alvo, nunca feature.",
                    },
                }
            )
    save_json(channel_dir / "records.json", {"records": records})
    with (channel_dir / "all_uploads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "duration_seconds", "is_short_candidate"])
        for vid in video_ids:
            writer.writerow([vid, durations.get(vid), vid in set(shorts)])
    summary = {
        "channel_id": channel_id,
        "sector": selected["sector"],
        "label": selected["label"],
        "status": "ok",
        "collected_at": collected_at,
        "channel_video_count_reported": channel.get("statistics", {}).get("videoCount"),
        "uploads_enumerated": len(video_ids),
        "durations_returned": len(durations),
        "shorts_candidates": len(shorts),
        "shorts_records": len(records),
        "playlist_pages": page,
    }
    save_json(channel_dir / "done.json", summary)
    return summary


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("YOUTUBE_API_KEY"):
        raise SystemExit("ERRO: preencha YOUTUBE_API_KEY no .env.")
    run_dir = OUTPUT_ROOT / args.run_id
    channels = json.loads(SELECTION.read_text(encoding="utf-8"))["channels"]
    if args.channel:
        channels = [item for item in channels if item["channel_id"] in set(args.channel)]

    summaries = []
    for selected in channels:
        done = run_dir / "raw" / selected["channel_id"] / "done.json"
        if done.exists():
            summaries.append(json.loads(done.read_text(encoding="utf-8")))
            continue
        print(f"{selected['sector']} | {selected['label']}...", flush=True)
        try:
            summaries.append(collect_channel(selected, run_dir, args.max_seconds))
        except RuntimeError as exc:
            summaries.append(
                {"channel_id": selected["channel_id"], "status": "error", "error": str(exc)[:500]}
            )
            if "quotaExceeded" in str(exc):
                print("Cota esgotada; rode de novo amanhã com o mesmo --run-id.")
                break
        print(
            f"  {summaries[-1].get('status')}: {summaries[-1].get('shorts_records', 0)} Shorts",
            flush=True,
        )

    records = []
    for summary in summaries:
        path = run_dir / "raw" / summary["channel_id"] / "records.json"
        if summary.get("status") == "ok" and path.exists():
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    save_json(run_dir / "normalized_records.json", {"records": records})
    save_json(
        run_dir / "run_metadata.json",
        {
            "run_id": args.run_id,
            "updated_at": utc_now(),
            "max_seconds": args.max_seconds,
            "selection_file": "config/selected_channels.json",
            "criterion": "Duração até max_seconds na Data API; a API não marca Short.",
            "channels": summaries,
            "shorts_records": len(records),
            "complete": all(item.get("status") == "ok" for item in summaries)
            and len(summaries) == len(channels),
        },
    )
    print(
        json.dumps(
            {
                "shorts_records": len(records),
                "channels_ok": sum(s.get("status") == "ok" for s in summaries),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
