"""Busca canais de criadores de moda e beleza que publicam Shorts.

Usa só a YouTube Data API v3 oficial: ``search.list`` para achar canais,
``channels.list`` para país e inscritos, e os 50 últimos uploads
(``playlistItems.list`` + ``videos.list``) para medir quanto o canal publica
em Shorts. Critérios fixados antes da busca:

* país BR e pelo menos ``--min-subscribers`` inscritos;
* pelo menos ``--min-shorts`` Shorts (até 180 s) entre os 50 últimos uploads;
* último upload há no máximo 90 dias;
* likes visíveis em pelo menos metade dos 50 últimos uploads;
* fora da seleção atual.

O resultado é uma lista de candidatos; a decisão de criador contra marca é
conferida à mão pelo título e pela descrição.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_channel_shorts_history import duration_seconds  # noqa: E402
from youtube_extract import load_dotenv, save_json, utc_now, youtube_get  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION = PROJECT_ROOT / "config" / "selected_channels.json"
OUTPUT = PROJECT_ROOT / "data" / "youtube" / "channel_discovery"

QUERIES = {
    "moda": [
        "look do dia moda feminina",
        "dicas de moda feminina",
        "estilo pessoal consultora de imagem",
        "moda masculina dicas de estilo",
        "provador de roupas achadinhos",
        "moda plus size dicas",
        "tendências de moda 2026",
        "como combinar roupas",
        "estilo masculino looks",
        "moda acessível looks baratos",
        "guarda-roupa cápsula",
        "influenciadora de moda brasileira",
    ],
    "beleza": [
        "tutorial de maquiagem",
        "skincare rotina de cuidados",
        "resenha de maquiagem",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="creators-2026-10-21")
    parser.add_argument("--min-subscribers", type=int, default=100_000)
    parser.add_argument("--min-shorts", type=int, default=25)
    parser.add_argument("--sector", choices=sorted(QUERIES), help="Limita a busca a um setor.")
    return parser.parse_args()


def search_channels(sector: str, query: str) -> list[dict]:
    response = youtube_get(
        "search",
        {"part": "snippet", "type": "channel", "q": query, "regionCode": "BR", "relevanceLanguage": "pt", "maxResults": 50},
    )
    return [
        {"sector": sector, "query": query, "channel_id": item["snippet"]["channelId"], "title": item["snippet"]["title"]}
        for item in response.get("items", [])
    ]


def channel_details(ids: list[str]) -> dict[str, dict]:
    details: dict[str, dict] = {}
    for start in range(0, len(ids), 50):
        response = youtube_get(
            "channels", {"part": "snippet,statistics,contentDetails", "id": ",".join(ids[start : start + 50])}
        )
        for item in response.get("items", []):
            details[item["id"]] = item
    return details


def recent_uploads(uploads_playlist: str) -> list[dict]:
    items = youtube_get("playlistItems", {"part": "contentDetails", "playlistId": uploads_playlist, "maxResults": 50})
    ids = [item["contentDetails"]["videoId"] for item in items.get("items", [])]
    if not ids:
        return []
    videos = youtube_get("videos", {"part": "contentDetails,statistics,snippet", "id": ",".join(ids)})
    return videos.get("items", [])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    selected = {item["channel_id"] for item in json.loads(SELECTION.read_text(encoding="utf-8"))["channels"]}

    found: dict[str, dict] = {}
    for sector, queries in QUERIES.items():
        if args.sector and sector != args.sector:
            continue
        for query in queries:
            for hit in search_channels(sector, query):
                found.setdefault(hit["channel_id"], hit)
    details = channel_details(sorted(found))
    now = datetime.now(UTC)
    candidates = []
    for channel_id, hit in found.items():
        item = details.get(channel_id)
        if not item or channel_id in selected:
            continue
        stats, snippet = item.get("statistics", {}), item.get("snippet", {})
        subscribers = int(stats.get("subscriberCount", 0) or 0)
        if snippet.get("country") != "BR" or subscribers < args.min_subscribers:
            continue
        uploads = recent_uploads(item["contentDetails"]["relatedPlaylists"]["uploads"])
        durations = [duration_seconds(video["contentDetails"].get("duration")) for video in uploads]
        shorts = sum(1 for value in durations if value is not None and 0 < value <= 180)
        likes_visible = sum("likeCount" in video.get("statistics", {}) for video in uploads)
        last = max((video["snippet"]["publishedAt"] for video in uploads), default=None)
        recent = last is not None and datetime.fromisoformat(last.replace("Z", "+00:00")) >= now - timedelta(days=90)
        record = {
            "sector": hit["sector"],
            "query": hit["query"],
            "channel_id": channel_id,
            "title": snippet.get("title"),
            "description": re.sub(r"\s+", " ", snippet.get("description", ""))[:240],
            "subscribers": subscribers,
            "video_count": int(stats.get("videoCount", 0) or 0),
            "shorts_in_last_50": shorts,
            "likes_visible_in_last_50": likes_visible,
            "last_upload": last,
            "passes": shorts >= args.min_shorts and recent and likes_visible >= len(uploads) / 2,
        }
        candidates.append(record)
    candidates.sort(key=lambda row: (row["sector"], not row["passes"], -row["subscribers"]))
    save_json(
        OUTPUT / args.run_id / "candidates.json",
        {"created_at": utc_now(), "queries": QUERIES, "criteria": vars(args), "candidates": candidates},
    )
    for row in candidates:
        if row["passes"]:
            print(
                f"{row['sector']:7s} | {row['title'][:38]:38s} | {row['subscribers']:>10,} | "
                f"Shorts {row['shorts_in_last_50']:2d}/50 | {row['channel_id']} | {row['description'][:90]}"
            )
    print(f"{sum(r['passes'] for r in candidates)} aprovados de {len(candidates)} canais BR com inscritos suficientes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
