"""Primeira extração pequena da YouTube Data API v3.

Fluxo:
1. Consulta um canal pelo channel_id.
2. Descobre a playlist de uploads.
3. Lista até N vídeos dessa playlist.
4. Consulta detalhes/metadados dos vídeos.
5. Salva respostas brutas e registros normalizados.

Uso:
    python youtube_extract.py

Não depende de bibliotecas externas.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://www.googleapis.com/youtube/v3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "youtube" / "runs"


def load_dotenv(path: Path) -> None:
    """Carrega um .env simples sem sobrescrever variáveis já existentes."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def youtube_get(resource: str, params: dict[str, str | int]) -> dict:
    query = {**params, "key": os.environ["YOUTUBE_API_KEY"]}
    url = f"{BASE_URL}/{resource}?{urlencode(query)}"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"YouTube API retornou HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Falha de rede ao consultar a YouTube API: {exc.reason}") from exc


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("YOUTUBE_API_KEY")
    channel_id = os.getenv("YOUTUBE_CHANNEL_ID")
    max_videos = int(os.getenv("YOUTUBE_MAX_VIDEOS", "5"))

    if not api_key:
        print("ERRO: preencha YOUTUBE_API_KEY no arquivo .env", file=sys.stderr)
        return 1
    if not channel_id:
        print("ERRO: preencha YOUTUBE_CHANNEL_ID no arquivo .env", file=sys.stderr)
        return 1
    if not 1 <= max_videos <= 50:
        print("ERRO: YOUTUBE_MAX_VIDEOS deve estar entre 1 e 50", file=sys.stderr)
        return 1

    collected_at = utc_now()
    run_dir = OUTPUT_DIR / collected_at.replace(":", "-")
    raw_dir = run_dir / "raw"

    print(f"Consultando canal {channel_id}...")
    channel_response = youtube_get(
        "channels",
        {"part": "snippet,contentDetails,statistics", "id": channel_id},
    )
    save_json(raw_dir / "channels.json", channel_response)

    channels = channel_response.get("items", [])
    if not channels:
        print("ERRO: nenhum canal encontrado para esse channel_id", file=sys.stderr)
        return 1

    channel = channels[0]
    uploads_playlist_id = (
        channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    )
    if not uploads_playlist_id:
        print("ERRO: a resposta não trouxe a playlist de uploads", file=sys.stderr)
        return 1

    print(f"Listando até {max_videos} vídeos da playlist {uploads_playlist_id}...")
    playlist_response = youtube_get(
        "playlistItems",
        {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": max_videos,
        },
    )
    save_json(raw_dir / "playlistItems.json", playlist_response)

    video_ids = [
        item.get("contentDetails", {}).get("videoId")
        for item in playlist_response.get("items", [])
    ]
    video_ids = [video_id for video_id in video_ids if video_id]

    videos_response = {"kind": "youtube#videoListResponse", "items": []}
    if video_ids:
        print(f"Consultando detalhes de {len(video_ids)} vídeos...")
        videos_response = youtube_get(
            "videos",
            {
                "part": "snippet,contentDetails,statistics,status",
                "id": ",".join(video_ids),
            },
        )
    save_json(raw_dir / "videos.json", videos_response)

    videos_by_id = {item.get("id"): item for item in videos_response.get("items", [])}
    records = []
    for item in playlist_response.get("items", []):
        content = item.get("contentDetails", {})
        snippet = item.get("snippet", {})
        video_id = content.get("videoId")
        video = videos_by_id.get(video_id, {})
        video_snippet = video.get("snippet", {})
        video_details = video.get("contentDetails", {})
        statistics = video.get("statistics", {})

        records.append(
            {
                "video_id": video_id,
                "channel_id": channel.get("id", channel_id),
                "channel_title": channel.get("snippet", {}).get("title"),
                "uploads_playlist_id": uploads_playlist_id,
                "playlist_position": snippet.get("position"),
                "published_at": video_snippet.get("publishedAt") or content.get("videoPublishedAt"),
                "collected_at": collected_at,
                "title": video_snippet.get("title"),
                "description": video_snippet.get("description"),
                "tags": video_snippet.get("tags"),
                "category_id": video_snippet.get("categoryId"),
                "duration": video_details.get("duration"),
                "definition": video_details.get("definition"),
                "view_count": statistics.get("viewCount"),
                "like_count": statistics.get("likeCount"),
                "comment_count": statistics.get("commentCount"),
                "source": {
                    "api": "YouTube Data API v3",
                    "channel_endpoint": "channels.list",
                    "playlist_endpoint": "playlistItems.list",
                    "video_endpoint": "videos.list",
                    "channel_id_requested": channel_id,
                },
            }
        )

    save_json(run_dir / "normalized_records.json", {"records": records})
    save_json(
        run_dir / "run_metadata.json",
        {
            "collected_at": collected_at,
            "channel_id_requested": channel_id,
            "uploads_playlist_id": uploads_playlist_id,
            "max_videos_requested": max_videos,
            "video_ids_returned": video_ids,
            "raw_responses": ["raw/channels.json", "raw/playlistItems.json", "raw/videos.json"],
        },
    )

    print(f"Concluído: {len(records)} registros salvos em {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
