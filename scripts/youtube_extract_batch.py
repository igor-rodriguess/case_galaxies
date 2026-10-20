"""Extrai um lote paginado e rastreável de vários canais.

Cada página da playlist e cada resposta de detalhes é preservada em ``raw``.
O limite é por canal e pode ultrapassar os 50 itens máximos de uma chamada.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from youtube_extract import load_dotenv, save_json, utc_now, youtube_get

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "youtube" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coleta paginada dos canais selecionados.")
    parser.add_argument(
        "--max-videos-per-channel",
        type=int,
        help="Quantidade máxima por canal; padrão: YOUTUBE_MAX_VIDEOS ou 5.",
    )
    parser.add_argument("--run-id", help="Nome estável e retomável da coleta.")
    return parser.parse_args()


def chunks(values: list[str], size: int = 50) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("YOUTUBE_API_KEY"):
        print("ERRO: preencha YOUTUBE_API_KEY no .env", file=sys.stderr)
        return 1

    selected_path = PROJECT_ROOT / "config" / "selected_channels.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    channels_to_collect = selected.get("channels", [])
    max_videos = args.max_videos_per_channel or int(
        os.getenv("YOUTUBE_MAX_VIDEOS", "5")
    )
    if not channels_to_collect:
        print("ERRO: selected_channels.json não contém canais", file=sys.stderr)
        return 1
    if not 1 <= max_videos <= 5_000:
        print("ERRO: o limite por canal deve estar entre 1 e 5000", file=sys.stderr)
        return 1

    collected_at = utc_now()
    run_dir = OUTPUT_DIR / (args.run_id or collected_at.replace(":", "-"))
    if run_dir.exists():
        print(f"ERRO: a coleta já existe: {run_dir}", file=sys.stderr)
        return 1
    records = []
    results = []

    for selected_channel in channels_to_collect:
        channel_id = selected_channel["channel_id"]
        sector = selected_channel["sector"]
        expected_label = selected_channel["label"]
        channel_dir = run_dir / "raw" / channel_id
        print(f"Consultando {channel_id}...")
        try:
            channel_response = youtube_get(
                "channels",
                {"part": "snippet,contentDetails,statistics", "id": channel_id},
            )
            save_json(channel_dir / "channels.json", channel_response)
            channels = channel_response.get("items", [])
            if not channels:
                results.append({"channel_id": channel_id, "status": "not_found"})
                continue

            channel = channels[0]
            uploads_id = (
                channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            )
            playlist_items = []
            page_token = None
            page_number = 0
            while len(playlist_items) < max_videos:
                page_number += 1
                request_params = {
                    "part": "snippet,contentDetails",
                    "playlistId": uploads_id,
                    "maxResults": min(50, max_videos - len(playlist_items)),
                }
                if page_token:
                    request_params["pageToken"] = page_token
                playlist_response = youtube_get("playlistItems", request_params)
                save_json(
                    channel_dir / f"playlistItems-page-{page_number:04d}.json",
                    playlist_response,
                )
                page_items = playlist_response.get("items", [])
                playlist_items.extend(page_items)
                page_token = playlist_response.get("nextPageToken")
                if not page_items or not page_token:
                    break
            video_ids = [
                item.get("contentDetails", {}).get("videoId")
                for item in playlist_items
            ]
            video_ids = [video_id for video_id in video_ids if video_id]
            video_items = []
            for batch_number, video_batch in enumerate(chunks(video_ids), start=1):
                videos_response = youtube_get(
                    "videos",
                    {
                        "part": "snippet,contentDetails,statistics,status",
                        "id": ",".join(video_batch),
                    },
                )
                save_json(
                    channel_dir / f"videos-batch-{batch_number:04d}.json",
                    videos_response,
                )
                video_items.extend(videos_response.get("items", []))
            videos_by_id = {item.get("id"): item for item in video_items}

            for item in playlist_items:
                content = item.get("contentDetails", {})
                snippet = item.get("snippet", {})
                video_id = content.get("videoId")
                video = videos_by_id.get(video_id, {})
                vs = video.get("snippet", {})
                vc = video.get("contentDetails", {})
                stats = video.get("statistics", {})
                records.append(
                    {
                        "video_id": video_id,
                        "sector": sector,
                        "selected_channel_label": expected_label,
                        "channel_id": channel.get("id", channel_id),
                        "channel_title": channel.get("snippet", {}).get("title"),
                        "channel_subscriber_count_at_collection": channel.get("statistics", {}).get(
                            "subscriberCount"
                        ),
                        "channel_video_count_at_collection": channel.get("statistics", {}).get(
                            "videoCount"
                        ),
                        "uploads_playlist_id": uploads_id,
                        "playlist_position": snippet.get("position"),
                        "published_at": vs.get("publishedAt") or content.get("videoPublishedAt"),
                        "collected_at": collected_at,
                        "title": vs.get("title"),
                        "description": vs.get("description"),
                        "tags": vs.get("tags"),
                        "category_id": vs.get("categoryId"),
                        "duration": vc.get("duration"),
                        "definition": vc.get("definition"),
                        "view_count": stats.get("viewCount"),
                        "like_count": stats.get("likeCount"),
                        "comment_count": stats.get("commentCount"),
                        "source": {
                            "api": "YouTube Data API v3",
                            "endpoints": ["channels.list", "playlistItems.list", "videos.list"],
                            "raw_directory": f"raw/{channel_id}",
                            "selection_file": "config/selected_channels.json",
                        },
                    }
                )
            results.append(
                {
                    "sector": sector,
                    "selected_channel_label": expected_label,
                    "channel_id": channel_id,
                    "channel_title": channel.get("snippet", {}).get("title"),
                    "status": "ok",
                    "videos": len(video_ids),
                    "playlist_pages": page_number,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "sector": sector,
                    "selected_channel_label": expected_label,
                    "channel_id": channel_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    save_json(run_dir / "normalized_records.json", {"records": records})
    save_json(
        run_dir / "run_metadata.json",
        {
            "collected_at": collected_at,
            "selection_version": selected.get("selection_version"),
            "selection_method": selected.get("selection_method"),
            "channels_requested": channels_to_collect,
            "max_videos_per_channel": max_videos,
            "results": results,
        },
    )
    print(f"Concluído: {len(records)} vídeos em {len(channels_to_collect)} canais.")
    print(f"Saída: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
