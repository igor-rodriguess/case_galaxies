"""Fila prospectiva: descobre Shorts novos e lê as views em idades fixas.

Rodar uma vez por dia. Em cada execução:

1. **Descoberta.** Lê o topo da playlist de uploads dos 30 canais e registra
   cada vídeo publicado a partir de ``PROSPECTIVE_START`` ainda não visto. O
   título, a descrição, as tags e a thumbnail são guardados no primeiro contato,
   o mais perto possível da publicação. Só Shorts (até 180 s) entram na fila.
2. **Leituras.** Para cada Short da fila, lê as métricas nas idades de 1, 3, 7,
   14 e 30 dias, cada uma dentro de ±18 h. Janela fora da tolerância é marcada
   como perdida; nada é interpolado. A leitura de 30 dias é o rótulo da
   validação final (`docs/validacao_final_prospectiva.md`).

Estado em ``data/youtube/prospective/``: ``registry.json`` (vídeos e janelas),
``observations.jsonl`` (uma linha por leitura) e ``raw/`` (respostas da API).

Uso:
    python scripts/track_prospective_shorts.py
    python scripts/track_prospective_shorts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from collect_channel_shorts_history import duration_seconds  # noqa: E402
from youtube_extract import load_dotenv, save_json, utc_now, youtube_get  # noqa: E402

from galaxies.prospective import (  # noqa: E402
    HORIZON_DAYS,
    MIN_VIDEOS_PER_SECTOR,
    PROSPECTIVE_START,
    is_tracked_short,
    window_status,
)

STATE_DIR = PROJECT_ROOT / "data" / "youtube" / "prospective"
SELECTION = PROJECT_ROOT / "config" / "selected_channels.json"
MAX_DISCOVERY_PAGES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Mostra o estado sem chamar a API.")
    return parser.parse_args()


def load_registry() -> dict[str, dict]:
    path = STATE_DIR / "registry.json"
    return json.loads(path.read_text(encoding="utf-8"))["videos"] if path.exists() else {}


def save_registry(videos: dict[str, dict]) -> None:
    save_json(
        STATE_DIR / "registry.json",
        {
            "schema_version": "0.1.0",
            "prospective_start": PROSPECTIVE_START.isoformat(),
            "updated_at": utc_now(),
            "videos": videos,
        },
    )


def download_thumbnail(video_id: str) -> str | None:
    target = STATE_DIR / "thumbnails" / f"{video_id}.jpg"
    if target.exists():
        return target.relative_to(STATE_DIR).as_posix()
    try:
        with urlopen(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg", timeout=30) as response:
            content = response.read()
    except (URLError, TimeoutError):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target.relative_to(STATE_DIR).as_posix()


def discover(videos: dict[str, dict], raw_dir: Path) -> int:
    channels = json.loads(SELECTION.read_text(encoding="utf-8"))["channels"]
    new_ids: dict[str, dict] = {}
    for selected in channels:
        uploads = "UU" + selected["channel_id"][2:]
        page_token = None
        for page in range(1, MAX_DISCOVERY_PAGES + 1):
            params = {"part": "contentDetails", "playlistId": uploads, "maxResults": 50}
            if page_token:
                params["pageToken"] = page_token
            response = youtube_get("playlistItems", params)
            save_json(raw_dir / f"playlist-{selected['channel_id']}-{page}.json", response)
            reached_old = False
            for item in response.get("items", []):
                details = item.get("contentDetails", {})
                published = pd.Timestamp(details.get("videoPublishedAt") or "1970-01-01T00:00:00Z")
                if published < PROSPECTIVE_START:
                    reached_old = True
                    continue
                if details["videoId"] not in videos:
                    new_ids[details["videoId"]] = selected
            page_token = response.get("nextPageToken")
            if reached_old or not page_token:
                break

    ids = list(new_ids)
    for number, start in enumerate(range(0, len(ids), 50), start=1):
        batch = ids[start : start + 50]
        response = youtube_get(
            "videos", {"part": "snippet,contentDetails,status", "id": ",".join(batch)}
        )
        raw_file = raw_dir / f"discovery-{number:03d}.json"
        save_json(raw_file, response)
        for video in response.get("items", []):
            selected = new_ids[video["id"]]
            snippet, details = video.get("snippet", {}), video.get("contentDetails", {})
            published = pd.Timestamp(snippet.get("publishedAt"))
            seconds = duration_seconds(details.get("duration"))
            tracked = is_tracked_short(published, seconds)
            videos[video["id"]] = {
                "video_id": video["id"],
                "sector": selected["sector"],
                "channel_id": selected["channel_id"],
                "channel_label": selected["label"],
                "published_at": published.isoformat(),
                "duration_seconds": seconds,
                "tracked": tracked,
                "first_seen_at": utc_now(),
                "at_discovery": {
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "tags": snippet.get("tags"),
                    "category_id": snippet.get("categoryId"),
                    "privacy_status": video.get("status", {}).get("privacyStatus"),
                    "thumbnail_file": download_thumbnail(video["id"]) if tracked else None,
                    "raw_response": raw_file.relative_to(STATE_DIR).as_posix(),
                },
                "observed_ages": [],
                "missed_ages": [],
            }
    return len(ids)


def observe(videos: dict[str, dict], raw_dir: Path, now: pd.Timestamp) -> int:
    due: dict[str, list[int]] = {}
    for video in videos.values():
        if not video["tracked"]:
            continue
        status = window_status(
            pd.Timestamp(video["published_at"]), now, set(video["observed_ages"])
        )
        video["missed_ages"] = sorted(set(video["missed_ages"]) | set(status["missed"]))
        if status["due"]:
            due[video["video_id"]] = status["due"]

    ids = list(due)
    written = 0
    with (STATE_DIR / "observations.jsonl").open("a", encoding="utf-8") as handle:
        for number, start in enumerate(range(0, len(ids), 50), start=1):
            batch = ids[start : start + 50]
            response = youtube_get("videos", {"part": "statistics", "id": ",".join(batch)})
            observed_at = pd.Timestamp(utc_now())
            raw_file = raw_dir / f"observations-{number:03d}.json"
            save_json(raw_file, response)
            returned = {
                item["id"]: item.get("statistics", {}) for item in response.get("items", [])
            }
            for video_id in batch:
                video = videos[video_id]
                stats = returned.get(video_id)
                age = (observed_at - pd.Timestamp(video["published_at"])).total_seconds() / 86_400
                for target in due[video_id]:
                    line = {
                        "video_id": video_id,
                        "sector": video["sector"],
                        "channel_id": video["channel_id"],
                        "target_age_day": target,
                        "age_days": round(age, 4),
                        "observed_at": observed_at.isoformat(),
                        "status": "ok" if stats is not None else "not_returned",
                        "view_count": int(stats["viewCount"])
                        if stats and "viewCount" in stats
                        else None,
                        "like_count": int(stats["likeCount"])
                        if stats and "likeCount" in stats
                        else None,
                        "comment_count": int(stats["commentCount"])
                        if stats and "commentCount" in stats
                        else None,
                        "raw_response": raw_file.relative_to(STATE_DIR).as_posix(),
                    }
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                    video["observed_ages"] = sorted(set(video["observed_ages"]) | {target})
                    written += 1
    return written


def summary(videos: dict[str, dict]) -> dict:
    tracked = [video for video in videos.values() if video["tracked"]]
    labelled = Counter(
        video["sector"] for video in tracked if HORIZON_DAYS in video["observed_ages"]
    )
    missed = Counter(video["sector"] for video in tracked if HORIZON_DAYS in video["missed_ages"])
    return {
        "videos_seen": len(videos),
        "shorts_tracked": Counter(video["sector"] for video in tracked),
        "labels_30d": labelled,
        "missed_30d": missed,
        "ready_to_open": all(
            labelled.get(sector, 0) >= MIN_VIDEOS_PER_SECTOR
            for sector in ("automotivo", "moda", "beleza")
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    videos = load_registry()
    if args.dry_run:
        print(json.dumps(summary(videos), ensure_ascii=False, indent=2))
        return 0
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("YOUTUBE_API_KEY"):
        raise SystemExit("ERRO: preencha YOUTUBE_API_KEY no .env.")
    now = pd.Timestamp(utc_now())
    raw_dir = STATE_DIR / "raw" / now.strftime("%Y-%m-%dT%H-%M-%SZ")
    discovered = discover(videos, raw_dir)
    save_registry(videos)
    written = observe(videos, raw_dir, now)
    save_registry(videos)
    print(
        json.dumps(
            {"new_videos": discovered, "observations_written": written, **summary(videos)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
