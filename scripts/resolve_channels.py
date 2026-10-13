"""Pesquisa os canais candidatos e salva as respostas para validação manual."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from youtube_extract import load_dotenv, save_json, utc_now, youtube_get

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("YOUTUBE_API_KEY"):
        print("ERRO: YOUTUBE_API_KEY não configurada", file=sys.stderr)
        return 1

    config = json.loads(
        (PROJECT_ROOT / "config" / "channel_candidates.json").read_text(encoding="utf-8")
    )
    collected_at = utc_now()
    output = {
        "collected_at": collected_at,
        "source": {
            "api": "YouTube Data API v3",
            "endpoint": "search.list",
            "parameters": {
                "type": "channel",
                "maxResults": 5,
                "regionCode": "BR",
                "relevanceLanguage": "pt",
            },
        },
        "queries": [],
    }

    for candidate in config["candidates"]:
        print(f"Pesquisando: {candidate['sector']} / {candidate['label']}")
        response = youtube_get(
            "search",
            {
                "part": "snippet",
                "q": candidate["query"],
                "type": "channel",
                "maxResults": 5,
                "regionCode": "BR",
                "relevanceLanguage": "pt",
            },
        )
        matches = []
        for rank, item in enumerate(response.get("items", []), start=1):
            snippet = item.get("snippet", {})
            matches.append(
                {
                    "rank": rank,
                    "channel_id": item.get("id", {}).get("channelId"),
                    "channel_title": snippet.get("channelTitle") or snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt"),
                }
            )
        output["queries"].append({**candidate, "matches": matches})

    path = (
        PROJECT_ROOT
        / "data"
        / "youtube"
        / "channel_resolution"
        / f"resolution-{collected_at.replace(':', '-')}.json"
    )
    save_json(path, output)
    print(f"Resultados salvos em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
