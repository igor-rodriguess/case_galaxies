"""Coleta snapshots imutáveis de vídeos novos para o futuro alvo de 30 dias.

O YouTube Data API só revela métricas atuais. Este coletor não tenta reconstruir
o passado: identifica vídeos que estão, hoje, nas janelas de 1/3/7/14/30 dias
e salva a resposta bruta com horário de observação. Rode diariamente pelo
Agendador do Windows ou CI; repetir no mesmo dia não duplica o mesmo vídeo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from youtube_extract import load_dotenv, save_json, utc_now, youtube_get  # noqa: E402

TARGET_AGES = (1, 3, 7, 14, 30)
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "youtube" / "daily_snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coleta snapshots diários de métricas YouTube.")
    parser.add_argument("--records", type=Path, help="normalized_records.json de origem.")
    parser.add_argument("--tolerance-hours", type=float, default=18.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def latest_records() -> Path:
    candidates = list((PROJECT_ROOT / "data" / "youtube" / "runs").glob("*/normalized_records.json"))
    if not candidates:
        raise FileNotFoundError("Nenhum normalized_records.json encontrado.")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.parent.name))


def candidate_rows(records: list[dict], now: pd.Timestamp, tolerance_hours: float) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["video_id", "published_at"]).copy()
    age_days = (now - frame["published_at"]).dt.total_seconds() / 86400
    targets = pd.DataFrame({day: (age_days - day).abs() for day in TARGET_AGES})
    frame["snapshot_age_day"] = targets.idxmin(axis=1).astype(int)
    distance = (age_days - frame["snapshot_age_day"]).abs() * 24
    return frame.loc[distance <= tolerance_hours].copy()


def main() -> int:
    args = parse_args()
    if args.tolerance_hours <= 0 or args.tolerance_hours > 24:
        raise SystemExit("ERRO: --tolerance-hours deve estar entre 0 e 24.")
    load_dotenv(PROJECT_ROOT / ".env")
    if not args.dry_run and not os.getenv("YOUTUBE_API_KEY"):
        raise SystemExit("ERRO: preencha YOUTUBE_API_KEY no .env.")

    records_path = args.records or latest_records()
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    now = pd.Timestamp(datetime.now(UTC))
    candidates = candidate_rows(payload.get("records", []), now, args.tolerance_hours)
    day_dir = SNAPSHOT_ROOT / now.strftime("%Y-%m-%d")
    manifest_path = day_dir / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"items": []}
    already = {(item["video_id"], item["target_age_day"]) for item in prior.get("items", [])}
    candidates = candidates.loc[
        ~candidates.apply(lambda row: (row.video_id, int(row.snapshot_age_day)) in already, axis=1)
    ]
    if args.dry_run:
        print(json.dumps({"records": str(records_path), "eligible": len(candidates)}, ensure_ascii=False))
        return 0

    items = list(prior.get("items", []))
    for batch_start in range(0, len(candidates), 50):
        batch = candidates.iloc[batch_start : batch_start + 50]
        response = youtube_get("videos", {"part": "snippet,contentDetails,statistics,status", "id": ",".join(batch.video_id)})
        raw_path = day_dir / "raw" / f"videos-{batch_start // 50 + 1:04d}.json"
        save_json(raw_path, response)
        returned = {item.get("id"): item for item in response.get("items", [])}
        for _, source in batch.iterrows():
            video = returned.get(source.video_id)
            items.append(
                {
                    "video_id": source.video_id,
                    "channel_id": source.get("channel_id"),
                    "sector": source.get("sector"),
                    "published_at": source.published_at.isoformat(),
                    "target_age_day": int(source.snapshot_age_day),
                    "observed_at": utc_now(),
                    "status": "ok" if video else "not_returned",
                    "raw_response": str(raw_path.relative_to(day_dir)),
                    "metrics": (video or {}).get("statistics", {}),
                    "source": {"api": "YouTube Data API v3", "endpoint": "videos.list", "records": str(records_path)},
                }
            )
    manifest = {
        "schema_version": "0.1.0",
        "observed_date": now.strftime("%Y-%m-%d"),
        "target_ages_days": list(TARGET_AGES),
        "tolerance_hours": args.tolerance_hours,
        "records_source": str(records_path),
        "items": items,
    }
    save_json(manifest_path, manifest)
    print(json.dumps({"snapshot": str(manifest_path), "new_items": len(candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
