"""Baixa e recorta mídia de um vídeo YouTube expressamente autorizado.

O script aceita apenas IDs presentes no plano caso-controle congelado. O rótulo
viral e as métricas não saem do plano: eles nunca são enviados ao provedor VLM.
Os arquivos são ignorados pelo Git, mas cada download recebe um manifesto com
origem, permissão declarada, hash, formato e timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "data" / "features" / "plans" / "case-control-opening-v1" / "manifest.json"
MEDIA_ROOT = PROJECT_ROOT / "data" / "media" / "authorized_youtube"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planned_video(video_id: str) -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    for item in plan["queue"]:
        if item["video_id"] == video_id:
            return {
                "video_id": item["video_id"],
                "sector": item["sector"],
                "channel_id": item["channel_id"],
                "channel_title": item["channel_title"],
                "published_at": item["published_at"],
                "format": item["format"],
                "duration_seconds": item["duration_seconds"],
                "position": item["position"],
            }
    raise SystemExit(f"ERRO: {video_id} não pertence à fila caso-controle autorizada.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa uma abertura YouTube autorizada e rastreável.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--source-permission",
        required=True,
        help="Registro curto da permissão dos canais para este uso.",
    )
    parser.add_argument("--run-id", default="youtube-media-pilot")
    parser.add_argument("--clip-seconds", type=float, default=15.0)
    parser.add_argument("--max-height", type=int, default=360)
    parser.add_argument("--max-filesize-mb", type=int, default=180)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Gera novamente o clip quando ele precisa de nova normalização.",
    )
    return parser.parse_args()


def find_downloaded_file(output_dir: Path, pattern: str) -> Path:
    candidates = [path for path in output_dir.glob(pattern) if path.suffix != ".part"]
    if len(candidates) != 1:
        raise RuntimeError(f"Esperava um único arquivo fonte em {output_dir}, encontrei: {candidates}")
    return candidates[0]


def main() -> int:
    args = parse_args()
    if args.clip_seconds <= 0:
        raise SystemExit("ERRO: --clip-seconds deve ser positivo.")
    if args.max_height < 144:
        raise SystemExit("ERRO: --max-height deve ser pelo menos 144.")
    if args.max_filesize_mb <= 0:
        raise SystemExit("ERRO: --max-filesize-mb deve ser positivo.")
    if not PLAN_PATH.is_file():
        raise SystemExit(f"ERRO: plano não encontrado: {PLAN_PATH}")
    try:
        import imageio_ffmpeg
        import yt_dlp
    except ImportError as exc:
        raise SystemExit('ERRO: instale com pip install -e ".[youtube-media]".') from exc

    planned = planned_video(args.video_id)
    output_dir = MEDIA_ROOT / args.run_id / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / "opening.mp4"
    manifest_path = output_dir / "media_manifest.json"
    source_url = f"https://www.youtube.com/watch?v={args.video_id}"

    if clip_path.is_file() and manifest_path.is_file() and not args.force:
        print(f"Clip já disponível: {clip_path}")
        return 0

    started_at = utc_now()
    save_json(
        manifest_path,
        {
            "schema_version": "0.1.0",
            "status": "started",
            "started_at": started_at,
            "permission": args.source_permission,
            "source": {"provider": "YouTube", "watch_url": source_url, **planned},
            "download_policy": {
                "max_height": args.max_height,
                "max_filesize_mb": args.max_filesize_mb,
                "requested_opening_seconds": args.clip_seconds,
            },
        },
    )
    download_options = {
        "noplaylist": True,
        "max_filesize": args.max_filesize_mb * 1024 * 1024,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        # yt-dlp não reconhece sempre o FFmpeg empacotado pelo Python. Baixamos as
        # faixas separadas e invocamos explicitamente o binário validado abaixo.
        with yt_dlp.YoutubeDL(
            {
                **download_options,
                "format": f"bestvideo[height<={args.max_height}]/bestvideo",
                "outtmpl": str(output_dir / "source_video.%(ext)s"),
            }
        ) as downloader:
            info = downloader.extract_info(source_url, download=True)
        with yt_dlp.YoutubeDL(
            {
                **download_options,
                "format": "bestaudio/best",
                "outtmpl": str(output_dir / "source_audio.%(ext)s"),
            }
        ) as downloader:
            downloader.extract_info(source_url, download=True)
        video_path = find_downloaded_file(output_dir, "source_video.*")
        audio_path = find_downloaded_file(output_dir, "source_audio.*")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-t",
            str(args.clip_seconds),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            # TwelveLabs exige pelo menos 360 px em ambos os lados. O filtro
            # preserva a proporção: 360p de altura em paisagem ou 360p de largura
            # em retrato. Upscaling é declarado no manifesto, não é um ganho de
            # informação visual.
            "-vf",
            "scale=w='if(gte(iw,ih),-2,360)':h='if(gte(iw,ih),360,-2)'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        video_path.unlink()
        audio_path.unlink()
        save_json(
            manifest_path,
            {
                "schema_version": "0.1.0",
                "status": "complete",
                "started_at": started_at,
                "completed_at": utc_now(),
                "permission": args.source_permission,
                "source": {
                    "provider": "YouTube",
                    "watch_url": source_url,
                    "webpage_url": info.get("webpage_url"),
                    "extractor": info.get("extractor_key"),
                    **planned,
                },
                "download_policy": {
                    "max_height": args.max_height,
                    "max_filesize_mb": args.max_filesize_mb,
                    "requested_opening_seconds": args.clip_seconds,
                },
                "normalization": {
                    "minimum_dimension_pixels": 360,
                    "method": "aspect-ratio-preserving FFmpeg scale",
                    "note": "Upscaling atende ao requisito do provedor; não cria detalhe visual novo.",
                },
                "file": {
                    "name": clip_path.name,
                    "bytes": clip_path.stat().st_size,
                    "sha256": sha256_file(clip_path),
                    "duration_seconds_requested": args.clip_seconds,
                },
            },
        )
    except Exception as exc:
        save_json(
            manifest_path,
            {
                "schema_version": "0.1.0",
                "status": "failed",
                "started_at": started_at,
                "failed_at": utc_now(),
                "permission": args.source_permission,
                "source": {"provider": "YouTube", "watch_url": source_url, **planned},
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    print(f"Clip pronto: {clip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
