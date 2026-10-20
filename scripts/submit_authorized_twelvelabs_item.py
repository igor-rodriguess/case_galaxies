"""Submete um item autorizado da fila caso-controle ao TwelveLabs.

Baixa apenas o vídeo solicitado, gera a abertura de 15 segundos e cria uma
tarefa assíncrona. A coleta é feita depois por ``twelvelabs_resume_task.py``;
assim, a tarefa não reenvia mídia quando uma execução local termina cedo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from download_authorized_youtube_clip import planned_video  # noqa: E402

from galaxies.dataset import load_latest_vlm_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submete um item caso-controle autorizado.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--source-permission", required=True)
    parser.add_argument("--media-run-id", default="twelvelabs-case-control-media-v1")
    parser.add_argument("--feature-run-id", default="twelvelabs-case-control-v1")
    parser.add_argument("--clip-seconds", type=float, default=15.0)
    parser.add_argument("--max-height", type=int, default=360)
    parser.add_argument("--max-filesize-mb", type=int, default=180)
    parser.add_argument("--force-media", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> int:
    args = parse_args()
    if args.video_id in load_latest_vlm_results():
        print(f"Ignorado: {args.video_id} já possui análise VLM válida.")
        return 0
    item = planned_video(args.video_id)
    media_path = (
        PROJECT_ROOT
        / "data"
        / "media"
        / "authorized_youtube"
        / args.media_run_id
        / args.video_id
        / "opening.mp4"
    )
    run(
        [
            sys.executable,
            "scripts/download_authorized_youtube_clip.py",
            # Usar a forma com "=" evita que ``argparse`` trate um ID
            # legítimo que começa com hífen como uma nova opção.
            f"--video-id={args.video_id}",
            "--source-permission",
            args.source_permission,
            "--run-id",
            args.media_run_id,
            "--clip-seconds",
            str(args.clip_seconds),
            "--max-height",
            str(args.max_height),
            "--max-filesize-mb",
            str(args.max_filesize_mb),
            *( ["--force"] if args.force_media else []),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/twelvelabs_opening_pilot.py",
            f"--video-id={args.video_id}",
            "--sector",
            item["sector"],
            "--media-file",
            str(media_path),
            "--source-permission",
            args.source_permission,
            "--start",
            "0",
            "--end",
            str(args.clip_seconds),
            "--run-id",
            args.feature_run_id,
            "--submit-only",
        ]
    )
    print(f"Item submetido: {args.video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
