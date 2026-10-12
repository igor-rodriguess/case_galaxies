"""Executa o extrator validado de abertura nos vídeos ainda não processados.

O lote é retomável: antes de chamar a API, procura um opening_vlm.json válido para
cada video_id em qualquer execução anterior. Resultados existentes são reutilizados.

Uso:
    python scripts/gemini_opening_batch.py --dry-run
    python scripts/gemini_opening_batch.py
    python scripts/gemini_opening_batch.py --run-id batch-opening-2026-10-12
"""

from __future__ import annotations

import argparse
import json
import locale
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOUTUBE_RUNS_DIR = PROJECT_ROOT / "data" / "youtube" / "runs"
FEATURE_RUNS_DIR = PROJECT_ROOT / "data" / "features" / "runs"
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "gemini_opening_pilot.py"
DEFAULT_MODEL = "gemini-3.8-flash"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def latest_youtube_records() -> tuple[Path, list[dict]]:
    candidates = sorted(
        (
            run_dir
            for run_dir in YOUTUBE_RUNS_DIR.iterdir()
            if run_dir.is_dir() and (run_dir / "normalized_records.json").exists()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Nenhuma coleta normalizada do YouTube foi encontrada.")

    run_dir = candidates[0]
    payload = json.loads(
        (run_dir / "normalized_records.json").read_text(encoding="utf-8")
    )
    records = payload.get("records", [])
    if not records:
        raise ValueError("A coleta mais recente não contém registros de vídeo.")

    ids = [record.get("video_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("A coleta contém video_ids duplicados.")
    return run_dir, records


def existing_successes() -> dict[str, dict]:
    """Retorna o resultado mais recente de cada vídeo já processado com sucesso."""
    successes: dict[str, dict] = {}
    if not FEATURE_RUNS_DIR.exists():
        return successes

    for result_path in FEATURE_RUNS_DIR.rglob("opening_vlm.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            video_id = result.get("identity", {}).get("video_id")
            completed_at = result.get("provenance", {}).get("completed_at", "")
        except (OSError, json.JSONDecodeError):
            continue
        if not video_id:
            continue

        previous = successes.get(video_id)
        if previous is None or completed_at > previous["completed_at"]:
            successes[video_id] = {
                "completed_at": completed_at,
                "result": relative_path(result_path),
            }
    return successes


def is_valid_run_id(value: str) -> bool:
    cleaned = value.replace("-", "").replace("_", "").replace(".", "")
    return bool(cleaned) and cleaned.isalnum() and value not in {".", ".."}


def refresh_summary(manifest: dict) -> None:
    counts = Counter(video["status"] for video in manifest["videos"])
    available = counts["reused_existing"] + counts["completed"]
    manifest["summary"] = {
        "videos_total": len(manifest["videos"]),
        "results_available": available,
        "reused_existing": counts["reused_existing"],
        "completed_in_batch": counts["completed"],
        "pending": counts["pending"],
        "failed": counts["failed"],
        "quota_blocked": counts["quota_blocked"],
        "running": counts["running"],
    }
    manifest["updated_at"] = utc_now()


def build_manifest(
    *,
    batch_id: str,
    source_run: Path,
    records: list[dict],
    successes: dict[str, dict],
    model: str,
    start: float,
    end: float,
    fps: float,
    max_attempts_per_video: int,
) -> dict:
    videos = []
    for record in records:
        video_id = record["video_id"]
        existing = successes.get(video_id)
        videos.append(
            {
                "video_id": video_id,
                "sector": record.get("sector"),
                "channel_id": record.get("channel_id"),
                "channel_title": record.get("channel_title"),
                "published_at": record.get("published_at"),
                "duration_iso8601": record.get("duration"),
                "status": "reused_existing" if existing else "pending",
                "result": existing["result"] if existing else None,
                "error": None,
            }
        )

    manifest = {
        "schema_version": "0.1.0",
        "batch_id": batch_id,
        "status": "planned",
        "created_at": utc_now(),
        "updated_at": None,
        "source": {
            "youtube_run": relative_path(source_run),
            "selection": "Todos os vídeos únicos da coleta normalizada mais recente.",
        },
        "protocol": {
            "provider": "Google Gemini API",
            "model": model,
            "start_seconds": start,
            "end_seconds": end,
            "requested_fps": fps,
            "prompt_version": "galaxies-opening-v0.1.0",
            "performance_metrics_sent": False,
            "max_attempts_per_video": max_attempts_per_video,
            "resume_behavior": "Reutiliza opening_vlm.json existente por video_id.",
        },
        "videos": videos,
        "summary": {},
    }
    refresh_summary(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processa, de forma retomável, as aberturas ainda não extraídas."
    )
    parser.add_argument("--run-id", help="ID opcional para criar ou retomar o lote.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Espera entre vídeos para reduzir pressão na API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limita quantos vídeos pendentes serão tentados nesta execução.",
    )
    parser.add_argument(
        "--max-attempts-per-video",
        type=int,
        default=1,
        help=(
            "Tentativas por vídeo. O padrão 1 preserva a cota gratuita; "
            "falhas podem ser retomadas depois."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gera o plano e o manifesto sem chamar a API.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")

    args = parse_args()
    if args.start < 0 or args.end <= args.start:
        print("ERRO: o intervalo precisa satisfazer 0 <= start < end", file=sys.stderr)
        return 1
    if not 0 < args.fps <= 24:
        print("ERRO: use um FPS maior que 0 e menor ou igual a 24", file=sys.stderr)
        return 1
    if args.delay_seconds < 0:
        print("ERRO: delay-seconds não pode ser negativo", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("ERRO: limit precisa ser maior ou igual a 1", file=sys.stderr)
        return 1
    if not 1 <= args.max_attempts_per_video <= 5:
        print(
            "ERRO: max-attempts-per-video deve ficar entre 1 e 5",
            file=sys.stderr,
        )
        return 1

    generated_id = "batch-" + utc_now().replace(":", "-")
    batch_id = args.run_id or generated_id
    if not is_valid_run_id(batch_id):
        print("ERRO: run-id contém caracteres inválidos", file=sys.stderr)
        return 1

    source_run, records = latest_youtube_records()
    successes = existing_successes()
    manifest = build_manifest(
        batch_id=batch_id,
        source_run=source_run,
        records=records,
        successes=successes,
        model=args.model,
        start=args.start,
        end=args.end,
        fps=args.fps,
        max_attempts_per_video=args.max_attempts_per_video,
    )
    batch_dir = FEATURE_RUNS_DIR / batch_id
    manifest_path = batch_dir / "batch_manifest.json"
    save_json(manifest_path, manifest)

    pending = [video for video in manifest["videos"] if video["status"] == "pending"]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Lote: {batch_id}", flush=True)
    print(f"Vídeos na coleta: {len(records)}", flush=True)
    print(
        f"Resultados já existentes: {manifest['summary']['reused_existing']}",
        flush=True,
    )
    print(f"Vídeos pendentes selecionados: {len(pending)}", flush=True)
    print(f"Manifesto: {manifest_path}", flush=True)

    if args.dry_run:
        manifest["status"] = "planned"
        refresh_summary(manifest)
        save_json(manifest_path, manifest)
        print("Dry-run concluído; nenhuma chamada à API foi realizada.", flush=True)
        return 0

    manifest["status"] = "running"
    manifest["started_at"] = utc_now()
    refresh_summary(manifest)
    save_json(manifest_path, manifest)

    try:
        for position, video in enumerate(pending, start=1):
            video_id = video["video_id"]
            video["status"] = "running"
            video["attempted_at"] = utc_now()
            refresh_summary(manifest)
            save_json(manifest_path, manifest)

            print(
                f"[{position}/{len(pending)}] {video['sector']} | "
                f"{video['channel_title']} | {video_id}",
                flush=True,
            )
            command = [
                sys.executable,
                str(PILOT_SCRIPT),
                "--video-id",
                video_id,
                "--start",
                str(args.start),
                "--end",
                str(args.end),
                "--fps",
                str(args.fps),
                "--model",
                args.model,
                "--run-id",
                batch_id,
                "--max-attempts",
                str(args.max_attempts_per_video),
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=locale.getpreferredencoding(False),
                errors="replace",
                check=False,
            )
            if completed.stdout:
                print(completed.stdout.rstrip(), flush=True)

            output_dir = batch_dir / "videos" / video_id
            result_path = output_dir / "opening_vlm.json"
            error_path = output_dir / "error.json"
            video["return_code"] = completed.returncode
            video["finished_at"] = utc_now()
            if completed.returncode == 75:
                video["status"] = "quota_blocked"
                video["error"] = (
                    relative_path(error_path) if error_path.exists() else None
                )
                refresh_summary(manifest)
                manifest["status"] = "paused_quota_exhausted"
                manifest["pause_reason"] = {
                    "provider_status": 429,
                    "provider_error": "RESOURCE_EXHAUSTED",
                    "decision": (
                        "Execução interrompida automaticamente no primeiro "
                        "erro de cota."
                    ),
                }
                save_json(manifest_path, manifest)
                print(
                    "Cota esgotada; lote pausado com progresso preservado.",
                    flush=True,
                )
                return 75
            if completed.returncode == 0 and result_path.exists():
                video["status"] = "completed"
                video["result"] = relative_path(result_path)
                video["error"] = None
            else:
                video["status"] = "failed"
                video["error"] = (
                    relative_path(error_path) if error_path.exists() else None
                )

            refresh_summary(manifest)
            save_json(manifest_path, manifest)
            print(
                "Progresso disponível: "
                f"{manifest['summary']['results_available']}/"
                f"{manifest['summary']['videos_total']}; "
                f"falhas: {manifest['summary']['failed']}",
                flush=True,
            )

            if position < len(pending) and args.delay_seconds:
                time.sleep(args.delay_seconds)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["interrupted_at"] = utc_now()
        for video in manifest["videos"]:
            if video["status"] == "running":
                video["status"] = "pending"
        refresh_summary(manifest)
        save_json(manifest_path, manifest)
        print("Lote interrompido; o progresso foi salvo e pode ser retomado.", flush=True)
        return 130

    refresh_summary(manifest)
    if manifest["summary"]["failed"]:
        manifest["status"] = "completed_with_errors"
    elif manifest["summary"]["pending"]:
        manifest["status"] = "partially_completed"
    else:
        manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    refresh_summary(manifest)
    save_json(manifest_path, manifest)

    print(
        f"Lote encerrado com status {manifest['status']}: "
        f"{manifest['summary']['results_available']}/"
        f"{manifest['summary']['videos_total']} resultados disponíveis.",
        flush=True,
    )
    return 0 if manifest["summary"]["results_available"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
