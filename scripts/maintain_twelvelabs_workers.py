"""Mantém uma quantidade limitada de análises TwelveLabs em andamento.

Este worker é idempotente: retoma ativos existentes antes de criar qualquer
novo, consulta o carregador oficial para não repetir resultados válidos e pula
fontes que já falharam permanentemente. Ele foi feito para ser chamado por uma
automação periódica, não para substituir a auditoria humana da origem.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.dataset import load_latest_vlm_results  # noqa: E402

DEFAULT_RUN_ID = "twelvelabs-case-control-v1"
DEFAULT_PLAN = PROJECT_ROOT / "data" / "features" / "plans" / "case-control-opening-v1" / "manifest.json"
TERMINAL_SOURCE_FAILURES = {"source_permission_denied", "source_download_unavailable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retoma e preenche workers TwelveLabs.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--max-active", type=int, default=5)
    parser.add_argument("--source-permission", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def planned_ids(plan_path: Path) -> list[str]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    queue = payload.get("queue")
    if not isinstance(queue, list):
        raise ValueError("Plano caso-controle não possui uma fila válida.")
    ids = [str(item.get("video_id") or "") for item in queue if isinstance(item, dict)]
    if not ids or any(not video_id for video_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Fila possui video_id ausente ou duplicado.")
    return ids


def worker_state(run_dir: Path) -> tuple[list[str], set[str]]:
    """Retorna ativos retomáveis e falhas permanentes de origem."""
    videos_dir = run_dir / "videos"
    active: list[str] = []
    blocked: set[str] = set()
    if not videos_dir.exists():
        return active, blocked
    for video_dir in videos_dir.iterdir():
        if not video_dir.is_dir():
            continue
        if (video_dir / "opening_vlm.json").exists():
            continue
        error_path = video_dir / "error.json"
        if error_path.exists():
            text = error_path.read_text(encoding="utf-8", errors="replace")
            try:
                kind = json.loads(text).get("kind")
            except json.JSONDecodeError:
                kind = None
            if kind in TERMINAL_SOURCE_FAILURES or "403" in text:
                blocked.add(video_dir.name)
            continue
        if (video_dir / "request_manifest.json").exists():
            active.append(video_dir.name)
    return sorted(active), blocked


def pending_ids(queue: list[str], completed: set[str], blocked: set[str], active: set[str]) -> list[str]:
    return [video_id for video_id in queue if video_id not in completed | blocked | active]


def option_value(option: str, value: str) -> str:
    """Forma segura para valores que podem começar com hífen."""
    return f"{option}={value}"


def invoke(command: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    print("DRY-RUN" if dry_run else "EXEC", " ".join(command[:4]), flush=True)
    if not dry_run:
        try:
            result = subprocess.run(
                command,
                check=False,
                cwd=PROJECT_ROOT,
                timeout=180,
                text=True,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as error:
            # Uma submissão pode ultrapassar o tempo local depois de já ter
            # criado o manifesto e a tarefa remota. Não derrubamos o worker:
            # a próxima execução encontra o manifesto e retoma a consulta.
            stdout = error.stdout or ""
            stderr = (error.stderr or "") + "\nSubmissão excedeu 180s; será retomada pelo manifesto existente."
            result = subprocess.CompletedProcess(command, returncode=124, stdout=stdout, stderr=stderr)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result
    return None


def record_source_issue(run_dir: Path, video_id: str, kind: str, message: str) -> None:
    """Persiste uma indisponibilidade terminal da fonte para não repeti-la."""
    path = run_dir / "videos" / video_id / "error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "kind": kind,
                "message": message[-4_000:],
                "failed_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def record_source_block(run_dir: Path, video_id: str, message: str) -> None:
    """Persiste um 403 de origem para que a próxima execução não o repita."""
    record_source_issue(run_dir, video_id, "source_permission_denied", message)


def is_deterministic_download_failure(message: str) -> bool:
    """Falha repetível: download terminou sem arquivo-fonte processável."""
    return "Esperava um único arquivo fonte" in message and "encontrei: []" in message


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_active <= 5:
        raise SystemExit("ERRO: --max-active deve estar entre 1 e 5.")
    queue = planned_ids(args.plan)
    completed = set(load_latest_vlm_results())
    run_dir = PROJECT_ROOT / "data" / "features" / "runs" / args.run_id
    active, blocked = worker_state(run_dir)

    # Retomar primeiro garante que ativos prontos gerem resultado antes de
    # consumirmos novos créditos/transferências.
    for video_id in active:
        invoke(
            [
                sys.executable,
                "scripts/twelvelabs_resume_task.py",
                "--run-id",
                args.run_id,
                option_value("--video-id", video_id),
                "--poll-seconds",
                "2",
                "--timeout-seconds",
                "15",
            ],
            dry_run=args.dry_run,
        )

    completed = set(load_latest_vlm_results())
    active, blocked_after_resume = worker_state(run_dir)
    blocked |= blocked_after_resume
    slots = max(args.max_active - len(active), 0)
    selected = pending_ids(queue, completed, blocked, set(active))[:slots]
    submitted: list[str] = []
    for video_id in selected:
        # ``argparse`` interpreta um valor que começa com "-" como opção se
        # ele vier depois de ``--video-id``. O formato com "=" preserva IDs
        # YouTube legítimos como ``-0kEAmYHXdo``.
        video_id_argument = option_value("--video-id", video_id)
        result = invoke(
            [
                sys.executable,
                "scripts/submit_authorized_twelvelabs_item.py",
                video_id_argument,
                "--source-permission",
                args.source_permission,
            ],
            dry_run=args.dry_run,
        )
        if result is None or result.returncode == 0:
            submitted.append(video_id)
        if result and result.returncode:
            message = result.stdout + result.stderr
            if "403" in message:
                record_source_block(run_dir, video_id, message)
            elif is_deterministic_download_failure(message):
                record_source_issue(run_dir, video_id, "source_download_unavailable", message)

    print(
        json.dumps(
            {
                "valid_results": len(completed),
                "active_before_fill": len(active),
                "source_blocked": len(blocked),
                "submitted": submitted,
                "remaining_queue": len(pending_ids(queue, completed, blocked, set(active) | set(submitted))),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
