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
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOUTUBE_RUNS_DIR = PROJECT_ROOT / "data" / "youtube" / "runs"
FEATURE_RUNS_DIR = PROJECT_ROOT / "data" / "features" / "runs"
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "gemini_opening_pilot.py"
DEFAULT_MODEL = "gemini-3.8-flash"
SECTOR_ORDER = ("automotivo", "moda", "beleza")
COLLECTOR_LOCK_PATH = FEATURE_RUNS_DIR / ".locks" / "gemini-opening-batch.lock.json"
VIDEO_CLAIMS_DIR = FEATURE_RUNS_DIR / ".claims" / "gemini-opening"
DEFAULT_LEASE_SECONDS = 45 * 60


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _acquire_lease(
    path: Path,
    *,
    resource: str,
    owner: str,
    now: datetime | None = None,
    ttl_seconds: int = DEFAULT_LEASE_SECONDS,
) -> tuple[str | None, dict | None]:
    """Cria uma reserva exclusiva e reaproveita somente reservas expiradas."""
    current = now or datetime.now(UTC)
    token = uuid4().hex
    payload = {
        "schema_version": "0.1.0",
        "resource": resource,
        "owner": owner,
        "pid": os.getpid(),
        "token": token,
        "claimed_at": current.isoformat().replace("+00:00", "Z"),
        "expires_at": (current + timedelta(seconds=ttl_seconds))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return token, payload
        except FileExistsError:
            existing = _read_json(path)
            expires_at = _parse_utc(existing.get("expires_at")) if existing else None
            if expires_at is None or expires_at > current:
                return None, existing
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return None, existing
    return None, _read_json(path)


def _release_lease(path: Path, token: str | None) -> None:
    """Libera apenas a reserva criada pelo processo atual."""
    if not token:
        return
    existing = _read_json(path)
    if existing is None or existing.get("token") != token:
        return
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def acquire_execution_lock(
    batch_id: str,
    *,
    now: datetime | None = None,
    ttl_seconds: int = DEFAULT_LEASE_SECONDS,
) -> tuple[str | None, dict | None]:
    return _acquire_lease(
        COLLECTOR_LOCK_PATH,
        resource="gemini_opening_batch",
        owner=batch_id,
        now=now,
        ttl_seconds=ttl_seconds,
    )


def release_execution_lock(token: str | None) -> None:
    _release_lease(COLLECTOR_LOCK_PATH, token)


def claim_video(
    video_id: str,
    batch_id: str,
    *,
    now: datetime | None = None,
    ttl_seconds: int = DEFAULT_LEASE_SECONDS,
) -> tuple[str | None, dict | None]:
    allowed = "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if not video_id or any(character not in allowed for character in video_id):
        raise ValueError("video_id contém caracteres inválidos para a reserva.")
    return _acquire_lease(
        VIDEO_CLAIMS_DIR / f"{video_id}.json",
        resource=f"gemini_opening_video:{video_id}",
        owner=batch_id,
        now=now,
        ttl_seconds=ttl_seconds,
    )


def release_video_claim(video_id: str, token: str | None) -> None:
    _release_lease(VIDEO_CLAIMS_DIR / f"{video_id}.json", token)


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def latest_youtube_records() -> tuple[Path, list[dict]]:
    candidates = [
        run_dir
        for run_dir in YOUTUBE_RUNS_DIR.iterdir()
        if run_dir.is_dir() and (run_dir / "normalized_records.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("Nenhuma coleta normalizada do YouTube foi encontrada.")

    run_dir = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    payload = json.loads((run_dir / "normalized_records.json").read_text(encoding="utf-8"))
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
        prompt_version = result.get("provenance", {}).get("extractor", {}).get("prompt_version", "")
        validation_status = result.get("validation", {}).get("status")
        if str(prompt_version).startswith("galaxies-opening-v0.2") and validation_status != "complete":
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


def existing_permission_denials(
    runs_dir: Path = FEATURE_RUNS_DIR,
) -> dict[str, dict]:
    """Retorna vídeos que o provedor recusou explicitamente com HTTP 403.

    Esses erros são ligados ao recurso de origem, não à disponibilidade geral da
    API. Mantê-los fora de novas filas evita consumir chamadas repetidas sem
    apagar o registro nem transformar ausência em resultado.
    """
    blocked: dict[str, dict] = {}
    if not runs_dir.exists():
        return blocked

    for error_path in runs_dir.rglob("error.json"):
        try:
            error = json.loads(error_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        message = str(error.get("message") or "")
        if "HTTP 403" not in message or "PERMISSION_DENIED" not in message:
            continue
        video_id = error_path.parent.name
        failed_at = str(error.get("failed_at") or "")
        previous = blocked.get(video_id)
        if previous is None or failed_at > previous["failed_at"]:
            blocked[video_id] = {
                "failed_at": failed_at,
                "error": relative_path(error_path),
            }
    return blocked


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
        "provider_unavailable": counts["provider_unavailable"],
        "quota_blocked": counts["quota_blocked"],
        "source_blocked": counts["source_blocked"],
        "running": counts["running"],
    }
    manifest["updated_at"] = utc_now()


def balance_pending(videos: list[dict]) -> list[dict]:
    """Intercala setores e canais para usar a cota diária de forma representativa."""
    by_sector_and_channel: dict[str, dict[str, deque[dict]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for video in videos:
        sector = str(video.get("sector") or "unknown")
        channel_id = str(video.get("channel_id") or "unknown")
        by_sector_and_channel[sector][channel_id].append(video)

    sector_queues: dict[str, deque[dict]] = {}
    ordered_sectors = [
        *[sector for sector in SECTOR_ORDER if sector in by_sector_and_channel],
        *sorted(set(by_sector_and_channel) - set(SECTOR_ORDER)),
    ]
    for sector in ordered_sectors:
        channel_queues = by_sector_and_channel[sector]
        channel_ids = list(channel_queues)
        sector_queue: deque[dict] = deque()
        while any(channel_queues.values()):
            for channel_id in channel_ids:
                if channel_queues[channel_id]:
                    sector_queue.append(channel_queues[channel_id].popleft())
        sector_queues[sector] = sector_queue

    balanced: list[dict] = []
    while any(sector_queues.values()):
        for sector in ordered_sectors:
            if sector_queues[sector]:
                balanced.append(sector_queues[sector].popleft())
    return balanced


def _parse_timestamp(value: str) -> datetime:
    """Interpreta timestamps ISO da coleta como instantes UTC."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def performance_strata(
    records: list[dict],
    *,
    minimum_age_days: float,
) -> dict[str, str]:
    """Cria estratos de desempenho observado para priorizar a amostra VLM.

    A pontuação é um resíduo de log(views) contra log(idade), ajustado na
    própria coleta e ordenado dentro de cada canal. Ela serve apenas para
    selecionar vídeos para anotação audiovisual; não é usada como feature nem
    substitui o alvo de views em horizonte fixo.
    """
    if minimum_age_days < 0:
        raise ValueError("minimum_age_days não pode ser negativo.")

    eligible: list[tuple[dict, float, float]] = []
    for record in records:
        try:
            observed_at = _parse_timestamp(str(record["collected_at"]))
            published_at = _parse_timestamp(str(record["published_at"]))
            age_days = max((observed_at - published_at).total_seconds() / 86_400, 0.0)
            views = max(float(record.get("view_count") or 0), 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if age_days >= minimum_age_days:
            eligible.append((record, age_days, views))

    if len(eligible) < 2:
        return {}

    age_logs = [math.log1p(age_days) for _, age_days, _ in eligible]
    view_logs = [math.log1p(views) for _, _, views in eligible]
    age_mean = sum(age_logs) / len(age_logs)
    view_mean = sum(view_logs) / len(view_logs)
    variance = sum((value - age_mean) ** 2 for value in age_logs)
    slope = (
        sum(
            (age_log - age_mean) * (view_log - view_mean)
            for age_log, view_log in zip(age_logs, view_logs, strict=False)
        )
        / variance
        if variance
        else 0.0
    )
    intercept = view_mean - slope * age_mean

    by_channel: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for record, age_days, views in eligible:
        channel_id = str(record.get("channel_id") or "unknown")
        score = math.log1p(views) - (intercept + slope * math.log1p(age_days))
        by_channel[channel_id].append((score, str(record["video_id"])))

    strata: dict[str, str] = {}
    for channel_scores in by_channel.values():
        ordered = sorted(channel_scores)
        size = len(ordered)
        for index, (_, video_id) in enumerate(ordered):
            percentile = (index + 0.5) / size
            if percentile <= 0.20:
                strata[video_id] = "baixo"
            elif percentile >= 0.80:
                strata[video_id] = "viral"
            elif 0.40 <= percentile <= 0.60:
                strata[video_id] = "tipico"
            else:
                strata[video_id] = "intermediario"
    return strata


def performance_stratified_pending(videos: list[dict]) -> list[dict]:
    """Intercala setor, estrato e canal, priorizando a cauda viral.

    A ordem garante que cada bloco curto contenha exemplos virais, típicos e
    baixos dos três setores, em vez de consumir apenas uploads recentes. Vídeos
    recentes demais para pontuar entram por último, com ausência explícita.
    """
    tier_order = ("viral", "tipico", "baixo", "intermediario", "recente_ou_sem_score")
    by_sector_tier_channel: dict[str, dict[str, dict[str, deque[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(deque))
    )
    for video in videos:
        sector = str(video.get("sector") or "unknown")
        tier = str(video.get("performance_stratum") or "recente_ou_sem_score")
        channel_id = str(video.get("channel_id") or "unknown")
        by_sector_tier_channel[sector][tier][channel_id].append(video)

    ordered_sectors = [
        *[sector for sector in SECTOR_ORDER if sector in by_sector_tier_channel],
        *sorted(set(by_sector_tier_channel) - set(SECTOR_ORDER)),
    ]
    sector_queues: dict[str, deque[dict]] = {}
    for sector in ordered_sectors:
        tier_queues: dict[str, deque[dict]] = {}
        tiers = [
            *[tier for tier in tier_order if tier in by_sector_tier_channel[sector]],
            *sorted(set(by_sector_tier_channel[sector]) - set(tier_order)),
        ]
        for tier in tiers:
            channels = by_sector_tier_channel[sector][tier]
            tier_queue: deque[dict] = deque()
            while any(channels.values()):
                for channel_id in sorted(channels):
                    if channels[channel_id]:
                        tier_queue.append(channels[channel_id].popleft())
            tier_queues[tier] = tier_queue

        sector_queue: deque[dict] = deque()
        while any(tier_queues.values()):
            for tier in tiers:
                if tier_queues[tier]:
                    sector_queue.append(tier_queues[tier].popleft())
        sector_queues[sector] = sector_queue

    selected: list[dict] = []
    while any(sector_queues.values()):
        for sector in ordered_sectors:
            if sector_queues[sector]:
                selected.append(sector_queues[sector].popleft())
    return selected


def build_manifest(
    *,
    batch_id: str,
    source_run: Path,
    records: list[dict],
    successes: dict[str, dict],
    permission_denials: dict[str, dict],
    model: str,
    start: float,
    end: float,
    fps: float,
    max_attempts_per_video: int,
    max_consecutive_provider_failures: int,
    selection_order: str,
    performance_strata_by_video: dict[str, str],
    performance_minimum_age_days: float,
) -> dict:
    videos = []
    for record in records:
        video_id = record["video_id"]
        existing = successes.get(video_id)
        blocked = permission_denials.get(video_id)
        status = "reused_existing" if existing else "source_blocked" if blocked else "pending"
        videos.append(
            {
                "video_id": video_id,
                "sector": record.get("sector"),
                "channel_id": record.get("channel_id"),
                "channel_title": record.get("channel_title"),
                "published_at": record.get("published_at"),
                "duration_iso8601": record.get("duration"),
                "performance_stratum": performance_strata_by_video.get(
                    video_id, "recente_ou_sem_score"
                ),
                "status": status,
                "result": existing["result"] if existing else None,
                "error": blocked["error"] if blocked and not existing else None,
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
            "max_consecutive_provider_failures": max_consecutive_provider_failures,
            "resume_behavior": (
                "Reutiliza opening_vlm.json existente e não repete recursos "
                "já recusados com HTTP 403 por video_id."
            ),
            "selection_order": selection_order,
            "performance_stratification": {
                "enabled": selection_order == "performance_stratified",
                "definition": (
                    "Percentil de views ajustadas pela idade, dentro do canal; "
                    "usado apenas para selecionar a amostra audiovisual."
                ),
                "minimum_age_days": performance_minimum_age_days,
                "tiers": ["baixo", "tipico", "intermediario", "viral"],
                "recent_or_unscored_label": "recente_ou_sem_score",
            },
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
        "--selection-order",
        choices=("balanced", "performance_stratified", "source"),
        default="balanced",
        help=(
            "Ordem dos pendentes; performance_stratified prioriza, por canal, "
            "vídeos virais, típicos e baixos ajustados pela idade."
        ),
    )
    parser.add_argument(
        "--case-control-plan",
        type=Path,
        help=(
            "Manifesto gerado por build_case_control_queue.py. Quando informado, "
            "a fila de video_ids desse plano substitui a seleção ampla; os rótulos "
            "de auditoria do plano não são encaminhados ao Gemini."
        ),
    )
    parser.add_argument(
        "--performance-minimum-age-days",
        type=float,
        default=30.0,
        help=(
            "Idade mínima para estratificar desempenho observado; vídeos mais "
            "novos permanecem como recente_ou_sem_score."
        ),
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
        "--max-consecutive-provider-failures",
        type=int,
        default=3,
        help="Pausa o lote após falhas temporárias consecutivas do provedor.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gera o plano e o manifesto sem chamar a API.",
    )
    return parser.parse_args()


def records_from_case_control_plan(plan_path: Path, records: list[dict]) -> list[dict]:
    """Recorta e ordena os registros conforme uma fila caso-controle auditada."""
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Não foi possível ler o plano caso-controle: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plano caso-controle inválido: {plan_path}") from exc
    queue = payload.get("queue")
    if not isinstance(queue, list) or not queue:
        raise ValueError("O plano caso-controle não contém uma fila não vazia.")
    by_id = {str(record["video_id"]): record for record in records}
    selected: list[dict] = []
    seen: set[str] = set()
    for item in queue:
        if not isinstance(item, dict):
            raise ValueError("O plano caso-controle contém item inválido.")
        video_id = str(item.get("video_id") or "")
        if not video_id or video_id in seen:
            raise ValueError("O plano caso-controle contém video_id ausente ou duplicado.")
        if video_id not in by_id:
            raise ValueError(f"O plano referencia vídeo ausente da coleta: {video_id}")
        selected.append(by_id[video_id])
        seen.add(video_id)
    return selected


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    if args.performance_minimum_age_days < 0:
        print("ERRO: performance-minimum-age-days não pode ser negativo", file=sys.stderr)
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
    if not 1 <= args.max_consecutive_provider_failures <= 10:
        print(
            "ERRO: max-consecutive-provider-failures deve ficar entre 1 e 10",
            file=sys.stderr,
        )
        return 1

    generated_id = "batch-" + utc_now().replace(":", "-")
    batch_id = args.run_id or generated_id
    if not is_valid_run_id(batch_id):
        print("ERRO: run-id contém caracteres inválidos", file=sys.stderr)
        return 1

    if args.dry_run:
        return run_batch(args, batch_id)

    lock_token, active_lock = acquire_execution_lock(batch_id)
    if lock_token is None:
        owner = active_lock.get("owner") if active_lock else "desconhecido"
        expires_at = active_lock.get("expires_at") if active_lock else "desconhecido"
        print(
            "Coletor Gemini já está em execução "
            f"(lote {owner}; reserva até {expires_at}). Nenhum vídeo foi enviado.",
            flush=True,
        )
        return 73
    try:
        return run_batch(args, batch_id)
    finally:
        release_execution_lock(lock_token)


def run_batch(args: argparse.Namespace, batch_id: str) -> int:

    source_run, records = latest_youtube_records()
    plan_path: Path | None = None
    if args.case_control_plan is not None:
        plan_path = args.case_control_plan.resolve()
        records = records_from_case_control_plan(plan_path, records)
    successes = existing_successes()
    permission_denials = existing_permission_denials()
    strata_by_video = performance_strata(
        records,
        minimum_age_days=args.performance_minimum_age_days,
    )
    manifest = build_manifest(
        batch_id=batch_id,
        source_run=source_run,
        records=records,
        successes=successes,
        permission_denials=permission_denials,
        model=args.model,
        start=args.start,
        end=args.end,
        fps=args.fps,
        max_attempts_per_video=args.max_attempts_per_video,
        max_consecutive_provider_failures=args.max_consecutive_provider_failures,
        selection_order=args.selection_order,
        performance_strata_by_video=strata_by_video,
        performance_minimum_age_days=args.performance_minimum_age_days,
    )
    if plan_path is not None:
        manifest["source"]["selection"] = (
            "Fila caso-controle auditada; os rótulos de seleção permanecem no plano "
            "e não são enviados ao extrator."
        )
        manifest["source"]["case_control_plan"] = relative_path(plan_path)
    batch_dir = FEATURE_RUNS_DIR / batch_id
    manifest_path = batch_dir / "batch_manifest.json"
    save_json(manifest_path, manifest)

    pending = [video for video in manifest["videos"] if video["status"] == "pending"]
    if args.selection_order == "balanced":
        pending = balance_pending(pending)
    elif args.selection_order == "performance_stratified":
        pending = performance_stratified_pending(pending)
    if args.limit is not None:
        pending = pending[: args.limit]
    manifest["selection"] = {
        "order": args.selection_order,
        "selected_candidates": [
            {
                "position": position,
                "video_id": video["video_id"],
                "sector": video.get("sector"),
                "channel_id": video.get("channel_id"),
                "channel_title": video.get("channel_title"),
                "performance_stratum": video.get("performance_stratum"),
            }
            for position, video in enumerate(pending, start=1)
        ],
    }
    save_json(manifest_path, manifest)

    print(f"Lote: {batch_id}", flush=True)
    print(f"Vídeos na coleta: {len(records)}", flush=True)
    print(
        f"Resultados já existentes: {manifest['summary']['reused_existing']}",
        flush=True,
    )
    print(
        f"Vídeos bloqueados na origem: {manifest['summary']['source_blocked']}",
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
        consecutive_provider_failures = 0
        for position, video in enumerate(pending, start=1):
            video_id = video["video_id"]
            claim_token, active_claim = claim_video(video_id, batch_id)
            if claim_token is None:
                video["status"] = "pending"
                video["deferred_reason"] = {
                    "reason": "claimed_by_another_worker",
                    "owner": active_claim.get("owner") if active_claim else None,
                    "expires_at": active_claim.get("expires_at") if active_claim else None,
                }
                refresh_summary(manifest)
                save_json(manifest_path, manifest)
                print(
                    f"[{position}/{len(pending)}] {video_id} reservado por outro worker; adiado.",
                    flush=True,
                )
                continue
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
                f"--video-id={video_id}",
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
            try:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            finally:
                release_video_claim(video_id, claim_token)
            if completed.stdout:
                print(completed.stdout.rstrip(), flush=True)

            output_dir = batch_dir / "videos" / video_id
            result_path = output_dir / "opening_vlm.json"
            error_path = output_dir / "error.json"
            video["return_code"] = completed.returncode
            video["finished_at"] = utc_now()
            if completed.returncode == 75:
                video["status"] = "quota_blocked"
                video["error"] = relative_path(error_path) if error_path.exists() else None
                refresh_summary(manifest)
                manifest["status"] = "paused_quota_exhausted"
                manifest["pause_reason"] = {
                    "provider_status": 429,
                    "provider_error": "RESOURCE_EXHAUSTED",
                    "decision": ("Execução interrompida automaticamente no primeiro erro de cota."),
                }
                save_json(manifest_path, manifest)
                print(
                    "Cota esgotada; lote pausado com progresso preservado.",
                    flush=True,
                )
                return 75
            if completed.returncode == 69:
                video["status"] = "provider_unavailable"
                video["error"] = relative_path(error_path) if error_path.exists() else None
                consecutive_provider_failures += 1
                refresh_summary(manifest)
                if consecutive_provider_failures >= args.max_consecutive_provider_failures:
                    manifest["status"] = "paused_provider_unavailable"
                    manifest["pause_reason"] = {
                        "provider_error": "UNAVAILABLE",
                        "consecutive_failures": consecutive_provider_failures,
                        "decision": (
                            "Execução pausada automaticamente para preservar a cota; "
                            "tente novamente mais tarde."
                        ),
                    }
                    save_json(manifest_path, manifest)
                    print(
                        "Provedor temporariamente indisponível; lote pausado com "
                        "progresso preservado.",
                        flush=True,
                    )
                    return 69
                save_json(manifest_path, manifest)
                continue
            if completed.returncode == 0 and result_path.exists():
                video["status"] = "completed"
                video["result"] = relative_path(result_path)
                video["error"] = None
                consecutive_provider_failures = 0
            else:
                video["status"] = "failed"
                video["error"] = relative_path(error_path) if error_path.exists() else None
                consecutive_provider_failures = 0

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
