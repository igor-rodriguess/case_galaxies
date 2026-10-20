"""Piloto TwelveLabs para abertura de vídeo com origem autorizada.

O TwelveLabs não aceita páginas do YouTube. Use um MP4 local autorizado ou URL
direta para a mídia bruta. O resultado segue o schema ``opening_vlm.json`` já
usado pelo mapa de features, preservando a resposta original do provedor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_RUNS_DIR = PROJECT_ROOT / "data" / "features" / "runs"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gemini_opening_pilot import build_prompt, validate_analysis  # noqa: E402

DEFAULT_MODEL = "pegasus1.6"
PROMPT_VERSION = "galaxies-opening-v0.2.0-closed-taxonomy"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def result_text(value: Any) -> str:
    """Extrai o texto do resultado sem assumir uma única forma de SDK."""
    payload = serializable(value)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "output", "data", "content"):
            candidate = payload.get(key)
            if isinstance(candidate, str):
                return candidate
    raise ValueError("A resposta TwelveLabs não contém texto JSON utilizável.")


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    result = json.loads(cleaned.strip())
    if not isinstance(result, dict):
        raise ValueError("A análise TwelveLabs precisa ser um objeto JSON.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa uma abertura autorizada com TwelveLabs.")
    parser.add_argument("--video-id", required=True, help="ID interno/YouTube usado para rastreabilidade.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--media-url", help="URL direta de arquivo de mídia, por exemplo .mp4.")
    source.add_argument("--media-file", type=Path, help="Arquivo local autorizado de vídeo.")
    parser.add_argument("--sector", required=True, choices=("automotivo", "moda", "beleza"))
    parser.add_argument("--source-permission", required=True, help="Registro curto da permissão/origem.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="twelvelabs-opening-pilot")
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Cria o ativo e a tarefa, mas deixa a coleta do resultado para o retomador.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start < 0 or args.end <= args.start:
        raise SystemExit("ERRO: use 0 <= start < end.")
    if args.media_url and "youtube.com" in args.media_url.lower():
        raise SystemExit("ERRO: TwelveLabs requer URL direta de mídia, não página do YouTube.")
    if args.media_file and not args.media_file.is_file():
        raise SystemExit(f"ERRO: arquivo não encontrado: {args.media_file}")
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("TWELVELABS_API_KEY")
    if not api_key:
        raise SystemExit("ERRO: preencha TWELVELABS_API_KEY no .env.")
    try:
        from twelvelabs import TwelveLabs
        from twelvelabs.types import AnalyzePromptV2, VideoContext_AssetId
    except ImportError as exc:
        raise SystemExit('ERRO: instale com pip install -e ".[twelvelabs]".') from exc

    started_at = utc_now()
    output_dir = FEATURE_RUNS_DIR / args.run_id / "videos" / args.video_id
    prompt = build_prompt(args.start, args.end, args.fps)
    manifest = {
        "schema_version": "0.1.0",
        "started_at": started_at,
        "identity": {"video_id": args.video_id},
        "source": {"sector": args.sector, "permission": args.source_permission},
        "extractor": {"provider": "TwelveLabs", "model_requested": args.model, "prompt_version": PROMPT_VERSION},
        "clip": {"start_seconds": args.start, "end_seconds": args.end, "requested_fps": args.fps},
        "media": {"kind": "url" if args.media_url else "local_file", "url": args.media_url, "file_name": args.media_file.name if args.media_file else None},
        "warning": "O provedor pode processar o ativo inteiro; o prompt restringe a anotação ao intervalo solicitado.",
        "prompt": prompt,
    }
    save_json(output_dir / "request_manifest.json", manifest)

    client = TwelveLabs(api_key=api_key)
    print("Enviando ativo autorizado à TwelveLabs...", flush=True)
    if args.media_url:
        asset = client.assets.create(method="url", url=args.media_url)
    else:
        with args.media_file.open("rb") as source_file:
            asset = client.assets.create(method="direct", file=source_file)
    asset_id = str(asset.id)
    print(f"Ativo criado: {asset_id}", flush=True)
    manifest["asset"] = {"id": asset_id, "status": str(asset.status)}
    save_json(output_dir / "request_manifest.json", manifest)
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        asset = client.assets.retrieve(asset_id)
        print(f"Estado do ativo: {asset.status}", flush=True)
        if str(asset.status) == "ready":
            break
        if str(asset.status) == "failed":
            raise RuntimeError(f"Ativo TwelveLabs falhou: {serializable(asset)}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Tempo limite aguardando o ativo TwelveLabs.")
        time.sleep(args.poll_seconds)
    manifest["asset"]["status"] = "ready"
    save_json(output_dir / "request_manifest.json", manifest)

    task = client.analyze_async.tasks.create(
        model_name=args.model,
        video=VideoContext_AssetId(asset_id=asset_id),
        prompt_v_2=AnalyzePromptV2(input_text=prompt),
        temperature=0,
        max_tokens=4096,
    )
    task_id = str(task.task_id)
    print(f"Tarefa criada: {task_id}", flush=True)
    manifest["task"] = {"id": task_id, "status": str(task.status)}
    save_json(output_dir / "request_manifest.json", manifest)
    if args.submit_only:
        print("Tarefa enviada; use twelvelabs_resume_task.py para coletar o resultado.")
        return 0
    while True:
        task = client.analyze_async.tasks.retrieve(task_id)
        status = str(task.status)
        print(f"Estado da tarefa: {status}", flush=True)
        if status == "ready":
            break
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"Tarefa TwelveLabs {status}: {serializable(task)}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Tempo limite aguardando análise TwelveLabs.")
        time.sleep(args.poll_seconds)
    manifest["task"]["status"] = "ready"
    save_json(output_dir / "request_manifest.json", manifest)

    raw = serializable(task.result)
    save_json(output_dir / "raw_response.json", raw)
    analysis = parse_json_response(result_text(task.result))
    validate_analysis(analysis)
    result = {
        "schema_version": "0.1.0",
        "identity": manifest["identity"],
        "provenance": {
            "source": manifest["source"],
            "extractor": manifest["extractor"],
            "clip": manifest["clip"],
            "started_at": started_at,
            "completed_at": utc_now(),
            "asset_id": asset_id,
            "task_id": task_id,
            "raw_response_file": "raw_response.json",
            "request_manifest_file": "request_manifest.json",
        },
        "response_metadata": {"provider": "TwelveLabs"},
        "validation": {"status": "complete", "validator": "required-opening-fields-v0.1.0"},
        "analysis": analysis,
    }
    save_json(output_dir / "opening_vlm.json", result)
    print(f"Análise concluída: {output_dir / 'opening_vlm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
