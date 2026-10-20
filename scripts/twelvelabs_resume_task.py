"""Retoma uma tarefa TwelveLabs existente sem reenviar mídia nem duplicar custo."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from twelvelabs_opening_pilot import (  # noqa: E402
    FEATURE_RUNS_DIR,
    load_dotenv,
    parse_json_response,
    result_text,
    save_json,
    serializable,
    utc_now,
    validate_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retoma uma análise TwelveLabs sem reenviar o ativo.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = FEATURE_RUNS_DIR / args.run_id / "videos" / args.video_id
    manifest_path = output_dir / "request_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"ERRO: manifesto não encontrado: {manifest_path}")
    import json

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_id = manifest.get("task", {}).get("id")
    if not task_id:
        raise SystemExit("ERRO: o manifesto não possui task.id para retomar.")
    if (output_dir / "opening_vlm.json").is_file():
        print(f"Análise já concluída: {output_dir / 'opening_vlm.json'}")
        return 0

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("TWELVELABS_API_KEY")
    if not api_key:
        raise SystemExit("ERRO: preencha TWELVELABS_API_KEY no .env.")
    try:
        from twelvelabs import TwelveLabs
    except ImportError as exc:
        raise SystemExit('ERRO: instale com pip install -e ".[twelvelabs]".') from exc

    client = TwelveLabs(api_key=api_key)
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        task = client.analyze_async.tasks.retrieve(task_id)
        status = str(task.status)
        print(f"Estado da tarefa: {status}", flush=True)
        manifest.setdefault("task", {})["status"] = status
        manifest["task"]["last_checked_at"] = utc_now()
        save_json(manifest_path, manifest)
        if status == "ready":
            break
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"Tarefa TwelveLabs {status}: {serializable(task)}")
        if time.monotonic() >= deadline:
            print("A tarefa ainda está processando; retome depois com este mesmo comando.")
            return 2
        time.sleep(args.poll_seconds)

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
            "started_at": manifest["started_at"],
            "completed_at": utc_now(),
            "asset_id": manifest["asset"]["id"],
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
