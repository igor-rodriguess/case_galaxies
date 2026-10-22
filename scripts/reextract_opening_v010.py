"""Reextrai aberturas com o pedido v0.1.0 exato, trocando só o modelo.

O protocolo principal da avaliação final é `galaxies-opening-v0.1.0` com o
Gemini 3.5 Flash-Lite, janela de 0 a 3 s e 4 FPS (`docs/escopo_avaliacao_final.md`). Os
pedidos v0.1.0 já enviados são idênticos byte a byte, exceto pelo `video_id`.
Este script confere essa igualdade, usa o pedido como referência e o reenvia para
os vídeos cujo resultado v0.1.0 veio de outro modelo. O piloto atual monta o
prompt v0.2.0, por isso não serve para esta reextração.

Vídeos da fila caso-controle ficam de fora: não entram na regressão.

Uso:
    python scripts/reextract_opening_v010.py --dry-run
    python scripts/reextract_opening_v010.py --run-id reextract-v010-flash-lite-2026-10-20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gemini_opening_pilot as pilot  # noqa: E402

from galaxies.dataset import load_latest_vlm_results, nested  # noqa: E402

PROMPT_VERSION = "galaxies-opening-v0.1.0"
PLACEHOLDER = "<VIDEO_ID>"
CASE_CONTROL_PLAN = (
    PROJECT_ROOT / "data" / "features" / "plans" / "case-control-opening-v1" / "manifest.json"
)
DATA_EXCLUSION = [
    "title",
    "description",
    "channel_title",
    "view_count",
    "like_count",
    "comment_count",
    "subscriber_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reextrai aberturas v0.1.0 com outro modelo.")
    parser.add_argument("--run-id", default="reextract-v010-flash-lite")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--from-model", default="gemini-3.8-flash")
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-consecutive-unavailable", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalized_request(payload: dict, video_id: str) -> str:
    """Serializa o pedido com o video_id trocado por um marcador fixo."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).replace(video_id, PLACEHOLDER)


def reference_request(results: dict[str, dict]) -> tuple[str, Path, int]:
    """Exige que todos os pedidos v0.1.0 gravados sejam iguais fora o video_id."""
    variants: dict[str, Path] = {}
    count = 0
    for video_id, result in results.items():
        if nested(result, "provenance", "extractor", "prompt_version") != PROMPT_VERSION:
            continue
        path = Path(result["_source_path"]).parent / "api_request.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        variants.setdefault(normalized_request(payload, video_id), path)
        count += 1
    if len(variants) != 1:
        raise SystemExit(f"ERRO: {len(variants)} variantes do pedido v0.1.0; esperado 1.")
    template, path = next(iter(variants.items()))
    return template, path, count


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    pilot.load_dotenv(PROJECT_ROOT / ".env")

    results = load_latest_vlm_results()
    template, template_path, template_uses = reference_request(results)
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    reference = json.loads(template)
    clip_part, prompt_part = reference["contents"][0]["parts"]
    clip = clip_part["video_metadata"]
    start = float(clip["start_offset"].rstrip("s"))
    end = float(clip["end_offset"].rstrip("s"))
    fps = float(clip["fps"])
    prompt = prompt_part["text"]

    plan = json.loads(CASE_CONTROL_PLAN.read_text(encoding="utf-8"))
    case_control = {item["video_id"] for item in plan["queue"]}
    targets = sorted(
        video_id
        for video_id, result in results.items()
        if nested(result, "provenance", "extractor", "prompt_version") == PROMPT_VERSION
        and nested(result, "provenance", "extractor", "model_requested") == args.from_model
        and video_id not in case_control
    )
    run_dir = pilot.FEATURE_RUNS_DIR / args.run_id
    manifest_path = run_dir / "batch_manifest.json"
    summary = {
        "reference_request": template_path.relative_to(PROJECT_ROOT).as_posix(),
        "reference_request_sha256": template_hash,
        "reference_request_uses": template_uses,
        "clip": {"start_seconds": start, "end_seconds": end, "requested_fps": fps},
        "targets": len(targets),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERRO: preencha GEMINI_API_KEY no .env.")

    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "schema_version": "0.1.0",
            "run_id": args.run_id,
            "reason": (
                "Homogeneizar o extrator da base principal: mesmo pedido v0.1.0, "
                f"modelo {args.model} no lugar de {args.from_model}."
            ),
            "model": args.model,
            "replaces_model": args.from_model,
            "prompt_version": PROMPT_VERSION,
            **summary,
            "items": {},
        }
    )
    unavailable_streak = 0
    for video_id in targets:
        item = manifest["items"].get(video_id, {})
        if item.get("status") == "completed":
            continue
        previous = results[video_id]
        output_dir = run_dir / "videos" / video_id
        source_path, source_record = pilot.find_video_record(video_id)
        identity = source_record.get("identity", {})
        started_at = pilot.utc_now()
        request_manifest = {
            "schema_version": "0.1.0",
            "run_id": args.run_id,
            "started_at": started_at,
            "identity": {
                "video_id": video_id,
                "channel_id": identity.get("channel_id"),
                "youtube_watch_url": identity.get("youtube_watch_url"),
            },
            "source": {
                "youtube_record": source_path.relative_to(PROJECT_ROOT).as_posix(),
                "sector": source_record.get("classification", {}).get("sector"),
            },
            "extractor": {
                "provider": "Google Gemini API",
                "operation": "models.generateContent",
                "model_requested": args.model,
                "prompt_version": PROMPT_VERSION,
                "processing_mode": "static (default)",
            },
            "clip": summary["clip"],
            "prompt": prompt,
            "data_exclusion": DATA_EXCLUSION,
            "reextraction": {
                "replaces_result": Path(previous["_source_path"])
                .relative_to(PROJECT_ROOT)
                .as_posix(),
                "replaces_model": args.from_model,
                "reference_request_sha256": template_hash,
            },
        }
        pilot.save_json(output_dir / "request_manifest.json", request_manifest)
        try:
            payload, raw_response = pilot.request_gemini(
                api_key=api_key,
                model=args.model,
                video_url=identity["youtube_watch_url"],
                start_seconds=start,
                end_seconds=end,
                fps=fps,
                prompt=prompt,
                max_attempts=args.max_attempts,
            )
            if normalized_request(payload, video_id) != template:
                raise SystemExit("ERRO: o pedido enviado divergiu do pedido v0.1.0 de referência.")
            analysis = pilot.parse_json_response(pilot.extract_text(raw_response))
        except (
            pilot.QuotaExhaustedError,
            pilot.ProviderUnavailableError,
            RuntimeError,
            ValueError,
        ) as exc:
            status = (
                "quota_blocked"
                if isinstance(exc, pilot.QuotaExhaustedError)
                else "provider_unavailable"
                if isinstance(exc, pilot.ProviderUnavailableError)
                else "failed"
            )
            pilot.save_json(
                output_dir / "error.json",
                {
                    "schema_version": "0.1.0",
                    "failed_at": pilot.utc_now(),
                    "error_type": type(exc).__name__,
                    "message": str(exc).replace(api_key, "<redacted>"),
                },
            )
            manifest["items"][video_id] = {"status": status, "error": type(exc).__name__}
            pilot.save_json(manifest_path, manifest)
            print(f"{video_id}: {status} ({type(exc).__name__})")
            if status == "quota_blocked":
                break
            unavailable_streak = unavailable_streak + 1 if status == "provider_unavailable" else 0
            if unavailable_streak >= args.max_consecutive_unavailable:
                print("Pausado: provedor indisponível em chamadas seguidas.")
                break
            continue

        unavailable_streak = 0
        pilot.save_json(output_dir / "api_request.json", payload)
        pilot.save_json(output_dir / "raw_response.json", raw_response)
        result = {
            "schema_version": "0.1.0",
            "identity": request_manifest["identity"],
            "provenance": {
                "source": request_manifest["source"],
                "extractor": request_manifest["extractor"],
                "clip": request_manifest["clip"],
                "started_at": started_at,
                "completed_at": pilot.utc_now(),
                "raw_response_file": "raw_response.json",
                "request_manifest_file": "request_manifest.json",
                "reextraction": request_manifest["reextraction"],
            },
            "response_metadata": {
                "model_version": raw_response.get("modelVersion"),
                "usage_metadata": raw_response.get("usageMetadata"),
            },
            "analysis": analysis,
        }
        pilot.save_json(output_dir / "opening_vlm.json", result)
        manifest["items"][video_id] = {
            "status": "completed",
            "completed_at": result["provenance"]["completed_at"],
        }
        pilot.save_json(manifest_path, manifest)
        print(f"{video_id}: completed")
        time.sleep(args.delay_seconds)

    statuses = [item["status"] for item in manifest["items"].values()]
    manifest["summary"] = {status: statuses.count(status) for status in sorted(set(statuses))}
    manifest["summary"]["pending"] = len(set(targets) - set(manifest["items"]))
    pilot.save_json(manifest_path, manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
