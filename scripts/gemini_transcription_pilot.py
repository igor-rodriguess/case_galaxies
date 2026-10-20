"""Transcreve um trecho autorizado de vídeo por URL, preservando timestamps.

O artefato é separado do VLM visual e ligado ao restante da base por ``video_id``.
Não envia título, canal ou métricas de desempenho ao provedor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gemini_opening_pilot import (  # noqa: E402
    DEFAULT_MODEL,
    FEATURE_RUNS_DIR,
    ProviderUnavailableError,
    QuotaExhaustedError,
    extract_text,
    find_video_record,
    load_dotenv,
    parse_json_response,
    request_gemini,
    save_json,
    utc_now,
)

PROMPT_VERSION = "galaxies-transcript-segments-v0.1.0"


def build_prompt(start_seconds: float, end_seconds: float) -> str:
    return f"""
Transcreva somente a fala realmente audível entre {start_seconds:g} e
{end_seconds:g} segundos do vídeo fornecido. Não use título, descrição, canal,
legendas externas, popularidade ou conhecimento prévio. Não descreva imagens.

Regras:
- preserve o idioma e as palavras audíveis; não traduza nem complete trechos;
- use null e uma lista vazia quando não houver fala inteligível;
- os timestamps são relativos ao início do vídeo e devem permanecer dentro do intervalo;
- divida a fala em segmentos curtos quando houver pausa ou mudança de locutor;
- marque sobreposição com música ou ruído sem inventar o conteúdo;
- responda apenas com JSON válido, sem markdown, no esquema abaixo.

{{
  "schema_version": "0.1.0",
  "speech_present": null,
  "language": null,
  "segments": [
    {{
      "start_seconds": null,
      "end_seconds": null,
      "text": null,
      "speaker_label": null,
      "confidence": 0.0,
      "music_overlap": null,
      "noise_overlap": null
    }}
  ],
  "full_text": null,
  "uncertainties": []
}}
""".strip()


def validate_transcript(
    analysis: dict[str, Any], *, start_seconds: float, end_seconds: float
) -> list[str]:
    errors: list[str] = []
    if analysis.get("speech_present") not in {True, False, None}:
        errors.append("speech_present deve ser booleano ou null")
    segments = analysis.get("segments")
    if not isinstance(segments, list):
        return [*errors, "segments deve ser uma lista"]
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"segments[{index}] deve ser objeto")
            continue
        begin = segment.get("start_seconds")
        finish = segment.get("end_seconds")
        text = segment.get("text")
        if not isinstance(begin, (int, float)) or not isinstance(finish, (int, float)):
            errors.append(f"segments[{index}] não possui timestamps numéricos")
        elif not start_seconds <= float(begin) <= float(finish) <= end_seconds:
            errors.append(f"segments[{index}] está fora do intervalo solicitado")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"segments[{index}] não possui texto")
    if analysis.get("speech_present") is False and segments:
        errors.append("segments deve ser vazio quando speech_present é false")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcreve um trecho com timestamps via Gemini.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--model", help=f"Modelo Gemini; padrão: {DEFAULT_MODEL}.")
    parser.add_argument("--run-id")
    parser.add_argument("--max-attempts", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERRO: preencha GEMINI_API_KEY no arquivo .env", file=sys.stderr)
        return 1
    if args.start < 0 or args.end <= args.start or args.end - args.start > 60:
        print("ERRO: use um intervalo válido de no máximo 60 segundos", file=sys.stderr)
        return 1
    if not 0 < args.fps <= 4 or not 1 <= args.max_attempts <= 5:
        print("ERRO: fps deve ficar entre 0 e 4 e max-attempts entre 1 e 5", file=sys.stderr)
        return 1
    if args.run_id and (
        not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum()
        or args.run_id in {".", ".."}
    ):
        print("ERRO: run-id contém caracteres inválidos", file=sys.stderr)
        return 1

    source_path, source_record = find_video_record(args.video_id)
    identity = source_record.get("identity", {})
    video_url = identity.get("youtube_watch_url")
    if not video_url:
        print("ERRO: o registro curado não contém youtube_watch_url", file=sys.stderr)
        return 1

    model = args.model or os.getenv("GEMINI_TRANSCRIPTION_MODEL") or DEFAULT_MODEL
    started_at = utc_now()
    run_id = args.run_id or f"transcript-{started_at.replace(':', '-')}"
    output_dir = FEATURE_RUNS_DIR / run_id / "videos" / args.video_id
    prompt = build_prompt(args.start, args.end)
    request_manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "started_at": started_at,
        "identity": {
            "video_id": args.video_id,
            "channel_id": identity.get("channel_id"),
            "youtube_watch_url": video_url,
        },
        "source": {
            "youtube_record": source_path.relative_to(PROJECT_ROOT).as_posix(),
            "sector": source_record.get("classification", {}).get("sector"),
        },
        "extractor": {
            "provider": "Google Gemini API",
            "operation": "models.generateContent",
            "model_requested": model,
            "prompt_version": PROMPT_VERSION,
        },
        "clip": {
            "start_seconds": args.start,
            "end_seconds": args.end,
            "requested_fps": args.fps,
        },
        "prompt": prompt,
        "data_exclusion": [
            "title",
            "description",
            "channel_title",
            "view_count",
            "like_count",
            "comment_count",
            "subscriber_count",
        ],
    }
    save_json(output_dir / "request_manifest.json", request_manifest)

    try:
        request_payload, raw_response = request_gemini(
            api_key=api_key,
            model=model,
            video_url=video_url,
            start_seconds=args.start,
            end_seconds=args.end,
            fps=args.fps,
            prompt=prompt,
            max_attempts=args.max_attempts,
        )
        # Persiste a resposta antes do parsing para permitir auditoria de JSON malformado.
        save_json(output_dir / "api_request.json", request_payload)
        save_json(output_dir / "raw_response.json", raw_response)
        analysis = parse_json_response(extract_text(raw_response))
        validation_errors = validate_transcript(
            analysis,
            start_seconds=args.start,
            end_seconds=args.end,
        )
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
    except QuotaExhaustedError as exc:
        code = 75
        caught_error = exc
        error = {"provider_status": 429, "retryable_after_quota_reset": True}
        message = f"ERRO DE COTA: {exc}"
    except ProviderUnavailableError as exc:
        code = 69
        caught_error = exc
        error = {"provider_status": exc.status_code, "retryable": True}
        message = f"ERRO TEMPORÁRIO DO PROVEDOR: {exc}"
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        code = 1
        caught_error = exc
        error = {}
        message = f"ERRO: {exc}"
    else:
        result = {
            "schema_version": "0.1.0",
            "identity": request_manifest["identity"],
            "provenance": {
                "source": request_manifest["source"],
                "extractor": request_manifest["extractor"],
                "clip": request_manifest["clip"],
                "started_at": started_at,
                "completed_at": utc_now(),
                "raw_response_file": "raw_response.json",
                "request_manifest_file": "request_manifest.json",
            },
            "response_metadata": {
                "model_version": raw_response.get("modelVersion"),
                "usage_metadata": raw_response.get("usageMetadata"),
            },
            "analysis": analysis,
        }
        save_json(output_dir / "transcript.json", result)
        print(f"Transcrição concluída: {output_dir / 'transcript.json'}")
        return 0

    error.update(
        {
            "schema_version": "0.1.0",
            "failed_at": utc_now(),
            "error_type": type(caught_error).__name__,
            "message": str(caught_error).replace(api_key, "<redacted>"),
        }
    )
    save_json(output_dir / "error.json", error)
    print(message, file=sys.stderr)
    print(f"Diagnóstico salvo em: {output_dir}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
