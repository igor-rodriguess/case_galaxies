"""Piloto de análise dos segundos iniciais de um vídeo público do YouTube.

O script envia ao Gemini somente um intervalo temporal do vídeo por URL pública.
Ele não baixa nem armazena o arquivo audiovisual e não envia métricas de desempenho.

Uso, a partir da raiz do projeto:
    python scripts/gemini_opening_pilot.py --video-id gmi2iRLj1Q4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOUTUBE_RUNS_DIR = PROJECT_ROOT / "data" / "youtube" / "runs"
FEATURE_RUNS_DIR = PROJECT_ROOT / "data" / "features" / "runs"
DEFAULT_MODEL = "gemini-3.8-flash"
PROMPT_VERSION = "galaxies-opening-v0.1.0"


class QuotaExhaustedError(RuntimeError):
    """Indica que a cota do projeto/modelo foi esgotada."""


class ProviderUnavailableError(RuntimeError):
    """Indica falha temporária do provedor que pode ser retomada depois."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


def load_dotenv(path: Path) -> None:
    """Carrega um .env simples sem sobrescrever o ambiente atual."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def find_video_record(video_id: str) -> tuple[Path, dict]:
    """Localiza o JSON curado do vídeo, priorizando a coleta mais recente."""
    run_dirs = sorted(
        (path for path in YOUTUBE_RUNS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        matches = list((run_dir / "curated" / "videos").rglob(f"{video_id}.json"))
        if matches:
            source_path = matches[0]
            return source_path, json.loads(source_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Não encontrei {video_id}.json em nenhuma coleta curada do YouTube.")


def build_prompt(start_seconds: float, end_seconds: float, fps: float) -> str:
    return f"""
Você é um anotador de conteúdo audiovisual para pesquisa. Analise exclusivamente o
intervalo de {start_seconds:g} a {end_seconds:g} segundos fornecido nesta requisição,
amostrado a {fps:g} FPS. Use apenas o que puder observar ou ouvir nesse intervalo.

Regras:
- Não use título, descrição, canal, popularidade, visualizações, curtidas ou comentários.
- Não estime desempenho, viralidade, qualidade publicitária ou intenção do criador.
- Não complete informações por conhecimento externo.
- Se algo não estiver observável, use null, "unknown" ou uma lista vazia.
- Separe observações diretas de interpretações.
- Para texto na tela e fala, transcreva apenas o que estiver realmente legível/audível.
- Associe as observações importantes a timestamps em segundos.
- Responda somente com JSON válido, sem markdown, seguindo exatamente esta estrutura:

{{
  "schema_version": "0.1.0",
  "analysis_scope": {{
    "start_seconds": {start_seconds:g},
    "end_seconds": {end_seconds:g},
    "requested_fps": {fps:g}
  }},
  "visual": {{
    "first_visible_element": {{
      "value": null,
      "timestamp_seconds": null,
      "confidence": 0.0
    }},
    "people": {{
      "present": null,
      "count_estimate": null,
      "faces_visible": null,
      "confidence": 0.0
    }},
    "product_or_object_in_focus": {{
      "present": null,
      "description": null,
      "brand_if_legible": null,
      "confidence": 0.0
    }},
    "on_screen_text": [],
    "setting": {{
      "description": null,
      "confidence": 0.0
    }},
    "shot_and_camera": {{
      "framing": "unknown",
      "camera_movement": "unknown",
      "confidence": 0.0
    }},
    "scene_changes": [],
    "motion_intensity": {{
      "label": "unknown",
      "confidence": 0.0
    }}
  }},
  "audio": {{
    "speech_present": null,
    "speech_start_seconds": null,
    "transcript_excerpt": null,
    "language": null,
    "music_present": null,
    "sound_effects": [],
    "notes": null,
    "confidence": 0.0
  }},
  "interpretation": {{
    "opening_type": {{
      "label": "unknown",
      "allowed_labels": [
        "question", "statement", "demonstration", "product_reveal",
        "person_speaking", "montage", "text_hook", "other", "unknown"
      ],
      "reason": null,
      "confidence": 0.0
    }},
    "product_prominence": {{
      "label": "unknown",
      "allowed_labels": ["none", "low", "medium", "high", "unknown"],
      "reason": null,
      "confidence": 0.0
    }},
    "call_to_action": {{
      "present": null,
      "text_or_description": null,
      "confidence": 0.0
    }}
  }},
  "evidence": [],
  "uncertainties": []
}}

Cada item de "on_screen_text" deve conter "text", "timestamp_seconds" e
"legibility". Cada item de "scene_changes" deve conter "timestamp_seconds" e
"description". Cada evidência deve conter "timestamp_seconds", "modality"
("visual", "audio" ou "audiovisual") e "observation". Valores de confidence
devem ficar entre 0 e 1. Não repita a lista allowed_labels fora dos locais indicados.
""".strip()


def extract_text(raw_response: dict) -> str:
    candidates = raw_response.get("candidates", [])
    if not candidates:
        raise ValueError("A API não retornou candidatos de resposta.")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part["text"] for part in parts if part.get("text") and not part.get("thought")]
    if not texts:
        raise ValueError("A resposta não contém uma parte textual utilizável.")
    return "\n".join(texts).strip()


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    parsed = json.loads(cleaned.strip())
    if not isinstance(parsed, dict):
        raise ValueError("O conteúdo retornado não é um objeto JSON.")
    return parsed


def request_gemini(
    *,
    api_key: str,
    model: str,
    video_url: str,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    prompt: str,
    max_attempts: int,
) -> tuple[dict, dict]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "file_data": {"file_uri": video_url},
                        "video_metadata": {
                            "start_offset": f"{start_seconds:g}s",
                            "end_offset": f"{end_seconds:g}s",
                            "fps": fps,
                        },
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 4096,
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-goog-api-key": api_key,
        },
    )

    transient_statuses = {429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=90) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").replace(api_key, "<redacted>")
            if exc.code == 429 and ("RESOURCE_EXHAUSTED" in body or "Quota exceeded" in body):
                raise QuotaExhaustedError(f"Gemini API retornou HTTP {exc.code}: {body}") from exc
            if exc.code in transient_statuses and attempt < max_attempts:
                delay_seconds = 5 * attempt
                print(
                    f"AVISO: HTTP {exc.code} temporário; nova tentativa "
                    f"{attempt + 1}/{max_attempts} em {delay_seconds}s."
                )
                time.sleep(delay_seconds)
                continue
            if exc.code in transient_statuses:
                raise ProviderUnavailableError(
                    f"Gemini API temporariamente indisponível (HTTP {exc.code}): {body}",
                    status_code=exc.code,
                ) from exc
            raise RuntimeError(f"Gemini API retornou HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt < max_attempts:
                delay_seconds = 5 * attempt
                print(
                    "AVISO: falha temporária de rede; nova tentativa "
                    f"{attempt + 1}/{max_attempts} em {delay_seconds}s."
                )
                time.sleep(delay_seconds)
                continue
            reason = getattr(exc, "reason", str(exc))
            raise ProviderUnavailableError(
                f"Falha temporária de rede ao consultar a Gemini API: {reason}"
            ) from exc

    return payload, raw_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa uma abertura curta por URL pública usando Gemini."
    )
    parser.add_argument("--video-id", required=True, help="ID do vídeo já coletado.")
    parser.add_argument("--start", type=float, default=0.0, help="Início do trecho, em segundos.")
    parser.add_argument("--end", type=float, default=3.0, help="Fim do trecho, em segundos.")
    parser.add_argument("--fps", type=float, default=4.0, help="Taxa de amostragem solicitada.")
    parser.add_argument("--model", help=f"Modelo Gemini. Padrão: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--run-id",
        help=(
            "Identificador opcional para agrupar vários vídeos na mesma execução. "
            "Aceita somente letras, números, ponto, hífen e sublinhado."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Número máximo de tentativas da chamada. Padrão: 3.",
    )
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
    if args.start < 0 or args.end <= args.start:
        print("ERRO: o intervalo precisa satisfazer 0 <= start < end", file=sys.stderr)
        return 1
    if not 0 < args.fps <= 24:
        print("ERRO: use um FPS maior que 0 e menor ou igual a 24", file=sys.stderr)
        return 1
    if not 1 <= args.max_attempts <= 5:
        print("ERRO: max-attempts deve ficar entre 1 e 5", file=sys.stderr)
        return 1
    if args.run_id and (
        not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum()
        or args.run_id in {".", ".."}
    ):
        print("ERRO: run-id contém caracteres inválidos", file=sys.stderr)
        return 1

    model = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    source_path, source_record = find_video_record(args.video_id)
    identity = source_record.get("identity", {})
    video_url = identity.get("youtube_watch_url")
    if not video_url:
        print("ERRO: o registro curado não contém youtube_watch_url", file=sys.stderr)
        return 1

    started_at = utc_now()
    run_id = args.run_id or started_at.replace(":", "-")
    output_dir = FEATURE_RUNS_DIR / run_id / "videos" / args.video_id
    prompt = build_prompt(args.start, args.end, args.fps)

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
            "processing_mode": "static (default)",
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

    print(f"Vídeo: {args.video_id}")
    print(f"Trecho enviado: {args.start:g}s-{args.end:g}s a {args.fps:g} FPS")
    print(f"Modelo solicitado: {model}")
    print("Aguardando a análise do Gemini...")

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
        response_text = extract_text(raw_response)
        analysis = parse_json_response(response_text)
    except QuotaExhaustedError as exc:
        save_json(
            output_dir / "error.json",
            {
                "schema_version": "0.1.0",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "provider_status": 429,
                "retryable_after_quota_reset": True,
                "message": str(exc).replace(api_key, "<redacted>"),
            },
        )
        print(f"ERRO DE COTA: {exc}", file=sys.stderr)
        print(f"Diagnóstico salvo em: {output_dir}", file=sys.stderr)
        return 75
    except ProviderUnavailableError as exc:
        save_json(
            output_dir / "error.json",
            {
                "schema_version": "0.1.0",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "provider_status": exc.status_code,
                "retryable": True,
                "message": str(exc).replace(api_key, "<redacted>"),
            },
        )
        print(f"ERRO TEMPORÁRIO DO PROVEDOR: {exc}", file=sys.stderr)
        print(f"Diagnóstico salvo em: {output_dir}", file=sys.stderr)
        return 69
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        save_json(
            output_dir / "error.json",
            {
                "schema_version": "0.1.0",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "message": str(exc).replace(api_key, "<redacted>"),
            },
        )
        print(f"ERRO: {exc}", file=sys.stderr)
        print(f"Diagnóstico salvo em: {output_dir}", file=sys.stderr)
        return 1

    save_json(output_dir / "api_request.json", request_payload)
    save_json(output_dir / "raw_response.json", raw_response)
    completed_at = utc_now()
    result = {
        "schema_version": "0.1.0",
        "identity": request_manifest["identity"],
        "provenance": {
            "source": request_manifest["source"],
            "extractor": request_manifest["extractor"],
            "clip": request_manifest["clip"],
            "started_at": started_at,
            "completed_at": completed_at,
            "raw_response_file": "raw_response.json",
            "request_manifest_file": "request_manifest.json",
        },
        "response_metadata": {
            "model_version": raw_response.get("modelVersion"),
            "usage_metadata": raw_response.get("usageMetadata"),
        },
        "analysis": analysis,
    }
    save_json(output_dir / "opening_vlm.json", result)

    print("Análise concluída.")
    print(f"Resultado principal: {output_dir / 'opening_vlm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
