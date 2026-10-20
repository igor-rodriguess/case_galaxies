"""Analisa localmente a abertura de um vídeo com SmolVLM2-500M.

O script não persiste frames: decodifica somente o intervalo solicitado para a
memória, executa o modelo em CPU e salva a resposta e a proveniência em JSON.

Exemplos:
    python scripts/smolvlm_opening_pilot.py --synthetic
    python scripts/smolvlm_opening_pilot.py --video-file C:\\videos\\exemplo.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_RUNS_DIR = PROJECT_ROOT / "data" / "features" / "runs"
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPT_VERSION = "galaxies-opening-local-v0.1.0"
ALLOWED_OPENING_TYPES = (
    "question",
    "statement",
    "demonstration",
    "product_reveal",
    "person_speaking",
    "montage",
    "text_hook",
    "other",
    "unknown",
)


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


def resize_frame(image: Any, longest_edge: int = 384) -> Any:
    """Reduz um PIL.Image sem ampliar imagens pequenas."""
    width, height = image.size
    scale = min(1.0, longest_edge / max(width, height))
    if scale == 1.0:
        return image.convert("RGB")
    return image.resize((round(width * scale), round(height * scale))).convert("RGB")


def sample_video_frames(
    path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    longest_edge: int = 384,
) -> tuple[list[Any], list[float]]:
    """Amostra o primeiro frame disponível para cada timestamp solicitado."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depende do extra local-vlm
        raise RuntimeError("Instale o extra local-vlm para decodificar vídeos.") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Arquivo audiovisual não encontrado: {path}")

    targets: list[float] = []
    timestamp = start_seconds
    while timestamp < end_seconds - 1e-9:
        targets.append(round(timestamp, 6))
        timestamp += 1 / fps

    frames: list[Any] = []
    sampled_at: list[float] = []
    target_index = 0
    last_frame = None
    last_time = None

    with av.open(str(path)) as container:
        video_streams = container.streams.video
        if not video_streams:
            raise ValueError("O arquivo não contém uma faixa de vídeo.")
        stream = video_streams[0]
        for frame in container.decode(stream):
            frame_time = float(frame.time or 0.0)
            if frame_time < start_seconds:
                continue
            if frame_time >= end_seconds and target_index >= len(targets):
                break

            image = resize_frame(frame.to_image(), longest_edge)
            last_frame = image
            last_time = frame_time
            while target_index < len(targets) and frame_time >= targets[target_index] - 1e-6:
                frames.append(image.copy())
                sampled_at.append(round(frame_time, 6))
                target_index += 1
            if target_index >= len(targets):
                break

    if last_frame is not None:
        while target_index < len(targets):
            frames.append(last_frame.copy())
            sampled_at.append(round(float(last_time or 0.0), 6))
            target_index += 1
    if not frames:
        raise ValueError("Nenhum frame foi decodificado no intervalo solicitado.")
    return frames, sampled_at


def synthetic_opening_frames(fps: float = 4.0) -> tuple[list[Any], list[float]]:
    """Cria uma abertura controlada para avaliar reconhecimento e JSON."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 384, 216
    font = ImageFont.load_default(size=28)
    small_font = ImageFont.load_default(size=18)
    frames: list[Any] = []
    sampled_at: list[float] = []

    for index in range(round(3 * fps)):
        timestamp = index / fps
        sampled_at.append(timestamp)
        if timestamp < 1:
            image = Image.new("RGB", (width, height), "#d51f5d")
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((55, 55, 329, 165), radius=20, fill="white")
            draw.text((91, 91), "NOVO BATOM", fill="#141414", font=font)
        elif timestamp < 2:
            image = Image.new("RGB", (width, height), "#f2dfd0")
            draw = ImageDraw.Draw(image)
            draw.rectangle((159, 93, 225, 195), fill="#161616")
            draw.rectangle((170, 45, 214, 95), fill="#b70032")
            draw.text((90, 12), "BATOM VERMELHO", fill="#141414", font=small_font)
        else:
            image = Image.new("RGB", (width, height), "#151515")
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((66, 60, 318, 156), radius=16, fill="#ffd447")
            draw.text((88, 93), "COMPRE AGORA", fill="#101010", font=font)
        frames.append(image)
    return frames, sampled_at


def build_prompt(sampled_at: list[float]) -> str:
    timestamps = ", ".join(f"{value:g}" for value in sampled_at)
    labels = ", ".join(ALLOWED_OPENING_TYPES)
    return f"""
Analise apenas os frames ordenados da abertura de um vídeo. Os timestamps, na
mesma ordem, são: [{timestamps}] segundos. Não invente áudio, fala, marca ou
contexto que não esteja visível. Responda somente com um objeto JSON válido,
sem markdown e sem texto antes ou depois, usando exatamente esta estrutura:

{{
  "visual": {{
    "first_visible_element": null,
    "people_present": null,
    "faces_visible": null,
    "product_in_focus": {{"present": null, "description": null}},
    "on_screen_text": [],
    "setting": null,
    "framing": "unknown",
    "camera_movement": "unknown",
    "scene_change_count": null,
    "motion_intensity": "unknown"
  }},
  "audio": {{"analyzed": false}},
  "interpretation": {{
    "opening_type": "unknown",
    "product_prominence": "unknown",
    "call_to_action": {{"present": null, "text": null}}
  }},
  "evidence": [],
  "uncertainties": []
}}

opening_type deve ser um de: {labels}.
product_prominence deve ser none, low, medium, high ou unknown.
motion_intensity deve ser low, medium, high ou unknown.
Cada item de on_screen_text deve ter text e timestamp_seconds.
Cada item de evidence deve ter timestamp_seconds e observation.
""".strip()


def build_compact_prompt() -> str:
    """Prompt curto para medir a obediência de um modelo pequeno."""
    return """
Inspect the ordered video frames. Return ONLY valid JSON, with no markdown and
no explanation:
{"visible_text":[],"product_present":null,"product_description":null,
"people_present":null,"scene_change_count":null,"opening_type":"unknown",
"product_prominence":"unknown","cta_present":null,"cta_text":null}
opening_type: product_reveal, text_hook, demonstration, montage, other, unknown.
product_prominence: none, low, medium, high, unknown.
Use only visible evidence. Do not infer audio.
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extrai um único objeto JSON, tolerando apenas cercas de markdown."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("O modelo não retornou um objeto JSON.")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("A resposta do modelo não é um objeto JSON.")
    return parsed


def validate_analysis_shape(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"visual", "audio", "interpretation", "evidence", "uncertainties"}
    missing = sorted(required - set(analysis))
    if missing:
        errors.append(f"campos ausentes: {', '.join(missing)}")
    opening_type = analysis.get("interpretation", {}).get("opening_type")
    if opening_type not in ALLOWED_OPENING_TYPES:
        errors.append(f"opening_type inválido: {opening_type!r}")
    if analysis.get("audio") != {"analyzed": False}:
        errors.append("audio deve registrar exatamente que não foi analisado")
    return errors


def validate_compact_shape(analysis: dict[str, Any]) -> list[str]:
    required = {
        "visible_text",
        "product_present",
        "product_description",
        "people_present",
        "scene_change_count",
        "opening_type",
        "product_prominence",
        "cta_present",
        "cta_text",
    }
    missing = sorted(required - set(analysis))
    return [f"campos ausentes: {', '.join(missing)}"] if missing else []


def run_inference(
    *,
    frames: list[Any],
    prompt: str,
    model_id: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, float | int]]:
    try:
        import psutil
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - depende do extra local-vlm
        raise RuntimeError("Instale o extra local-vlm antes de executar o piloto.") from exc

    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    process = psutil.Process()
    baseline_rss = process.memory_info().rss

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    loaded_rss = process.memory_info().rss

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_sample_frames": False},
    ).to("cpu")

    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    inference_seconds = time.perf_counter() - inference_started
    prompt_tokens = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, prompt_tokens:]
    raw_text = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
    peak_rss = process.memory_info().rss

    metrics: dict[str, float | int] = {
        "frame_count": len(frames),
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_only.shape[1]),
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "rss_before_mb": round(baseline_rss / 1024**2, 1),
        "rss_after_load_mb": round(loaded_rss / 1024**2, 1),
        "rss_after_inference_mb": round(peak_rss / 1024**2, 1),
    }
    return raw_text, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piloto local de abertura com SmolVLM2-500M.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-file", type=Path, help="Arquivo audiovisual autorizado.")
    source.add_argument("--synthetic", action="store_true", help="Usa o vídeo sintético controlado.")
    parser.add_argument("--video-id", help="Identificador rastreável; padrão: nome do arquivo.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--prompt-mode",
        choices=("full", "compact"),
        default="full",
        help="Contrato completo ou diagnóstico compacto para o modelo de 500M.",
    )
    parser.add_argument("--run-id", help="Identificador do teste.")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.start < 0 or args.end <= args.start:
        print("ERRO: o intervalo precisa satisfazer 0 <= start < end", file=sys.stderr)
        return 2
    if not 0 < args.fps <= 8:
        print("ERRO: fps deve ficar entre 0 e 8 para o piloto local", file=sys.stderr)
        return 2
    if not 64 <= args.max_new_tokens <= 1024:
        print("ERRO: max-new-tokens deve ficar entre 64 e 1024", file=sys.stderr)
        return 2

    started_at = utc_now()
    if args.synthetic:
        frames, sampled_at = synthetic_opening_frames(args.fps)
        video_id = args.video_id or "synthetic-lipstick-opening"
        source = {"type": "synthetic_control", "expected_duration_seconds": 3}
    else:
        assert args.video_file is not None
        frames, sampled_at = sample_video_frames(
            args.video_file.resolve(),
            start_seconds=args.start,
            end_seconds=args.end,
            fps=args.fps,
        )
        video_id = args.video_id or args.video_file.stem
        source = {
            "type": "authorized_local_file",
            "file_name": args.video_file.name,
            "absolute_path_stored": False,
        }

    run_id = args.run_id or f"smolvlm-pilot-{started_at.replace(':', '-')}"
    output_dir = FEATURE_RUNS_DIR / run_id / "videos" / video_id
    prompt = build_prompt(sampled_at) if args.prompt_mode == "full" else build_compact_prompt()

    print(f"Modelo: {args.model}")
    print(f"Dispositivo: CPU | frames em memória: {len(frames)}")
    print("Carregando checkpoint e executando inferência local...")
    try:
        raw_text, metrics = run_inference(
            frames=frames,
            prompt=prompt,
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
        )
        parse_error = None
        validation_errors: list[str] = []
        try:
            analysis = extract_json_object(raw_text)
            validation_errors = (
                validate_analysis_shape(analysis)
                if args.prompt_mode == "full"
                else validate_compact_shape(analysis)
            )
        except (ValueError, json.JSONDecodeError) as exc:
            analysis = None
            parse_error = str(exc)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        save_json(
            output_dir / "error.json",
            {
                "schema_version": "0.1.0",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    completed_at = utc_now()
    result = {
        "schema_version": "0.1.0",
        "identity": {"video_id": video_id},
        "provenance": {
            "source": source,
            "extractor": {
                "provider": "local",
                "library": "Hugging Face Transformers",
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "prompt_mode": args.prompt_mode,
                "device": "cpu",
                "python": platform.python_version(),
            },
            "clip": {
                "start_seconds": args.start,
                "end_seconds": args.end,
                "requested_fps": args.fps,
                "sampled_timestamps": sampled_at,
                "frames_persisted": False,
                "audio_analyzed": False,
            },
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "quality": {
            "json_parsed": analysis is not None,
            "schema_valid": analysis is not None and not validation_errors,
            "parse_error": parse_error,
            "validation_errors": validation_errors,
        },
        "runtime": metrics,
        "analysis": analysis,
        "raw_model_text": raw_text,
    }
    save_json(output_dir / "opening_vlm.json", result)

    print(f"Inferência concluída em {metrics['inference_seconds']}s.")
    print(f"JSON interpretável: {result['quality']['json_parsed']}")
    print(f"Schema mínimo válido: {result['quality']['schema_valid']}")
    print(f"Resultado: {output_dir / 'opening_vlm.json'}")
    return 0 if result["quality"]["schema_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
