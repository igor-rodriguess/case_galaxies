"""Piloto local quantizado de abertura com Qwen2.5-VL-3B.

Usa bitsandbytes em 4 bits para caber em uma máquina Windows com 16 GB de RAM.
Os frames são mantidos somente em memória e o áudio não é analisado.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

if __package__:
    from scripts.smolvlm_opening_pilot import (
        FEATURE_RUNS_DIR,
        build_compact_prompt,
        extract_json_object,
        sample_video_frames,
        save_json,
        synthetic_opening_frames,
        utc_now,
        validate_compact_shape,
    )
else:
    from smolvlm_opening_pilot import (
        FEATURE_RUNS_DIR,
        build_compact_prompt,
        extract_json_object,
        sample_video_frames,
        save_json,
        synthetic_opening_frames,
        utc_now,
        validate_compact_shape,
    )

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT_VERSION = "galaxies-opening-qwen-compact-v0.1.0"


def run_inference(
    *,
    frames: list[Any],
    prompt: str,
    model_id: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, float | int | str]]:
    try:
        import bitsandbytes as bnb
        import psutil
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as exc:  # pragma: no cover - depende do extra local-vlm
        raise RuntimeError("Instale o extra local-vlm antes de executar o Qwen.") from exc

    available_gb = psutil.virtual_memory().available / 1024**3
    swap_free_gb = psutil.swap_memory().free / 1024**3
    if available_gb < 0.75 or available_gb + swap_free_gb < 8:
        raise MemoryError(
            f"Somente {available_gb:.2f} GB de RAM e {swap_free_gb:.2f} GB de "
            "paginação estão disponíveis; "
            "feche outros programas antes de carregar o Qwen quantizado."
        )

    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map={"": "cpu"},
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    loaded_rss = process.memory_info().rss

    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
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

    metrics: dict[str, float | int | str] = {
        "frame_count": len(frames),
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_only.shape[1]),
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "rss_before_mb": round(baseline_rss / 1024**2, 1),
        "rss_after_load_mb": round(loaded_rss / 1024**2, 1),
        "rss_after_inference_mb": round(process.memory_info().rss / 1024**2, 1),
        "model_memory_mb": round(model.get_memory_footprint() / 1024**2, 1),
        "quantization": "bitsandbytes-nf4-double-quant",
        "bitsandbytes_version": bnb.__version__,
        "physical_memory_available_before_load_gb": round(available_gb, 2),
        "swap_available_before_load_gb": round(swap_free_gb, 2),
    }
    return raw_text, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piloto Qwen2.5-VL-3B local em 4 bits.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-file", type=Path, help="Arquivo audiovisual autorizado.")
    source.add_argument("--synthetic", action="store_true", help="Usa o controle sintético.")
    parser.add_argument("--video-id", help="Identificador rastreável; padrão: nome do arquivo.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
    if not 0 < args.fps <= 4:
        print("ERRO: fps deve ficar entre 0 e 4 no Qwen local", file=sys.stderr)
        return 2
    if not 64 <= args.max_new_tokens <= 512:
        print("ERRO: max-new-tokens deve ficar entre 64 e 512", file=sys.stderr)
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

    run_id = args.run_id or f"qwen-pilot-{started_at.replace(':', '-')}"
    output_dir = FEATURE_RUNS_DIR / run_id / "videos" / video_id
    prompt = build_compact_prompt()
    print(f"Modelo: {args.model}")
    print(f"Dispositivo: CPU 4-bit | frames em memória: {len(frames)}")
    print("Carregando checkpoint quantizado e executando inferência...")

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
            validation_errors = validate_compact_shape(analysis)
        except (ValueError, json.JSONDecodeError) as exc:
            analysis = None
            parse_error = str(exc)
    except (ImportError, MemoryError, OSError, RuntimeError, ValueError) as exc:
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
        print(f"Diagnóstico: {output_dir / 'error.json'}", file=sys.stderr)
        return 1

    result = {
        "schema_version": "0.1.0",
        "identity": {"video_id": video_id},
        "provenance": {
            "source": source,
            "extractor": {
                "provider": "local",
                "library": "Hugging Face Transformers + bitsandbytes",
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
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
            "completed_at": utc_now(),
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
    print(f"Schema compacto válido: {result['quality']['schema_valid']}")
    print(f"Resultado: {output_dir / 'opening_vlm.json'}")
    return 0 if result["quality"]["schema_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
