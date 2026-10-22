"""Extrai embeddings pré-publicação de thumbnail, título e título+descrição.

O processo é retomável: cada artefato guarda ``video_id``, vetor, modelo e
manifesto. Métricas, rótulos e campos pós-publicação nunca são entregues aos
modelos de embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_WEIGHTS = "laion2b_s34b_b79k"
RESNET18_MODEL = "torchvision:resnet18"
RESNET18_WEIGHTS = "IMAGENET1K_V1"
DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-base"
THUMBNAIL_DIR = PROJECT_ROOT / "data" / "features" / "thumbnails"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "features" / "embeddings"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_records() -> tuple[Path, list[dict]]:
    candidates: list[tuple[Path, list[dict]]] = []
    for run in (PROJECT_ROOT / "data" / "youtube" / "runs").iterdir():
        source = run / "normalized_records.json"
        if source.is_file():
            payload = json.loads(source.read_text(encoding="utf-8"))
            candidates.append((source, list(payload.get("records", []))))
    if not candidates:
        raise FileNotFoundError("Nenhum normalized_records.json encontrado.")
    # A expansão consolidada é a maior coleta; um piloto recente não pode
    # silenciosamente encolher a base usada na avaliação.
    return max(candidates, key=lambda item: (len(item[1]), item[0].stat().st_mtime_ns))


def feature_map_records(path: Path) -> tuple[Path, list[dict]]:
    """Lê IDs de um mapa JSONL para extrair somente thumbnails.

    A população de avaliação pode ser maior que a coleta inicial de metadados.
    Em especial, a Fase 2 tem 14 mil Shorts no histórico completo. Receber o
    mapa explicitamente impede que um piloto menor escolha silenciosamente a
    fonte e crie cobertura enviesada de embeddings.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Mapa de features não encontrado: {path}")
    frame = pd.read_json(path, lines=True)
    if "video_id" not in frame:
        raise ValueError("O mapa de features precisa conter video_id.")
    ids = frame["video_id"].dropna().astype(str).drop_duplicates().sort_values()
    return path, [{"video_id": video_id} for video_id in ids]


def load_existing(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(pd.read_parquet(path, columns=["video_id"])["video_id"].astype(str))


def download_thumbnail(video_id: str, *, force: bool) -> tuple[Path | None, str | None]:
    target = THUMBNAIL_DIR / f"{video_id}.jpg"
    if target.is_file() and not force:
        return target, None
    request = urllib.request.Request(
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        headers={"User-Agent": "GalaxiesResearch/0.1 (authorized analytics)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
        if len(data) < 256:
            raise ValueError("thumbnail vazia ou curta demais")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        return target, None
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def text_value(record: dict, include_description: bool) -> str:
    title = str(record.get("title") or "").strip()
    if not include_description:
        return title
    description = str(record.get("description") or "").strip()[:300]
    return f"{title}\n{description}".strip()


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    previous = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    merged = pd.concat([previous, frame], ignore_index=True).drop_duplicates("video_id", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    # O arquivo é consultado por auditorias enquanto a extração roda. Escrever em
    # um temporário e substituir ao final impede que um leitor enxergue um
    # Parquet parcialmente gravado.
    temporary = path.with_name(f"{path.stem}.writing{path.suffix}")
    merged.to_parquet(temporary, index=False)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai embeddings de thumbnail e texto de vídeos YouTube.")
    parser.add_argument("--limit", type=int, help="Limita itens para piloto; omita para toda a base.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Threads de CPU do encoder visual (padrão: 1 para execuções compartilhadas).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    parser.add_argument(
        "--thumbnail-workers",
        type=int,
        default=8,
        help="Downloads paralelos de thumbnails (padrão: 8).",
    )
    parser.add_argument("--force-thumbnails", action="store_true")
    parser.add_argument("--only", choices=("thumb", "title", "title_desc"))
    parser.add_argument(
        "--thumbnail-backend",
        choices=("open_clip", "resnet18"),
        default="open_clip",
        help=(
            "Encoder visual. resnet18 é uma alternativa pré-treinada estável quando "
            "open_clip não estiver carregável; grava em um Parquet separado."
        ),
    )
    parser.add_argument(
        "--feature-map",
        type=Path,
        help=(
            "Mapa JSONL cuja população deve receber thumbnails. Só pode ser usado "
            "com --only thumb, pois o mapa não contém texto original."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("ERRO: --limit deve ser >= 1.")
    if args.batch_size < 1:
        raise SystemExit("ERRO: --batch-size deve ser >= 1.")
    if args.torch_threads < 1:
        raise SystemExit("ERRO: --torch-threads deve ser >= 1.")
    if args.thumbnail_workers < 1:
        raise SystemExit("ERRO: --thumbnail-workers deve ser >= 1.")
    selected = {args.only} if args.only else {"thumb", "title", "title_desc"}
    if args.feature_map and selected != {"thumb"}:
        raise SystemExit("ERRO: --feature-map exige --only thumb; texto vem da coleta original.")
    if args.thumbnail_backend != "open_clip" and "thumb" not in selected:
        raise SystemExit("ERRO: --thumbnail-backend só pode ser usado quando thumbnails forem extraídas.")
    print("Inicializando dependências de embedding...", flush=True)
    torch = None
    if "thumb" in selected:
        try:
            import torch
            torch.set_num_threads(args.torch_threads)
        except ImportError as exc:
            raise SystemExit('ERRO: instale com pip install -e ".[embeddings]".') from exc

    print("Dependências carregadas; selecionando população...", flush=True)
    source, records = (
        feature_map_records(args.feature_map) if args.feature_map else latest_records()
    )
    records = sorted(records, key=lambda row: str(row["video_id"]))
    if args.limit:
        records = records[: args.limit]
    print(f"Coleta selecionada: {source} ({len(records)} vídeos solicitados).", flush=True)
    if torch is None:
        device = "cpu" if args.device == "auto" else args.device
    else:
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
    started_at = utc_now()
    manifest = {
        "schema_version": "0.1.0",
        "started_at": started_at,
        "source_records": str(source),
        "records_requested": len(records),
        "models": {
            "thumbnail": (
                f"open_clip:{DEFAULT_CLIP_MODEL}/{DEFAULT_CLIP_WEIGHTS}"
                if args.thumbnail_backend == "open_clip"
                else f"{RESNET18_MODEL}/{RESNET18_WEIGHTS}"
            ),
            "text": DEFAULT_TEXT_MODEL,
        },
        "device": device,
        "inputs": "Somente thumbnail, título e os primeiros 300 caracteres da descrição, disponíveis antes da publicação.",
    }
    save_json(EMBEDDING_DIR / "manifest.json", manifest)
    print("Manifesto de execução registrado.", flush=True)

    if "thumb" in selected:
        assert torch is not None
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit('ERRO: instale com pip install -e ".[embeddings]".') from exc
        if args.thumbnail_backend == "open_clip":
            try:
                import open_clip
            except ImportError as exc:
                raise SystemExit('ERRO: instale com pip install -e ".[embeddings]".') from exc
            model, _, preprocess = open_clip.create_model_and_transforms(
                DEFAULT_CLIP_MODEL,
                pretrained=DEFAULT_CLIP_WEIGHTS,
                device=device,
            )
            target = EMBEDDING_DIR / "thumb.parquet"
            model_name = DEFAULT_CLIP_MODEL
            weights_name = DEFAULT_CLIP_WEIGHTS
        else:
            from torch import nn
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = nn.Identity()
            model = model.to(device)
            preprocess = weights.transforms()
            target = EMBEDDING_DIR / "thumb_resnet18.parquet"
            model_name = RESNET18_MODEL
            weights_name = RESNET18_WEIGHTS
        print("Lendo checkpoint de thumbnails...", flush=True)
        completed = load_existing(target)
        print(f"Checkpoint lido: {len(completed)} thumbnails já vetorizadas.", flush=True)
        candidates = [row for row in records if str(row["video_id"]) not in completed]
        images: list[tuple[str, Path]] = []
        failures: list[dict] = []
        print(
            f"Thumbnails pendentes: {len(candidates)}; baixando com {args.thumbnail_workers} workers...",
            flush=True,
        )
        # A ordem de conclusão da rede não é a ordem da fila: guardamos o id junto
        # do arquivo e reordenamos depois, para manter resultados determinísticos.
        with ThreadPoolExecutor(max_workers=args.thumbnail_workers) as executor:
            futures = {
                executor.submit(download_thumbnail, str(row["video_id"]), force=args.force_thumbnails): str(row["video_id"])
                for row in candidates
            }
            for index, future in enumerate(as_completed(futures), start=1):
                video_id = futures[future]
                try:
                    path, error = future.result()
                except Exception as exc:  # preserva falhas inesperadas por vídeo
                    path, error = None, f"{type(exc).__name__}: {exc}"
                if path is None:
                    failures.append({"video_id": video_id, "error": error, "observed_at": utc_now()})
                else:
                    images.append((video_id, path))
                if index % 100 == 0 or index == len(candidates):
                    print(f"  thumbnails: {index}/{len(candidates)}", flush=True)
        images.sort(key=lambda item: item[0])
        failure_name = (
            "thumbnail_failures.json"
            if args.thumbnail_backend == "open_clip"
            else f"thumbnail_{args.thumbnail_backend}_failures.json"
        )
        save_json(EMBEDDING_DIR / failure_name, failures)
        print(
            f"Inicializando {args.thumbnail_backend} para {len(images)} thumbnails válidas...",
            flush=True,
        )
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(images), args.batch_size):
                batch = images[offset : offset + args.batch_size]
                tensor = torch.stack([preprocess(Image.open(path).convert("RGB")) for _, path in batch]).to(device)
                vectors = (
                    model.encode_image(tensor)
                    if args.thumbnail_backend == "open_clip"
                    else model(tensor)
                )
                vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
                rows = [
                    {
                        "video_id": video_id,
                        "embedding": vector.astype(float).tolist(),
                        "model": model_name,
                        "weights": weights_name,
                        "created_at": utc_now(),
                    }
                    for (video_id, _), vector in zip(batch, vectors, strict=True)
                ]
                # Checkpoint por lote: uma interrupção não força reprocessar a base.
                write_table(target, rows)
                if (offset // args.batch_size + 1) % 10 == 0 or offset + len(batch) == len(images):
                    print(f"  vetores visuais: {offset + len(batch)}/{len(images)}", flush=True)

    if selected & {"title", "title_desc"}:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit('ERRO: instale com pip install -e ".[embeddings]".') from exc
        print("Inicializando modelo de texto...", flush=True)
        text_model = SentenceTransformer(DEFAULT_TEXT_MODEL, device=device)
        for kind, with_description in (("title", False), ("title_desc", True)):
            if kind not in selected:
                continue
            target = EMBEDDING_DIR / f"{kind}.parquet"
            completed = load_existing(target)
            candidates = [row for row in records if str(row["video_id"]) not in completed]
            print(f"{kind}: {len(candidates)} textos pendentes.", flush=True)
            for offset in range(0, len(candidates), args.batch_size):
                batch = candidates[offset : offset + args.batch_size]
                texts = ["passage: " + text_value(row, with_description) for row in batch]
                vectors = text_model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False)
                rows = [
                    {
                        "video_id": str(row["video_id"]),
                        "embedding": vector.astype(float).tolist(),
                        "model": DEFAULT_TEXT_MODEL,
                        "created_at": utc_now(),
                        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                    for row, text, vector in zip(batch, texts, vectors, strict=True)
                ]
                write_table(target, rows)
                if (offset // args.batch_size + 1) % 10 == 0 or offset + len(batch) == len(candidates):
                    print(f"  {kind}: {offset + len(batch)}/{len(candidates)}", flush=True)

    manifest["completed_at"] = utc_now()
    manifest["status"] = "complete"
    save_json(EMBEDDING_DIR / "manifest.json", manifest)
    print(f"Embeddings em: {EMBEDDING_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
