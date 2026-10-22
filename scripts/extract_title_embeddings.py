"""Embedding do título dos Shorts com o multilingual-e5-base (cache local).

O título é escrito antes da publicação, então é um sinal legítimo. Nada de
views, likes ou comentários entra aqui. Saída: um vetor normalizado por vídeo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "data" / "youtube" / "shorts_history" / "shorts-history-2026-10-21" / "normalized_records.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "features" / "embeddings" / "title_e5.parquet")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    records = json.loads(args.records.read_text(encoding="utf-8"))["records"]
    frame = pd.DataFrame({"video_id": [r["video_id"] for r in records], "title": [r.get("title") or "" for r in records]})
    frame = frame.drop_duplicates("video_id", keep="last").reset_index(drop=True)
    model = SentenceTransformer(args.model, device="cpu")
    vectors = model.encode(
        ["query: " + title for title in frame["title"]],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    frame["embedding"] = list(np.asarray(vectors, dtype=np.float32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[["video_id", "title", "embedding"]].to_parquet(args.output, index=False)
    print(f"{len(frame)} títulos -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
