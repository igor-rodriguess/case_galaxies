"""Executa e salva uma dobra LOCO de embeddings por vez.

O executor é deliberadamente granular: em máquinas com limite curto de processo,
cada canal pode terminar e ser persistido sem repetir dobras já concluídas.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.embedding_evaluation import (  # noqa: E402
    PRE_REGISTERED_CONFIGS,
    build_embedding_frame,
    latest_normalized_records,
    score_channel,
)
from galaxies.agent.protocol import latest_feature_map  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa uma dobra LOCO de embeddings.")
    parser.add_argument("--config", choices=[item.name for item in PRE_REGISTERED_CONFIGS], required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-map", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--embeddings-dir", type=Path, default=PROJECT_ROOT / "data/features/embeddings")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = next(item for item in PRE_REGISTERED_CONFIGS if item.name == args.config)
    feature_map = args.feature_map or latest_feature_map(MODELING_RUNS_DIR)
    records = args.records or latest_normalized_records(PROJECT_ROOT)
    frame, vault_channels = build_embedding_frame(feature_map, args.embeddings_dir, records)
    output_dir = PROJECT_ROOT / "data/modeling/runs" / args.run_id / "folds" / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.channel}.parquet"
    if output_path.exists():
        print(f"Já concluída: {output_path}")
        return 0

    scores, components = score_channel(frame, args.channel, config, args.seed)
    temporary_path = output_path.with_suffix(".writing.parquet")
    scores.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    metadata = {
        "schema_version": "0.1.0",
        "protocol": "viral-intra-canal-v1",
        "channel_id": args.channel,
        "config": config.name,
        "pca_components_used": components,
        "feature_map": str(feature_map),
        "records": str(records),
        "vault_channels_excluded": list(vault_channels),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    (output_path.with_suffix(".json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Concluída: {args.channel} ({len(scores)} vídeos; PCA={components})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
