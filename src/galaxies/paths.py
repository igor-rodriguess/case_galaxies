"""Caminhos centrais do projeto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
YOUTUBE_RUNS_DIR = DATA_DIR / "youtube" / "runs"
FEATURE_RUNS_DIR = DATA_DIR / "features" / "runs"
MODELING_RUNS_DIR = DATA_DIR / "modeling" / "runs"
MODELS_DIR = PROJECT_ROOT / "models"
