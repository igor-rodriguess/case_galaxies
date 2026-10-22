from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from galaxies.agent.protocol import is_forbidden
from scripts.video_pilot import (
    BOT_CHECK_LIMIT,
    CLIP_PROBES,
    SCALAR_VIDEO,
    _cuts,
    _frames,
    block_streak,
    expansion_decision,
    pick_sample,
)


def _train(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for channel, size in counts.items():
        for index in range(size):
            rows.append(
                {
                    "video_id": f"{channel}-{index}",
                    "source_channel_id": channel,
                    "sector": "moda",
                    "published_at": f"2025-01-{index % 28 + 1:02d}T00:00:00Z",
                    "duration_seconds": 30.0,
                }
            )
    return pd.DataFrame(rows)


def test_sample_caps_each_channel_and_is_reproducible() -> None:
    train = _train({"a": 40, "b": 10, "c": 3})
    picked = pick_sample(train, per_channel=25, seed=42)
    assert picked.groupby("source_channel_id").size().to_dict() == {"a": 25, "b": 10, "c": 3}
    assert picked["video_id"].tolist() == pick_sample(train, per_channel=25, seed=42)["video_id"].tolist()


def test_smaller_sample_is_subset_of_larger_with_same_seed() -> None:
    train = _train({"a": 200, "b": 150})
    small = set(pick_sample(train, per_channel=25, seed=7)["video_id"])
    large = set(pick_sample(train, per_channel=100, seed=7)["video_id"])
    assert small <= large


def test_bot_check_streak_resets_on_any_other_result() -> None:
    bot = "ERROR: [youtube] x: Sign in to confirm you're not a bot."
    streak = 0
    for status in (bot, bot, "ok", bot):
        streak = block_streak(streak, status)
    assert streak == 1
    for _ in range(BOT_CHECK_LIMIT - 1):
        streak = block_streak(streak, bot)
    assert streak == BOT_CHECK_LIMIT
    assert block_streak(streak, "ERROR: Video unavailable") == 0


def _table(delta: float, ci_low: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"config": "hgb_pct_base", "auc_macro_channel": 0.55, "delta_vs_hgb_pct_base": 0.0, "delta_ci_low": 0.0},
            {"config": "hgb_pct_video", "auc_macro_channel": 0.55 + delta, "delta_vs_hgb_pct_base": delta, "delta_ci_low": ci_low},
            {"config": "hgb_pct_video_clip16", "auc_macro_channel": 0.54, "delta_vs_hgb_pct_base": -0.01, "delta_ci_low": -0.05},
            # A referência logística não conta como "config com vídeo".
            {"config": "logistic_top10_video", "auc_macro_channel": 0.90, "delta_vs_hgb_pct_base": 0.35, "delta_ci_low": 0.2},
        ]
    )


@pytest.mark.parametrize(
    ("delta", "ci_low", "expected"),
    [(0.05, 0.01, True), (0.05, -0.01, False), (0.015, 0.001, False), (0.02, 0.0001, True)],
)
def test_expansion_needs_gain_and_interval_above_zero(delta: float, ci_low: float, expected: bool) -> None:
    best, expand = expansion_decision(_table(delta, ci_low))
    assert best["config"] == "hgb_pct_video"
    assert expand is expected


def test_video_features_never_use_audience_metrics() -> None:
    assert not [column for column in SCALAR_VIDEO if is_forbidden(column)]
    assert not [column for column in SCALAR_VIDEO if any(word in column for word in ("view", "like", "comment"))]
    assert set(CLIP_PROBES) <= set(SCALAR_VIDEO)


def _synthetic_clip(path: Path, fps: int = 30) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 160))
    if not writer.isOpened():
        pytest.skip("codec mp4v indisponível")
    for index in range(4 * fps):
        value = 10 if index < 2 * fps else 240
        writer.write(np.full((160, 96, 3), value, dtype=np.uint8))
    writer.release()
    return path


def test_rhythm_features_find_the_single_cut(tmp_path: Path) -> None:
    pytest.importorskip("scenedetect")
    clip = _synthetic_clip(tmp_path / "corte.mp4")
    frames, rhythm = _frames(clip)
    cuts = _cuts(clip)
    assert len(frames) == 5
    assert cuts["cut_count"] == 1
    assert cuts["first_cut_seconds"] == pytest.approx(2.0, abs=0.1)
    assert 0.3 < rhythm["brightness_mean"] < 0.7
    assert rhythm["motion_first_3s"] > rhythm["motion_mean"] * 0.5
