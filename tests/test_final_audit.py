from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxies.final_audit import (
    FinalScope,
    column_role,
    duplicate_titles,
    duration_format,
    minimum_age_rule,
    near_duplicate_pairs,
    selection_check,
    snapshot_growth,
    unusable_columns,
    vlm_protocol_ok,
)


def test_column_role_blocks_post_publication_and_time_index() -> None:
    assert column_role("like_count_snapshot") == "target_only"
    assert column_role("target_view_count") == "target_only"
    assert column_role("prior_views_median_observed") == "post_publication_history"
    assert column_role("channel_subscriber_count_snapshot") == "snapshot_contaminated"
    assert column_role("prior_video_count") == "time_index"
    assert column_role("vlm_model_version") == "extraction_protocol"
    assert column_role("days_since_previous_upload") == "candidate_feature"
    assert column_role("opening_type") == "candidate_feature"


def test_duration_format_buckets() -> None:
    assert duration_format(45) == "ate_60s"
    assert duration_format(60) == "ate_60s"
    assert duration_format(61) == "61_180s"
    assert duration_format(180) == "61_180s"
    assert duration_format(181) == "acima_180s"
    assert duration_format(None) == "desconhecido"


def test_vlm_protocol_ok_requires_model_prompt_and_window() -> None:
    scope = FinalScope()
    frame = pd.DataFrame(
        {
            "vlm_available": [True, True, True, False],
            "vlm_prompt_version": [scope.vlm_prompt_version] * 3 + [None],
            "vlm_model_requested": [scope.vlm_model, "gemini-3.8-flash", scope.vlm_model, None],
            "vlm_interval_start_seconds": [0.0, 0.0, 0.0, None],
            "vlm_interval_end_seconds": [3.0, 3.0, 15.0, None],
            "vlm_requested_fps": [4.0, 4.0, 4.0, None],
        }
    )
    assert vlm_protocol_ok(frame, scope).tolist() == [True, False, False, False]


def _snapshots(age_days: list[float], growth: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_1 = pd.Timestamp("2026-10-16T00:00:00Z")
    observed_2 = observed_1 + pd.Timedelta(days=4)
    views_1 = np.full(len(age_days), 1_000.0)
    first = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(len(age_days))],
            "published_at": [(observed_1 - pd.Timedelta(days=age)).isoformat() for age in age_days],
            "observed_at": observed_1.isoformat(),
            "view_count": views_1,
        }
    )
    second = pd.DataFrame(
        {
            "video_id": first["video_id"],
            "observed_at": observed_2.isoformat(),
            "view_count": np.expm1(np.log1p(views_1) + np.asarray(growth) * 4),
        }
    )
    return first, second


def test_snapshot_growth_measures_daily_log_growth() -> None:
    first, second = _snapshots([10.0, 40.0], [0.05, 0.0])
    growth = snapshot_growth(first, second)
    assert growth["age_days_1"].tolist() == pytest.approx([10.0, 40.0])
    assert growth["days_between"].tolist() == pytest.approx([4.0, 4.0])
    assert growth["daily_log_growth"].tolist() == pytest.approx([0.05, 0.0])


def test_snapshot_growth_rejects_second_reading_before_first() -> None:
    first, second = _snapshots([10.0], [0.0])
    second["observed_at"] = "2026-10-15T00:00:00Z"
    with pytest.raises(ValueError, match="posterior"):
        snapshot_growth(first, second)


def test_minimum_age_rule_picks_first_candidate_under_threshold() -> None:
    first, second = _snapshots(
        [8, 9, 10, 15, 16, 17, 31, 32], [0.05, 0.04, 0.06, 0.008, 0.009, 0.01, 0.0, 0.0]
    )
    chosen, rows = minimum_age_rule(snapshot_growth(first, second))
    assert chosen == 14
    assert [row["passes"] for row in rows] == [False, True, True]


def test_minimum_age_rule_falls_back_when_nothing_passes() -> None:
    first, second = _snapshots([8, 15, 31], [0.05, 0.05, 0.05])
    chosen, _ = minimum_age_rule(snapshot_growth(first, second), fallback_days=30)
    assert chosen == 30


def test_unusable_columns_flags_missing_and_constant() -> None:
    quality = pd.DataFrame(
        {
            "sector": ["moda"] * 4,
            "column": ["a", "b", "c", "d"],
            "missing_share": [1.0, 0.0, 0.2, 0.6],
            "distinct_values": [0, 1, 5, 1],
        }
    )
    flagged = unusable_columns(quality).set_index("column")
    assert flagged["reason"].to_dict() == {
        "a": "100% ausente",
        "b": "constante",
        "d": "constante nos observados",
    }
    assert flagged["action"].to_dict() == {"a": "remover", "b": "remover", "d": "recodificar"}


def test_duplicate_titles_only_within_channel() -> None:
    frame = pd.DataFrame(
        {
            "sector": ["moda"] * 3,
            "channel_id": ["a", "a", "b"],
            "video_id": ["1", "2", "3"],
            "title": ["Look do Dia!", "look do dia", "Look do dia"],
            "published_at": ["2026-01-01"] * 3,
        }
    )
    assert duplicate_titles(frame)["video_id"].tolist() == ["1", "2"]


def test_near_duplicate_pairs_uses_cosine() -> None:
    embeddings = pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.99, 0.01]), np.array([0.0, 1.0])],
        }
    )
    pairs = near_duplicate_pairs(embeddings, ["a", "b", "c"])
    assert pairs[["video_id_a", "video_id_b"]].values.tolist() == [["a", "b"]]


def test_selection_check_reports_within_channel_percentiles() -> None:
    frame = pd.DataFrame({"sector": ["moda"] * 4, "channel_id": ["a"] * 4})
    score = pd.Series([1.0, 2.0, 3.0, 4.0])
    selected = pd.Series([False, False, True, True])
    table = selection_check(frame, selected, score).set_index("group")
    assert table.loc["analisados", "mean_percentile"] == pytest.approx(0.875)
    assert table.loc["nao_analisados", "mean_percentile"] == pytest.approx(0.375)
