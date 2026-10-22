import numpy as np
import pandas as pd
import pytest

from galaxies.viral_dataset import (
    audit_boosted,
    clean_shorts,
    normalized_title,
    outlier_labels,
    shorts_limit_seconds,
)


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {
        "channel_id": "c1",
        "sector": "moda",
        "collected_at": "2026-10-21T12:00:00Z",
        "privacy_status": "public",
        "title": "titulo",
        "view_count": "100",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_shorts_limit_follows_the_platform_era() -> None:
    published = pd.Series(pd.to_datetime(["2021-07-11", "2021-07-12", "2024-10-14", "2024-10-15"], utc=True))
    limit = shorts_limit_seconds(published)
    assert np.isnan(limit[0])
    assert limit.tolist()[1:] == [60.0, 60.0, 180.0]


def test_clean_drops_non_shorts_prospective_and_reuploads() -> None:
    raw = _raw(
        [
            {"video_id": "antigo", "published_at": "2020-01-01T00:00:00Z", "duration_seconds": 30},
            {"video_id": "longo_2023", "published_at": "2023-05-01T00:00:00Z", "duration_seconds": 90, "title": "a"},
            {"video_id": "ok_2023", "published_at": "2023-05-02T00:00:00Z", "duration_seconds": 45, "title": "b"},
            {"video_id": "ok_2025", "published_at": "2025-05-01T00:00:00Z", "duration_seconds": 150, "title": "c"},
            {"video_id": "reupload", "published_at": "2025-06-01T00:00:00Z", "duration_seconds": 150, "title": "C!!"},
            {"video_id": "futuro", "published_at": "2026-10-21T01:00:00Z", "duration_seconds": 30, "title": "d"},
        ]
    )
    clean, funnel = clean_shorts(raw)
    assert sorted(clean["video_id"]) == ["ok_2023", "ok_2025"]
    assert funnel["restantes"].iloc[-1] == 2
    assert funnel["removidos"].sum() == len(raw) - 2


def test_normalized_title_ignores_case_accents_and_emoji() -> None:
    assert normalized_title("Carro NOVO é bom?! 🚗") == normalized_title("carro novo e bom")


def _clean_channel(views: list[float], start: str = "2024-11-01") -> pd.DataFrame:
    published = pd.date_range(start, periods=len(views), freq="D", tz="UTC")
    collected = pd.Timestamp("2026-10-01", tz="UTC")
    return pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(len(views))],
            "channel_id": "c1",
            "sector": "moda",
            "published": published,
            "published_at": published.astype(str),
            "views": views,
            "age_days": (collected - published).days.astype(float),
        }
    )


def test_baseline_uses_only_earlier_shorts_of_the_channel() -> None:
    views = [100.0] * 12 + [1000.0, 50.0]
    labels = outlier_labels(_clean_channel(views), history=30, min_history=10)
    spike = labels.set_index("video_id").loc["v12"]
    after = labels.set_index("video_id").loc["v13"]
    assert spike["baseline_median_views"] == 100.0  # o próprio vídeo não entra na mediana
    assert spike["viral"] == 1
    assert after["baseline_median_views"] == 100.0  # mediana resiste ao pico anterior
    assert after["viral"] == 0


def test_young_videos_and_short_history_stay_unlabelled() -> None:
    views = [100.0] * 15
    frame = _clean_channel(views)
    frame.loc[14, "age_days"] = 10.0
    labels = outlier_labels(frame, min_age_days=60, history=30, min_history=10).set_index("video_id")
    assert labels.loc["v3", "label_status"] == "histórico curto"
    assert labels.loc["v14", "label_status"] == "jovem demais"
    assert pd.isna(labels.loc["v14", "viral"])
    assert labels.loc["v12", "label_status"] == "rotulado"


@pytest.mark.parametrize(("ratio", "expected"), [(2.99, 0), (3.0, 1), (10.0, 1)])
def test_viral_threshold_is_three_times_the_channel_median(ratio: float, expected: int) -> None:
    labels = outlier_labels(_clean_channel([100.0] * 10 + [100.0 * ratio]), min_history=10)
    assert labels.iloc[-1]["viral"] == expected


def _labelled_channel(likes: list[float | None], views: float = 1000.0, channel: str = "c1") -> pd.DataFrame:
    frame = _clean_channel([views] * len(likes))
    frame["channel_id"] = channel
    frame["likes"] = likes
    return outlier_labels(frame, min_history=3)


def test_boosted_short_leaves_the_label_but_normal_low_engagement_stays() -> None:
    # Taxa normal de 5%; um vídeo com 0,1% (impulsionado) e um com 2% (viral orgânico típico).
    likes = [50.0] * 12 + [1.0, 20.0]
    audited = audit_boosted(_labelled_channel(likes)).set_index("video_id")
    assert audited.loc["v12", "label_status"] == "provável impulsionado"
    assert pd.isna(audited.loc["v12", "viral"])
    assert audited.loc["v13", "label_status"] == "rotulado"


def test_boosted_threshold_uses_only_earlier_shorts() -> None:
    # Um pico de likes depois do vídeo não muda a régua dele.
    likes = [50.0] * 10 + [4.0] + [5000.0] * 3
    audited = audit_boosted(_labelled_channel(likes)).set_index("video_id")
    assert audited.loc["v10", "prior_median_like_rate"] == pytest.approx(0.05)
    assert audited.loc["v10", "label_status"] == "provável impulsionado"


def test_channel_with_hidden_likes_is_not_labelled() -> None:
    audited = audit_boosted(_labelled_channel([None] * 14))
    assert set(audited.loc[audited["prior_shorts"] >= 3, "label_status"]) == {"likes ocultos"}
    assert audited["viral"].isna().all()
