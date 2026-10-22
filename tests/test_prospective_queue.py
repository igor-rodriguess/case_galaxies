from __future__ import annotations

import pandas as pd

from galaxies.prospective import PROSPECTIVE_START, is_tracked_short, window_status

PUBLISHED = pd.Timestamp("2026-10-22T12:00:00Z")


def test_day_30_is_due_only_inside_tolerance() -> None:
    inside = window_status(PUBLISHED, PUBLISHED + pd.Timedelta(days=30, hours=17), set())
    assert 30 in inside["due"]
    early = window_status(PUBLISHED, PUBLISHED + pd.Timedelta(days=29, hours=5), set())
    assert 30 in early["pending"]


def test_window_after_tolerance_is_missed_not_reused() -> None:
    late = window_status(PUBLISHED, PUBLISHED + pd.Timedelta(days=31, hours=1), set())
    assert 30 in late["missed"]
    assert 30 not in late["due"]


def test_observed_ages_are_never_due_again() -> None:
    status = window_status(PUBLISHED, PUBLISHED + pd.Timedelta(days=3), {3})
    assert 3 not in status["due"]
    assert 1 in status["missed"]


def test_only_reserved_shorts_are_tracked() -> None:
    assert is_tracked_short(PROSPECTIVE_START, 45.0)
    assert is_tracked_short(PROSPECTIVE_START, 180.0)
    assert not is_tracked_short(PROSPECTIVE_START, 181.0)
    assert not is_tracked_short(PROSPECTIVE_START, None)
    assert not is_tracked_short(PROSPECTIVE_START - pd.Timedelta(seconds=1), 30.0)
