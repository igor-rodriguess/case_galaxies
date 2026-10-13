from __future__ import annotations

from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from scripts.gemini_opening_pilot import (
    ProviderUnavailableError,
    QuotaExhaustedError,
    request_gemini,
)


def call_request() -> None:
    request_gemini(
        api_key="test-key",
        model="test-model",
        video_url="https://youtube.test/watch?v=video",
        start_seconds=0,
        end_seconds=3,
        fps=4,
        prompt="test",
        max_attempts=1,
    )


def http_error(status: int, body: str) -> HTTPError:
    return HTTPError(
        url="https://generativelanguage.googleapis.com",
        code=status,
        msg="provider error",
        hdrs=None,
        fp=BytesIO(body.encode("utf-8")),
    )


def test_request_classifies_provider_unavailable() -> None:
    error = http_error(503, '{"error":{"status":"UNAVAILABLE"}}')
    with (
        patch("scripts.gemini_opening_pilot.urlopen", side_effect=error),
        pytest.raises(ProviderUnavailableError) as captured,
    ):
        call_request()

    assert captured.value.status_code == 503


def test_request_classifies_quota_exhaustion() -> None:
    error = http_error(
        429,
        '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded"}}',
    )
    with (
        patch("scripts.gemini_opening_pilot.urlopen", side_effect=error),
        pytest.raises(QuotaExhaustedError),
    ):
        call_request()
