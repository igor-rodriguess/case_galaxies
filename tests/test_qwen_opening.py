from __future__ import annotations

from scripts.qwen_opening_pilot import DEFAULT_MODEL, PROMPT_VERSION


def test_qwen_pilot_uses_smallest_official_checkpoint() -> None:
    assert DEFAULT_MODEL == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_qwen_pilot_records_its_protocol_version() -> None:
    assert PROMPT_VERSION == "galaxies-opening-qwen-compact-v0.1.0"
