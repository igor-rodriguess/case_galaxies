from __future__ import annotations

import pandas as pd
import pytest

from galaxies.agent.embedding_evaluation import text_statistics


def test_text_statistics_uses_only_the_supplied_prepublication_text() -> None:
    records = pd.DataFrame(
        [
            {
                "video_id": "video-1",
                "title": "TOP 10! Será que funciona?",
                "description": "Descrição para teste.",
            }
        ]
    )

    result = text_statistics(records).iloc[0]

    assert result["video_id"] == "video-1"
    assert result["title_word_count"] == 5
    assert result["title_question_count"] == 1
    assert result["title_exclamation_count"] == 1
    assert result["title_number_token_count"] == 1
    assert result["description_word_count"] == 3
    assert result["title_uppercase_ratio"] == pytest.approx(4 / 18)
