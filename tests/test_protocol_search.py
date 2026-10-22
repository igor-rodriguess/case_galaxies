from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.agent.protocol import AgentDataset
from galaxies.agent.protocol_search import (
    PRIMARY_VLM_COHORTS,
    harmonize_primary_protocols,
    isolate_protocol,
    preregistered_protocol_specs,
)


def test_isolate_protocol_never_mixes_provider_window_or_prompt() -> None:
    frame = pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "vlm_available": [True, True, True],
            "vlm_prompt_version": [
                "galaxies-opening-v0.1.0",
                "galaxies-opening-v0.1.0",
                "galaxies-opening-v0.2.0-closed-taxonomy",
            ],
            "vlm_model_version": ["gemini-3.5-flash-lite", "other", "pegasus1.6"],
            "vlm_interval_end_seconds": [3.0, 3.0, 15.0],
            "in_population": [True, True, True],
        }
    )
    dataset = AgentDataset(frame, Path("features.jsonl"), "abc", (), [])
    isolated = isolate_protocol(dataset, PRIMARY_VLM_COHORTS[0])
    assert isolated.frame.loc[isolated.frame["vlm_available"], "video_id"].tolist() == ["a"]


def test_preregistered_specs_have_single_reference_and_no_postpublication_features() -> None:
    specs = preregistered_protocol_specs()
    assert sum(name == "reference_metadata_logistic" for name, _ in specs) == 1
    assert {spec.model for _, spec in specs} >= {
        "logistic",
        "hist_gb",
        "extra_trees",
        "catboost",
        "catboost_ranker",
        "mlp",
    }
    for _, spec in specs:
        numeric, categorical = spec.columns()
        assert "target_view_count" not in numeric + categorical
        assert np.isfinite(spec.train_min_age_days)


def test_harmonize_primary_protocols_preserves_domain_without_using_it_as_feature() -> None:
    frame = pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "vlm_available": [True, True, True],
            "vlm_prompt_version": [
                "galaxies-opening-v0.1.0",
                "galaxies-opening-v0.2.0-closed-taxonomy",
                "other",
            ],
            "vlm_model_version": ["gemini-3.5-flash-lite", "pegasus1.6", "other"],
            "vlm_interval_end_seconds": [3.0, 15.0, 3.0],
            "in_population": [True, True, True],
        }
    )
    dataset = AgentDataset(frame, Path("features.jsonl"), "abc", (), [])
    harmonized = harmonize_primary_protocols(dataset)
    assert harmonized.frame["vlm_available"].tolist() == [True, True, False]
    assert harmonized.frame["vlm_protocol_cohort"].notna().tolist() == [True, True, False]
    for _, spec in preregistered_protocol_specs():
        numeric, categorical = spec.columns()
        assert "vlm_protocol_cohort" not in numeric + categorical
