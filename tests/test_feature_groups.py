from galaxies.config import load_sector_config
from galaxies.modeling.feature_groups import feature_sets


def test_feature_groups_cover_config_without_overlap() -> None:
    config = load_sector_config("beleza")
    groups = feature_sets(config)

    assert "duration_seconds" in groups["metadata"]["numeric"]
    assert "prior_views_mean_observed" in groups["channel_history"]["numeric"]
    assert "motion_intensity" in groups["audiovisual"]["categorical"]
    assert "beauty_swatch_present" in groups["audiovisual"]["categorical"]

    expected = set(config["features"]["numeric"] + config["features"]["categorical"])
    combined = set(
        groups["combined"]["numeric"] + groups["combined"]["categorical"]
    )
    base_parts = [
        set(groups[name]["numeric"] + groups[name]["categorical"])
        for name in ("metadata", "channel_history", "audiovisual")
    ]

    assert combined == expected
    assert not (base_parts[0] & base_parts[1])
    assert not (base_parts[0] & base_parts[2])
    assert not (base_parts[1] & base_parts[2])
