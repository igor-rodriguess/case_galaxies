from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from galaxies.agent.evaluation import (
    auc_intra_channel,
    bootstrap_top_fraction,
    channel_top_counts,
    fit_predict,
    paired_comparison,
    run_experiment,
    temporal_split,
    top_fraction_metrics,
    training_rows_and_labels,
)
from galaxies.agent.hypotheses import (
    GeminiGenerator,
    OfflineGenerator,
    neighbours,
    parse_specs,
)
from galaxies.agent.ledger import AgentPaths, Ledger
from galaxies.agent.protocol import (
    MIN_EVAL_AGE_DAYS,
    build_dataset,
    is_forbidden,
    normalize_vlm_value,
)
from galaxies.agent.spec import ExperimentSpec


def synthetic_raw(seed: int = 0, channels_per_sector: int = 5, videos: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for sector in ("automotivo", "moda", "beleza"):
        for channel in range(channels_per_sector):
            channel_id = f"{sector}-{channel}"
            level = rng.normal(9, 1)
            for index in range(videos):
                age = float(rng.uniform(5, 400))
                duration = float(rng.choice([45, 120, 600, 900]))
                quality = rng.normal()
                views = np.expm1(
                    level + 0.4 * np.log1p(age) + 0.8 * quality + 0.3 * (duration < 180)
                )
                rows.append(
                    {
                        "video_id": f"{channel_id}-{index}",
                        "channel_id": channel_id,
                        "sector": sector,
                        "published_at": (start + pd.Timedelta(days=index * 3)).isoformat(),
                        "target_age_days": age,
                        "target_view_count": int(views),
                        "duration_seconds": duration,
                        "title_character_count": int(40 + 20 * quality + rng.normal(0, 5)),
                        "description_character_count": int(rng.integers(0, 500)),
                        "tag_count": int(rng.integers(0, 15)),
                        "publish_hour_utc": int(rng.integers(0, 24)),
                        "publish_weekday_utc": int(rng.integers(0, 7)),
                        "category_id": "22",
                        "days_since_previous_upload": float(rng.uniform(0, 10)),
                        "prior_upload_interval_median_days": float(rng.uniform(0, 5)),
                        "prior_video_count": index,
                        "vlm_available": bool(index % 2 == 0),
                        "framing": rng.choice(["medium shot", "Close-up", "wide shot"]),
                        "camera_movement": rng.choice(["static", "panning", "handheld"]),
                        "like_count_snapshot": int(views * 0.05),
                        "prior_views_mean_observed": float(views),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def dataset():
    return build_dataset(synthetic_raw(), Path("synthetic.jsonl"), "synthetic000001")


def test_label_marks_top_decile_per_channel_only_for_mature_videos(dataset) -> None:
    frame = dataset.frame
    mature = frame[frame["in_population"]]
    assert (mature["target_age_days"] >= MIN_EVAL_AGE_DAYS).all()
    assert set(frame.loc[~frame["in_population"], "viral"]) <= {-1}
    share = mature.groupby("channel_id")["viral"].mean()
    assert share.between(0.05, 0.2).all()


def test_vault_has_one_channel_per_sector_and_never_enters_evaluation(dataset) -> None:
    vault_sectors = dataset.frame.loc[dataset.frame["in_vault"], "sector"].unique()
    assert sorted(vault_sectors) == ["automotivo", "beleza", "moda"]
    assert len(dataset.vault_channels) == 3
    for track in ("full", "vlm"):
        assert not dataset.evaluable(track)["channel_id"].isin(dataset.vault_channels).any()


def test_post_publication_columns_are_forbidden() -> None:
    for column in (
        "prior_video_count",
        "like_count_snapshot",
        "prior_views_mean_observed",
        "target_view_count",
        "channel_id",
    ):
        assert is_forbidden(column)
    assert not is_forbidden("days_since_previous_upload")
    spec = ExperimentSpec(
        hypothesis="todas as features",
        track="vlm",
        feature_groups=["metadata", "cadence", "audiovisual"],
    )
    numeric, categorical = spec.columns()
    assert not any(is_forbidden(column) for column in numeric + categorical)


def test_audiovisual_requires_vlm_track() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec(
            hypothesis="vazamento de trilha", track="full", feature_groups=["audiovisual"]
        )


def test_spec_key_ignores_text_and_normalizes_groups() -> None:
    first = ExperimentSpec(hypothesis="A primeira versão", feature_groups=["metadata", "cadence"])
    second = ExperimentSpec(
        hypothesis="Outra redação", rationale="x", feature_groups=["cadence", "metadata"]
    )
    assert first.key() == second.key()
    assert (
        ExperimentSpec(hypothesis="taxonomia sem efeito", vlm_taxonomy="normalized").vlm_taxonomy
        == "raw"
    )


def test_clean_contrast_uses_only_training_channel_quantiles() -> None:
    train = pd.DataFrame(
        {
            "channel_id": ["a"] * 10,
            "adjusted_performance": list(range(10)),
            "viral": [0] * 9 + [1],
        }
    )
    spec = ExperimentSpec(hypothesis="contraste limpo", train_label="clean20vs50")
    selected, labels = training_rows_and_labels(spec, train)

    assert selected["adjusted_performance"].tolist() == [0, 1, 2, 3, 4, 8, 9]
    assert labels.tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_temporal_split_holds_out_the_latest_quarter_per_channel() -> None:
    rows = pd.DataFrame(
        {
            "channel_id": ["a"] * 4 + ["b"] * 4,
            "published_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"] * 2, utc=True
            ),
        }
    )
    train, test = temporal_split(rows)

    assert test.groupby("channel_id").size().to_dict() == {"a": 1, "b": 1}
    assert train.groupby("channel_id").size().to_dict() == {"a": 3, "b": 3}
    assert set(test["published_at"].dt.day) == {4}


def test_known_channel_mode_adds_only_past_history_features(dataset) -> None:
    rows = dataset.evaluable("full")
    train, test = temporal_split(rows)
    spec = ExperimentSpec(
        hypothesis="Canal conhecido usa histórico defasado e seguro.",
        feature_groups=["metadata"],
        evaluation_mode="known_channel",
    )
    scores = fit_predict(spec, train, test, seed=0, known_channel=True)
    assert len(scores) == len(test)
    assert np.isfinite(scores).all()


def test_taxonomy_maps_free_text() -> None:
    assert normalize_vlm_value("framing", "Medium_Shot") == "medium"
    assert normalize_vlm_value("framing", "medium close-up") == "medium_close_up"
    assert normalize_vlm_value("camera_movement", "estático") == "static"
    assert normalize_vlm_value("camera_movement", "algo novo") == "other"


def test_auc_intra_channel_counts_only_pairs_inside_each_channel() -> None:
    frame = pd.DataFrame(
        {
            "channel_id": ["a", "a", "a", "b", "b"],
            "viral": [1, 0, 0, 1, 0],
            # Canal b tem notas muito maiores; isso não deve importar.
            "score": [0.9, 0.1, 0.95, 10.0, 5.0],
        }
    )
    # a: viral 0.9 vence 0.1 e perde para 0.95 (1 de 2); b: 1 de 1 → 2/3
    assert auc_intra_channel(frame, "score") == pytest.approx(2 / 3)
    metrics = top_fraction_metrics(frame, "score")
    assert metrics["recall_top10"] == pytest.approx(0.5)


def test_top_fraction_bootstrap_returns_bounded_intervals() -> None:
    frame = pd.DataFrame(
        {
            "channel_id": ["a", "a", "a", "b", "b", "b"],
            "viral": [1, 0, 0, 0, 1, 0],
            "score": [0.9, 0.2, 0.1, 0.3, 0.8, 0.1],
        }
    )
    intervals = bootstrap_top_fraction(channel_top_counts(frame, "score"), 100, 7)
    assert 0 <= intervals["recall_ci_low"] <= intervals["recall_ci_high"] <= 1
    assert 0 <= intervals["precision_ci_low"] <= intervals["precision_ci_high"] <= 1
    assert intervals["lift_ci_low"] >= 0


def test_paired_comparison_detects_a_clear_winner() -> None:
    index = pd.Index([f"c{i}" for i in range(20)], name="channel_id")
    better = pd.DataFrame({"pairs": 10, "correct": 8.0}, index=index)
    worse = pd.DataFrame({"pairs": 10, "correct": 5.0}, index=index)
    result = paired_comparison(better, worse, 200, 0)
    assert result["diff"] == pytest.approx(0.3)
    assert result["p_better"] == 1.0


def test_per_sector_training_never_sees_other_sectors(dataset, monkeypatch) -> None:
    seen: list[set[str]] = []
    import galaxies.agent.evaluation as evaluation

    def spy(spec, train, test, seed):
        seen.append(set(train["sector"]))
        return np.zeros(len(test))

    monkeypatch.setattr(evaluation, "_fit_predict", spy)
    rows = dataset.evaluable("full")
    spec = ExperimentSpec(hypothesis="por setor", pooling="per_sector")
    fit_predict(spec, rows, rows[rows["sector"] == "moda"], 0)
    assert seen == [{"moda"}]


def test_run_experiment_is_loco_and_finite(dataset, monkeypatch) -> None:
    import galaxies.agent.evaluation as evaluation

    original = evaluation.fit_predict

    def guarded(spec, train, test, seed):
        if test["channel_id"].nunique() == 1:  # chamadas LOCO
            assert not set(train["channel_id"]) & set(test["channel_id"])
        return original(spec, train, test, seed)

    monkeypatch.setattr(evaluation, "fit_predict", guarded)
    spec = ExperimentSpec(
        hypothesis="teste", model="logistic", feature_groups=["metadata", "cadence"]
    )
    result = run_experiment(spec, dataset, bootstrap_iterations=50)
    assert result.predictions["score_loco"].notna().all()
    assert 0 <= result.loco["auc_intra"] <= 1
    assert result.loco["channels"] == dataset.evaluable("full")["channel_id"].nunique()


def test_offline_generator_skips_tried_and_then_explores_neighbours() -> None:
    generator = OfflineGenerator()
    first = generator.propose({"tried_keys": []}, 3)
    assert len(first.specs) == 3
    tried = [spec.key() for spec in first.specs]
    second = generator.propose({"tried_keys": tried}, 3)
    assert not {spec.key() for spec in second.specs} & set(tried)
    champion = first.specs[0]
    assert all(spec.key() != champion.key() for spec in neighbours(champion))


def test_gemini_proposals_are_validated_and_fall_back_on_failure() -> None:
    payload = {
        "proposals": [
            {
                "hypothesis": "Árvores rasas por formato",
                "rationale": "r",
                "expected_effect": "e",
                "track": "full",
                "feature_groups": ["metadata"],
                "pooling": "pooled",
                "format_strategy": "segmented",
                "model": "hist_gb",
                "complexity": "low",
                "train_min_age_days": 60,
                "vlm_taxonomy": "raw",
            },
            {"hypothesis": "Inválida", "track": "full", "feature_groups": ["audiovisual"]},
        ],
        "backlog": [{"idea": "Thumbnails", "why": "CTR", "needs": "coletar imagens"}],
    }

    def transport(endpoint, body, key):
        assert "responseSchema" in body["generationConfig"]
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}

    proposal = GeminiGenerator("chave", transport=transport).propose({"tried_keys": []}, 4)
    assert proposal.source == "gemini"
    assert len(proposal.specs) == 1 and proposal.specs[0].train_min_age_days == 60
    assert len(proposal.invalid) == 1
    assert proposal.backlog[0]["idea"] == "Thumbnails"

    def broken(endpoint, body, key):
        raise ValueError("falha com chave no texto")

    fallback = GeminiGenerator("chave", transport=broken, max_attempts=1).propose(
        {"tried_keys": []}, 2
    )
    assert fallback.source == "offline" and len(fallback.specs) == 2
    assert "chave" not in fallback.note.replace("<redacted>", "")


def test_parse_specs_reports_reasons() -> None:
    specs, invalid = parse_specs(
        [{"hypothesis": "ok ok"}, {"hypothesis": "x", "model": "transformer"}]
    )
    assert len(specs) == 1 and len(invalid) == 1


@pytest.mark.parametrize("model", ["catboost", "catboost_ranker", "mlp"])
def test_new_model_families_fit_without_post_publication_features(dataset, model) -> None:
    rows = dataset.evaluable("vlm")
    spec = ExperimentSpec(
        hypothesis=f"avaliar {model}",
        track="vlm",
        feature_groups=["metadata", "audiovisual"],
        model=model,
        pooling="pooled",
    )
    train = rows.iloc[: max(80, len(rows) // 2)]
    test = rows.iloc[-20:]
    scores = fit_predict(spec, train, test, seed=2026)
    assert len(scores) == len(test)
    assert np.isfinite(scores).all()


def test_full_graph_run_offline_writes_ledger_and_report(dataset, tmp_path) -> None:
    pytest.importorskip("langgraph")
    from galaxies.agent.graph import AgentContext, run_agent

    paths = AgentPaths(tmp_path / "agent", "teste")
    paths.ensure()
    ledger = Ledger(paths, dataset.fingerprint)
    logs: list[str] = []
    context = AgentContext(
        dataset=dataset,
        ledger=ledger,
        paths=paths,
        generator=OfflineGenerator(),
        bootstrap_iterations=50,
        open_vault=True,
        log=logs.append,
    )
    state = run_agent(context, rounds=2, per_round=2)
    report = Path(state["report_path"]).read_text(encoding="utf-8")
    assert "Campeões" in report and "Cofre" in report
    records = [json.loads(line) for line in paths.ledger.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert records[0]["status"] == "referência"
    summary = json.loads((paths.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiments_in_run"] == 4

    # Uma segunda execução na mesma base não repete configurações.
    paths2 = AgentPaths(tmp_path / "agent", "teste-2")
    paths2.ensure()
    ledger2 = Ledger(paths2, dataset.fingerprint)
    assert len(ledger2.tried_keys()) == 4
    context2 = AgentContext(
        dataset=dataset,
        ledger=ledger2,
        paths=paths2,
        generator=OfflineGenerator(),
        bootstrap_iterations=50,
        log=logs.append,
    )
    run_agent(context2, rounds=1, per_round=2)
    keys = [record["key"] for record in ledger2.records]
    assert len(keys) == len(set(keys)) == 6


def test_label_does_not_carry_age_inside_channels(dataset) -> None:
    from galaxies.agent.protocol import within_channel_age_correlation

    mature = dataset.frame[dataset.frame["in_population"]]
    rho = within_channel_age_correlation(
        mature.assign(label=mature["viral"].astype(float)), "label"
    )
    assert abs(rho) < 0.15


def test_score_that_tracks_age_is_flagged(dataset) -> None:
    from galaxies.agent.evaluation import summarize_scores

    rows = dataset.evaluable("full")[["channel_id", "viral", "target_age_days"]].copy()
    rows["score"] = -rows["target_age_days"]
    summary = summarize_scores(rows, "score")
    assert summary["age_rho"] == pytest.approx(-1.0)
