"""Ranking de Shorts com sinais de conteúdo e desvio do padrão do canal.

Protocolo (fixado antes de olhar qualquer resultado):

* População, rótulo as-of, unidades e corte externo são os mesmos da Fase 2
  (``format_ranking``). Métrica principal: AUC média por canal real.
* Features só pré-publicação: metadados, cadência, estilo do título, embedding
  do título (e5), embedding da thumbnail (ResNet-18) e desvios em relação aos
  20 Shorts anteriores do mesmo canal. Os desvios usam apenas atributos que o
  criador conhece ao publicar (datas, durações, textos e imagens), nunca views,
  likes ou comentários de vídeos anteriores.
* PCA, imputação, codificação e modelo são ajustados só no treino de cada dobra.
* A lista ``CONFIGS`` é fechada. A escolha é uma configuração global: a maior
  AUC média por canal somando as três dobras internas de calendário. Empate fica
  com a mais simples (ordem da lista).
* ``--stage outer`` só roda se ``frozen_selection.json`` já existir e lê o
  corte externo uma única vez, com IC pareado por canal contra a referência.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.format_ranking import (  # noqa: E402
    bootstrap_channels,
    channel_level,
    format_as_of_labels,
    inner_calendar_folds,
    outer_split,
    ranking_metrics,
    shorts_population,
    unit_counts,
    with_format_units,
)
from galaxies.agent.protocol import (  # noqa: E402
    CADENCE_NUMERIC,
    METADATA_CATEGORICAL,
    METADATA_NUMERIC,
    load_dataset,
)
from galaxies.modeling.preprocessing import make_preprocessor  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

HISTORY = 20

STYLE_NUMERIC = [
    "title_word_count",
    "title_upper_ratio",
    "title_emoji_count",
    "title_question",
    "title_exclamation",
    "title_digit_count",
    "title_hashtag_count",
]
RELATIVE_NUMERIC = [
    "rel_duration_z",
    "rel_title_length_z",
    "rel_tag_count_diff",
    "rel_hour_distance",
    "rel_gap_ratio",
    "rel_word_count_z",
]
NOVELTY_NUMERIC = ["title_sim_mean", "title_sim_max", "thumb_sim_mean", "thumb_sim_max"]


@dataclass(frozen=True)
class Config:
    name: str
    model: str  # logistic | hgb_reg | cat_reg | cat_rank | ensemble
    target: str  # top10 | percentile
    groups: tuple[str, ...]
    title_pca: int = 0
    thumb_pca: int = 0
    members: tuple[str, ...] = ()


CONFIGS = (
    Config("reference_logistic_top10", "logistic", "top10", ("base",)),
    Config("hgb_pct_base", "hgb_reg", "percentile", ("base",)),
    Config("hgb_pct_relative_style", "hgb_reg", "percentile", ("base", "style", "relative")),
    Config("hgb_pct_relative_novelty", "hgb_reg", "percentile", ("base", "style", "relative", "novelty")),
    Config("hgb_pct_text16", "hgb_reg", "percentile", ("base", "style", "relative", "novelty"), title_pca=16),
    Config("hgb_pct_text16_thumb16", "hgb_reg", "percentile", ("base", "style", "relative", "novelty"), 16, 16),
    Config("cat_pct_text16_thumb16", "cat_reg", "percentile", ("base", "style", "relative", "novelty"), 16, 16),
    Config("cat_rank_text16_thumb16", "cat_rank", "percentile", ("base", "style", "relative", "novelty"), 16, 16),
    Config(
        "ensemble_rank_mean",
        "ensemble",
        "percentile",
        (),
        members=("hgb_pct_text16_thumb16", "cat_pct_text16_thumb16", "cat_rank_text16_thumb16"),
    ),
)
BY_NAME = {config.name: config for config in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inner", "outer"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--feature-map",
        type=Path,
        default=MODELING_RUNS_DIR / "feature-map-shorts-operational-180s-20261021" / "feature_map.jsonl",
    )
    parser.add_argument("--title-embeddings", type=Path, default=ROOT / "data" / "features" / "embeddings" / "title_e5.parquet")
    parser.add_argument("--thumb-embeddings", type=Path, default=ROOT / "data" / "features" / "embeddings" / "thumb_resnet18.parquet")
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --- Features ---------------------------------------------------------------


def _style(title: str) -> dict[str, float]:
    letters = [ch for ch in title if ch.isalpha()]
    return {
        "title_word_count": float(len(title.split())),
        "title_upper_ratio": float(sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0,
        "title_emoji_count": float(sum(unicodedata.category(ch) == "So" for ch in title)),
        "title_question": float("?" in title),
        "title_exclamation": float("!" in title),
        "title_digit_count": float(len(re.findall(r"\d+", title))),
        "title_hashtag_count": float(title.count("#")),
    }


def _rolling_relative(rows: pd.DataFrame) -> pd.DataFrame:
    """Desvio de cada Short em relação aos ``HISTORY`` anteriores do canal."""
    out = pd.DataFrame(index=rows.index)
    ordered = rows.sort_values("_published")
    grouped = ordered.groupby("source_channel_id", sort=False)

    def prior(column: str, stat: str) -> pd.Series:
        return grouped[column].transform(
            lambda series: getattr(series.shift(1).rolling(HISTORY, min_periods=3), stat)()
        )

    for column, name in (
        ("log_duration", "rel_duration_z"),
        ("title_character_count", "rel_title_length_z"),
        ("title_word_count", "rel_word_count_z"),
    ):
        mean, std = prior(column, "mean"), prior(column, "std")
        out.loc[ordered.index, name] = ((ordered[column] - mean) / std.where(std > 0)).to_numpy()
    out.loc[ordered.index, "rel_tag_count_diff"] = (ordered["tag_count"] - prior("tag_count", "mean")).to_numpy()
    hour_median = prior("publish_hour_utc", "median")
    distance = (ordered["publish_hour_utc"] - hour_median).abs()
    out.loc[ordered.index, "rel_hour_distance"] = np.minimum(distance, 24 - distance).to_numpy()
    out.loc[ordered.index, "rel_gap_ratio"] = (
        ordered["days_since_previous_upload"] / ordered["prior_upload_interval_median_days"].where(lambda s: s > 0)
    ).to_numpy()
    return out


def _novelty(rows: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Similaridade com os ``HISTORY`` títulos/thumbnails anteriores do canal."""
    out = pd.DataFrame(np.nan, index=rows.index, columns=[f"{prefix}_sim_mean", f"{prefix}_sim_max"])
    for _, group in rows.sort_values("_published").groupby("source_channel_id", sort=False):
        matrix = np.stack(group[column].to_numpy()).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        for position in range(3, len(group)):
            history = matrix[max(0, position - HISTORY) : position]
            similarity = history @ matrix[position]
            out.loc[group.index[position]] = (float(similarity.mean()), float(similarity.max()))
    return out


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    dataset = load_dataset(args.feature_map)
    rows = with_format_units(shorts_population(dataset.evaluable("full"), args.max_seconds))
    rows["_published"] = pd.to_datetime(rows["published_at"], utc=True)
    titles = pd.read_parquet(args.title_embeddings).drop_duplicates("video_id", keep="last")
    titles = titles.rename(columns={"embedding": "_title_embedding", "title": "_title"})
    thumbs = pd.read_parquet(args.thumb_embeddings, columns=["video_id", "embedding"]).drop_duplicates("video_id", keep="last")
    thumbs = thumbs.rename(columns={"embedding": "_thumb_embedding"})
    rows = rows.merge(titles, on="video_id", how="left", validate="one_to_one")
    rows = rows.merge(thumbs, on="video_id", how="left", validate="one_to_one")
    for column in ("_title_embedding", "_thumb_embedding"):
        missing = int(rows[column].isna().sum())
        if missing:
            raise ValueError(f"Cobertura incompleta em {column}: faltam {missing} de {len(rows)}.")
    rows.index = pd.RangeIndex(len(rows))
    style = pd.DataFrame([_style(str(title)) for title in rows["_title"]], index=rows.index)
    rows = pd.concat([rows, style], axis=1)
    rows = pd.concat(
        [
            rows,
            _rolling_relative(rows),
            _novelty(rows, "_title_embedding", "title"),
            _novelty(rows, "_thumb_embedding", "thumb"),
        ],
        axis=1,
    )
    return rows


# --- Modelos ----------------------------------------------------------------


def _columns(config: Config) -> tuple[list[str], list[str]]:
    numeric = list(METADATA_NUMERIC) + list(CADENCE_NUMERIC)
    if "style" in config.groups:
        numeric += STYLE_NUMERIC
    if "relative" in config.groups:
        numeric += RELATIVE_NUMERIC
    if "novelty" in config.groups:
        numeric += NOVELTY_NUMERIC
    return numeric, list(METADATA_CATEGORICAL) + ["sector"]


def _matrix(config: Config, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    numeric, categorical = _columns(config)
    preprocessor = make_preprocessor({"features": {"numeric": numeric, "categorical": categorical}})
    x_train = [preprocessor.fit_transform(train[numeric + categorical])]
    x_test = [preprocessor.transform(test[numeric + categorical])]
    for column, components in (("_title_embedding", config.title_pca), ("_thumb_embedding", config.thumb_pca)):
        if not components:
            continue
        pca = PCA(n_components=components, random_state=seed).fit(np.stack(train[column].to_numpy()))
        x_train.append(pca.transform(np.stack(train[column].to_numpy())))
        x_test.append(pca.transform(np.stack(test[column].to_numpy())))
    dense = [np.asarray(part.todense()) if hasattr(part, "todense") else np.asarray(part) for part in x_train]
    dense_test = [np.asarray(part.todense()) if hasattr(part, "todense") else np.asarray(part) for part in x_test]
    return np.hstack(dense), np.hstack(dense_test)


def _percentile_target(train: pd.DataFrame) -> np.ndarray:
    """Percentil do desempenho ajustado por idade dentro da unidade, só no treino."""
    return train.groupby("channel_id")["adjusted_performance"].rank(pct=True).to_numpy()


def score(config: Config, train: pd.DataFrame, test: pd.DataFrame, seed: int, cache: dict[str, np.ndarray]) -> np.ndarray:
    if config.name in cache:
        return cache[config.name]
    if config.model == "ensemble":
        parts = [pd.Series(score(BY_NAME[name], train, test, seed, cache)) for name in config.members]
        frame = pd.DataFrame({i: part for i, part in enumerate(parts)})
        frame["unit"] = test["channel_id"].to_numpy()
        ranks = frame.groupby("unit").rank(pct=True)
        result = ranks.mean(axis=1).to_numpy()
        cache[config.name] = result
        return result
    x_train, x_test = _matrix(config, train, test, seed)
    if config.model == "logistic":
        model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)
        model.fit(x_train, train["viral"].to_numpy())
        result = model.predict_proba(x_test)[:, 1]
    elif config.model == "hgb_reg":
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, l2_regularization=1.0, random_state=seed
        )
        model.fit(x_train, _percentile_target(train))
        result = model.predict(x_test)
    elif config.model == "cat_reg":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=5, random_seed=seed, verbose=False, thread_count=8)
        model.fit(x_train, _percentile_target(train))
        result = model.predict(x_test)
    elif config.model == "cat_rank":
        from catboost import CatBoost, Pool

        order = np.argsort(train["channel_id"].to_numpy(), kind="stable")
        groups = train["channel_id"].to_numpy()[order]
        pool = Pool(x_train[order], label=_percentile_target(train)[order], group_id=groups)
        model = CatBoost(
            {"loss_function": "YetiRank", "iterations": 500, "depth": 4, "learning_rate": 0.05, "random_seed": seed, "verbose": False, "thread_count": 8}
        )
        model.fit(pool)
        result = model.predict(x_test)
    else:
        raise ValueError(config.model)
    cache[config.name] = np.asarray(result, dtype=float)
    return cache[config.name]


def counts_for(scores: np.ndarray, test: pd.DataFrame) -> pd.DataFrame:
    scored = test[["channel_id", "source_channel_id", "sector", "viral"]].copy()
    scored["score"] = scores
    return channel_level(unit_counts(scored, "score"))


def age_rho(scores: np.ndarray, test: pd.DataFrame) -> float:
    """Spearman médio entre nota e idade dentro da unidade (atalho de idade)."""
    values = []
    frame = pd.DataFrame({"score": scores, "age": test["target_age_days"].to_numpy(), "unit": test["channel_id"].to_numpy()})
    for _, group in frame.groupby("unit"):
        if len(group) >= 10 and group["score"].nunique() > 1:
            values.append(spearmanr(group["score"], group["age"]).statistic)
    return float(np.nanmean(values)) if values else float("nan")


# --- Etapas -----------------------------------------------------------------


def run_inner(rows: pd.DataFrame, output: Path, seed: int) -> None:
    train, _, cutoffs = outer_split(rows)
    observations = []
    for fold, (inner_train, inner_valid) in enumerate(inner_calendar_folds(train)):
        labelled_train, labelled_valid = format_as_of_labels(inner_train, inner_valid)
        cache: dict[str, np.ndarray] = {}
        for config in CONFIGS:
            counts = counts_for(score(config, labelled_train, labelled_valid, seed, cache), labelled_valid)
            for (sector, channel), row in counts.iterrows():
                if row["pairs"]:
                    observations.append(
                        {"fold": fold, "sector": sector, "channel": channel, "config": config.name, "pairs": row["pairs"], "correct": row["correct"]}
                    )
            print(f"dobra {fold} {config.name}: {ranking_metrics(counts)['auc_macro_channel']:.3f}", flush=True)
    table = pd.DataFrame(observations)
    table.to_csv(output / "inner_folds.csv", index=False)
    per_channel = table.groupby(["config", "sector", "channel"])[["pairs", "correct"]].sum()
    per_channel["auc"] = per_channel["correct"] / per_channel["pairs"]
    summary = per_channel.groupby("config")["auc"].agg(inner_auc="mean", channels="size").reset_index()
    by_sector = per_channel.groupby(["config", "sector"])["auc"].mean().unstack()
    summary = summary.merge(by_sector.reset_index(), on="config")
    order = {config.name: position for position, config in enumerate(CONFIGS)}
    summary["order"] = summary["config"].map(order)
    summary = summary.sort_values(["inner_auc", "order"], ascending=[False, True])
    chosen = str(summary.iloc[0]["config"])
    summary.drop(columns="order").to_csv(output / "inner_selection.csv", index=False)
    frozen = {
        "chosen": chosen,
        "configs": [asdict(config) for config in CONFIGS],
        "history": HISTORY,
        "sector_cutoffs": cutoffs,
        "seed": seed,
    }
    (output / "frozen_selection.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.drop(columns="order").to_string(index=False))
    print(f"Escolhido nas dobras internas: {chosen}")


def run_outer(rows: pd.DataFrame, output: Path, seed: int, iterations: int) -> None:
    frozen_path = output / "frozen_selection.json"
    if not frozen_path.exists():
        raise SystemExit("Rode --stage inner antes: a escolha precisa estar congelada.")
    if (output / "report.json").exists():
        raise SystemExit("O corte externo deste run já foi lido; não leia de novo.")
    frozen_text = frozen_path.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    chosen = BY_NAME[frozen["chosen"]]
    train, test, _ = outer_split(rows)
    cache: dict[str, np.ndarray] = {}
    candidate_scores = score(chosen, train, test, seed, cache)
    reference_scores = score(CONFIGS[0], train, test, seed, cache)
    candidate = counts_for(candidate_scores, test)
    reference = counts_for(reference_scores, test)
    report: dict[str, object] = {
        "protocol": "shorts-content-ranking-v1",
        "frozen_selection_sha256": hashlib.sha256(frozen_text.encode("utf-8")).hexdigest(),
        "chosen": chosen.name,
        "population": {"train_rows": len(train), "test_rows": len(test), "test_channels": int(test["source_channel_id"].nunique())},
        "overall": {
            **ranking_metrics(candidate),
            "reference_auc": ranking_metrics(reference)["auc_macro_channel"],
            "age_rho": age_rho(candidate_scores, test),
            "ci": bootstrap_channels(candidate, reference, iterations, seed),
        },
        "sectors": [],
        "promotion_rule": "Promover só se o IC95% do delta pareado contra a referência ficar acima de zero e |rho idade| < 0,20.",
    }
    for sector in sorted(test["sector"].unique()):
        mask = (test["sector"] == sector).to_numpy()
        part_c, part_r = candidate.loc[[sector]], reference.loc[[sector]]
        report["sectors"].append(
            {
                "sector": sector,
                **ranking_metrics(part_c),
                "reference_auc": ranking_metrics(part_r)["auc_macro_channel"],
                "age_rho": age_rho(candidate_scores[mask], test.loc[mask]),
                "ci": bootstrap_channels(part_c, part_r, iterations, seed),
            }
        )
    per_channel = candidate.assign(auc=candidate["correct"] / candidate["pairs"], reference_auc=reference["correct"] / reference["pairs"])
    per_channel.reset_index().to_csv(output / "outer_channels.csv", index=False)
    predictions = test[["video_id", "sector", "source_channel_id", "viral"]].copy()
    predictions["score"] = candidate_scores
    predictions["reference_score"] = reference_scores
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


def main() -> int:
    args = parse_args()
    output = MODELING_RUNS_DIR / args.run_id
    output.mkdir(parents=True, exist_ok=args.stage == "outer")
    rows = load_rows(args)
    if args.stage == "inner":
        run_inner(rows, output, args.seed)
    else:
        run_outer(rows, output, args.seed, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
