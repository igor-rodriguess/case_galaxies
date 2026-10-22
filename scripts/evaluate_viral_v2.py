"""Ranking de Shorts na base v2 (rótulo de outlier contra a mediana do canal).

Mesmo protocolo de ``evaluate_shorts_content_ranking.py``, com as mesmas nove
configurações (fixadas antes desta base existir):

* população: ``data/dataset_v2`` só com vídeos rotulados (maduros e com
  histórico); features calculadas sobre a base limpa inteira, só com atributos
  conhecidos na publicação e só com vídeos anteriores do canal;
* corte de calendário por setor (``calendar_split``) e três dobras internas no
  treino; escolha global pela AUC média por canal nas dobras;
* ``--stage outer`` lê o corte externo uma vez, com IC pareado por canal contra
  a referência logística e Spearman nota×idade dentro do canal.

O alvo de treino das configurações "pct" é o percentil do ``log_outlier_score``
dentro do canal; a referência logística usa ``viral`` (score >= 3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from galaxies.agent.evaluation import calendar_split  # noqa: E402
from galaxies.agent.format_ranking import bootstrap_channels, inner_calendar_folds, ranking_metrics  # noqa: E402
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402
from scripts import evaluate_shorts_content_ranking as content  # noqa: E402

EMBEDDINGS = ROOT / "data" / "features" / "embeddings"
AGE_SHORTCUT_LIMIT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inner", "outer"], required=True)
    parser.add_argument("--run-id", default="viral-v2-ranking-20261021")
    parser.add_argument("--data-version", choices=["v2", "v3"], default="v2")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _cadence(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["channel_id", "_published"])
    gaps = ordered.groupby("channel_id")["_published"].diff().dt.total_seconds() / 86400.0
    prior_median = gaps.groupby(ordered["channel_id"]).transform(lambda s: s.shift(1).expanding().median())
    return pd.DataFrame(
        {"days_since_previous_upload": gaps, "prior_upload_interval_median_days": prior_median}, index=ordered.index
    )


def load_rows(data_version: str = "v2") -> pd.DataFrame:
    data = ROOT / "data" / f"dataset_{data_version}"
    clean = pd.read_parquet(data / "shorts_clean.parquet")
    labels = pd.read_parquet(data / "shorts_labels.parquet", columns=["video_id", "label_status", "viral", "log_outlier_score"])
    rows = clean.merge(labels, on="video_id", validate="one_to_one")
    rows["_published"] = pd.to_datetime(rows["published_at"], utc=True)
    rows["source_channel_id"] = rows["channel_id"]
    rows["_title"] = rows["title"].fillna("")
    rows["duration_seconds"] = rows["duration_seconds"].astype(float)
    rows["log_duration"] = np.log1p(rows["duration_seconds"])
    rows["is_short"] = 1.0
    rows["title_character_count"] = rows["_title"].str.len().astype(float)
    rows["description_character_count"] = rows["description"].fillna("").str.len().astype(float)
    rows["tag_count"] = rows["tags"].map(lambda tags: float(len(tags)) if isinstance(tags, (list, np.ndarray)) else 0.0)
    rows["publish_hour_utc"] = rows["_published"].dt.hour.astype(float)
    rows["publish_weekday_utc"] = rows["_published"].dt.weekday.astype(str)
    rows["category_id"] = rows["category_id"].astype(str)
    rows = rows.join(_cadence(rows))

    # title_e5.parquet cobre a coleta de 21/10; title_e5_*.parquet, coletas posteriores.
    titles = pd.concat(
        [pd.read_parquet(path, columns=["video_id", "embedding"]) for path in sorted(EMBEDDINGS.glob("title_e5*.parquet"))],
        ignore_index=True,
    )
    thumbs = pd.read_parquet(EMBEDDINGS / "thumb_resnet18.parquet", columns=["video_id", "embedding"])
    rows = rows.merge(titles.drop_duplicates("video_id", keep="last").rename(columns={"embedding": "_title_embedding"}), on="video_id", how="left")
    rows = rows.merge(thumbs.drop_duplicates("video_id", keep="last").rename(columns={"embedding": "_thumb_embedding"}), on="video_id", how="left")
    for column in ("_title_embedding", "_thumb_embedding"):
        missing = int(rows[column].isna().sum())
        if missing:
            raise ValueError(f"Cobertura incompleta em {column}: faltam {missing} de {len(rows)}.")
    rows.index = pd.RangeIndex(len(rows))
    style = pd.DataFrame([content._style(title) for title in rows["_title"]], index=rows.index)
    rows = pd.concat([rows, style], axis=1)
    rows = pd.concat(
        [
            rows,
            content._rolling_relative(rows),
            content._novelty(rows, "_title_embedding", "title"),
            content._novelty(rows, "_thumb_embedding", "thumb"),
        ],
        axis=1,
    )
    labelled = rows.loc[rows["label_status"] == "rotulado"].copy()
    labelled["viral"] = labelled["viral"].astype(int)
    labelled["adjusted_performance"] = labelled["log_outlier_score"]  # alvo "pct" das configurações
    labelled["target_age_days"] = labelled["age_days"]  # só para a checagem de idade
    return labelled


def run_inner(rows: pd.DataFrame, output: Path, seed: int) -> None:
    train, _, cutoffs = calendar_split(rows)
    observations = []
    for fold, (inner_train, inner_valid) in enumerate(inner_calendar_folds(train)):
        inner_valid = inner_valid.loc[inner_valid["channel_id"].isin(set(inner_train["channel_id"]))]
        cache: dict[str, np.ndarray] = {}
        for config in content.CONFIGS:
            counts = content.counts_for(content.score(config, inner_train, inner_valid, seed, cache), inner_valid)
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
    summary = summary.merge(per_channel.groupby(["config", "sector"])["auc"].mean().unstack().reset_index(), on="config")
    order = {config.name: position for position, config in enumerate(content.CONFIGS)}
    summary["order"] = summary["config"].map(order)
    summary = summary.sort_values(["inner_auc", "order"], ascending=[False, True]).drop(columns="order")
    chosen = str(summary.iloc[0]["config"])
    summary.to_csv(output / "inner_selection.csv", index=False)
    frozen = {"chosen": chosen, "configs": [asdict(c) for c in content.CONFIGS], "sector_cutoffs": cutoffs, "seed": seed, "label": "outlier_score >= 3 (base v2)"}
    (output / "frozen_selection.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Escolhido nas dobras internas: {chosen}")


def run_outer(rows: pd.DataFrame, output: Path, seed: int, iterations: int) -> None:
    frozen_path = output / "frozen_selection.json"
    if not frozen_path.exists():
        raise SystemExit("Rode --stage inner antes.")
    if (output / "report.json").exists():
        raise SystemExit("O corte externo deste run já foi lido; não leia de novo.")
    frozen_text = frozen_path.read_text(encoding="utf-8")
    chosen = content.BY_NAME[json.loads(frozen_text)["chosen"]]
    train, test, _ = calendar_split(rows)
    cache: dict[str, np.ndarray] = {}
    candidate_scores = content.score(chosen, train, test, seed, cache)
    reference_scores = content.score(content.CONFIGS[0], train, test, seed, cache)
    candidate = content.counts_for(candidate_scores, test)
    reference = content.counts_for(reference_scores, test)

    def block(part_c: pd.DataFrame, part_r: pd.DataFrame, mask: np.ndarray) -> dict[str, object]:
        return {
            **ranking_metrics(part_c),
            "reference_auc": ranking_metrics(part_r)["auc_macro_channel"],
            "reference_lift_top10": ranking_metrics(part_r)["lift_top10"],
            "age_rho": content.age_rho(candidate_scores[mask], test.loc[mask]),
            "ci": bootstrap_channels(part_c, part_r, iterations, seed),
        }

    report: dict[str, object] = {
        "protocol": "viral-v2-ranking-v1",
        "frozen_selection_sha256": hashlib.sha256(frozen_text.encode("utf-8")).hexdigest(),
        "chosen": chosen.name,
        "population": {"train_rows": len(train), "test_rows": len(test), "test_channels": int(test["channel_id"].nunique()), "test_viral_rate": float(test["viral"].mean())},
        "overall": block(candidate, reference, np.ones(len(test), dtype=bool)),
        "sectors": [
            {"sector": sector, **block(candidate.loc[[sector]], reference.loc[[sector]], (test["sector"] == sector).to_numpy())}
            for sector in sorted(test["sector"].unique())
        ],
        "promotion_rule": f"Promover só se o IC95% do delta pareado ficar acima de zero e |rho idade| < {AGE_SHORTCUT_LIMIT}.",
    }
    overall = report["overall"]
    report["promoted"] = bool(overall["ci"]["delta_auc_macro"][0] > 0 and abs(overall["age_rho"]) < AGE_SHORTCUT_LIMIT)
    predictions = test[["video_id", "sector", "channel_id", "viral", "log_outlier_score", "age_days"]].copy()
    predictions["score"] = candidate_scores
    predictions["reference_score"] = reference_scores
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


def main() -> int:
    args = parse_args()
    output = MODELING_RUNS_DIR / args.run_id
    output.mkdir(parents=True, exist_ok=args.stage == "outer")
    rows = load_rows(args.data_version)
    print(f"{len(rows)} Shorts rotulados; taxa viral {rows['viral'].mean():.3f}", flush=True)
    if args.stage == "inner":
        run_inner(rows, output, args.seed)
    else:
        run_outer(rows, output, args.seed, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
