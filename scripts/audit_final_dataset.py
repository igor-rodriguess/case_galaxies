"""Auditoria final da base (Parte 1 da avaliação final).

Gera tabelas, figuras e a lista congelada da base principal a partir do escopo de
`docs/escopo_avaliacao_final.md`. Não treina nem avalia modelos.

Uso:
    python scripts/audit_final_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.agent.protocol import age_adjusted_performance  # noqa: E402
from galaxies.dataset import (  # noqa: E402
    build_feature_rows,
    load_latest_vlm_results,
    load_youtube_documents,
    nested,
)
from galaxies.final_audit import (  # noqa: E402
    AUDIT_VERSION,
    FORMATS,
    SECTORS,
    FinalScope,
    analysis_hash,
    case_control_roles,
    column_role,
    duplicate_titles,
    duration_format,
    growth_by_age,
    minimum_age_rule,
    missingness,
    near_duplicate_pairs,
    selection_check,
    snapshot_growth,
    unusable_columns,
    vlm_cardinality,
    vlm_protocol_ok,
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "final" / "auditoria"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "final-auditoria"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "features" / "embeddings"
FEATURE_RUNS_DIR = PROJECT_ROOT / "data" / "features" / "runs"
VLM_CATEGORICAL = [
    "framing",
    "camera_movement",
    "motion_intensity",
    "audio_language",
    "opening_type",
    "product_prominence",
]
AGE_EDGES = (0, 1, 3, 7, 14, 21, 30, 60, 90, 180, 365, 730, 100_000)

# Paleta de referência (skill dataviz): três primeiros slots, validados em todos os pares.
SECTOR_COLORS = {"automotivo": "#2a78d6", "moda": "#eb6834", "beleza": "#1baf7a"}
FORMAT_COLORS = {"ate_60s": "#86b6ef", "61_180s": "#2a78d6", "acima_180s": "#104281"}
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def to_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def load_frame(scope: FinalScope) -> tuple[pd.DataFrame, dict]:
    youtube_run = PROJECT_ROOT / scope.youtube_run
    cutoff = datetime.fromisoformat(scope.vlm_completed_before.replace("Z", "+00:00"))
    rows, _ = build_feature_rows(youtube_run, vlm_completed_before=cutoff)
    frame = pd.DataFrame(rows)
    documents = load_youtube_documents(youtube_run)
    titles = {
        nested(doc, "identity", "video_id"): nested(doc, "publication", "title")
        for doc in documents
    }
    frame["title"] = frame["video_id"].map(titles)

    vlm = load_latest_vlm_results(completed_before=cutoff)
    frame["vlm_model_requested"] = frame["video_id"].map(
        lambda vid: (
            nested(vlm[vid], "provenance", "extractor", "model_requested") if vid in vlm else None
        )
    )
    frame["vlm_analysis_hash"] = frame["video_id"].map(
        lambda vid: analysis_hash(vlm[vid].get("analysis")) if vid in vlm else None
    )
    orphan_vlm = sorted(set(vlm) - set(frame["video_id"]))

    plan = json.loads((PROJECT_ROOT / scope.case_control_plan).read_text(encoding="utf-8"))
    roles = case_control_roles(plan)
    frame["sample_role"] = np.where(
        frame["video_id"].isin(roles),
        frame["video_id"].map(roles),
        np.where(frame["vlm_available"], "lote", "sem_vlm"),
    )
    frame["format"] = frame["duration_seconds"].map(
        lambda value: duration_format(value, scope.short_max_seconds)
    )
    frame["age_days"] = frame["target_age_days"].astype(float)
    frame["published"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["protocol_ok"] = vlm_protocol_ok(frame, scope)
    return frame, {"vlm_results_total": len(vlm), "orphan_vlm_ids": orphan_vlm}


def second_snapshot(scope: FinalScope) -> tuple[pd.DataFrame, Path]:
    path = PROJECT_ROOT / scope.second_snapshot / "manifest.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return pd.DataFrame(items), path


def funnel(frame: pd.DataFrame, min_age: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    steps = {
        "base_data_api": pd.Series(True, index=frame.index),
        "com_vlm": frame["vlm_available"].astype(bool),
    }
    steps["fora_caso_controle"] = steps["com_vlm"] & ~frame["sample_role"].isin(["case", "control"])
    steps["protocolo_principal"] = steps["fora_caso_controle"] & frame["protocol_ok"]
    steps["idade_minima"] = steps["protocolo_principal"] & (frame["age_days"] >= min_age)
    table = pd.DataFrame(
        {name: mask.groupby(frame["sector"]).sum() for name, mask in steps.items()}
    ).T.reindex(columns=list(SECTORS))
    table["total"] = table.sum(axis=1)
    return table.reset_index(names="etapa"), steps


def channel_table(frame: pd.DataFrame, main: pd.Series) -> pd.DataFrame:
    rows = []
    for (sector, channel), group in frame.groupby(["sector", "channel_id"]):
        selected = group.loc[main.loc[group.index]]
        rows.append(
            {
                "sector": sector,
                "channel_id": channel,
                "base_videos": len(group),
                "main_videos": len(selected),
                "main_share_ate_180s": float((selected["format"] != "acima_180s").mean())
                if len(selected)
                else None,
                "main_first_published": selected["published"].min(),
                "main_last_published": selected["published"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["sector", "main_videos"], ascending=[True, False])


def coverage_table(frame: pd.DataFrame, main: pd.Series, eligible: pd.Series) -> pd.DataFrame:
    thumb = set(pd.read_parquet(EMBEDDINGS_DIR / "thumb.parquet", columns=["video_id"])["video_id"])
    title = set(pd.read_parquet(EMBEDDINGS_DIR / "title.parquet", columns=["video_id"])["video_id"])
    title_desc = set(
        pd.read_parquet(EMBEDDINGS_DIR / "title_desc.parquet", columns=["video_id"])["video_id"]
    )
    full_transcripts = {
        path.parent.name
        for pattern in ("transcript*.json", "*transcription*.json")
        for path in FEATURE_RUNS_DIR.glob(f"transcript-pilot*/videos/*/{pattern}")
    }
    rows = []
    for sector in SECTORS:
        in_sector = frame["sector"] == sector
        selected = frame.loc[main & in_sector]
        population = frame.loc[eligible & in_sector]
        rows.append(
            {
                "sector": sector,
                "main_videos": len(selected),
                "eligible_population": len(population),
                "vlm_protocol_share_of_eligible": len(selected) / len(population)
                if len(population)
                else None,
                "thumb_embedding": float(selected["video_id"].isin(thumb).mean()),
                "title_embedding": float(selected["video_id"].isin(title).mean()),
                "title_desc_embedding": float(selected["video_id"].isin(title_desc).mean()),
                "speech_in_0_3s": float(selected["transcript_available"].astype(bool).mean()),
                "full_transcript": float(selected["video_id"].isin(full_transcripts).mean()),
            }
        )
    return pd.DataFrame(rows)


def style_axes(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_growth(growth: pd.DataFrame, rule_rows: list[dict], scope: FinalScope, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style_axes(axis)
    data = growth.loc[growth["age_days_1"] > 0]
    axis.scatter(
        data["age_days_1"], data["daily_log_growth"], s=8, color="#b8b7b1", alpha=0.5, linewidths=0
    )
    edges = np.geomspace(1, data["age_days_1"].max() + 1, 25)
    centers = np.sqrt(edges[:-1] * edges[1:])
    buckets = pd.cut(data["age_days_1"], edges)
    for sector in SECTORS:
        medians = (
            data.loc[data["sector"] == sector]
            .groupby(
                pd.cut(data.loc[data["sector"] == sector, "age_days_1"], edges), observed=False
            )["daily_log_growth"]
            .median()
        )
        axis.plot(
            centers, medians.to_numpy(), color=SECTOR_COLORS[sector], linewidth=2, label=sector
        )
    pooled = data.groupby(buckets, observed=False)["daily_log_growth"].median()
    axis.plot(
        centers, pooled.to_numpy(), color=INK, linewidth=2, linestyle="--", label="três setores"
    )
    axis.axhline(scope.growth_threshold_per_day, color=INK_SECONDARY, linewidth=1)
    axis.annotate(
        f"limite {scope.growth_threshold_per_day:g}/dia",
        (1.1, scope.growth_threshold_per_day),
        textcoords="offset points",
        xytext=(0, 4),
        color=INK_SECONDARY,
        fontsize=9,
    )
    for row in rule_rows:
        axis.axvline(row["candidate_days"], color=GRID, linewidth=1)
    axis.set_xscale("log")
    axis.set_ylim(-0.01, max(0.2, float(np.nanquantile(data["daily_log_growth"], 0.97))))
    axis.set_xlabel("Idade na primeira leitura (dias, escala log)", color=INK_SECONDARY)
    axis.set_ylabel("Crescimento diário de log1p(views)", color=INK_SECONDARY)
    axis.set_title("Crescimento das views entre as duas leituras, por idade", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_coverage(frame: pd.DataFrame, main: pd.Series, path: Path) -> None:
    edges = (0, 7, 30, 90, 365, 100_000)
    labels = ["<7 d", "7-30 d", "30-90 d", "90 d-1 a", ">1 a"]
    bucket = pd.cut(frame["age_days"], edges, labels=labels, right=False)
    share = main.groupby([frame["sector"], bucket], observed=False).mean().unstack()
    figure, axis = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    style_axes(axis)
    width = 0.26
    positions = np.arange(len(labels))
    for offset, sector in enumerate(SECTORS):
        values = share.loc[sector, labels].to_numpy(dtype=float)
        axis.bar(
            positions + (offset - 1) * (width + 0.02),
            values,
            width=width,
            color=SECTOR_COLORS[sector],
            label=sector,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.set_xlabel("Idade no snapshot de 16/10", color=INK_SECONDARY)
    axis.set_ylabel("Vídeos na base principal", color=INK_SECONDARY)
    axis.set_title("Cobertura da base principal por idade do vídeo", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_views(selected: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    for axis, sector in zip(axes, SECTORS, strict=True):
        style_axes(axis)
        group = selected.loc[selected["sector"] == sector]
        data = [
            np.log1p(group.loc[group["format"] == fmt, "target_view_count"].astype(float))
            for fmt in FORMATS
        ]
        parts = axis.boxplot(
            data, widths=0.55, patch_artist=True, medianprops={"color": INK, "linewidth": 2}
        )
        for patch, fmt in zip(parts["boxes"], FORMATS, strict=True):
            patch.set_facecolor(FORMAT_COLORS[fmt])
            patch.set_edgecolor(SURFACE)
        axis.set_xticks([1, 2, 3], ["≤60 s", "61-180 s", ">180 s"])
        for position, values in enumerate(data, start=1):
            axis.annotate(
                f"n={len(values)}",
                (position, axis.get_ylim()[0]),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )
        axis.set_title(sector, color=INK, loc="left", fontsize=10)
    axes[0].set_ylabel("log1p(views) no snapshot de 16/10", color=INK_SECONDARY)
    figure.suptitle("Views da base principal por formato", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> int:
    scope = FinalScope()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    frame, vlm_info = load_frame(scope)
    second, second_path = second_snapshot(scope)
    first = frame.rename(
        columns={"target_observed_at": "observed_at", "target_view_count": "view_count"}
    )
    growth = snapshot_growth(first, second)
    growth = growth.merge(frame[["video_id", "sector"]], on="video_id")
    min_age, rule_rows = minimum_age_rule(
        growth,
        scope.age_candidates_days,
        scope.age_window_days,
        scope.growth_threshold_per_day,
        scope.fallback_min_age_days,
    )

    funnel_table, steps = funnel(frame, min_age)
    main_mask = steps["idade_minima"]
    eligible = frame["age_days"] >= min_age
    selected = frame.loc[main_mask].copy()

    second_by_id = second.set_index("video_id")
    title_changed = frame["title"] != frame["video_id"].map(second_by_id["title"])
    not_public = frame["video_id"].map(second_by_id["privacy_status"]).fillna("") != "public"

    feature_columns = [
        column
        for column in frame.columns
        if column_role(column) == "candidate_feature"
        and column
        not in {
            "title",
            "vlm_model_requested",
            "vlm_analysis_hash",
            "sample_role",
            "format",
            "age_days",
            "published",
            "protocol_ok",
        }
    ]
    roles = pd.DataFrame({"column": frame.columns, "role": [column_role(c) for c in frame.columns]})
    quality = missingness(selected, feature_columns)
    unusable = unusable_columns(quality)
    usable_by_sector = {
        sector: sorted(
            set(feature_columns) - set(unusable.loc[unusable["sector"] == sector, "column"])
        )
        for sector in SECTORS
    }

    score = pd.Series(np.nan, index=frame.index)
    population = frame.loc[eligible]
    score.loc[population.index] = age_adjusted_performance(population)
    selection = selection_check(frame.loc[eligible], main_mask.loc[eligible], score.loc[eligible])

    thumbs = pd.read_parquet(EMBEDDINGS_DIR / "thumb.parquet", columns=["video_id", "embedding"])
    near_dupes = near_duplicate_pairs(thumbs, selected["video_id"].tolist())
    title_dupes = duplicate_titles(selected)
    repeated_hash = selected.loc[
        selected["vlm_analysis_hash"].duplicated(keep=False),
        ["video_id", "sector", "vlm_analysis_hash"],
    ]

    periods = (
        selected.groupby(["sector", selected["published"].dt.year]).size().unstack(fill_value=0)
    )
    formats = selected.groupby(["sector", "format"]).size().unstack(fill_value=0)
    roles_by_sector = (
        frame.loc[frame["vlm_available"]]
        .groupby(["sector", "sample_role"])
        .size()
        .unstack(fill_value=0)
    )
    log_views = np.log1p(selected["target_view_count"].astype(float))
    view_summary = log_views.groupby([selected["sector"], selected["format"]]).describe()
    age_summary = selected.groupby("sector")["age_days"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    channels = channel_table(frame, main_mask)
    coverage = coverage_table(frame, main_mask, eligible)
    cardinality = vlm_cardinality(selected, VLM_CATEGORICAL)
    growth_table = growth_by_age(growth, AGE_EDGES)

    sensitivities = {
        "idade_14_dias": steps["protocolo_principal"] & (frame["age_days"] >= 14),
        "idade_30_dias": steps["protocolo_principal"]
        & (frame["age_days"] >= scope.sensitivity_min_age_days),
        "ate_180s": main_mask & (frame["format"] != "acima_180s"),
        "todos_os_extratores_v010": steps["fora_caso_controle"]
        & (frame["vlm_prompt_version"] == scope.vlm_prompt_version)
        & (frame["age_days"] >= min_age),
    }
    sensitivity_counts = {
        name: {
            sector: {
                "videos": int((mask & (frame["sector"] == sector)).sum()),
                "channels": int(
                    frame.loc[mask & (frame["sector"] == sector), "channel_id"].nunique()
                ),
                "channels_10plus": int(
                    (
                        frame.loc[mask & (frame["sector"] == sector), "channel_id"].value_counts()
                        >= 10
                    ).sum()
                ),
            }
            for sector in SECTORS
        }
        for name, mask in {"principal": main_mask, **sensitivities}.items()
    }

    base_columns = [
        "video_id",
        "sector",
        "channel_id",
        "published_at",
        "age_days",
        "format",
        "duration_seconds",
        "sample_role",
        "vlm_model_requested",
        "vlm_prompt_version",
    ]
    base = frame.loc[main_mask, base_columns].copy()
    for name, mask in sensitivities.items():
        base[f"in_{name}"] = mask.loc[base.index]
    base.sort_values(["sector", "channel_id", "published_at"]).to_csv(
        OUTPUT_DIR / "base_principal.csv", index=False
    )
    outside_main = ~main_mask & (
        sensitivities["idade_30_dias"] | sensitivities["todos_os_extratores_v010"]
    )
    sensitivity_only = frame.loc[outside_main, base_columns]
    sensitivity_only.to_csv(OUTPUT_DIR / "fora_da_principal_mas_em_sensibilidade.csv", index=False)

    funnel_table.to_csv(OUTPUT_DIR / "funil.csv", index=False)
    channels.to_csv(OUTPUT_DIR / "canais.csv", index=False)
    periods.to_csv(OUTPUT_DIR / "periodo_por_ano.csv")
    roles.to_csv(OUTPUT_DIR / "papeis_das_colunas.csv", index=False)
    quality.to_csv(OUTPUT_DIR / "ausencia_por_feature.csv", index=False)
    unusable.to_csv(OUTPUT_DIR / "features_descartadas.csv", index=False)
    growth_table.to_csv(OUTPUT_DIR / "crescimento_por_idade.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "vies_de_selecao.csv", index=False)
    cardinality.to_csv(OUTPUT_DIR / "cardinalidade_vlm.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "cobertura.csv", index=False)
    near_dupes.to_csv(OUTPUT_DIR / "thumbnails_quase_identicas.csv", index=False)
    title_dupes.to_csv(OUTPUT_DIR / "titulos_repetidos.csv", index=False)

    plot_growth(growth, rule_rows, scope, FIGURE_DIR / "crescimento_por_idade.png")
    plot_coverage(frame, main_mask, FIGURE_DIR / "cobertura_por_idade.png")
    plot_views(selected, FIGURE_DIR / "views_por_formato.png")

    records_path = PROJECT_ROOT / scope.youtube_run / "normalized_records.json"
    audit = {
        "audit_version": AUDIT_VERSION,
        "scope": scope.as_dict(),
        "reproducibility": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pyarrow")
            },
            "inputs_sha256": {
                scope.youtube_run + "/normalized_records.json": sha256(records_path),
                scope.second_snapshot + "/manifest.json": sha256(second_path),
                scope.case_control_plan: sha256(PROJECT_ROOT / scope.case_control_plan),
                "data/features/embeddings/thumb.parquet": sha256(EMBEDDINGS_DIR / "thumb.parquet"),
            },
            "base_principal_sha256": sha256(OUTPUT_DIR / "base_principal.csv"),
        },
        "vlm": {
            **vlm_info,
            "with_youtube_record": int(frame["vlm_available"].sum()),
            "models": frame.loc[frame["vlm_available"], "vlm_model_requested"]
            .value_counts()
            .to_dict(),
            "prompt_versions": frame.loc[frame["vlm_available"], "vlm_prompt_version"]
            .value_counts()
            .to_dict(),
            "roles_by_sector": roles_by_sector.to_dict(orient="index"),
        },
        "minimum_age": {"chosen_days": min_age, "rule": rule_rows},
        "funnel": to_records(funnel_table),
        "samples": sensitivity_counts,
        "formats": formats.to_dict(orient="index"),
        "periods_by_year": {
            sector: {str(k): int(v) for k, v in row.items()} for sector, row in periods.iterrows()
        },
        "age_days": age_summary.round(2).to_dict(orient="index"),
        "log1p_views_by_format": {
            f"{s}|{f}": row for (s, f), row in view_summary.round(3).to_dict(orient="index").items()
        },
        "snapshot_quality": {
            "second_snapshot_videos": len(second),
            "views_decreased": int(growth["views_decreased"].sum()),
            "title_changed_base": int(title_changed.sum()),
            "title_changed_main": int(title_changed.loc[main_mask].sum()),
            "not_public_second_snapshot": int(not_public.sum()),
            "days_between_median": float(growth["days_between"].median()),
        },
        "duplicates": {
            "video_id_in_base": int(frame["video_id"].duplicated().sum()),
            "repeated_titles_in_main_same_channel": len(title_dupes),
            "thumbnail_pairs_cosine_097_in_main": len(near_dupes),
            "identical_vlm_analyses_in_main": len(repeated_hash),
        },
        "columns": {
            "roles": roles["role"].value_counts().to_dict(),
            "candidate_features": feature_columns,
            "usable_by_sector": usable_by_sector,
        },
        "coverage": to_records(coverage),
        "selection_check": to_records(selection),
        "vlm_cardinality": to_records(cardinality),
    }
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"minimum_age_days": min_age, "main": sensitivity_counts["principal"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
