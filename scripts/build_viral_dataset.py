"""Gera a base de Shorts (v2 ou v3: camadas limpa e rótulo) e o retrato descritivo.

Saídas:
    data/dataset_<versão>/shorts_clean.parquet     camada 2, um Short válido por linha
    data/dataset_<versão>/shorts_labels.parquet    camada 3, rótulo de outlier
    data/dataset_<versão>/funnel.csv               quantos saíram em cada regra
    reports/final/dataset_<versão>/retrato_base_<versão>.md e figura
Nenhum modelo é treinado aqui.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.viral_dataset import (  # noqa: E402
    HISTORY,
    MIN_AGE_DAYS,
    MIN_HISTORY,
    VIRAL_K,
    audit_boosted,
    clean_shorts,
    outlier_labels,
)

HISTORY_ROOT = ROOT / "data" / "youtube" / "shorts_history"
SOURCES = {
    "v2": ["shorts-history-2026-10-21"],
    # v3 soma os criadores de moda e beleza coletados em 21/10/2026 e audita impulsionados.
    "v3": ["shorts-history-2026-10-21", "shorts-history-2026-10-21-criadores"],
}
SELECTION = ROOT / "config" / "selected_channels.json"
SECTOR_COLORS = {"automotivo": "#0065FF", "beleza": "#F574FF", "moda": "#00A88F"}
AGE_BINS = [MIN_AGE_DAYS, 180, 365, 730, np.inf]
AGE_LABELS = ["60–180 d", "180 d–1 ano", "1–2 anos", "> 2 anos"]


def pct(value: float) -> str:
    return f"{value:.1%}".replace(".", ",")


def table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=sorted(SOURCES), default="v2")
    version = parser.parse_args().version
    OUT = ROOT / "data" / f"dataset_{version}"
    REPORT = ROOT / "reports" / "final" / f"dataset_{version}"
    raw = pd.concat(
        [pd.DataFrame(json.loads((HISTORY_ROOT / run / "normalized_records.json").read_text(encoding="utf-8"))["records"]) for run in SOURCES[version]],
        ignore_index=True,
    )
    clean, funnel = clean_shorts(raw)
    types = {item["channel_id"]: item.get("channel_type", "não classificado") for item in json.loads(SELECTION.read_text(encoding="utf-8"))["channels"]}
    clean["channel_type"] = clean["channel_id"].map(types).fillna("não classificado")
    labels = outlier_labels(clean)
    if version != "v2":
        labels = audit_boosted(labels)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    keep = [
        "video_id", "channel_id", "channel_title", "sector", "published_at", "collected_at", "title",
        "description", "tags", "category_id", "duration_seconds", "definition", "age_days", "title_key", "channel_type",
    ]
    clean[keep].to_parquet(OUT / "shorts_clean.parquet", index=False)
    labels[
        ["video_id", "channel_id", "sector", "published_at", "age_days", "views", "prior_shorts",
         "baseline_median_views", "outlier_score", "log_outlier_score", "label_status", "viral", "channel_type"]
    ].to_parquet(OUT / "shorts_labels.parquet", index=False)
    funnel.to_csv(OUT / "funnel.csv", index=False)

    labelled = labels.loc[labels["label_status"] == "rotulado"].copy()
    labelled["viral"] = labelled["viral"].astype(int)
    labelled["faixa_idade"] = pd.cut(labelled["age_days"], AGE_BINS, labels=AGE_LABELS, right=False)
    by_sector = labelled.groupby("sector").agg(
        canais=("channel_id", "nunique"), rotulados=("viral", "size"), virais=("viral", "sum"), taxa=("viral", "mean")
    ).reset_index()
    by_type = labelled.groupby(["sector", "channel_type"]).agg(
        canais=("channel_id", "nunique"), rotulados=("viral", "size"), taxa=("viral", "mean")
    ).reset_index()
    status_by_type = pd.crosstab([labels["sector"], labels["channel_type"]], labels["label_status"]).reset_index()
    by_channel = labelled.groupby(["sector", "channel_id"]).agg(
        canal=("channel_id", "first"), rotulados=("viral", "size"), virais=("viral", "sum"),
        taxa=("viral", "mean"), mediana_views=("views", "median"),
    ).reset_index()
    titles = clean.drop_duplicates("channel_id").set_index("channel_id")["channel_title"]
    by_channel["canal"] = by_channel["channel_id"].map(titles)
    by_age = labelled.groupby("faixa_idade", observed=True).agg(rotulados=("viral", "size"), taxa=("viral", "mean")).reset_index()
    by_year = labelled.assign(ano=pd.to_datetime(labelled["published_at"], utc=True).dt.year).groupby("ano").agg(
        rotulados=("viral", "size"), taxa=("viral", "mean")
    ).reset_index()
    sensitivity = pd.DataFrame(
        [{"k": k, "taxa_viral": pct(float((labelled["outlier_score"] >= k).mean()))} for k in (2.0, 3.0, 5.0, 10.0)]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    bins = np.linspace(-2, 2.5, 60)
    for sector, color in SECTOR_COLORS.items():
        values = labelled.loc[labelled["sector"] == sector, "log_outlier_score"].dropna()
        axes[0].hist(values, bins=bins, histtype="step", linewidth=1.6, color=color, density=True, label=sector)
    axes[0].axvline(np.log10(VIRAL_K), color="#0F1729", linestyle="--", linewidth=1)
    axes[0].text(np.log10(VIRAL_K) + 0.05, axes[0].get_ylim()[1] * 0.9, f"viral: ≥ {VIRAL_K:g}× a mediana", fontsize=9)
    axes[0].set_xlabel("log10(views ÷ mediana dos 30 Shorts anteriores do canal)")
    axes[0].set_ylabel("densidade")
    axes[0].legend(frameon=False)
    axes[0].set_title("Distribuição do outlier score por setor")
    axes[1].bar(by_age["faixa_idade"].astype(str), by_age["taxa"], color="#0065FF")
    for position, row in enumerate(by_age.itertuples()):
        count = f"{row.rotulados:,}".replace(",", ".")
        axes[1].text(position, row.taxa + 0.004, f"{pct(row.taxa)}\nn={count}", ha="center", fontsize=9)
    axes[1].set_ylim(0, by_age["taxa"].max() * 1.22)
    axes[1].set_ylabel("taxa de virais")
    axes[1].set_title("Taxa de virais por idade do vídeo na coleta (diagnóstico de viés)")
    for axis in axes:
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.savefig(REPORT / f"retrato_base_{version}.png", dpi=160)

    by_sector_fmt = by_sector.assign(taxa=by_sector["taxa"].map(pct))
    by_channel_fmt = by_channel.sort_values(["sector", "taxa"], ascending=[True, False]).assign(
        taxa=lambda f: f["taxa"].map(pct), mediana_views=lambda f: f["mediana_views"].round(0).astype(int)
    )[["sector", "canal", "rotulados", "virais", "taxa", "mediana_views"]]
    lines = [
        f"# Base {version} de Shorts: limpeza e rótulo de outlier",
        "",
        "Gerado por `scripts/build_viral_dataset.py` a partir da coleta de 21/10/2026",
        "(`normalized_records.json`). Nenhum modelo foi treinado nesta base ainda.",
        "",
        "## Regras fixadas antes de qualquer modelo",
        "",
        "- Short = público, publicado a partir de 12/07/2021 (lançamento global), com até 60 s",
        "  antes de 15/10/2024 e até 180 s a partir dessa data.",
        "- Período prospectivo (desde 21/10/2026) fica fora: é a validação final.",
        "- Reupload: mesmo canal, título normalizado e duração; fica o primeiro publicado.",
        f"- Rótulo só para vídeos com pelo menos {MIN_AGE_DAYS:g} dias na coleta (**HIPÓTESE**: views de Short",
        "  estabilizam em cerca de 2 meses; não medido nesta base).",
        f"- `outlier_score` = views ÷ mediana das views dos {HISTORY} Shorts anteriores do mesmo canal",
        f"  (mínimo de {MIN_HISTORY} anteriores).",
        f"- `viral` = `outlier_score` ≥ {VIRAL_K:g} (**HIPÓTESE**: convenção de outlier \"3× a mediana\").",
        "- Views só aparecem no rótulo, nunca como feature.",
        "",
        "## Funil de limpeza",
        "",
        *table(funnel),
        "",
        "## Rótulo",
        "",
        *table(labels["label_status"].value_counts().rename_axis("situação").reset_index(name="vídeos")),
        "",
        *table(by_sector_fmt),
        "",
        "Sensibilidade da taxa de virais ao limiar k (descritivo; o k oficial continua 3):",
        "",
        *table(sensitivity),
        "",
        "## Tipo de canal e auditoria de impulsionados",
        "",
        "Tipo de canal é classificação manual (`config/selected_channels.json`). Na v3, Shorts com likes/view",
        "abaixo de 10% da mediana dos 30 anteriores do canal saem do rótulo como `provável impulsionado`,",
        "e canais com likes ocultos em mais da metade dos vídeos saem como `likes ocultos`.",
        "",
        *table(status_by_type),
        "",
        *table(by_type.assign(taxa=by_type["taxa"].map(pct))),
        "",
        "## Viés de idade (ler antes de modelar)",
        "",
        "A linha de base usa Shorts mais antigos do canal, que tiveram mais tempo para acumular views.",
        "Se a taxa de virais cresce com a idade, parte do rótulo é idade, e o modelo precisa ser",
        "conferido contra esse atalho.",
        "",
        *table(by_age.assign(taxa=by_age["taxa"].map(pct), faixa_idade=by_age["faixa_idade"].astype(str))),
        "",
        *table(by_year.assign(taxa=by_year["taxa"].map(pct))),
        "",
        f"![Retrato da base {version}](retrato_base_{version}.png)",
        "",
        "## Por canal",
        "",
        *table(by_channel_fmt),
        "",
    ]
    (REPORT / f"retrato_base_{version}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
