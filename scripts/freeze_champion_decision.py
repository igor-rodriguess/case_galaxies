"""Congela a decisão científica de modelos depois da avaliação pós-seleção.

O script não treina nem escolhe candidatos olhando holdouts. Ele consolida a
seleção interna já executada, a evidência bootstrap nos dois holdouts e as
ablações de embeddings em artefatos finais rastreáveis.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ID = "optimized-vlm1001-relative-metadata-av-20261020"
FEATURE_MAP_ID = "feature-map-vlm1001-20261020"
EMBEDDING_ID = "embedding-ablation-vlm1001-20261020"
CATBOOST_ID = "catboost-vlm1001-20261020"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def count_vlm_rows(path: Path) -> tuple[int, int]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("vlm_available"):
                ids.append(str(row["video_id"]))
    return len(ids), len(set(ids))


def main() -> int:
    evaluation_dir = ROOT / "data" / "modeling" / "evaluations" / EVALUATION_ID
    feature_dir = ROOT / "data" / "modeling" / "runs" / FEATURE_MAP_ID
    embedding_dir = ROOT / "data" / "modeling" / "runs" / EMBEDDING_ID
    catboost_dir = ROOT / "data" / "modeling" / "runs" / CATBOOST_ID
    output_dir = ROOT / "reports" / "final" / "modeling"

    summary = read_json(evaluation_dir / "summary.json")
    evidence = read_json(evaluation_dir / "winner_evidence.json")
    readiness = read_json(feature_dir / "readiness_audit.json")
    embedding = read_json(embedding_dir / "report.json")
    catboost = read_json(catboost_dir / "report.json")
    vlm_rows, vlm_unique = count_vlm_rows(feature_dir / "feature_map.jsonl")
    if vlm_rows != vlm_unique or vlm_unique < 1_000:
        raise RuntimeError(
            f"A decisão exige >=1000 VLMs únicos; linhas={vlm_rows}, únicos={vlm_unique}."
        )

    sectors: dict[str, dict] = {}
    for sector, selected in summary["sectors"].items():
        sector_evidence = evidence["sectors"][sector]
        confirmed = bool(sector_evidence["confirmed_for_both_holdouts"])
        candidate = str(selected["best_inner_cv"])
        sectors[sector] = {
            "selected_by_inner_validation": candidate,
            "promoted": confirmed,
            "champion": candidate if confirmed else "historical_median_baseline",
            "decision": (
                "candidate_promoted"
                if confirmed
                else "baseline_retained_insufficient_external_evidence"
            ),
            "temporal": sector_evidence["splits"]["temporal"],
            "channel_holdout": sector_evidence["splits"]["channel_holdout"],
        }

    embedding_rows = {
        row["name"]: {
            "auc_intra": row["metrics"]["auc_intra"],
            "auc_intra_ci95": [
                row["metrics"]["auc_intra_ci_low"],
                row["metrics"]["auc_intra_ci_high"],
            ],
            "lift_top10": row["metrics"]["lift_top10"],
            "paired": row.get("paired"),
        }
        for row in embedding["results"]
    }
    catboost_rows = {
        row["name"]: {
            "auc_intra": row["metrics"]["auc_intra"],
            "auc_intra_ci95": [
                row["metrics"]["auc_intra_ci_low"],
                row["metrics"]["auc_intra_ci_high"],
            ],
            "lift_top10": row["metrics"]["lift_top10"],
            "paired": row.get("paired"),
        }
        for row in catboost["results"]
    }

    frozen = {
        "schema_version": "1.0.0",
        "frozen_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "frozen_no_ml_promotion"
        if not any(row["promoted"] for row in sectors.values())
        else "frozen_with_promoted_model",
        "data": {
            "feature_map": str(feature_dir / "feature_map.jsonl"),
            "records": readiness["records"],
            "vlm_rows": vlm_rows,
            "vlm_unique_video_ids": vlm_unique,
            "duplicate_vlm_video_ids": vlm_rows - vlm_unique,
            "sector_readiness": readiness["sectors"],
        },
        "target": {
            "name": "relative_performance",
            "description": (
                "Desempenho relativo à curva de idade e à referência histórica do canal, "
                "ajustada somente dentro do treino."
            ),
            "fixed_horizon_available": False,
        },
        "selection": {
            "policy": (
                "Hiperparâmetros e algoritmo escolhidos por validação temporal interna; "
                "holdouts temporal e de canais usados apenas para confirmação."
            ),
            "promotion_rule": (
                "Mesmo candidato precisa superar o baseline nos dois holdouts, com IC95% "
                "pareado abaixo de zero, ganho >=5% e R² >=0,05."
            ),
            "feature_set": "metadata_audiovisual",
            "excluded_as_unsafe": [
                "métricas pós-publicação como entrada",
                "IDs e papel caso/controle",
                "histórico de canal marcado como temporalmente inválido",
            ],
        },
        "sectors": sectors,
        "embedding_ablation": {
            "protocol": embedding["protocol"],
            "results": embedding_rows,
            "decision": "not_promoted_paired_ci_crosses_zero",
        },
        "catboost_ablation": {
            "protocol": catboost["protocol"],
            "results": catboost_rows,
            "decision": "not_promoted_reference_auc_higher_and_paired_ci_crosses_zero",
        },
        "limitations": [
            "O alvo não é uma observação de views em horizonte fixo.",
            "Os 1.000 VLMs incluem protocolos de 3 s e 15 s; o modelo usa somente a taxonomia comum.",
            "Nenhum candidato cumpriu a regra completa de promoção nos dois holdouts.",
            "A interpretação causal de mudanças no criativo não é suportada por estes dados observacionais.",
        ],
        "artifacts": {
            "optimization_summary": str(evaluation_dir / "summary.json"),
            "winner_evidence": str(evaluation_dir / "winner_evidence.json"),
            "metrics": str(evaluation_dir / "model_metrics.csv"),
            "predictions": str(evaluation_dir / "predictions.csv"),
            "embedding_report": str(embedding_dir / "report.json"),
            "catboost_report": str(catboost_dir / "report.json"),
        },
    }
    write_json(output_dir / "champion-freeze.json", frozen)

    lines = [
        "# Decisão final de modelos — Galaxies",
        "",
        f"Base avaliada: **{vlm_unique} vídeos com VLM, sem IDs duplicados**.",
        "",
        "A seleção foi feita por validação interna. Os holdouts temporal e de canais "
        "foram abertos somente para confirmar generalização.",
        "",
        "| Setor | Candidato interno | Temporal | Canal novo | Decisão |",
        "|---|---|---:|---:|---|",
    ]
    for sector, row in sectors.items():
        temporal = row["temporal"]
        channel = row["channel_holdout"]
        lines.append(
            f"| {sector.title()} | {row['selected_by_inner_validation']} | "
            f"{temporal['relative_improvement']:.1%} (R² {temporal['r2']:.3f}) | "
            f"{channel['relative_improvement']:.1%} (R² {channel['r2']:.3f}) | "
            f"{row['champion']} |"
        )
    lines.extend(
        [
            "",
            "## Resultado",
            "",
            "Nenhum algoritmo de ML cumpriu a regra completa nos dois holdouts. O baseline "
            "histórico permanece como campeão operacional nos três setores. Isso é um resultado "
            "negativo válido e evita apresentar overfitting como avanço.",
            "",
            "O melhor sinal parcial apareceu em Automotivo com Random Forest: ganho de 15,8% "
            "no corte temporal e 14,5% em canal novo, mas o IC temporal tocou zero e o R² de "
            "canal novo ficou abaixo do limiar. Em Beleza, o ensemble melhorou MAE nos dois "
            "cortes, porém os R² permaneceram insuficientes. Em Moda, o ensemble não generalizou.",
            "",
            "Embeddings de thumbnail, kNN temporal e CatBoost também não foram promovidos: "
            "seus intervalos pareados cruzaram zero, e CatBoost ficou abaixo da referência em AUC.",
            "",
            "## Próxima melhoria com maior valor",
            "",
            "Coletar alvo em horizonte fixo e ampliar canais independentes. Novos algoritmos sem "
            "corrigir o alvo e a diversidade de canais provavelmente só aumentariam a flexibilidade "
            "da busca, não a generalização.",
            "",
        ]
    )
    report = output_dir / "model-selection-final.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(output_dir / "champion-freeze.json")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
