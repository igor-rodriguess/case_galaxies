"""Constrói um mapa tabular rastreável a partir dos JSONs existentes.

Esta camada não treina modelos. Ela conecta os registros curados do YouTube aos
resultados mais recentes do extrator multimodal usando ``video_id`` e explicita
quando uma variável ainda não é temporalmente válida para treino.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import Any

from .config import SECTORS, load_sector_config
from .paths import FEATURE_RUNS_DIR, MODELING_RUNS_DIR, MODELS_DIR, YOUTUBE_RUNS_DIR

FEATURE_MAP_SCHEMA_VERSION = "0.1.0"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido em {path}: {exc.msg}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"O documento JSON em {path} deve ser um objeto.")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}:{line_number}") from exc
    return rows


def parse_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    return None if number is None else int(number)


def nested(value: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def latest_youtube_run(runs_dir: Path = YOUTUBE_RUNS_DIR) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Diretório de coletas não encontrado: {runs_dir}")
    candidates = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "curated" / "videos").exists()
    )
    if not candidates:
        raise FileNotFoundError("Nenhuma coleta curada do YouTube foi encontrada.")
    return candidates[-1]


def load_youtube_documents(run_dir: Path | None = None) -> list[dict[str, Any]]:
    selected_run = run_dir or latest_youtube_run()
    video_dir = selected_run / "curated" / "videos"
    documents: list[dict[str, Any]] = []
    for path in sorted(video_dir.rglob("*.json")):
        document = read_json(path)
        video_id = nested(document, "identity", "video_id")
        channel_id = nested(document, "identity", "channel_id")
        sector = nested(document, "classification", "sector")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"Registro sem video_id válido: {path}")
        if not isinstance(channel_id, str) or not channel_id.strip():
            raise ValueError(f"Registro sem channel_id válido: {path}")
        if sector not in SECTORS:
            raise ValueError(f"Registro com setor inválido em {path}: {sector!r}")
        document["_source_path"] = path
        documents.append(document)
    ids = [nested(document, "identity", "video_id") for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError(f"A coleta {selected_run.name} contém video_id duplicado.")
    return documents


def load_latest_vlm_results(
    runs_dir: Path = FEATURE_RUNS_DIR,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not runs_dir.exists():
        return latest
    for path in sorted(runs_dir.rglob("opening_vlm.json")):
        try:
            result = read_json(path)
        except OSError as exc:
            raise OSError(f"Não foi possível ler o resultado VLM {path}.") from exc
        video_id = nested(result, "identity", "video_id")
        if not video_id:
            raise ValueError(f"Resultado VLM sem video_id: {path}")
        completed_at = nested(result, "provenance", "completed_at", default="")
        completed_timestamp = parse_datetime(completed_at)
        completed_key = (
            completed_timestamp or datetime.min.replace(tzinfo=UTC),
            str(path),
        )
        previous = latest.get(video_id)
        if previous is None or completed_key > previous["_completed_key"]:
            result["_source_path"] = path
            result["_completed_at"] = completed_at
            result["_completed_key"] = completed_key
            latest[video_id] = result
    return latest


def _days_between(start: Any, end: Any) -> float | None:
    start_dt = parse_datetime(start)
    end_dt = parse_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds() / 86_400


def _extract_target(document: dict[str, Any]) -> dict[str, Any]:
    """Prefere um horizonte fixo quando o futuro esquema passar a fornecê-lo."""
    horizons = document.get("metrics_horizons") or document.get("metrics_by_horizon")
    if isinstance(horizons, dict):
        for key in ("day_30", "30_days", "30"):
            candidate = horizons.get(key)
            if isinstance(candidate, dict) and candidate.get("view_count") is not None:
                return {
                    "target_view_count": optional_int(candidate.get("view_count")),
                    "target_observed_at": candidate.get("observed_at"),
                    "target_horizon_days": 30,
                    "target_is_fixed_horizon": True,
                    "target_source": f"metrics_horizons.{key}",
                }

    snapshot = document.get("metrics_snapshot", {})
    return {
        "target_view_count": optional_int(snapshot.get("view_count")),
        "target_observed_at": snapshot.get("observed_at"),
        "target_horizon_days": None,
        "target_is_fixed_horizon": False,
        "target_source": "metrics_snapshot",
    }


def _word_count(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    return len(text.split()) if text else 0


def _mean_confidence(analysis: dict[str, Any]) -> float | None:
    paths = (
        ("visual", "first_visible_element", "confidence"),
        ("visual", "people", "confidence"),
        ("visual", "product_or_object_in_focus", "confidence"),
        ("visual", "setting", "confidence"),
        ("visual", "shot_and_camera", "confidence"),
        ("visual", "motion_intensity", "confidence"),
        ("audio", "confidence"),
        ("interpretation", "opening_type", "confidence"),
        ("interpretation", "product_prominence", "confidence"),
        ("interpretation", "call_to_action", "confidence"),
    )
    values = [optional_float(nested(analysis, *path)) for path in paths]
    observed = [value for value in values if value is not None]
    return mean(observed) if observed else None


def _empty_vlm_features() -> dict[str, Any]:
    return {
        "vlm_available": False,
        "vlm_prompt_version": None,
        "vlm_model_version": None,
        "vlm_interval_start_seconds": None,
        "vlm_interval_end_seconds": None,
        "vlm_requested_fps": None,
        "vlm_mean_confidence": None,
        "people_present": None,
        "people_count_estimate": None,
        "faces_visible": None,
        "product_in_focus": None,
        "brand_legible": None,
        "on_screen_text_count": None,
        "setting_observed": None,
        "framing": None,
        "camera_movement": None,
        "scene_change_count": None,
        "motion_intensity": None,
        "speech_present": None,
        "speech_start_seconds": None,
        "transcript_excerpt_word_count": None,
        "audio_language": None,
        "music_present": None,
        "sound_effect_count": None,
        "opening_type": None,
        "product_prominence": None,
        "call_to_action_present": None,
        "evidence_count": None,
        "uncertainty_count": None,
        "beauty_swatch_present": None,
        "beauty_skin_application_present": None,
        "beauty_before_after_present": None,
        "fashion_try_on_present": None,
        "fashion_full_body_present": None,
        "fashion_garment_detail_present": None,
        "automotive_car_closeup_present": None,
        "automotive_engine_sound_present": None,
        "automotive_driving_pov_present": None,
        "automotive_interior_present": None,
        "vlm_result_path": None,
    }


def _extract_vlm_features(result: dict[str, Any] | None) -> dict[str, Any]:
    features = _empty_vlm_features()
    if result is None:
        return features

    analysis = result.get("analysis", {})
    text_items = nested(analysis, "visual", "on_screen_text", default=[]) or []
    scene_changes = nested(analysis, "visual", "scene_changes", default=[]) or []
    sound_effects = nested(analysis, "audio", "sound_effects", default=[]) or []
    evidence = analysis.get("evidence") or []
    uncertainties = analysis.get("uncertainties") or []
    brand = nested(
        analysis,
        "visual",
        "product_or_object_in_focus",
        "brand_if_legible",
    )
    sector_specific = nested(
        analysis,
        "interpretation",
        "sector_specific",
        default={},
    )
    if not isinstance(sector_specific, dict):
        sector_specific = {}

    source_path = result.get("_source_path")
    features.update(
        {
            "vlm_available": True,
            "vlm_prompt_version": nested(result, "provenance", "extractor", "prompt_version"),
            "vlm_model_version": nested(result, "response_metadata", "model_version")
            or nested(result, "provenance", "extractor", "model_requested"),
            "vlm_interval_start_seconds": nested(result, "provenance", "clip", "start_seconds"),
            "vlm_interval_end_seconds": nested(result, "provenance", "clip", "end_seconds"),
            "vlm_requested_fps": nested(result, "provenance", "clip", "requested_fps"),
            "vlm_mean_confidence": _mean_confidence(analysis),
            "people_present": nested(analysis, "visual", "people", "present"),
            "people_count_estimate": optional_int(
                nested(analysis, "visual", "people", "count_estimate")
            ),
            "faces_visible": nested(analysis, "visual", "people", "faces_visible"),
            "product_in_focus": nested(analysis, "visual", "product_or_object_in_focus", "present"),
            "brand_legible": bool(brand) if brand is not None else None,
            "on_screen_text_count": len(text_items),
            "setting_observed": bool(nested(analysis, "visual", "setting", "description")),
            "framing": nested(analysis, "visual", "shot_and_camera", "framing"),
            "camera_movement": nested(analysis, "visual", "shot_and_camera", "camera_movement"),
            "scene_change_count": len(scene_changes),
            "motion_intensity": nested(analysis, "visual", "motion_intensity", "label"),
            "speech_present": nested(analysis, "audio", "speech_present"),
            "speech_start_seconds": optional_float(
                nested(analysis, "audio", "speech_start_seconds")
            ),
            "transcript_excerpt_word_count": _word_count(
                nested(analysis, "audio", "transcript_excerpt")
            ),
            "audio_language": nested(analysis, "audio", "language"),
            "music_present": nested(analysis, "audio", "music_present"),
            "sound_effect_count": len(sound_effects),
            "opening_type": nested(analysis, "interpretation", "opening_type", "label"),
            "product_prominence": nested(analysis, "interpretation", "product_prominence", "label"),
            "call_to_action_present": nested(
                analysis, "interpretation", "call_to_action", "present"
            ),
            "evidence_count": len(evidence),
            "uncertainty_count": len(uncertainties),
            "beauty_swatch_present": sector_specific.get("beauty_swatch_present"),
            "beauty_skin_application_present": sector_specific.get(
                "beauty_skin_application_present"
            ),
            "beauty_before_after_present": sector_specific.get("beauty_before_after_present"),
            "fashion_try_on_present": sector_specific.get("fashion_try_on_present"),
            "fashion_full_body_present": sector_specific.get("fashion_full_body_present"),
            "fashion_garment_detail_present": sector_specific.get("fashion_garment_detail_present"),
            "automotive_car_closeup_present": sector_specific.get("automotive_car_closeup_present"),
            "automotive_engine_sound_present": sector_specific.get(
                "automotive_engine_sound_present"
            ),
            "automotive_driving_pov_present": sector_specific.get("automotive_driving_pov_present"),
            "automotive_interior_present": sector_specific.get("automotive_interior_present"),
            "vlm_result_path": str(source_path) if source_path else None,
        }
    )
    return features


def _base_feature_row(
    document: dict[str, Any],
    vlm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = document.get("identity", {})
    publication = document.get("publication", {})
    details = document.get("content_details", {})
    channel = document.get("channel_snapshot", {})
    prediction_channel = document.get("channel_stats_at_prediction", {})
    classification = document.get("classification", {})
    snapshot = document.get("metrics_snapshot", {})
    published = parse_datetime(publication.get("published_at"))
    target = _extract_target(document)
    source_path = document.get("_source_path")

    row: dict[str, Any] = {
        "schema_version": FEATURE_MAP_SCHEMA_VERSION,
        "video_id": identity.get("video_id"),
        "channel_id": identity.get("channel_id"),
        "sector": classification.get("sector"),
        "published_at": publication.get("published_at"),
        "collected_at": nested(document, "provenance", "collected_at"),
        "youtube_record_path": str(source_path) if source_path else None,
        "duration_seconds": optional_float(details.get("duration_seconds")),
        "definition": details.get("definition"),
        "category_id": publication.get("category_id"),
        "title_character_count": len(publication.get("title") or ""),
        "description_character_count": len(publication.get("description") or ""),
        "tag_count": len(publication.get("tags") or []),
        "publish_hour_utc": published.hour if published else None,
        "publish_weekday_utc": published.weekday() if published else None,
        "channel_subscriber_count_snapshot": optional_float(
            channel.get("subscriber_count_observed_at_collection")
        ),
        "channel_video_count_snapshot": optional_float(
            channel.get("video_count_observed_at_collection")
        ),
        "channel_subscriber_count_at_prediction": optional_float(
            prediction_channel.get("subscriber_count")
            if isinstance(prediction_channel, dict)
            else None
        ),
        "channel_video_count_at_prediction": optional_float(
            prediction_channel.get("video_count") if isinstance(prediction_channel, dict) else None
        ),
        "like_count_snapshot": optional_float(snapshot.get("like_count")),
        "comment_count_snapshot": optional_float(snapshot.get("comment_count")),
        "target_performance_band": None,
    }
    row.update(target)
    row["target_age_days"] = _days_between(
        publication.get("published_at"), row.get("target_observed_at")
    )
    row.update(_extract_vlm_features(vlm_result))
    return row


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = [optional_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def _add_channel_history(rows: list[dict[str, Any]]) -> None:
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_channel[str(row.get("channel_id"))].append(row)

    for channel_rows in by_channel.values():
        ordered = sorted(
            channel_rows,
            key=lambda row: (
                parse_datetime(row.get("published_at")) or datetime.min.replace(tzinfo=UTC)
            ),
        )
        accumulated: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(ordered):
            current_time = parse_datetime(ordered[cursor].get("published_at"))
            same_time: list[dict[str, Any]] = []
            while cursor < len(ordered):
                candidate_time = parse_datetime(ordered[cursor].get("published_at"))
                if candidate_time != current_time:
                    break
                same_time.append(ordered[cursor])
                cursor += 1

            views = _numeric_values(accumulated, "target_view_count")
            likes = _numeric_values(accumulated, "like_count_snapshot")
            comments = _numeric_values(accumulated, "comment_count_snapshot")
            published_times = [parse_datetime(row.get("published_at")) for row in accumulated]
            published_times = [value for value in published_times if value is not None]
            intervals = [
                (right - left).total_seconds() / 86_400 for left, right in pairwise(published_times)
            ]
            history_valid: bool | None = None
            if accumulated and current_time is not None:
                observed_times = [
                    parse_datetime(row.get("target_observed_at")) for row in accumulated
                ]
                history_valid = all(
                    observed is not None and observed <= current_time for observed in observed_times
                )

            for row in same_time:
                prior_median = median(views) if views else None
                row.update(
                    {
                        "prior_video_count": len(accumulated),
                        "prior_views_mean_observed": mean(views) if views else None,
                        "prior_views_median_observed": prior_median,
                        "prior_likes_median_observed": median(likes) if likes else None,
                        "prior_comments_median_observed": (median(comments) if comments else None),
                        "prior_upload_interval_median_days": (
                            median(intervals) if intervals else None
                        ),
                        "days_since_previous_upload": (
                            (current_time - published_times[-1]).total_seconds() / 86_400
                            if current_time is not None and published_times
                            else None
                        ),
                        "channel_history_temporally_valid": history_valid,
                        "target_relative_to_prior_median": (
                            row["target_view_count"] / prior_median
                            if prior_median and row.get("target_view_count") is not None
                            else None
                        ),
                    }
                )
            accumulated.extend(same_time)


def build_feature_rows(
    youtube_run: Path | None = None,
    feature_runs_dir: Path = FEATURE_RUNS_DIR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_run = youtube_run or latest_youtube_run()
    documents = load_youtube_documents(selected_run)
    vlm_results = load_latest_vlm_results(feature_runs_dir)
    rows = []
    for document in documents:
        video_id = nested(document, "identity", "video_id")
        rows.append(_base_feature_row(document, vlm_results.get(video_id)))
    _add_channel_history(rows)
    rows.sort(key=lambda row: (row.get("sector") or "", row.get("published_at") or ""))

    manifest = {
        "schema_version": FEATURE_MAP_SCHEMA_VERSION,
        "created_at": utc_now(),
        "source": {
            "youtube_run": str(selected_run),
            "feature_runs_directory": str(feature_runs_dir),
            "join_key": "video_id",
        },
        "record_count": len(rows),
        "unique_video_count": len({row.get("video_id") for row in rows}),
        "vlm_result_count": sum(bool(row.get("vlm_available")) for row in rows),
        "notes": [
            "O mapa é achatado para modelagem, mas preserva referências às fontes.",
            "Targets de snapshot são exploratórios e não equivalem ao dia 30.",
            "Histórico só é válido quando suas métricas foram observadas antes do alvo.",
        ],
    }
    return rows, manifest


def audit_feature_rows(
    rows: list[dict[str, Any]],
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    duplicates = len(rows) - len({row.get("video_id") for row in rows})
    sectors: dict[str, Any] = {}
    for sector in SECTORS:
        selected = [row for row in rows if row.get("sector") == sector]
        config = load_sector_config(sector, models_dir)
        minimum = config["minimum_data"]
        target_rows = [row for row in selected if row.get("target_view_count") is not None]
        fixed_rows = [row for row in target_rows if row.get("target_is_fixed_horizon")]
        history_rows = [row for row in selected if (row.get("prior_video_count") or 0) > 0]
        valid_history = [
            row for row in history_rows if row.get("channel_history_temporally_valid") is True
        ]
        invalid_history_count = len(history_rows) - len(valid_history)
        vlm_rows = [row for row in selected if row.get("vlm_available")]
        channels = {row.get("channel_id") for row in selected if row.get("channel_id")}
        blockers: list[str] = []
        if len(selected) < minimum["rows"]:
            blockers.append(f"{len(selected)} linhas; mínimo configurado: {minimum['rows']}.")
        if len(channels) < minimum["channels"]:
            blockers.append(f"{len(channels)} canais; mínimo configurado: {minimum['channels']}.")
        if config["target"]["requires_fixed_horizon"] and len(fixed_rows) < len(selected):
            blockers.append("Nem todas as linhas têm visualizações observadas em horizonte fixo.")
        coverage = len(vlm_rows) / len(selected) if selected else 0.0
        if coverage < minimum["vlm_coverage"]:
            blockers.append(f"Cobertura VLM {coverage:.1%}; mínimo: {minimum['vlm_coverage']:.1%}.")
        if not history_rows:
            blockers.append(
                "Nenhuma linha possui vídeos anteriores para formar histórico do canal."
            )
        elif invalid_history_count:
            blockers.append(
                f"{invalid_history_count} linhas possuem histórico temporalmente inválido."
            )
        sectors[sector] = {
            "ready_for_final_training": not blockers,
            "rows": len(selected),
            "channels": len(channels),
            "rows_with_target": len(target_rows),
            "rows_with_fixed_horizon_target": len(fixed_rows),
            "rows_with_vlm": len(vlm_rows),
            "vlm_coverage": coverage,
            "rows_with_temporally_valid_history": len(valid_history),
            "rows_with_temporally_invalid_history": invalid_history_count,
            "blocking_reasons": blockers,
        }

    return {
        "schema_version": FEATURE_MAP_SCHEMA_VERSION,
        "audited_at": utc_now(),
        "records": len(rows),
        "duplicate_video_ids": duplicates,
        "sectors": sectors,
        "ready_for_final_training": duplicates == 0
        and all(item["ready_for_final_training"] for item in sectors.values()),
    }


def create_modeling_run(
    run_id: str | None = None,
    output_root: Path = MODELING_RUNS_DIR,
    youtube_run: Path | None = None,
    feature_runs_dir: Path = FEATURE_RUNS_DIR,
    models_dir: Path = MODELS_DIR,
) -> tuple[Path, dict[str, Any]]:
    generated_id = "feature-map-" + utc_now().replace(":", "-")
    selected_id = run_id or generated_id
    cleaned = selected_id.replace("-", "").replace("_", "").replace(".", "")
    if not cleaned.isalnum() or selected_id in {".", ".."}:
        raise ValueError("run_id contém caracteres inválidos.")

    rows, manifest = build_feature_rows(youtube_run, feature_runs_dir)
    audit = audit_feature_rows(rows, models_dir)
    run_dir = output_root / selected_id
    if run_dir.exists():
        raise FileExistsError(
            f"A execução {selected_id!r} já existe em {run_dir}; escolha outro run_id."
        )
    write_jsonl(run_dir / "feature_map.jsonl", rows)
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "readiness_audit.json", audit)
    return run_dir, audit


def latest_feature_map(runs_dir: Path = MODELING_RUNS_DIR) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(
            "Nenhum mapa de features existe. Execute build_modeling_dataset.py."
        )
    candidates = list(runs_dir.glob("*/feature_map.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            "Nenhum mapa de features existe. Execute build_modeling_dataset.py."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)
