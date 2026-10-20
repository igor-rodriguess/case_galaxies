"""Auditoria de qualidade para respostas VLM antes da modelagem.

Não altera resultados nem preenche ausências. O objetivo é detectar JSON parcial,
taxonomia fora do contrato e timestamps fora do trecho solicitado.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

TAXONOMY: dict[tuple[str, ...], set[str]] = {
    ("visual", "shot_and_camera", "framing"): {
        "pov", "extreme_close_up", "close_up", "medium_close_up", "medium", "wide", "other", "unknown"
    },
    ("visual", "shot_and_camera", "camera_movement"): {
        "static", "handheld", "zoom", "moving", "cut", "dynamic", "other", "unknown"
    },
    ("visual", "motion_intensity", "label"): {"low", "medium", "high", "unknown"},
    ("interpretation", "opening_type", "label"): {
        "question", "statement", "demonstration", "product_reveal", "person_speaking", "montage", "text_hook", "other", "unknown"
    },
    ("interpretation", "product_prominence", "label"): {
        "none", "low", "medium", "high", "unknown"
    },
}

V02_REQUIRED_PATHS = (
    ("analysis_scope",),
    ("visual", "first_visible_element"),
    ("visual", "people"),
    ("visual", "product_or_object_in_focus"),
    ("visual", "on_screen_text"),
    ("visual", "setting"),
    ("visual", "shot_and_camera", "framing"),
    ("visual", "shot_and_camera", "camera_movement"),
    ("visual", "scene_changes"),
    ("visual", "motion_intensity"),
    ("audio", "speech_present"),
    ("audio", "transcript_excerpt"),
    ("audio", "language"),
    ("audio", "music_present"),
    ("audio", "sound_effects"),
    ("interpretation", "opening_type", "label"),
    ("interpretation", "product_prominence", "label"),
    ("interpretation", "call_to_action", "present"),
    ("interpretation", "sector_specific"),
    ("evidence",),
    ("uncertainties",),
)


def nested(value: object, path: tuple[str, ...]) -> object:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def has_path(value: object, path: tuple[str, ...]) -> bool:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_result(result: dict[str, Any]) -> list[dict[str, str]]:
    """Retorna problemas objetivos de um resultado individual."""
    identity = result.get("identity") or {}
    provenance = result.get("provenance") or {}
    extractor = provenance.get("extractor") or {}
    prompt_version = str(extractor.get("prompt_version") or "unknown")
    video_id = str(identity.get("video_id") or "unknown")
    analysis = result.get("analysis") or {}
    issues: list[dict[str, str]] = []

    if prompt_version.startswith("galaxies-opening-v0.2"):
        if nested(result, ("validation", "status")) != "complete":
            issues.append({"video_id": video_id, "kind": "missing_validation_seal", "detail": prompt_version})
        for path in V02_REQUIRED_PATHS:
            if not has_path(analysis, path):
                issues.append({"video_id": video_id, "kind": "missing_required_path", "detail": ".".join(path)})

    for path, allowed in TAXONOMY.items():
        value = nested(analysis, path)
        if value is not None and str(value) not in allowed:
            issues.append(
                {"video_id": video_id, "kind": "taxonomy_drift", "detail": f"{'.'.join(path)}={value}"}
            )

    clip = provenance.get("clip") or {}
    start = _as_float(clip.get("start_seconds"))
    end = _as_float(clip.get("end_seconds"))
    if start is not None and end is not None:
        timestamp_paths = (
            ("visual", "on_screen_text"),
            ("visual", "scene_changes"),
            ("evidence",),
        )
        for path in timestamp_paths:
            items = nested(analysis, path)
            if not isinstance(items, list):
                continue
            for position, item in enumerate(items):
                if not isinstance(item, dict):
                    issues.append({"video_id": video_id, "kind": "invalid_evidence_item", "detail": ".".join(path)})
                    continue
                timestamp = _as_float(item.get("timestamp_seconds"))
                if timestamp is None or not start <= timestamp <= end:
                    issues.append(
                        {
                            "video_id": video_id,
                            "kind": "timestamp_outside_clip",
                            "detail": f"{'.'.join(path)}[{position}]={item.get('timestamp_seconds')}",
                        }
                    )

    speech = nested(analysis, ("audio", "speech_present"))
    transcript = nested(analysis, ("audio", "transcript_excerpt"))
    if speech is False and transcript not in (None, ""):
        issues.append({"video_id": video_id, "kind": "speech_transcript_conflict", "detail": "speech=false"})
    return issues


def audit_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Agrega auditoria de todos os resultados VLM mais recentes por vídeo."""
    issues: list[dict[str, str]] = []
    versions: Counter[str] = Counter()
    issues_by_version: Counter[str] = Counter()
    for result in results.values():
        version = str(nested(result, ("provenance", "extractor", "prompt_version")) or "unknown")
        versions[version] += 1
        result_issues = audit_result(result)
        issues_by_version[version] += len(result_issues)
        issues.extend(result_issues)
    issue_counts = Counter(issue["kind"] for issue in issues)
    affected = {issue["video_id"] for issue in issues}
    return {
        "results_audited": len(results),
        "results_with_issues": len(affected),
        "prompt_versions": dict(sorted(versions.items())),
        "issues_by_prompt_version": dict(sorted(issues_by_version.items())),
        "issues_by_kind": dict(sorted(issue_counts.items())),
        "issues": issues,
    }
