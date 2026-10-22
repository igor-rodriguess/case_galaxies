"""Piloto de features de vídeo (15 s iniciais) extraídas localmente.

Etapas:

``sample``    sorteia até ``--per-channel`` Shorts por canal, só do treino do
              corte externo (o teste externo e o período prospectivo ficam fora).
``download``  baixa os 15 s iniciais em até 360p com áudio (yt-dlp + ffmpeg).
``extract``   fala (faster-whisper base, int8), frames (CLIP ViT-B/32 LAION)
              e ritmo (cortes de cena, movimento, brilho, volume).
``evaluate``  dobras internas de calendário do treino, comparando o mesmo
              modelo com e sem as features de vídeo.

Regra fixada antes de olhar o resultado: expandir a extração para os 22 mil
Shorts só se a melhor configuração com vídeo superar ``hgb_pct_base`` em pelo
menos 0,02 de AUC média por canal **e** o IC95% pareado por canal ficar acima
de zero. Caso contrário, o piloto é registrado como negativo.

Nenhuma feature usa views, likes ou comentários. O vídeo é o que foi publicado;
o rótulo é o as-of da Fase 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxies.agent.format_ranking import (  # noqa: E402
    channel_level,
    format_as_of_labels,
    inner_calendar_folds,
    outer_split,
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
from galaxies.paths import MODELING_RUNS_DIR  # noqa: E402

PILOT_DIR = ROOT / "data" / "video_pilot"
CLIPS = PILOT_DIR / "clips"
SAMPLE = PILOT_DIR / "sample.csv"
FEATURES_GLOB = "video_features_*.parquet"
FEATURE_MAP = MODELING_RUNS_DIR / "feature-map-shorts-operational-180s-20261021" / "feature_map.jsonl"
WINDOW_SECONDS = 15.0
FRAME_TIMES = (0.0, 1.0, 3.0, 7.0, 14.0)

# Sondas zero-shot do CLIP, fixadas antes do piloto.
CLIP_PROBES = {
    "probe_person_face": "a close-up of a person's face looking at the camera",
    "probe_text_overlay": "a video frame with large text captions on screen",
    "probe_product": "a close-up of a product",
    "probe_car": "a car",
    "probe_outdoor": "an outdoor scene",
}

SCALAR_VIDEO = [
    "has_speech",
    "speech_words",
    "words_per_second",
    "first_speech_seconds",
    "speech_ratio",
    "transcript_characters",
    "rms_mean",
    "rms_first_second",
    "cut_count",
    "first_cut_seconds",
    "motion_mean",
    "motion_first_3s",
    "brightness_mean",
    "saturation_mean",
    "clip_drift_0_14",
    *CLIP_PROBES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sample", "download", "extract", "evaluate"])
    parser.add_argument("--per-channel", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pause", type=float, default=15.0, help="Segundos entre downloads.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--run-id", default="video-pilot-20261021")
    parser.add_argument("--whisper", default="base")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    return parser.parse_args()


def outer_train() -> pd.DataFrame:
    dataset = load_dataset(FEATURE_MAP)
    rows = with_format_units(shorts_population(dataset.evaluable("full"), 180.0))
    train, _, _ = outer_split(rows)
    return train


# --- sample -----------------------------------------------------------------


def pick_sample(train: pd.DataFrame, per_channel: int, seed: int) -> pd.DataFrame:
    """Até ``per_channel`` Shorts por canal, sorteados com semente fixa.

    Embaralha uma vez e corta por canal: amostras menores são subconjuntos das
    maiores com a mesma semente, então reduzir o piloto não escolhe a dedo.
    """
    shuffled = train.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("source_channel_id").head(per_channel).loc[
        :, ["video_id", "source_channel_id", "sector", "published_at", "duration_seconds"]
    ]


def stage_sample(args: argparse.Namespace) -> None:
    picked = pick_sample(outer_train(), args.per_channel, args.seed)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    picked.to_csv(SAMPLE, index=False)
    print(f"{len(picked)} Shorts de {picked['source_channel_id'].nunique()} canais -> {SAMPLE}")
    print(picked.groupby("sector").size().to_string())


# --- download ---------------------------------------------------------------


def _download(video_id: str, ffmpeg: str) -> tuple[str, str]:
    target = CLIPS / f"{video_id}.mp4"
    if target.exists() and target.stat().st_size > 10_000:
        return video_id, "existente"
    command = [
        sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--ffmpeg-location", ffmpeg,
        "-f", "bv*+ba/b", "-S", "res:360,vcodec:h264,acodec:m4a", "--merge-output-format", "mp4",
        "--download-sections", f"*0-{int(WINDOW_SECONDS)}", "-o", str(CLIPS / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/shorts/{video_id}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    if result.returncode == 0 and target.exists():
        return video_id, "ok"
    message = (result.stderr or result.stdout).strip()
    return video_id, message.splitlines()[-1][:300] if message else "falhou"


BOT_CHECK = "confirm you"
BOT_CHECK_LIMIT = 3


def block_streak(streak: int, status: str) -> int:
    """Conta verificações de robô seguidas; qualquer outro resultado zera."""
    return streak + 1 if BOT_CHECK in status else 0


def stage_download(args: argparse.Namespace) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    CLIPS.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(SAMPLE)["video_id"].tolist()
    failures: dict[str, str] = {}
    done = 0
    blocked_in_row = 0
    pending = [video_id for video_id in ids if not (CLIPS / f"{video_id}.mp4").exists()]
    print(f"{len(ids) - len(pending)} já baixados; {len(pending)} pendentes.", flush=True)
    # Um por vez e com pausa: o YouTube bloqueia o IP com rajadas paralelas.
    # Três verificações de robô seguidas encerram a rodada; não há tentativa
    # de contornar o bloqueio.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for video_id in pending:
            _, status = pool.submit(_download, video_id, ffmpeg).result()
            done += 1
            if status != "ok":
                failures[video_id] = status
            blocked_in_row = block_streak(blocked_in_row, status)
            if blocked_in_row >= BOT_CHECK_LIMIT:
                print("YouTube pediu verificação de robô 3 vezes seguidas; parando.", flush=True)
                break
            if done % 50 == 0:
                print(f"{done}/{len(pending)} ({len(failures)} falhas)", flush=True)
            time.sleep(args.pause)
    (PILOT_DIR / "download_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    have = sum((CLIPS / f"{video_id}.mp4").exists() for video_id in ids)
    print(f"Concluído: {have}/{len(ids)} clipes no disco; {len(failures)} falhas nesta rodada.")


# --- extract ----------------------------------------------------------------


def _frames(path: Path) -> tuple[list[np.ndarray], dict[str, float]]:
    """Frames-chave para o CLIP e medidas de ritmo a ~6 fps."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / 6))
    keyframes: dict[float, np.ndarray] = {}
    small, times = [], []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        seconds = index / fps
        if seconds > WINDOW_SECONDS:
            break
        for target in FRAME_TIMES:
            if target not in keyframes and seconds >= target:
                keyframes[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if index % step == 0:
            small.append(cv2.cvtColor(cv2.resize(frame, (72, 128)), cv2.COLOR_BGR2HSV).astype(np.float32))
            times.append(seconds)
        index += 1
    capture.release()
    if not small:
        raise ValueError("vídeo sem frames")
    stack = np.stack(small)
    diffs = np.abs(np.diff(stack[..., 2], axis=0)).mean(axis=(1, 2)) / 255.0 if len(stack) > 1 else np.zeros(1)
    times_arr = np.asarray(times[1:]) if len(times) > 1 else np.zeros(1)
    last = keyframes[max(keyframes)] if keyframes else None
    ordered = [keyframes.get(t, last) for t in FRAME_TIMES]
    return ordered, {
        "motion_mean": float(diffs.mean()),
        "motion_first_3s": float(diffs[times_arr <= 3.0].mean()) if (times_arr <= 3.0).any() else float(diffs.mean()),
        "brightness_mean": float(stack[..., 2].mean() / 255.0),
        "saturation_mean": float(stack[..., 1].mean() / 255.0),
    }


def _cuts(path: Path) -> dict[str, float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(str(path), ContentDetector(threshold=27.0), end_time=WINDOW_SECONDS, show_progress=False)
    starts = [scene[0].seconds for scene in scenes[1:]]
    return {"cut_count": float(len(starts)), "first_cut_seconds": float(starts[0]) if starts else WINDOW_SECONDS}


def _speech(path: Path, model) -> dict[str, object]:
    import imageio_ffmpeg

    # Decodifica com o ffmpeg do imageio: o decode_audio do faster-whisper
    # quebra com o PyAV instalado.
    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-t", str(WINDOW_SECONDS),
         "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    rms_first = float(np.sqrt(np.mean(audio[:16000] ** 2))) if len(audio) else 0.0
    segments, info = model.transcribe(audio, beam_size=1, vad_filter=True, condition_on_previous_text=False)
    segments = list(segments)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    words = len(text.split())
    spoken = sum(max(0.0, segment.end - segment.start) for segment in segments)
    return {
        "transcript": text,
        "language": info.language,
        "has_speech": float(words > 0),
        "speech_words": float(words),
        "words_per_second": float(words / WINDOW_SECONDS),
        "first_speech_seconds": float(segments[0].start) if segments else WINDOW_SECONDS,
        "speech_ratio": float(min(1.0, spoken / WINDOW_SECONDS)),
        "transcript_characters": float(len(text)),
        "rms_mean": rms,
        "rms_first_second": rms_first,
    }


def stage_extract(args: argparse.Namespace) -> None:
    import open_clip
    import torch
    from faster_whisper import WhisperModel
    from PIL import Image

    torch.set_num_threads(args.threads)
    output = PILOT_DIR / f"video_features_{args.shard}.parquet"
    clip_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    clip_model.eval()
    with torch.no_grad():
        probe_vectors = clip_model.encode_text(tokenizer(list(CLIP_PROBES.values())))
        probe_vectors = torch.nn.functional.normalize(probe_vectors, dim=-1)
    whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8", cpu_threads=args.threads)

    done = pd.read_parquet(output) if output.exists() else pd.DataFrame(columns=["video_id"])
    seen = set(done["video_id"])
    sample = pd.read_csv(SAMPLE)["video_id"].tolist()
    mine = [vid for position, vid in enumerate(sample) if position % args.shards == args.shard]
    ids = [vid for vid in mine if vid not in seen and (CLIPS / f"{vid}.mp4").exists()]
    print(f"{len(ids)} clipes a extrair ({len(seen)} já feitos).", flush=True)
    records, errors = [], {}
    started = time.time()
    for position, video_id in enumerate(ids, 1):
        path = CLIPS / f"{video_id}.mp4"
        try:
            frames, rhythm = _frames(path)
            with torch.no_grad():
                batch = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames])
                vectors = torch.nn.functional.normalize(clip_model.encode_image(batch), dim=-1)
            probes = (vectors[0:1] @ probe_vectors.T).squeeze(0).numpy()
            record = {
                "video_id": video_id,
                **rhythm,
                **_cuts(path),
                **_speech(path, whisper),
                "clip_first": vectors[0].numpy().astype(np.float32),
                "clip_mean": torch.nn.functional.normalize(vectors.mean(0), dim=-1).numpy().astype(np.float32),
                "clip_drift_0_14": float(1.0 - (vectors[0] @ vectors[-1]).item()),
                **{name: float(value) for name, value in zip(CLIP_PROBES, probes, strict=True)},
            }
            records.append(record)
        except Exception as error:
            errors[video_id] = repr(error)[:300]
        if position % 50 == 0 or position == len(ids):
            frame = pd.concat([done, pd.DataFrame(records)], ignore_index=True)
            frame.to_parquet(output, index=False)
            rate = (time.time() - started) / position
            print(f"{position}/{len(ids)} · {rate:.1f} s/clipe · {len(errors)} erros", flush=True)
    (PILOT_DIR / f"extract_errors_{args.shard}.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")


# --- evaluate ---------------------------------------------------------------


def _percentile(train: pd.DataFrame) -> np.ndarray:
    return train.groupby("channel_id")["adjusted_performance"].rank(pct=True).to_numpy()


def _design(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], clip_pca: int, seed: int):
    from sklearn.decomposition import PCA

    from galaxies.modeling.preprocessing import make_preprocessor

    categorical = [*list(METADATA_CATEGORICAL), "sector"]
    pre = make_preprocessor({"features": {"numeric": numeric, "categorical": categorical}})
    x_train = pre.fit_transform(train[numeric + categorical])
    x_test = pre.transform(test[numeric + categorical])
    x_train = np.asarray(x_train.todense()) if hasattr(x_train, "todense") else np.asarray(x_train)
    x_test = np.asarray(x_test.todense()) if hasattr(x_test, "todense") else np.asarray(x_test)
    if clip_pca:
        pca = PCA(n_components=clip_pca, random_state=seed).fit(np.stack(train["clip_mean"].to_numpy()))
        x_train = np.hstack([x_train, pca.transform(np.stack(train["clip_mean"].to_numpy()))])
        x_test = np.hstack([x_test, pca.transform(np.stack(test["clip_mean"].to_numpy()))])
    return x_train, x_test


CONFIGS = {
    # nome: (modelo, usa vídeo escalar, componentes PCA do CLIP)
    "reference_logistic_top10": ("logistic", False, 0),
    "hgb_pct_base": ("hgb", False, 0),
    "hgb_pct_video": ("hgb", True, 0),
    "hgb_pct_video_clip16": ("hgb", True, 16),
    "logistic_top10_video": ("logistic", True, 0),
}


def _score(name: str, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    model, video, clip_pca = CONFIGS[name]
    numeric = list(METADATA_NUMERIC) + list(CADENCE_NUMERIC) + (SCALAR_VIDEO if video else [])
    x_train, x_test = _design(train, test, numeric, clip_pca, seed)
    if model == "logistic":
        estimator = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(x_train, train["viral"].to_numpy())
        return estimator.predict_proba(x_test)[:, 1]
    estimator = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30, l2_regularization=1.0, random_state=seed
    ).fit(x_train, _percentile(train))
    return estimator.predict(x_test)


VIDEO_CONFIGS = ("hgb_pct_video", "hgb_pct_video_clip16")
MIN_GAIN = 0.02


def expansion_decision(table: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Regra fixada antes do piloto: ganho >= 0,02 e IC95% pareado acima de zero."""
    video_rows = table.loc[table["config"].isin(VIDEO_CONFIGS)]
    best = video_rows.sort_values("auc_macro_channel", ascending=False).iloc[0]
    return best, bool(best["delta_vs_hgb_pct_base"] >= MIN_GAIN and best["delta_ci_low"] > 0)


def stage_evaluate(args: argparse.Namespace) -> None:
    features = pd.concat([pd.read_parquet(path) for path in sorted(PILOT_DIR.glob(FEATURES_GLOB))], ignore_index=True)
    features = features.drop_duplicates("video_id", keep="last")
    train = outer_train()
    output = MODELING_RUNS_DIR / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    pilot_ids = set(features["video_id"])
    totals: dict[str, pd.DataFrame] = {}
    sizes = []
    for fold, (inner_train, inner_valid) in enumerate(inner_calendar_folds(train)):
        # Rótulo e limiar com todo o treino da dobra; só depois restringe ao piloto.
        labelled_train, labelled_valid = format_as_of_labels(inner_train, inner_valid)
        fold_train = labelled_train.loc[labelled_train["video_id"].isin(pilot_ids)].merge(features, on="video_id")
        fold_valid = labelled_valid.loc[labelled_valid["video_id"].isin(pilot_ids)].merge(features, on="video_id")
        sizes.append({"fold": fold, "train": len(fold_train), "valid": len(fold_valid), "valid_positives": int(fold_valid["viral"].sum())})
        for name in CONFIGS:
            scored = fold_valid[["channel_id", "source_channel_id", "sector", "viral"]].copy()
            scored["score"] = _score(name, fold_train, fold_valid, args.seed)
            counts = channel_level(unit_counts(scored, "score"))[["pairs", "correct"]]
            totals[name] = counts if name not in totals else totals[name].add(counts, fill_value=0)
        print(f"dobra {fold}: treino {len(fold_train)}, validação {len(fold_valid)}", flush=True)

    def macro(counts: pd.DataFrame) -> float:
        valid = counts.loc[counts["pairs"] > 0]
        return float((valid["correct"] / valid["pairs"]).mean())

    base = totals["hgb_pct_base"]
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, counts in totals.items():
        aligned = counts.reindex(base.index).fillna(0)
        keys = base.index[(base["pairs"] > 0) & (aligned["pairs"] > 0)]
        deltas, aucs = [], []
        for _ in range(args.bootstrap):
            pick = keys[rng.integers(0, len(keys), len(keys))]
            aucs.append(macro(aligned.loc[pick]))
            deltas.append(macro(aligned.loc[pick]) - macro(base.loc[pick]))
        rows.append(
            {
                "config": name,
                "auc_macro_channel": macro(aligned.loc[keys]),
                "auc_ci_low": float(np.quantile(aucs, 0.025)),
                "auc_ci_high": float(np.quantile(aucs, 0.975)),
                "delta_vs_hgb_pct_base": macro(aligned.loc[keys]) - macro(base.loc[keys]),
                "delta_ci_low": float(np.quantile(deltas, 0.025)),
                "delta_ci_high": float(np.quantile(deltas, 0.975)),
                "channels": len(keys),
            }
        )
    table = pd.DataFrame(rows)
    best, expand = expansion_decision(table)
    report = {
        "protocol": "video-pilot-inner-folds-v1",
        "pilot_videos": len(features),
        "folds": sizes,
        "results": rows,
        "decision_rule": "Expandir se a melhor config com vídeo ganhar >= 0,02 de AUC média por canal sobre hgb_pct_base com IC95% pareado acima de zero.",
        "best_video_config": str(best["config"]),
        "expand_to_full_dataset": expand,
    }
    table.to_csv(output / "pilot_results.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Decisão: {'expandir' if expand else 'não expandir'} (melhor com vídeo: {best['config']})")


def main() -> int:
    args = parse_args()
    {"sample": stage_sample, "download": stage_download, "extract": stage_extract, "evaluate": stage_evaluate}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
