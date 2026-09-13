from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from .pipeline import AnalysisPipeline, PipelineConfig


MAX_UPLOAD_BYTES = int(os.getenv("MEDIA_TIMELINE_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


app = FastAPI(title="Media Timeline Service", version="0.1.0")
pipeline = AnalysisPipeline(
    PipelineConfig(
        asr_backend=os.getenv("MEDIA_TIMELINE_ASR_BACKEND", "whisper"),
        verbatim_model=os.getenv("MEDIA_TIMELINE_VERBATIM_MODEL", "nyralabs/CrisperWhisper2.0_large"),
        alignment=os.getenv("MEDIA_TIMELINE_ALIGNMENT", "none"),
        delivery=os.getenv("MEDIA_TIMELINE_DELIVERY", "acoustic"),
        delivery_model=os.getenv("MEDIA_TIMELINE_DELIVERY_MODEL", "Qwen/Qwen2.5-Omni-3B"),
        language=os.getenv("MEDIA_TIMELINE_LANGUAGE", "en"),
        device=os.getenv("MEDIA_TIMELINE_DEVICE", "cpu"),
        whisper_model=os.getenv("MEDIA_TIMELINE_WHISPER_MODEL", "small.en"),
        model_cache=Path(os.getenv("MEDIA_TIMELINE_MODEL_CACHE", ".cache/media-timeline")),
        use_vocal_separation=_env_bool("MEDIA_TIMELINE_USE_VOCAL_SEPARATION", True),
        use_audio_events=_env_bool("MEDIA_TIMELINE_USE_AUDIO_EVENTS", True),
        use_scene_detection=_env_bool("MEDIA_TIMELINE_USE_SCENE_DETECTION", True),
        offline=_env_bool("MEDIA_TIMELINE_OFFLINE", False),
        silence_threshold_db=int(os.getenv("MEDIA_TIMELINE_SILENCE_THRESHOLD_DB", "-42")),
        scene_threshold=float(os.getenv("MEDIA_TIMELINE_SCENE_THRESHOLD", "0.24")),
        event_window_seconds=float(os.getenv("MEDIA_TIMELINE_EVENT_WINDOW_SECONDS", "2.0")),
        event_threshold=float(os.getenv("MEDIA_TIMELINE_EVENT_THRESHOLD", "0.07")),
        event_hop_seconds=float(os.getenv("MEDIA_TIMELINE_EVENT_HOP_SECONDS", "0.5")),
        vad_filter=_env_bool("MEDIA_TIMELINE_VAD_FILTER", False),
        speech_source=os.getenv("MEDIA_TIMELINE_SPEECH_SOURCE", "both"),
    )
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/analyze")
def analyze_video(
    video: UploadFile = File(...),
    segments_only: bool = Query(False),
    events_only: bool = Query(False),
    context_text: str | None = Form(None),
) -> dict | list:
    requested_suffix = Path(video.filename or "upload.mp4").suffix.lower()
    suffix = requested_suffix if requested_suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"} else ".bin"
    try:
        with tempfile.TemporaryDirectory(prefix="media-timeline-upload-") as directory:
            source = Path(directory) / f"input{suffix}"
            copied = 0
            with source.open("wb") as output:
                while chunk := video.file.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Video exceeds the upload-size limit")
                    output.write(chunk)
            result = pipeline.analyze(source, context_text=context_text).to_dict()
            if events_only:
                return result["events"]
            return result["segments"] if segments_only else result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        video.file.close()
