from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from .client import Wan3Client, Wan3Error
from .schema import DEFAULT_PROMPT, GenerateRequest, GenerateResult, JobAccepted, JobStatus, Resolution, AspectRatio

def _timeout() -> float:
    raw = os.getenv("WAN3_CLIENT_TIMEOUT")
    return float(raw) if raw else 600.0


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="Wan 3.0 Prime Reference-to-Video", version="0.1.0")
client = Wan3Client(client_timeout=_timeout())


def _http_error(exc: Wan3Error) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


async def _request_from_upload(
    audio: UploadFile,
    prompt: str | None,
    resolution: Resolution,
    aspect_ratio: AspectRatio,
    seed: int | None,
    webhook_url: str | None,
) -> tuple[GenerateRequest, Path]:
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}:
        suffix = ".wav"
    handle = tempfile.NamedTemporaryFile(prefix="wan3-audio-", suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        while chunk := await audio.read(1024 * 1024):
            handle.write(chunk)
    finally:
        handle.close()
        await audio.close()
    request = GenerateRequest(
        prompt=prompt or DEFAULT_PROMPT,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        seed=seed,
        audio_path=str(path),
        webhook_url=webhook_url,
    )
    return request, path


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/generate", response_model=GenerateResult)
async def generate(
    audio: UploadFile = File(...),
    prompt: str | None = Form(None),
    resolution: Resolution = Form("480p"),
    aspect_ratio: AspectRatio = Form("16:9"),
    seed: int | None = Form(None),
) -> GenerateResult:
    path: Path | None = None
    try:
        request, path = await _request_from_upload(audio, prompt, resolution, aspect_ratio, seed, None)
        return await client.generate(request)
    except Wan3Error as exc:
        raise _http_error(exc) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@app.post("/v1/jobs", response_model=JobAccepted)
async def submit_job(
    audio: UploadFile = File(...),
    prompt: str | None = Form(None),
    resolution: Resolution = Form("480p"),
    aspect_ratio: AspectRatio = Form("16:9"),
    seed: int | None = Form(None),
    webhook_url: str | None = Form(None),
) -> JobAccepted:
    path: Path | None = None
    try:
        request, path = await _request_from_upload(
            audio, prompt, resolution, aspect_ratio, seed, webhook_url,
        )
        return await client.submit(request)
    except Wan3Error as exc:
        raise _http_error(exc) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@app.get("/v1/jobs/{request_id}", response_model=JobStatus)
async def job_status(request_id: str, logs: bool = Query(True)) -> JobStatus:
    try:
        return await client.status(request_id, with_logs=logs)
    except Wan3Error as exc:
        raise _http_error(exc) from exc


@app.get("/v1/jobs/{request_id}/result", response_model=GenerateResult)
async def job_result(request_id: str) -> GenerateResult:
    try:
        return await client.result(request_id)
    except Wan3Error as exc:
        raise _http_error(exc) from exc
