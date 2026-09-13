from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from vstack.schema import Resolution as VStackResolution
from wan3.client import Wan3Client
from wan3.schema import DEFAULT_PROMPT, AspectRatio, Resolution as Wan3Resolution

from .pipeline import OrchestrateError, run

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="Vocals → Wan 3 → Stack Orchestrator", version="0.1.0")
client = Wan3Client()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/orchestrate")
async def orchestrate(
    video: UploadFile = File(...),
    prompt: str | None = Form(None),
    resolution: VStackResolution = Form("1080p"),
    wan3_resolution: Wan3Resolution = Form("480p"),
    aspect_ratio: AspectRatio = Form("16:9"),
    seed: int | None = Form(None),
) -> FileResponse:
    source = await _save_upload(video)
    handle = tempfile.NamedTemporaryFile(prefix="orchestrate-out-", suffix=".mp4", delete=False)
    destination = Path(handle.name)
    handle.close()
    try:
        await run(
            source,
            destination,
            client=client,
            prompt=prompt or DEFAULT_PROMPT,
            wan3_resolution=wan3_resolution,
            aspect_ratio=aspect_ratio,
            seed=seed,
            vstack_resolution=resolution,
        )
    except OrchestrateError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        source.unlink(missing_ok=True)
    return FileResponse(destination, media_type="video/mp4", filename="stacked.mp4")


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
    handle = tempfile.NamedTemporaryFile(prefix="orchestrate-in-", suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        while chunk := await upload.read(1024 * 1024):
            handle.write(chunk)
    finally:
        handle.close()
        await upload.close()
    return path
