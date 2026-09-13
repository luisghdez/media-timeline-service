from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from media_timeline.media import MediaError

from .compose import compose
from .schema import Resolution

app = FastAPI(title="9:16 Video Stack", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compose")
async def compose_videos(
    video1: UploadFile = File(...),
    video2: UploadFile = File(...),
    resolution: Resolution = Form("1080p"),
) -> FileResponse:
    top = await _save_upload(video1, "vstack-top-")
    bottom = await _save_upload(video2, "vstack-bottom-")
    handle = tempfile.NamedTemporaryFile(prefix="vstack-out-", suffix=".mp4", delete=False)
    destination = Path(handle.name)
    handle.close()
    try:
        compose(top, bottom, destination, resolution=resolution)
    except MediaError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        top.unlink(missing_ok=True)
        bottom.unlink(missing_ok=True)
    return FileResponse(destination, media_type="video/mp4", filename="stacked.mp4")


async def _save_upload(upload: UploadFile, prefix: str) -> Path:
    suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        while chunk := await upload.read(1024 * 1024):
            handle.write(chunk)
    finally:
        handle.close()
        await upload.close()
    return path
