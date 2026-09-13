from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fal_client
from fal_client import Completed, FalClientError, FalClientHTTPError, FalClientTimeoutError, InProgress, Queued

from media_timeline.media import MediaError, probe

from .schema import (
    GenerateRequest,
    GenerateResult,
    JobAccepted,
    JobStatus,
    VideoFile,
    assert_reference_audio_duration,
    output_duration_from_audio,
)

MODEL_ID = "alibaba/wan-3.0-prime/reference-to-video"

_STATUS_NAMES = {
    Queued: "IN_QUEUE",
    InProgress: "IN_PROGRESS",
    Completed: "COMPLETED",
}


class Wan3Error(Exception):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class Wan3Client:
    def __init__(self, client_timeout: float | None = None) -> None:
        self.client_timeout = client_timeout

    def require_api_key(self) -> None:
        if not os.getenv("FAL_KEY", "").strip():
            raise Wan3Error("FAL_KEY is not set", status_code=503)

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.require_api_key()
        prepared = await self.prepare(request)
        try:
            payload = await fal_client.subscribe_async(
                MODEL_ID,
                arguments=prepared.to_fal_arguments(),
                with_logs=True,
                client_timeout=self.client_timeout,
            )
        except FalClientTimeoutError as exc:
            raise Wan3Error(str(exc) or "Wan 3 request timed out", status_code=504) from exc
        except (FalClientHTTPError, FalClientError) as exc:
            raise Wan3Error(str(exc) or "Wan 3 request failed") from exc
        return _parse_result(payload)

    async def submit(self, request: GenerateRequest) -> JobAccepted:
        self.require_api_key()
        prepared = await self.prepare(request)
        try:
            handle = await fal_client.submit_async(
                MODEL_ID,
                arguments=prepared.to_fal_arguments(),
                webhook_url=prepared.webhook_url,
            )
        except (FalClientHTTPError, FalClientError) as exc:
            raise Wan3Error(str(exc) or "Failed to submit Wan 3 job") from exc
        return JobAccepted(request_id=handle.request_id)

    async def status(self, request_id: str, *, with_logs: bool = True) -> JobStatus:
        self.require_api_key()
        try:
            update = await fal_client.status_async(MODEL_ID, request_id, with_logs=with_logs)
        except (FalClientHTTPError, FalClientError) as exc:
            raise Wan3Error(str(exc) or "Failed to fetch Wan 3 job status") from exc
        return _parse_status(request_id, update)

    async def result(self, request_id: str) -> GenerateResult:
        self.require_api_key()
        try:
            payload = await fal_client.result_async(MODEL_ID, request_id)
        except FalClientTimeoutError as exc:
            raise Wan3Error(str(exc) or "Wan 3 result timed out", status_code=504) from exc
        except (FalClientHTTPError, FalClientError) as exc:
            raise Wan3Error(str(exc) or "Failed to fetch Wan 3 result") from exc
        return _parse_result(payload, request_id=request_id)

    async def prepare(self, request: GenerateRequest) -> GenerateRequest:
        if request.reference_audio_urls and request.duration is not None:
            return request
        if not request.audio_path:
            raise Wan3Error("audio_path is required", status_code=422)
        path = Path(request.audio_path)
        try:
            seconds = probe(path).duration
        except MediaError as exc:
            raise Wan3Error(str(exc), status_code=422) from exc
        try:
            assert_reference_audio_duration(seconds)
        except ValueError as exc:
            raise Wan3Error(str(exc), status_code=422) from exc
        self.require_api_key()
        try:
            url = await fal_client.upload_file_async(path)
        except (FalClientHTTPError, FalClientError) as exc:
            raise Wan3Error(str(exc) or "Failed to upload audio") from exc
        return request.model_copy(
            update={
                "duration": output_duration_from_audio(seconds),
                "reference_audio_urls": [url],
            }
        )


def _parse_result(payload: dict[str, Any], request_id: str | None = None) -> GenerateResult:
    video = payload.get("video") or {}
    if not video.get("url"):
        raise Wan3Error("Wan 3 response did not include a video URL")
    seed = payload.get("seed")
    if seed is None:
        raise Wan3Error("Wan 3 response did not include a seed")
    duration = payload.get("duration")
    if duration is None:
        duration = video.get("duration")
    return GenerateResult(
        video=VideoFile.model_validate(video),
        seed=int(seed),
        duration=None if duration is None else float(duration),
        actual_prompt=payload.get("actual_prompt"),
        request_id=request_id,
    )


def _parse_status(request_id: str, update: object) -> JobStatus:
    status = _STATUS_NAMES.get(type(update))
    if status is None:
        raise Wan3Error(f"Unexpected Wan 3 status: {type(update).__name__}")
    return JobStatus(
        request_id=request_id,
        status=status,
        position=getattr(update, "position", None),
        logs=getattr(update, "logs", None),
        metrics=getattr(update, "metrics", None),
        error=getattr(update, "error", None),
        error_type=getattr(update, "error_type", None),
    )
