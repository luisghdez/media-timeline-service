from __future__ import annotations

import os
from typing import Any

import fal_client
from fal_client import Completed, FalClientError, FalClientHTTPError, FalClientTimeoutError, InProgress, Queued

from .schema import GenerateRequest, GenerateResult, JobAccepted, JobStatus, VideoFile

MODEL_ID = "bytedance/seedance-2.5/text-to-video"

_STATUS_NAMES = {
    Queued: "IN_QUEUE",
    InProgress: "IN_PROGRESS",
    Completed: "COMPLETED",
}


class SeedanceError(Exception):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class SeedanceClient:
    def __init__(self, client_timeout: float | None = None) -> None:
        self.client_timeout = client_timeout

    def require_api_key(self) -> None:
        if not os.getenv("FAL_KEY", "").strip():
            raise SeedanceError("FAL_KEY is not set", status_code=503)

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.require_api_key()
        try:
            payload = await fal_client.subscribe_async(
                MODEL_ID,
                arguments=request.to_fal_arguments(),
                with_logs=True,
                client_timeout=self.client_timeout,
            )
        except FalClientTimeoutError as exc:
            raise SeedanceError(str(exc) or "Seedance request timed out", status_code=504) from exc
        except (FalClientHTTPError, FalClientError) as exc:
            raise SeedanceError(str(exc) or "Seedance request failed") from exc
        return _parse_result(payload)

    async def submit(self, request: GenerateRequest) -> JobAccepted:
        self.require_api_key()
        try:
            handle = await fal_client.submit_async(
                MODEL_ID,
                arguments=request.to_fal_arguments(),
                webhook_url=request.webhook_url,
            )
        except (FalClientHTTPError, FalClientError) as exc:
            raise SeedanceError(str(exc) or "Failed to submit Seedance job") from exc
        return JobAccepted(request_id=handle.request_id)

    async def status(self, request_id: str, *, with_logs: bool = True) -> JobStatus:
        self.require_api_key()
        try:
            update = await fal_client.status_async(MODEL_ID, request_id, with_logs=with_logs)
        except (FalClientHTTPError, FalClientError) as exc:
            raise SeedanceError(str(exc) or "Failed to fetch Seedance job status") from exc
        return _parse_status(request_id, update)

    async def result(self, request_id: str) -> GenerateResult:
        self.require_api_key()
        try:
            payload = await fal_client.result_async(MODEL_ID, request_id)
        except FalClientTimeoutError as exc:
            raise SeedanceError(str(exc) or "Seedance result timed out", status_code=504) from exc
        except (FalClientHTTPError, FalClientError) as exc:
            raise SeedanceError(str(exc) or "Failed to fetch Seedance result") from exc
        return _parse_result(payload, request_id=request_id)


def _parse_result(payload: dict[str, Any], request_id: str | None = None) -> GenerateResult:
    video = payload.get("video") or {}
    if not video.get("url"):
        raise SeedanceError("Seedance response did not include a video URL")
    return GenerateResult(
        video=VideoFile.model_validate(video),
        seed=int(payload["seed"]),
        request_id=request_id,
    )


def _parse_status(request_id: str, update: object) -> JobStatus:
    status = _STATUS_NAMES.get(type(update))
    if status is None:
        raise SeedanceError(f"Unexpected Seedance status: {type(update).__name__}")
    return JobStatus(
        request_id=request_id,
        status=status,
        position=getattr(update, "position", None),
        logs=getattr(update, "logs", None),
        metrics=getattr(update, "metrics", None),
        error=getattr(update, "error", None),
        error_type=getattr(update, "error_type", None),
    )
