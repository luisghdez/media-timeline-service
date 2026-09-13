from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Resolution = Literal["480p", "720p", "1080p"]
AspectRatio = Literal["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
BitrateMode = Literal["standard", "high"]
JobState = Literal["IN_QUEUE", "IN_PROGRESS", "COMPLETED"]

_DURATIONS = frozenset({"auto", *[str(seconds) for seconds in range(4, 31)]})


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    resolution: Resolution = "720p"
    duration: str = "auto"
    aspect_ratio: AspectRatio = "auto"
    generate_audio: bool = True
    bitrate_mode: BitrateMode = "standard"
    end_user_id: str | None = None
    webhook_url: str | None = None

    @field_validator("duration", mode="before")
    @classmethod
    def normalize_duration(cls, value: object) -> str:
        text = "auto" if value is None else str(value).strip()
        if text in _DURATIONS:
            return text
        raise ValueError("duration must be 'auto' or an integer from 4 to 30")

    def to_fal_arguments(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"webhook_url"}, exclude_none=True)
        return payload


class VideoFile(BaseModel):
    url: str
    content_type: str | None = None
    file_name: str | None = None
    file_size: int | None = None


class GenerateResult(BaseModel):
    video: VideoFile
    seed: int
    request_id: str | None = None


class JobAccepted(BaseModel):
    request_id: str


class JobStatus(BaseModel):
    request_id: str
    status: JobState
    position: int | None = None
    logs: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
