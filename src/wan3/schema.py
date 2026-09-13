from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_PROMPT = (
    "create a streamer reaction video using the exact audio attached. "
    "He maintains hands on keyboard and mouse."
)
MAX_REFERENCE_AUDIO_SECONDS = 15.0
MIN_OUTPUT_DURATION = 2
MAX_OUTPUT_DURATION = 30

Resolution = Literal["480p", "720p", "1080p"]
AspectRatio = Literal["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16"]
JobState = Literal["IN_QUEUE", "IN_PROGRESS", "COMPLETED"]


def output_duration_from_audio(seconds: float) -> int:
    return max(MIN_OUTPUT_DURATION, min(MAX_OUTPUT_DURATION, round(seconds)))


def assert_reference_audio_duration(seconds: float) -> None:
    if seconds > MAX_REFERENCE_AUDIO_SECONDS:
        raise ValueError(
            f"reference audio is {seconds:.3f}s; fal allows at most "
            f"{MAX_REFERENCE_AUDIO_SECONDS:.0f}s"
        )


class GenerateRequest(BaseModel):
    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1)
    resolution: Resolution = "480p"
    aspect_ratio: AspectRatio = "16:9"
    audio: bool = False
    duration: int | None = Field(default=None, ge=MIN_OUTPUT_DURATION, le=MAX_OUTPUT_DURATION)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    reference_audio_urls: list[str] = Field(default_factory=list)
    audio_path: str | None = None
    webhook_url: str | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def default_prompt(cls, value: object) -> object:
        if value is None:
            return DEFAULT_PROMPT
        if isinstance(value, str) and not value.strip():
            return DEFAULT_PROMPT
        return value

    def to_fal_arguments(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"webhook_url", "audio_path"}, exclude_none=True)
        return payload


class VideoFile(BaseModel):
    url: str
    content_type: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration: float | None = None
    num_frames: int | None = None


class GenerateResult(BaseModel):
    video: VideoFile
    seed: int
    duration: float | None = None
    actual_prompt: str | None = None
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
