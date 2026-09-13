"""Wan 3.0 Prime reference-to-video client (fal.ai)."""

from .client import MODEL_ID, Wan3Client
from .schema import DEFAULT_PROMPT, GenerateRequest, GenerateResult, JobStatus, VideoFile

__all__ = [
    "DEFAULT_PROMPT",
    "MODEL_ID",
    "GenerateRequest",
    "GenerateResult",
    "JobStatus",
    "Wan3Client",
    "VideoFile",
]
