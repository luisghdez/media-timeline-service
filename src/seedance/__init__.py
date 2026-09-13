"""Seedance 2.5 text-to-video client (fal.ai)."""

from .client import MODEL_ID, SeedanceClient
from .schema import GenerateRequest, GenerateResult, JobStatus, VideoFile

__all__ = [
    "MODEL_ID",
    "GenerateRequest",
    "GenerateResult",
    "JobStatus",
    "SeedanceClient",
    "VideoFile",
]
