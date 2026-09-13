from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fal_client import Completed, InProgress, Queued
from fastapi.testclient import TestClient

from wan3 import api
from wan3.client import Wan3Client, Wan3Error, _parse_status
from wan3.schema import (
    DEFAULT_PROMPT,
    GenerateRequest,
    GenerateResult,
    JobAccepted,
    VideoFile,
    assert_reference_audio_duration,
    output_duration_from_audio,
)


class StubClient:
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        assert request.audio_path
        return GenerateResult(
            video=VideoFile(url="https://example.test/out.mp4", content_type="video/mp4"),
            seed=42,
            duration=15.0,
        )

    async def submit(self, request: GenerateRequest) -> JobAccepted:
        assert request.audio_path
        return JobAccepted(request_id="job-1")

    async def status(self, request_id: str, *, with_logs: bool = True):
        from wan3.schema import JobStatus

        return JobStatus(request_id=request_id, status="COMPLETED")

    async def result(self, request_id: str) -> GenerateResult:
        return GenerateResult(
            video=VideoFile(url="https://example.test/out.mp4"),
            seed=7,
            duration=15.0,
            request_id=request_id,
        )


def test_health() -> None:
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate(monkeypatch) -> None:
    monkeypatch.setattr(api, "client", StubClient())
    response = TestClient(api.app).post(
        "/v1/generate",
        files={"audio": ("vocals.wav", b"fake-wav", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["video"]["url"] == "https://example.test/out.mp4"
    assert body["seed"] == 42
    assert body["duration"] == 15.0


def test_queue_flow(monkeypatch) -> None:
    monkeypatch.setattr(api, "client", StubClient())
    http = TestClient(api.app)
    submitted = http.post(
        "/v1/jobs",
        files={"audio": ("vocals.wav", b"fake-wav", "audio/wav")},
        data={"prompt": "a quiet forest"},
    )
    assert submitted.status_code == 200
    assert submitted.json() == {"request_id": "job-1"}
    status = http.get("/v1/jobs/job-1")
    assert status.json()["status"] == "COMPLETED"
    result = http.get("/v1/jobs/job-1/result")
    assert result.json()["request_id"] == "job-1"


def test_output_duration_from_audio() -> None:
    assert output_duration_from_audio(14.6) == 15
    assert output_duration_from_audio(1.2) == 2
    assert output_duration_from_audio(31) == 30


def test_reference_audio_cap() -> None:
    assert_reference_audio_duration(15.0)
    with pytest.raises(ValueError, match="15"):
        assert_reference_audio_duration(15.01)


def test_to_fal_arguments_defaults() -> None:
    args = GenerateRequest(
        reference_audio_urls=["https://example.test/audio.wav"],
        duration=15,
        webhook_url="https://example.test/hook",
        audio_path="/tmp/local.wav",
    ).to_fal_arguments()
    assert args["prompt"] == DEFAULT_PROMPT
    assert args["resolution"] == "480p"
    assert args["aspect_ratio"] == "16:9"
    assert args["audio"] is False
    assert args["duration"] == 15
    assert args["reference_audio_urls"] == ["https://example.test/audio.wav"]
    assert "webhook_url" not in args
    assert "audio_path" not in args


def test_prepare_rejects_long_reference(monkeypatch) -> None:
    monkeypatch.setattr("wan3.client.probe", lambda _: SimpleNamespace(duration=15.2))

    async def fail_upload(_path):
        raise AssertionError("upload should not run")

    monkeypatch.setattr("wan3.client.fal_client.upload_file_async", fail_upload)
    with pytest.raises(Wan3Error) as exc:
        asyncio.run(Wan3Client().prepare(GenerateRequest(audio_path="clip.wav")))
    assert exc.value.status_code == 422


def test_prepare_sets_duration_and_url(monkeypatch) -> None:
    monkeypatch.setattr("wan3.client.probe", lambda _: SimpleNamespace(duration=14.606))
    monkeypatch.setenv("FAL_KEY", "test-key")

    async def fake_upload(path):
        assert str(path).endswith("clip.wav")
        return "https://fal.media/audio.wav"

    monkeypatch.setattr("wan3.client.fal_client.upload_file_async", fake_upload)
    prepared = asyncio.run(Wan3Client().prepare(GenerateRequest(audio_path="clip.wav")))
    assert prepared.duration == 15
    assert prepared.reference_audio_urls == ["https://fal.media/audio.wav"]


def test_parse_status_maps_fal_types() -> None:
    queued = _parse_status("abc", Queued(position=2))
    assert queued.status == "IN_QUEUE"
    assert queued.position == 2
    progress = _parse_status("abc", InProgress(logs=[{"message": "rendering"}]))
    assert progress.status == "IN_PROGRESS"
    done = _parse_status("abc", Completed(logs=None, metrics={"infer": 1.2}))
    assert done.status == "COMPLETED"
    assert done.metrics == {"infer": 1.2}
