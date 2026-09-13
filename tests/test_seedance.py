from __future__ import annotations

from fal_client import Completed, InProgress, Queued
from fastapi.testclient import TestClient

from seedance import api
from seedance.client import _parse_status
from seedance.schema import GenerateRequest, GenerateResult, JobAccepted, VideoFile


class StubClient:
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            video=VideoFile(url="https://example.test/out.mp4", content_type="video/mp4"),
            seed=42,
        )

    async def submit(self, request: GenerateRequest) -> JobAccepted:
        assert request.prompt
        return JobAccepted(request_id="job-1")

    async def status(self, request_id: str, *, with_logs: bool = True):
        from seedance.schema import JobStatus

        return JobStatus(request_id=request_id, status="COMPLETED")

    async def result(self, request_id: str) -> GenerateResult:
        return GenerateResult(
            video=VideoFile(url="https://example.test/out.mp4"),
            seed=7,
            request_id=request_id,
        )


def test_health() -> None:
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate(monkeypatch) -> None:
    monkeypatch.setattr(api, "client", StubClient())
    response = TestClient(api.app).post("/v1/generate", json={"prompt": "an octopus plays football"})
    assert response.status_code == 200
    body = response.json()
    assert body["video"]["url"] == "https://example.test/out.mp4"
    assert body["seed"] == 42


def test_queue_flow(monkeypatch) -> None:
    monkeypatch.setattr(api, "client", StubClient())
    http = TestClient(api.app)
    submitted = http.post("/v1/jobs", json={"prompt": "a quiet forest", "duration": 8})
    assert submitted.status_code == 200
    assert submitted.json() == {"request_id": "job-1"}
    status = http.get("/v1/jobs/job-1")
    assert status.json()["status"] == "COMPLETED"
    result = http.get("/v1/jobs/job-1/result")
    assert result.json()["request_id"] == "job-1"


def test_rejects_invalid_duration() -> None:
    response = TestClient(api.app).post("/v1/generate", json={"prompt": "x", "duration": "3"})
    assert response.status_code == 422


def test_to_fal_arguments_omits_webhook() -> None:
    args = GenerateRequest(prompt="scene", webhook_url="https://example.test/hook").to_fal_arguments()
    assert args["prompt"] == "scene"
    assert "webhook_url" not in args


def test_parse_status_maps_fal_types() -> None:
    queued = _parse_status("abc", Queued(position=2))
    assert queued.status == "IN_QUEUE"
    assert queued.position == 2
    progress = _parse_status("abc", InProgress(logs=[{"message": "rendering"}]))
    assert progress.status == "IN_PROGRESS"
    done = _parse_status("abc", Completed(logs=None, metrics={"infer": 1.2}))
    assert done.status == "COMPLETED"
    assert done.metrics == {"infer": 1.2}
