from fastapi.testclient import TestClient

from media_timeline import api
from media_timeline.schema import AnalysisResult, Layer, Segment


class StubPipeline:
    def analyze(self, source, context_text=None):
        return AnalysisResult(
            duration=1.0,
            source={"filename": "input.mp4"},
            segments=[Segment(0.0, 1.0, [Layer(type="speech", text="Hello", confidence=0.9)])],
        )


def test_health() -> None:
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_env_bool(monkeypatch) -> None:
    monkeypatch.setenv("TEST_BOOLEAN", "yes")
    assert api._env_bool("TEST_BOOLEAN", False) is True


def test_analyze_upload(monkeypatch) -> None:
    monkeypatch.setattr(api, "pipeline", StubPipeline())
    response = TestClient(api.app).post(
        "/v1/analyze?segments_only=true",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    assert response.json()[0]["layers"][0]["text"] == "Hello"
