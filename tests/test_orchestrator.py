from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_timeline.media import MediaInfo
from orchestrator import api
from orchestrator.cli import build_parser
from orchestrator.pipeline import OrchestrateError, run
from vstack.schema import ComposeResult
from wan3.client import Wan3Error
from wan3.schema import GenerateRequest, GenerateResult, VideoFile


class StubClient:
    def __init__(self) -> None:
        self.calls: list[GenerateRequest] = []
        self.compose_calls: list[tuple[Path, Path, Path, str]] = []

    def require_api_key(self) -> None:
        return None

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.calls.append(request)
        return GenerateResult(
            video=VideoFile(url="https://example.test/wan3.mp4", content_type="video/mp4"),
            seed=99,
            duration=8.0,
        )


def _touch_source(path: Path) -> Path:
    path.write_bytes(b"fake-video")
    return path


def _install_pipeline_stubs(monkeypatch, *, vocals_duration: float = 8.0) -> StubClient:
    client = StubClient()

    def fake_extract(source: Path, destination: Path, sample_rate: int = 16_000, channels: int = 1) -> None:
        destination.write_bytes(b"mix")

    def fake_separate(source: Path, output_root: Path, model_name: str = "htdemucs") -> tuple[Path, Path]:
        vocals = output_root / model_name / source.stem / "vocals.wav"
        accompaniment = vocals.with_name("no_vocals.wav")
        vocals.parent.mkdir(parents=True, exist_ok=True)
        vocals.write_bytes(b"vocals")
        accompaniment.write_bytes(b"bg")
        return vocals, accompaniment

    def fake_probe(path: Path) -> MediaInfo:
        return MediaInfo(
            duration=vocals_duration if path.name == "vocals.wav" else 12.0,
            audio_codec="pcm_s16le",
            video_codec="h264",
            sample_rate=44100,
            channels=2,
            frame_rate=30.0,
        )

    def fake_download(url: str, destination: Path, *, timeout: float = 120.0) -> None:
        assert url == "https://example.test/wan3.mp4"
        destination.write_bytes(b"wan3")

    def fake_compose(top: Path, bottom: Path, destination: Path, *, resolution: str = "1080p") -> ComposeResult:
        client.compose_calls.append((top, bottom, destination, resolution))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stacked")
        return ComposeResult(output_path=destination, width=1080, height=1920, duration=8.0)

    monkeypatch.setattr("orchestrator.pipeline.extract_audio", fake_extract)
    monkeypatch.setattr("orchestrator.pipeline.separate_vocals", fake_separate)
    monkeypatch.setattr("orchestrator.pipeline.probe", fake_probe)
    monkeypatch.setattr("orchestrator.pipeline.download_file", fake_download)
    monkeypatch.setattr("orchestrator.pipeline.compose", fake_compose)
    return client


def test_run_stacks_wan3_over_source(monkeypatch, tmp_path: Path) -> None:
    source = _touch_source(tmp_path / "clip.mp4")
    destination = tmp_path / "stacked.mp4"
    work_dir = tmp_path / "work"
    client = _install_pipeline_stubs(monkeypatch)
    result = asyncio.run(
        run(source, destination, work_dir=work_dir, client=client, vstack_resolution="720p")
    )
    assert result.output_path == destination
    assert result.seed == 99
    assert result.vocals_path == work_dir / "vocals.wav"
    assert result.wan3_path == work_dir / "wan3.mp4"
    assert destination.read_bytes() == b"stacked"
    assert (work_dir / "vocals.wav").is_file()
    top, bottom, dest, resolution = client.compose_calls[0]
    assert top == work_dir / "wan3.mp4"
    assert bottom == source
    assert dest == destination
    assert resolution == "720p"
    assert client.calls[0].audio_path == str(work_dir / "vocals.wav")


def test_run_rejects_audio_over_15s_before_generate(monkeypatch, tmp_path: Path) -> None:
    source = _touch_source(tmp_path / "clip.mp4")
    client = _install_pipeline_stubs(monkeypatch, vocals_duration=15.5)
    with pytest.raises(OrchestrateError, match="at most 15s"):
        asyncio.run(run(source, tmp_path / "out.mp4", work_dir=tmp_path / "work", client=client))
    assert client.calls == []
    assert client.compose_calls == []


def test_run_missing_source(tmp_path: Path) -> None:
    with pytest.raises(OrchestrateError, match="video file not found"):
        asyncio.run(run(tmp_path / "missing.mp4", tmp_path / "out.mp4", client=StubClient()))


def test_run_missing_fal_key(monkeypatch, tmp_path: Path) -> None:
    source = _touch_source(tmp_path / "clip.mp4")
    monkeypatch.delenv("FAL_KEY", raising=False)

    class MissingKeyClient:
        def require_api_key(self) -> None:
            raise Wan3Error("FAL_KEY is not set", status_code=503)

    with pytest.raises(OrchestrateError, match="FAL_KEY") as exc_info:
        asyncio.run(run(source, tmp_path / "out.mp4", client=MissingKeyClient()))  # type: ignore[arg-type]
    assert exc_info.value.status_code == 503


def test_cli_parser() -> None:
    args = build_parser().parse_args(
        ["clip.mp4", "-o", "out.mp4", "--work-dir", "work", "--resolution", "720p", "--seed", "3"]
    )
    assert args.video == Path("clip.mp4")
    assert args.output == Path("out.mp4")
    assert args.work_dir == Path("work")
    assert args.resolution == "720p"
    assert args.seed == 3


def test_health() -> None:
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_orchestrate_endpoint(monkeypatch) -> None:
    client = _install_pipeline_stubs(monkeypatch)
    monkeypatch.setattr(api, "client", client)
    response = TestClient(api.app).post(
        "/v1/orchestrate",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
        data={"resolution": "720p"},
    )
    assert response.status_code == 200
    assert response.content == b"stacked"
    assert response.headers["content-type"].startswith("video/mp4")
    top, bottom, _dest, resolution = client.compose_calls[0]
    assert top.name == "wan3.mp4"
    assert bottom.name.startswith("orchestrate-in-")
    assert resolution == "720p"
