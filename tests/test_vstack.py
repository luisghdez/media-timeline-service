from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_timeline.media import MediaError, probe
from vstack import api
from vstack.compose import compose

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg required",
)


def _write_clip(
    path: Path,
    *,
    size: str,
    duration: float,
    color: str,
    audio: bool = False,
    frequency: int = 1000,
) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:r=10:d={duration}",
    ]
    if audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=16000:duration={duration}",
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ]
        )
    else:
        command.extend(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    command.append(str(path))
    subprocess.run(command, check=True)


def test_compose_is_9_16_and_uses_shortest_duration(tmp_path: Path) -> None:
    top = tmp_path / "top.mp4"
    bottom = tmp_path / "bottom.mp4"
    _write_clip(top, size="640x360", duration=0.8, color="red")
    _write_clip(bottom, size="360x640", duration=1.5, color="blue", audio=True)
    destination = tmp_path / "out.mp4"
    result = compose(top, bottom, destination, resolution="720p")
    assert result.width == 720
    assert result.height == 1280
    assert result.duration == pytest.approx(0.8, abs=0.08)
    info = probe(destination)
    assert info.audio_codec is not None
    assert info.width == 720
    assert info.height == 1280


def test_compose_keeps_video2_audio(tmp_path: Path) -> None:
    top = tmp_path / "top.mp4"
    bottom = tmp_path / "bottom.mp4"
    _write_clip(top, size="320x180", duration=1.0, color="green")
    _write_clip(bottom, size="180x320", duration=1.0, color="yellow", audio=True)
    destination = tmp_path / "out.mp4"
    compose(top, bottom, destination, resolution="720p")
    info = probe(destination)
    assert info.audio_codec is not None
    assert info.sample_rate is not None


def test_compose_rejects_bottom_without_audio(tmp_path: Path) -> None:
    top = tmp_path / "top.mp4"
    bottom = tmp_path / "bottom.mp4"
    _write_clip(top, size="320x180", duration=0.5, color="red")
    _write_clip(bottom, size="180x320", duration=0.5, color="blue", audio=False)
    with pytest.raises(MediaError, match="no audio stream"):
        compose(top, bottom, tmp_path / "out.mp4")


def test_api_health() -> None:
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_compose(tmp_path: Path) -> None:
    top = tmp_path / "top.mp4"
    bottom = tmp_path / "bottom.mp4"
    _write_clip(top, size="320x180", duration=0.5, color="red")
    _write_clip(bottom, size="180x320", duration=0.5, color="blue", audio=True)
    with top.open("rb") as top_handle, bottom.open("rb") as bottom_handle:
        response = TestClient(api.app).post(
            "/v1/compose",
            files={
                "video1": ("top.mp4", top_handle, "video/mp4"),
                "video2": ("bottom.mp4", bottom_handle, "video/mp4"),
            },
            data={"resolution": "720p"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/")
    saved = tmp_path / "api.mp4"
    saved.write_bytes(response.content)
    info = probe(saved)
    assert info.width == 720
    assert info.height == 1280
    assert info.audio_codec is not None
