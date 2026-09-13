from __future__ import annotations

import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from media_timeline.analyzers import separate_vocals
from media_timeline.media import MediaError, extract_audio, probe
from vstack.compose import compose
from vstack.schema import Resolution as VStackResolution
from wan3.client import Wan3Client, Wan3Error
from wan3.schema import (
    DEFAULT_PROMPT,
    AspectRatio,
    GenerateRequest,
    Resolution as Wan3Resolution,
    assert_reference_audio_duration,
)

from .schema import OrchestrateResult

MIX_SAMPLE_RATE = 44_100
MIX_CHANNELS = 2
DOWNLOAD_TIMEOUT = 120.0


class OrchestrateError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def download_file(url: str, destination: Path, *, timeout: float = DOWNLOAD_TIMEOUT) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise OrchestrateError(f"failed to download Wan 3 video: {exc}", status_code=502) from exc


async def run(
    source: Path,
    destination: Path,
    *,
    work_dir: Path | None = None,
    client: Wan3Client | None = None,
    prompt: str = DEFAULT_PROMPT,
    wan3_resolution: Wan3Resolution = "480p",
    aspect_ratio: AspectRatio = "16:9",
    seed: int | None = None,
    vstack_resolution: VStackResolution = "1080p",
    download_timeout: float = DOWNLOAD_TIMEOUT,
) -> OrchestrateResult:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise OrchestrateError(f"video file not found: {source}", status_code=400)

    wan3 = client or Wan3Client()
    try:
        wan3.require_api_key()
    except Wan3Error as exc:
        raise OrchestrateError(str(exc), status_code=exc.status_code) from exc

    owned_work_dir = work_dir is None
    root = work_dir.resolve() if work_dir is not None else Path(tempfile.mkdtemp(prefix="orchestrate-"))
    root.mkdir(parents=True, exist_ok=True)
    try:
        mix = root / "mix.wav"
        extract_audio(source, mix, sample_rate=MIX_SAMPLE_RATE, channels=MIX_CHANNELS)
        try:
            separated_vocals, _accompaniment = separate_vocals(mix, root / "separated")
        except RuntimeError as exc:
            raise OrchestrateError(f"vocal separation failed: {exc}", status_code=500) from exc
        vocals = root / "vocals.wav"
        shutil.copy2(separated_vocals, vocals)

        try:
            seconds = probe(vocals).duration
            assert_reference_audio_duration(seconds)
        except MediaError as exc:
            raise OrchestrateError(str(exc), status_code=422) from exc
        except ValueError as exc:
            raise OrchestrateError(str(exc), status_code=422) from exc

        request = GenerateRequest(
            prompt=prompt,
            resolution=wan3_resolution,
            aspect_ratio=aspect_ratio,
            seed=seed,
            audio_path=str(vocals),
        )
        try:
            generated = await wan3.generate(request)
        except Wan3Error as exc:
            raise OrchestrateError(str(exc), status_code=exc.status_code) from exc

        wan3_path = root / "wan3.mp4"
        download_file(generated.video.url, wan3_path, timeout=download_timeout)

        try:
            stacked = compose(wan3_path, source, destination, resolution=vstack_resolution)
        except MediaError as exc:
            raise OrchestrateError(str(exc), status_code=400) from exc

        return OrchestrateResult(
            output_path=stacked.output_path,
            vocals_path=vocals,
            wan3_path=wan3_path,
            seed=generated.seed,
            duration=stacked.duration,
            wan3_url=generated.video.url,
            width=stacked.width,
            height=stacked.height,
        )
    finally:
        if owned_work_dir:
            shutil.rmtree(root, ignore_errors=True)
