from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


@dataclass(slots=True)
class MediaInfo:
    duration: float
    audio_codec: str | None
    video_codec: str | None
    sample_rate: int | None
    channels: int | None
    frame_rate: float | None


def run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise MediaError(detail) from exc


def probe(path: Path) -> MediaInfo:
    completed = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,sample_rate,channels,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    return MediaInfo(
        duration=float(payload["format"]["duration"]),
        audio_codec=audio.get("codec_name"),
        video_codec=video.get("codec_name"),
        sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio.get("channels") else None,
        frame_rate=_parse_rate(video.get("r_frame_rate")),
    )


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def extract_audio(source: Path, destination: Path, sample_rate: int = 16_000) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def detect_silence(path: Path, threshold_db: int = -42, minimum_duration: float = 0.2) -> list[tuple[float, float]]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={threshold_db}dB:d={minimum_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", completed.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", completed.stderr)]
    intervals: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        if index < len(ends):
            intervals.append((start, ends[index]))
    return intervals


def detect_scene_changes(path: Path, threshold: float = 0.24) -> list[float]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-filter:v",
            f"select=gt(scene\\,{threshold}),metadata=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stdout + completed.stderr)]
    output: list[float] = []
    for timestamp in raw:
        if not output or timestamp - output[-1] >= 0.5:
            output.append(timestamp)
    return output
