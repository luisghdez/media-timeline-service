from __future__ import annotations

from pathlib import Path

from media_timeline.media import MediaError, probe, run

from .schema import CANVAS, ComposeResult, Resolution


def compose(
    top: Path,
    bottom: Path,
    destination: Path,
    *,
    resolution: Resolution = "1080p",
) -> ComposeResult:
    if not top.is_file():
        raise MediaError(f"video file not found: {top}")
    if not bottom.is_file():
        raise MediaError(f"video file not found: {bottom}")
    bottom_info = probe(bottom)
    if not bottom_info.audio_codec:
        raise MediaError(f"video 2 has no audio stream: {bottom}")
    duration = min(probe(top).duration, bottom_info.duration)

    width, height = CANVAS[resolution]
    top_height = height // 4
    bottom_height = height - top_height
    filter_graph = (
        f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,"
        f"scale={width}:{top_height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{top_height},setsar=1[top];"
        f"[1:v]trim=duration={duration},setpts=PTS-STARTPTS,"
        f"scale={width}:{bottom_height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{bottom_height},setsar=1[bot];"
        "[top][bot]vstack=inputs=2[v];"
        f"[1:a:0]atrim=duration={duration},asetpts=PTS-STARTPTS[a]"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(top),
            "-i",
            str(bottom),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(destination),
        ]
    )
    info = probe(destination)
    return ComposeResult(
        output_path=destination,
        width=info.width or width,
        height=info.height or height,
        duration=info.duration,
    )
