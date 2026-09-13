from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import AnalysisPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a confidence-scored audio timeline from a video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--model", default="small.en", help="faster-whisper model name")
    parser.add_argument("--event-threshold", type=float, default=0.07)
    parser.add_argument("--asr-backend", choices=("whisper", "crisper"), default="whisper")
    parser.add_argument("--verbatim-model", default="nyralabs/CrisperWhisper2.0_large")
    parser.add_argument("--alignment", choices=("none", "whisperx"), default="none")
    parser.add_argument("--delivery", choices=("none", "acoustic", "qwen"), default="acoustic")
    parser.add_argument("--delivery-model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu", help="Device for CrisperWhisper and alignment")
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/media-timeline"))
    parser.add_argument("--context", help="Source caption/title; retained as labeled context evidence")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-separation", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-scenes", action="store_true")
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable faster-whisper Silero VAD so non-speech regions are skipped",
    )
    parser.add_argument(
        "--asr-source",
        choices=("both", "mix", "vocals"),
        default="both",
        help="Audio sent to Whisper: full mix, Demucs vocals stem, or both then merged",
    )
    parser.add_argument("--segments-only", action="store_true", help="Emit only the segment array")
    parser.add_argument("--events-only", action="store_true", help="Emit the clean event list")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        whisper_model=args.model,
        event_threshold=args.event_threshold,
        asr_backend=args.asr_backend,
        verbatim_model=args.verbatim_model,
        alignment=args.alignment,
        delivery=args.delivery,
        delivery_model=args.delivery_model,
        language=args.language,
        device=args.device,
        model_cache=args.model_cache,
        use_vocal_separation=not args.no_separation,
        use_audio_events=not args.no_events,
        use_scene_detection=not args.no_scenes,
        offline=args.offline,
        vad_filter=args.vad,
        speech_source=args.asr_source,
    )
    result = AnalysisPipeline(config).analyze(args.video, args.artifacts_dir, args.context)
    full_payload = result.to_dict()
    if args.events_only:
        payload = full_payload["events"]
    elif args.segments_only:
        payload = full_payload["segments"]
    else:
        payload = full_payload
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
