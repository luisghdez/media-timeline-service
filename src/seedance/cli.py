from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .client import SeedanceClient
from .schema import GenerateRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate video with Seedance 2.5 via fal.ai")
    parser.add_argument("prompt")
    parser.add_argument("--resolution", default="720p", choices=["480p", "720p", "1080p"])
    parser.add_argument("--duration", default="auto")
    parser.add_argument(
        "--aspect-ratio",
        default="auto",
        choices=["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
    )
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--bitrate-mode", default="standard", choices=["standard", "high"])
    parser.add_argument("--end-user-id")
    parser.add_argument("--output", "-o", type=Path)
    return parser


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args = build_parser().parse_args()
    request = GenerateRequest(
        prompt=args.prompt,
        resolution=args.resolution,
        duration=args.duration,
        aspect_ratio=args.aspect_ratio,
        generate_audio=not args.no_audio,
        bitrate_mode=args.bitrate_mode,
        end_user_id=args.end_user_id,
    )
    result = _run(request)
    rendered = json.dumps(result.model_dump(exclude_none=True), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def _run(request: GenerateRequest):
    import asyncio

    return asyncio.run(SeedanceClient().generate(request))


if __name__ == "__main__":
    main()
