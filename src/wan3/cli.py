from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .client import Wan3Client
from .schema import DEFAULT_PROMPT, GenerateRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate silent video with Wan 3.0 Prime via fal.ai")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p", "1080p"])
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", "-o", type=Path)
    return parser


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args = build_parser().parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"audio file not found: {args.audio}")
    request = GenerateRequest(
        prompt=args.prompt,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        seed=args.seed,
        audio_path=str(args.audio.resolve()),
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

    return asyncio.run(Wan3Client().generate(request))


if __name__ == "__main__":
    main()
