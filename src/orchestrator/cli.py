from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from wan3.schema import DEFAULT_PROMPT

from .pipeline import OrchestrateError, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract vocals, generate a Wan 3 reaction clip, and stack it over the source video"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resolution", default="1080p", choices=("720p", "1080p"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--wan3-resolution", default="480p", choices=("480p", "720p", "1080p"))
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args = build_parser().parse_args()
    try:
        result = asyncio.run(
            run(
                args.video,
                args.output,
                work_dir=args.work_dir,
                prompt=args.prompt,
                wan3_resolution=args.wan3_resolution,
                seed=args.seed,
                vstack_resolution=args.resolution,
            )
        )
    except OrchestrateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.to_dict(), indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
