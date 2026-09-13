from __future__ import annotations

import argparse
import json
from pathlib import Path

from media_timeline.media import MediaError

from .compose import compose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stack two videos on a 9:16 canvas (top 1/4 + bottom 3/4, audio from video 2)"
    )
    parser.add_argument("video1", type=Path, help="Overlay video for the top quarter")
    parser.add_argument("video2", type=Path, help="Main video for the remaining space and soundtrack")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--resolution", default="1080p", choices=("720p", "1080p"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = compose(args.video1, args.video2, args.output, resolution=args.resolution)
    except MediaError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "width": result.width,
                "height": result.height,
                "duration": result.duration,
            },
            indent=2,
        )
        + "\n",
        end="",
    )


if __name__ == "__main__":
    main()
