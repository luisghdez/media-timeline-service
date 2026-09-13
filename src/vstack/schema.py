from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Resolution = Literal["720p", "1080p"]

CANVAS: dict[Resolution, tuple[int, int]] = {
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


@dataclass(slots=True)
class ComposeResult:
    output_path: Path
    width: int
    height: int
    duration: float
