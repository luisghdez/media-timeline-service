from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OrchestrateResult:
    output_path: Path
    vocals_path: Path
    wan3_path: Path
    seed: int
    duration: float
    wan3_url: str
    width: int
    height: int

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "output_path": str(self.output_path),
            "vocals_path": str(self.vocals_path),
            "wan3_path": str(self.wan3_path),
            "seed": self.seed,
            "duration": self.duration,
            "wan3_url": self.wan3_url,
            "width": self.width,
            "height": self.height,
        }
