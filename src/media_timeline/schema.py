from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

OUTPUT_LAYER_TYPES = frozenset({"speech", "vocal_reaction"})


@dataclass(slots=True)
class Evidence:
    source: str
    detail: str
    score: float | None = None


@dataclass(slots=True)
class Layer:
    type: str
    description: str | None = None
    text: str | None = None
    emotion: str | None = None
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)

    def signature(self) -> tuple[str, str | None, str | None, str | None]:
        return self.type, self.description, self.text, self.emotion


@dataclass(slots=True)
class TimedLayer:
    start: float
    end: float
    layer: Layer


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    layers: list[Layer]
    visual_context: str | None = None
    review_required: bool = False

    def signature(self) -> tuple[tuple[str, str | None, str | None, str | None], ...]:
        return tuple(sorted(layer.signature() for layer in self.layers))


@dataclass(slots=True)
class AnalysisResult:
    duration: float
    segments: list[Segment]
    source: dict[str, Any]
    events: list[TimedLayer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [
            {
                "start": event.start,
                "end": event.end,
                **asdict(event.layer),
                "review_required": event.layer.confidence < 0.6,
            }
            for event in self.events
        ]
        return _clean(payload)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    return value
