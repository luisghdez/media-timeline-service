from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json

OUTPUT_LAYER_TYPES = frozenset({"speech", "vocal_reaction"})


@dataclass(slots=True)
class Evidence:
    source: str
    detail: str
    score: float | None = None


@dataclass(slots=True)
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    kind: str = "word"
    timing_method: str = "asr_estimate"
    timing_confidence: float | None = None
    start_sample: int | None = None
    end_sample: int | None = None


@dataclass(slots=True)
class DeliverySpan:
    start: float
    end: float
    features: list[str] = field(default_factory=list)
    description: str | None = None
    confidence: float | None = None
    perceived_emotion: str | None = None
    emotion_confidence: float | None = None
    timing_method: str = "utterance_scope"
    evidence: list[Evidence] = field(default_factory=list)
    start_sample: int | None = None
    end_sample: int | None = None


@dataclass(slots=True)
class Layer:
    type: str
    description: str | None = None
    text: str | None = None
    emotion: str | None = None
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    verbatim_text: str | None = None
    transcript_mode: str | None = None
    words: list[Word] = field(default_factory=list)
    delivery: list[DeliverySpan] = field(default_factory=list)
    transcript_confidence: float | None = None
    timing_confidence: float | None = None
    timing_method: str = "unknown"
    review_reasons: list[str] = field(default_factory=list)
    context_hint: str | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def signature(self) -> str:
        # Timing and delivery are part of identity: identical phrases at different
        # times must never erase each other's word or annotation spans.
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    def needs_review(self) -> bool:
        return bool(self.review_reasons) or self.confidence < 0.6 or any(
            value is not None and value < 0.6
            for value in (self.transcript_confidence, self.timing_confidence)
        ) or any(
            word.confidence is not None and word.confidence < 0.6 for word in self.words
        ) or any(
            value is not None and value < 0.6
            for span in self.delivery
            for value in (span.confidence, span.emotion_confidence)
        )


@dataclass(slots=True)
class TimedLayer:
    start: float
    end: float
    layer: Layer
    start_sample: int | None = None
    end_sample: int | None = None


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    layers: list[Layer]
    visual_context: str | None = None
    review_required: bool = False

    def signature(self) -> tuple[str, ...]:
        return tuple(sorted(layer.signature() for layer in self.layers))


@dataclass(slots=True)
class AnalysisResult:
    duration: float
    segments: list[Segment]
    source: dict[str, Any]
    events: list[TimedLayer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [
            {
                "start": event.start,
                "end": event.end,
                "start_sample": event.start_sample,
                "end_sample": event.end_sample,
                **asdict(event.layer),
                "review_required": event.layer.needs_review(),
            }
            for event in self.events
        ]
        for segment, serialized in zip(self.segments, payload["segments"]):
            serialized["review_required"] = segment.review_required or any(
                layer.needs_review() for layer in segment.layers
            )
        return _clean(payload)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value
