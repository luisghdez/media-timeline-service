from __future__ import annotations

from copy import deepcopy

from .schema import OUTPUT_LAYER_TYPES, Evidence, Layer, Segment, TimedLayer


def fuse_timeline(
    duration: float,
    layers: list[TimedLayer],
    silence: list[tuple[float, float]],
    scene_changes: list[float],
    minimum_segment: float = 0.0,
) -> list[Segment]:
    """A presentation view of authoritative events; never move audio boundaries.

    Layer words/delivery retain absolute event times even when a visual cut splits
    the presentation. Silence is handled before fusion, not by midpoint deletion.
    minimum_segment is retained for callers but cannot override acoustic evidence.
    """
    boundaries = sorted({0.0, duration, *(
        max(0.0, min(duration, point))
        for item in layers for point in (item.start, item.end)
    ), *(point for point in scene_changes if 0 < point < duration)})
    segments: list[Segment] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = _deduplicate_layers([
            deepcopy(item.layer) for item in layers
            if item.start < end and item.end > start and item.layer.type in OUTPUT_LAYER_TYPES
        ])
        if not active:
            continue
        visual = "Visual scene transition at this boundary" if start in scene_changes else None
        segment = Segment(start, end, active, visual, any(layer.needs_review() for layer in active))
        if (segments and segments[-1].end == start and not visual
                and segments[-1].signature() == segment.signature()):
            segments[-1].end = end
        else:
            segments.append(segment)
    return segments


def prepare_layers(
    layers: list[TimedLayer],
    scene_changes: list[float],
    context_text: str | None = None,
) -> list[TimedLayer]:
    """Preserve lexical speech, distinct reactions, and provenance separately."""
    output: list[TimedLayer] = []
    seen: set[tuple[float, float, str]] = set()
    for original in sorted(layers, key=lambda item: (item.start, item.end, item.layer.type)):
        if original.layer.type not in OUTPUT_LAYER_TYPES or original.end <= original.start:
            continue
        item = deepcopy(original)
        if context_text and item.layer.type == "vocal_reaction":
            item.layer.context_hint = context_text
            item.layer.evidence.append(Evidence("context", "Unverified source caption/title; not a transcript"))
        key = (item.start, item.end, item.layer.signature())
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def remove_silent_overlaps(
    layers: list[TimedLayer],
    silence: list[tuple[float, float]],
    minimum_duration: float = 0.0,
) -> list[TimedLayer]:
    """Speech conflicts need review; subtract silence from reaction candidates.

    An energy threshold is not authority to delete quiet words. Splitting reactions
    retains *both* sides of every silence rather than dropping the entire tail.
    """
    output: list[TimedLayer] = []
    for original in layers:
        item = deepcopy(original)
        overlaps = [(a, b) for a, b in silence if a < item.end and b > item.start]
        if item.layer.type == "speech":
            if overlaps:
                item.layer.review_reasons.append("speech_overlaps_detected_silence")
            output.append(item)
            continue
        intervals = [(item.start, item.end)]
        for a, b in sorted(overlaps):
            remaining = []
            for start, end in intervals:
                if a >= end or b <= start:
                    remaining.append((start, end))
                else:
                    if start < a:
                        remaining.append((start, a))
                    if b < end:
                        remaining.append((b, end))
            intervals = remaining
        for start, end in intervals:
            if end <= start:
                continue
            part = deepcopy(item)
            part.start, part.end = start, end
            if (start, end) != (item.start, item.end):
                part.layer.evidence.append(Evidence("silence", "Candidate interval trimmed at detected silence"))
            output.append(part)
    return output


def _deduplicate_layers(layers: list[Layer]) -> list[Layer]:
    unique = {layer.signature(): layer for layer in layers}
    return sorted(unique.values(), key=lambda layer: (layer.type, -layer.confidence))
