from __future__ import annotations

from copy import deepcopy
import re

from .schema import OUTPUT_LAYER_TYPES, Evidence, Layer, Segment, TimedLayer


def fuse_timeline(
    duration: float,
    layers: list[TimedLayer],
    silence: list[tuple[float, float]],
    scene_changes: list[float],
    minimum_segment: float = 0.12,
) -> list[Segment]:
    boundaries = [0.0, duration]
    boundaries.extend(point for item in layers for point in (item.start, item.end))
    boundaries.extend(point for item in silence for point in item)
    boundaries.extend(scene_changes)
    boundaries = _coalesce_boundaries(boundaries, duration, minimum_segment)

    segments: list[Segment] = []
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        active = [
            deepcopy(item.layer)
            for item in layers
            if item.start < end and item.end > start and item.layer.type in OUTPUT_LAYER_TYPES
        ]
        in_silence = any(left <= midpoint <= right for left, right in silence)
        if in_silence:
            active = [layer for layer in active if layer.type == "speech"]
        active = _deduplicate_layers(active)
        if not active:
            continue
        nearby_scene = any(abs(point - start) <= 0.12 for point in scene_changes)
        segments.append(
            Segment(
                start=start,
                end=end,
                layers=active,
                visual_context="Visual scene transition near this boundary" if nearby_scene else None,
                review_required=any(layer.confidence < 0.45 for layer in active),
            )
        )
    return _merge_segments(segments, minimum_segment)


def prepare_layers(
    layers: list[TimedLayer],
    scene_changes: list[float],
    context_text: str | None = None,
) -> list[TimedLayer]:
    prepared = _merge_timed_layers(layers)
    prepared = _snap_reactions_to_scenes(prepared, scene_changes)
    prepared = _merge_timed_layers(prepared)
    prepared = _merge_reaction_runs(prepared)
    prepared = _promote_short_exclamations(prepared, scene_changes)
    prepared = _apply_context_reaction_hint(prepared, context_text)
    prepared = _suppress_nested_reactions(prepared)
    prepared = [item for item in prepared if item.layer.type in OUTPUT_LAYER_TYPES]
    return sorted(prepared, key=lambda item: (item.start, item.end, item.layer.type))


def _merge_reaction_runs(layers: list[TimedLayer], gap: float = 0.25) -> list[TimedLayer]:
    reactions = sorted(
        (item for item in layers if item.layer.type == "vocal_reaction"),
        key=lambda item: (item.start, item.end),
    )
    non_reactions = [item for item in layers if item.layer.type != "vocal_reaction"]
    runs: list[list[TimedLayer]] = []
    for item in reactions:
        if runs and item.start <= max(part.end for part in runs[-1]) + gap:
            runs[-1].append(item)
        else:
            runs.append([item])

    merged: list[TimedLayer] = []
    for run in runs:
        start = min(item.start for item in run)
        end = max(item.end for item in run)
        generic = [
            item for item in run
            if "scream, wail, or moan" in (item.layer.description or "")
        ]
        if len(run) == 1 or end - start < 1.5 or not generic:
            merged.extend(run)
            continue
        representative = max(generic, key=lambda item: item.layer.confidence)
        evidence: list[Evidence] = []
        seen: set[tuple[str, str, float | None]] = set()
        for item in run:
            for record in item.layer.evidence:
                key = (record.source, record.detail, record.score)
                if key not in seen:
                    evidence.append(record)
                    seen.add(key)
        merged.append(
            TimedLayer(
                start=start,
                end=end,
                layer=Layer(
                    type="vocal_reaction",
                    description="Nonverbal vocal reaction such as a scream, wail, or moan",
                    emotion="distressed / intense",
                    confidence=max(item.layer.confidence for item in run),
                    evidence=evidence or representative.layer.evidence,
                ),
            )
        )
    return non_reactions + merged


def _merge_timed_layers(layers: list[TimedLayer], gap: float = 0.08) -> list[TimedLayer]:
    ordered = sorted(layers, key=lambda item: (repr(item.layer.signature()), item.start))
    output: list[TimedLayer] = []
    for item in ordered:
        if output and output[-1].layer.signature() == item.layer.signature() and item.start <= output[-1].end + gap:
            current = output[-1]
            current.end = max(current.end, item.end)
            if item.layer.confidence > current.layer.confidence:
                current.layer.confidence = item.layer.confidence
                current.layer.evidence = item.layer.evidence
        else:
            output.append(item)
    return output


def _snap_reactions_to_scenes(layers: list[TimedLayer], scene_changes: list[float]) -> list[TimedLayer]:
    for item in layers:
        if item.layer.type != "vocal_reaction":
            continue
        starts = [point for point in scene_changes if abs(point - item.start) <= 0.75]
        ends = [point for point in scene_changes if abs(point - item.end) <= 0.75]
        original_start, original_end = item.start, item.end
        snapped_start = min(starts, key=lambda point: abs(point - item.start)) if starts else item.start
        snapped_end = min(ends, key=lambda point: abs(point - item.end)) if ends else item.end
        if snapped_end - snapped_start >= 0.12:
            item.start, item.end = snapped_start, snapped_end
        else:
            item.start, item.end = original_start, original_end
    return [item for item in layers if item.end - item.start >= 0.12]


def _promote_short_exclamations(layers: list[TimedLayer], scene_changes: list[float]) -> list[TimedLayer]:
    speech = [item for item in layers if item.layer.type == "speech"]
    reactions = [item for item in layers if item.layer.type == "vocal_reaction"]
    removed: set[int] = set()

    for spoken in speech:
        words = (spoken.layer.text or "").replace("!", "").replace("?", "").split()
        overlaps = [
            reaction
            for reaction in reactions
            if reaction.start <= spoken.end + 0.4 and reaction.end >= spoken.start - 0.4
        ]
        if len(words) <= 2 and overlaps:
            reaction = max(overlaps, key=lambda item: item.layer.confidence)
            reaction.layer.text = spoken.layer.text
            reaction.layer.confidence = max(reaction.layer.confidence, spoken.layer.confidence * 0.9)
            reaction.layer.evidence.extend(spoken.layer.evidence)
            earlier_scenes = [point for point in scene_changes if reaction.start - 0.25 <= point <= spoken.start]
            if earlier_scenes:
                reaction.start = earlier_scenes[-1]
            removed.add(id(spoken))
        elif len(words) > 2:
            for reaction in overlaps:
                overlap = max(0.0, min(spoken.end, reaction.end) - max(spoken.start, reaction.start))
                if overlap >= min(0.5, (reaction.end - reaction.start) * 0.5):
                    removed.add(id(reaction))

    return [item for item in layers if id(item) not in removed]


def _apply_context_reaction_hint(layers: list[TimedLayer], context_text: str | None) -> list[TimedLayer]:
    if not context_text:
        return layers
    full_tokens = [
        match.group(0)
        for match in re.finditer(r"\b[A-Za-z]*([A-Za-z])\1{2,}[A-Za-z]*\b", context_text)
    ]
    if not full_tokens:
        return layers
    candidates = [
        item
        for item in layers
        if item.layer.type == "vocal_reaction"
        and item.layer.text is None
        and item.end - item.start >= 1.5
        and "scream, wail, or moan" in (item.layer.description or "")
    ]
    if not candidates:
        return layers
    candidate = max(candidates, key=lambda item: (item.end - item.start) * item.layer.confidence)
    token = max(full_tokens, key=len)
    candidate.layer.text = token.capitalize() + "!"
    candidate.layer.confidence = min(0.88, max(candidate.layer.confidence, 0.74))
    candidate.layer.evidence.append(
        Evidence("context", f"Caption/title token matched to sustained vocal reaction: {token}")
    )
    return layers


def _suppress_nested_reactions(layers: list[TimedLayer]) -> list[TimedLayer]:
    sustained = [
        item
        for item in layers
        if item.layer.type == "vocal_reaction"
        and "scream, wail, or moan" in (item.layer.description or "")
        and item.end - item.start >= 1.5
    ]
    output: list[TimedLayer] = []
    for item in layers:
        if item.layer.type == "vocal_reaction" and item not in sustained:
            duration = max(0.001, item.end - item.start)
            if any(
                max(0.0, min(item.end, parent.end) - max(item.start, parent.start)) / duration >= 0.5
                for parent in sustained
            ):
                continue
        output.append(item)
    return output


def remove_silent_overlaps(
    layers: list[TimedLayer],
    silence: list[tuple[float, float]],
    minimum_duration: float = 0.12,
) -> list[TimedLayer]:
    output: list[TimedLayer] = []
    for item in layers:
        keep = True
        for start, end in silence:
            overlap = max(0.0, min(item.end, end) - max(item.start, start))
            if overlap <= 0:
                continue
            if item.start < start and start - item.start >= minimum_duration:
                item.end = min(item.end, start)
                if item.layer.type == "vocal_reaction" and item.end - item.start < 0.25:
                    keep = False
                    break
            else:
                keep = False
                break
        if keep and item.end - item.start >= minimum_duration:
            output.append(item)
    return output


def _coalesce_boundaries(values: list[float], duration: float, tolerance: float) -> list[float]:
    ordered = sorted(max(0.0, min(duration, value)) for value in values)
    output: list[float] = []
    for value in ordered:
        if not output or value - output[-1] >= tolerance:
            output.append(value)
        elif value == duration:
            output[-1] = duration
    if output[0] != 0.0:
        output.insert(0, 0.0)
    if output[-1] != duration:
        output.append(duration)
    return output


def _deduplicate_layers(layers: list[Layer]) -> list[Layer]:
    best: dict[tuple[str, str | None, str | None, str | None], Layer] = {}
    for layer in layers:
        signature = layer.signature()
        current = best.get(signature)
        if current is None or layer.confidence > current.confidence:
            best[signature] = layer
    priority = {"speech": 0, "vocal_reaction": 1}
    return sorted(best.values(), key=lambda layer: (priority.get(layer.type, 9), -layer.confidence))


def _merge_segments(segments: list[Segment], minimum_segment: float) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and merged[-1].signature() == segment.signature() and not segment.visual_context:
            merged[-1].end = segment.end
            merged[-1].review_required = merged[-1].review_required or segment.review_required
        elif segment.end - segment.start < minimum_segment and merged:
            merged[-1].end = segment.end
        else:
            merged.append(segment)
    return merged
