"""Explicit waveform clock mapping and optional acoustic forced alignment."""
from __future__ import annotations

import math
from pathlib import Path

from .schema import Evidence, TimedLayer


def stamp_timeline(items: list[TimedLayer], sample_rate: int, offset: float, duration: float) -> list[TimedLayer]:
    """Map analysis WAV seconds to source playback seconds, once, at export.

    Sample indices always refer to the extracted analysis WAV, before offset.
    Precision is a coordinate representation, not a claim of measured accuracy.
    """
    output = []
    for item in items:
        if not (math.isfinite(item.start) and math.isfinite(item.end)):
            continue
        item.start_sample = max(0, round(item.start * sample_rate))
        item.end_sample = max(0, round(item.end * sample_rate))
        item.start = max(0.0, min(duration, item.start + offset))
        item.end = max(0.0, min(duration, item.end + offset))
        for word in item.layer.words:
            if word.start is not None and word.end is not None:
                word.start_sample = max(0, round(word.start * sample_rate))
                word.end_sample = max(0, round(word.end * sample_rate))
                word.start = max(0.0, min(duration, word.start + offset))
                word.end = max(0.0, min(duration, word.end + offset))
        for span in item.layer.delivery:
            span.start_sample = max(0, round(span.start * sample_rate))
            span.end_sample = max(0, round(span.end * sample_rate))
            span.start = max(item.start, min(item.end, span.start + offset))
            span.end = max(item.start, min(item.end, span.end + offset))
        if item.end > item.start:
            output.append(item)
    return output


class ForcedAligner:
    def __init__(self, cache_dir: Path, language: str = "en", device: str = "cpu", offline: bool = False):
        self.cache_dir, self.language, self.device, self.offline = cache_dir, language, device, offline

    def align(self, items: list[TimedLayer], audio_path: Path) -> list[TimedLayer]:
        from whisperx.alignment import align, load_align_model
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError("Forced alignment requires the mono 16 kHz analysis waveform")
        model, metadata = load_align_model(
            self.language, self.device, model_dir=str(self.cache_dir), model_cache_only=self.offline,
        )
        duration = len(audio) / sr
        for item in items:
            if item.layer.type != "speech" or not item.layer.words:
                continue
            # One existing token per alignment segment preserves identity, including
            # repetitions; padded windows allow correction of ASR boundary drift.
            transcript = [
                {"start": max(0.0, (word.start if word.start is not None else item.start) - 0.25),
                 "end": min(duration, (word.end if word.end is not None else item.end) + 0.25),
                 "text": word.text}
                for word in item.layer.words
            ]
            segments = []
            for token in transcript:
                result = align([token], model, metadata, audio, self.device,
                               interpolate_method="nearest", return_char_alignments=True)
                # Alignment may split sentences or return no segment. Keep one
                # slot per input token so a failure cannot shift later words.
                segments.append({"chars": [char for part in result.get("segments", [])
                                           for char in part.get("chars", [])]})
            apply_alignment(item, segments)

        return items


def apply_alignment(item: TimedLayer, segments: list[dict]) -> None:
    """Never fabricate times for unsupported letters or silently lose a token."""
    complete = len(segments) == len(item.layer.words)
    previous_end = -1.0
    for index, word in enumerate(item.layer.words):
        segment = segments[index] if index < len(segments) else {}
        chars = [c for c in segment.get("chars", []) if c.get("char", "").isalnum()]
        expected = "".join(c.lower() for c in word.text if c.isalnum())
        observed = "".join(c.get("char", "").lower() for c in chars)
        valid = chars and expected == observed and all(
            isinstance(c.get("start"), (int, float)) and isinstance(c.get("end"), (int, float))
            and math.isfinite(c["start"]) and math.isfinite(c["end"])
            and 0 <= c["start"] < c["end"] for c in chars
        )
        if valid:
            start, end = min(c["start"] for c in chars), max(c["end"] for c in chars)
            valid = start >= previous_end
        if not valid:
            complete = False
            previous_end = max(previous_end, word.end if word.end is not None else previous_end)
            word.timing_method = "asr_estimate_alignment_failed"
            continue
        word.start, word.end = start, end
        word.timing_method = "forced_alignment"
        scores = [c["score"] for c in chars if isinstance(c.get("score"), (int, float))]
        word.timing_confidence = min(scores) if scores else None
        previous_end = end
    if complete:
        item.start = min(word.start for word in item.layer.words)
        item.end = max(word.end for word in item.layer.words)
        item.layer.timing_method = "forced_alignment"
        scores = [w.timing_confidence for w in item.layer.words if w.timing_confidence is not None]
        item.layer.timing_confidence = min(scores) if scores else None
    else:
        item.layer.timing_method = "mixed_alignment"
        item.layer.review_reasons.append("incomplete_forced_alignment")
        # Ensure the enclosing event still contains every retained word.
        item.start = min([item.start] + [w.start for w in item.layer.words if w.start is not None])
        item.end = max([item.end] + [w.end for w in item.layer.words if w.end is not None])
    item.layer.evidence.append(Evidence("alignment", "WhisperX character alignment; unaligned tokens retain ASR estimates"))


def refine_reaction_boundaries(items: list[TimedLayer], audio_path: Path) -> list[TimedLayer]:
    """Refine isolated, silence-bracketed reactions; keep ambiguous windows coarse.

    Energy cannot distinguish a scream from simultaneous speech or music. Only a
    single isolated active region is eligible, and the result remains an estimate.
    """
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    speech = [item for item in items if item.layer.type == "speech"]
    hop = max(1, round(sr * 0.01))
    for item in items:
        if item.layer.type != "vocal_reaction":
            continue
        if any(word.start < item.end and word.end > item.start for word in speech):
            continue
        left, right = max(0, round((item.start - 0.15) * sr)), min(len(audio), round((item.end + 0.15) * sr))
        chunk = audio[left:right]
        count = len(chunk) // hop
        if count < 5:
            continue
        rms = np.sqrt(np.mean(chunk[:count * hop].reshape(count, hop) ** 2, axis=1))
        threshold = max(0.005, float(rms.max()) * 0.12)
        active = rms >= threshold
        # Require quiet on both sides; never claim an ongoing sound is bounded.
        if active[:2].any() or active[-2:].any():
            continue
        indices = np.flatnonzero(active)
        if len(indices) == 0 or np.any(np.diff(indices) > 3):
            continue
        start, end = (left + int(indices[0]) * hop) / sr, (left + (int(indices[-1]) + 1) * hop) / sr
        if start >= item.end or end <= item.start:
            continue
        item.start, item.end = start, end
        item.layer.timing_method = "isolated_energy_envelope_10ms"
        item.layer.review_reasons = [r for r in item.layer.review_reasons if r != "coarse_reaction_timing"]
        item.layer.review_reasons.append("energy_boundary_requires_validation")
        item.layer.evidence.append(Evidence("boundary", "Single energy region bracketed by quiet; not class-specific localization"))
    return items
