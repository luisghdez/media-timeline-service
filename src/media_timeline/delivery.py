from __future__ import annotations

import re
from pathlib import Path

from .schema import Evidence, TimedLayer

_LEADING_OH = re.compile(r"^(oh+,?\s*)+", re.IGNORECASE)

def _nearby_lexical_tail(layers: list[TimedLayer], index: int) -> str | None:
    for candidate in layers[max(0, index - 1) : index + 2]:
        if candidate.layer.type != "speech" or not candidate.layer.text:
            continue
        clause = re.split(r"[,.!?]", candidate.layer.text)[0].strip()
        words = clause.split()
        if not words or len(words) > 3:
            continue
        if words[0].lower().strip("'") in {"oh", "ah", "my", "no", "whoa", "ow"}:
            return clause
    return None


def annotate_vocal_delivery(layers: list[TimedLayer], audio_path: str | Path) -> list[TimedLayer]:
    try:
        import soundfile as sf
    except ImportError:
        return layers

    path = Path(audio_path)
    if not path.is_file():
        return layers
    audio, sample_rate = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return layers

    last_stutter = -10.0
    ordered = sorted(layers, key=lambda item: (item.start, item.end))
    for index, item in enumerate(ordered):
        if item.layer.type != "vocal_reaction":
            continue
        if "gasp" not in (item.layer.description or "").lower():
            continue
        if item.start - last_stutter < 1.5:
            continue
        previous_end = ordered[index - 1].end if index else 0.0
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else item.end + 0.12
        window_start = max(previous_end, item.start - 0.55)
        window_end = min(item.end + 0.08, next_start)
        if window_end - window_start < 0.2:
            continue
        onsets = _energy_onsets(audio, sample_rate, window_start, window_end)
        if not _is_stuttered_onset_pattern(onsets):
            continue
        item.start = min(item.start, max(window_start, onsets[0] - 0.04))
        item.end = min(next_start, max(item.end, onsets[-1] + 0.08))
        lexical = item.layer.text or _nearby_lexical_tail(ordered, index)
        item.layer.delivery = "stuttered / panicked"
        item.layer.emotion = "panicked"
        item.layer.description = "Panicked, stuttered vocal reaction"
        item.layer.text = _render_stuttered_text(lexical, len(onsets))
        item.layer.evidence.append(
            Evidence("prosody", f"{len(onsets)} energy pulses in vocal stem", len(onsets))
        )
        last_stutter = item.start
    return ordered


def _energy_onsets(audio, sample_rate: int, start: float, end: float) -> list[float]:
    import numpy as np

    begin = max(0, int(start * sample_rate))
    stop = min(len(audio), int(end * sample_rate))
    chunk = np.asarray(audio[begin:stop], dtype="float32")
    if len(chunk) < sample_rate // 20:
        return []
    win = max(1, int(0.02 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    rms = np.array(
        [float(np.sqrt(np.mean(chunk[i : i + win] ** 2) + 1e-12)) for i in range(0, len(chunk) - win, hop)],
        dtype="float32",
    )
    if rms.size < 3:
        return []
    threshold = max(0.04, float(np.percentile(rms, 72)))
    onsets: list[float] = []
    for index in range(1, len(rms) - 1):
        if rms[index] < threshold or rms[index] < rms[index - 1] or rms[index] < rms[index + 1]:
            continue
        time = start + index * hop / sample_rate
        if not onsets or time - onsets[-1] >= 0.08:
            onsets.append(time)
    return onsets


def _is_stuttered_onset_pattern(onsets: list[float]) -> bool:
    if len(onsets) < 3:
        return False
    gaps = [onsets[index] - onsets[index - 1] for index in range(1, len(onsets))]
    median = sorted(gaps)[len(gaps) // 2]
    return 0.07 <= median <= 0.22


def _render_stuttered_text(text: str | None, pulses: int) -> str:
    rendered_pulses = min(max(pulses, 3), 4)
    stem = "O" + "-o" * (rendered_pulses - 2) + "-oh"
    remainder = _LEADING_OH.sub("", (text or "").strip()).strip(" !")
    if remainder:
        return f"{stem} {remainder}!"
    return f"{stem}!"
