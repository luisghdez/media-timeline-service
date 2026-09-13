from __future__ import annotations

import subprocess
import sys
import threading
import math
import re
from pathlib import Path

from .schema import Evidence, Layer, TimedLayer, Word


class SpeechAnalyzer:
    _models: dict[tuple[str, str, str], object] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        offline: bool = False,
        vad_filter: bool = False,
        language: str = "en",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.offline = offline
        self.vad_filter = vad_filter
        self.language = language

    def analyze(self, audio_path: Path) -> list[TimedLayer]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the 'asr' extra to enable speech transcription") from exc

        key = (self.model_name, "cpu", "int8")
        with self._lock:
            model = self._models.get(key)
            if model is None:
                model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.cache_dir),
                    local_files_only=self.offline,
                )
                self._models[key] = model

        segments, _ = model.transcribe(
            str(audio_path),
            beam_size=8,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            temperature=0,
            language=self.language,
        )
        output: list[TimedLayer] = []
        for segment in segments:
            if not segment.text.strip():
                continue
            output.extend(_split_segment(segment, self.model_name))
        return output


def _split_segment(segment: object, model_name: str, gap_seconds: float = 0.65,
                   transcript_mode: str = "asr", timing_method: str = "asr_estimate") -> list[TimedLayer]:
    # Low-probability syllables remain reviewable hypotheses, not silent deletions.
    words = list(segment.words or [])
    if not words:
        return [
            TimedLayer(
                start=float(segment.start),
                end=float(segment.end),
                layer=Layer(
                    type="speech",
                    text=segment.text.strip(),
                    confidence=0.0,
                    verbatim_text=segment.text.strip(),
                    transcript_mode=transcript_mode,
                    timing_method="segment_estimate",
                    review_reasons=["word_timestamps_unavailable", "transcript_confidence_unavailable"],
                    evidence=[Evidence("asr", model_name)],
                ),
            )
        ]

    groups: list[list[object]] = [[words[0]]]
    for word in words[1:]:
        if float(word.start) - float(groups[-1][-1].end) > gap_seconds:
            groups.append([word])
        else:
            groups[-1].append(word)

    output: list[TimedLayer] = []
    for group in groups:
        text = "".join(str(word.word) for word in group).strip()
        scores = [float(word.probability) for word in group if getattr(word, "probability", None) is not None]
        confidence = sum(scores) / len(scores) if scores else 0.0
        tokens = [Word(
            text=str(word.word).strip(), start=float(word.start), end=float(word.end),
            confidence=getattr(word, "probability", None),
            kind="partial_syllable" if str(word.word).strip().endswith("-") else "word",
            timing_method=timing_method,
        ) for word in group]
        for index, token in enumerate(tokens):
            normalized = token.text.strip("[] ,.!?").lower()
            if transcript_mode == "verbatim" and token.text.startswith("["):
                token.kind = "filler" if normalized in {"uh", "um", "erm", "er"} else "vocalization"
            elif index and normalized == tokens[index - 1].text.strip(" ,.!?").lower():
                token.kind = "repetition"
        reasons = []
        if getattr(segment, "no_speech_prob", 0) > 0.8:
            reasons.append("high_no_speech_probability")
        if len(scores) != len(group):
            reasons.append("transcript_confidence_unavailable")
        if any(word.end <= word.start for word in tokens):
            reasons.append("invalid_word_duration")
        output.append(
            TimedLayer(
                start=float(group[0].start),
                end=float(group[-1].end),
                layer=Layer(
                    type="speech",
                    text=_readable_text(text) if transcript_mode == "verbatim" else text,
                    verbatim_text=text,
                    transcript_mode=transcript_mode,
                    words=tokens,
                    transcript_confidence=confidence if scores else None,
                    timing_method=timing_method,
                    review_reasons=reasons,
                    confidence=confidence,
                    evidence=[Evidence("asr", model_name, confidence if scores else None)],
                ),
            )
        )
    if re.sub(r"\s+", "", segment.text) != re.sub(r"\s+", "", "".join(str(w.word) for w in words)):
        # Some backends omit unplaceable tokens from their word array. Preserve
        # the complete transcript and expose missing word times explicitly.
        from difflib import SequenceMatcher
        tokens = [token for event in output for token in event.layer.words]
        expected = segment.text.split()
        matcher = SequenceMatcher(None, expected, [token.text for token in tokens], autojunk=False)
        restored = []
        for tag, a, b, c, d in matcher.get_opcodes():
            if tag == "equal":
                restored.extend(tokens[c:d])
            else:
                restored.extend(Word(text=value, timing_method="unaligned") for value in expected[a:b])
        event = output[0]
        event.start, event.end = float(segment.start), float(segment.end)
        event.layer.verbatim_text = segment.text.strip()
        event.layer.text = _readable_text(segment.text.strip()) if transcript_mode == "verbatim" else segment.text.strip()
        event.layer.words = restored
        event.layer.review_reasons.append("transcript_word_mismatch")
        event.layer.timing_method = "segment_estimate"
        return [event]
    return output


def _readable_text(text: str) -> str:
    """Display-only normalization. Authoritative words/verbatim remain untouched."""
    def replace(match):
        parts = re.split(r"-\s*", match.group(0))
        return parts[-1] if all(parts[-1].lower().startswith(p.lower()) for p in parts[:-1]) else match.group(0)
    return re.sub(r"\b(?:\w+-\s*)+\w+", replace, text).strip()


class VerbatimSpeechAnalyzer:
    """Opt-in CrisperWhisper adapter, isolated from faster-whisper's CT2 runtime."""
    _models: dict[tuple[str, str], object] = {}

    def __init__(self, model_name: str, cache_dir: Path, offline: bool = False,
                 language: str = "en", device: str = "cpu"):
        self.model_name, self.cache_dir, self.offline = model_name, cache_dir, offline
        self.language, self.device = language, device

    def analyze(self, audio_path: Path) -> list[TimedLayer]:
        from crisperwhisper import CrisperWhisperModel
        from huggingface_hub import snapshot_download
        from types import SimpleNamespace
        import soundfile as sf

        # Resolve locally ourselves so offline mode never silently downloads.
        name = self.model_name
        if name in {"small", "medium", "turbo", "large"}:
            name = f"nyralabs/CrisperWhisper2.0_{name}"
        model_path = str(Path(name).resolve()) if Path(name).is_dir() else snapshot_download(
            name, cache_dir=str(self.cache_dir), local_files_only=self.offline,
        )
        key = (model_path, self.device)
        if key not in self._models:
            self._models[key] = CrisperWhisperModel(
                model_path, backend="transformers", device=self.device,
                compute_type="float32" if self.device == "cpu" else "float16",
            )
        result = self._models[key].transcribe(
            str(audio_path), language=self.language, mode="verbatim", word_timestamps=True,
        )
        _check_verbatim_quality(result)
        # This backend does not promise word probabilities. Unknown stays unknown.
        words = [SimpleNamespace(word=" " + w.word.strip(), start=w.start, end=w.end,
                                 probability=getattr(w, "probability", None)) for w in (result.words or [])]
        info = sf.info(str(audio_path))
        segment = SimpleNamespace(words=words, text=result.text, start=0.0,
                                  end=info.duration, no_speech_prob=0.0)
        return _split_segment(segment, f"crisperwhisper:{name}", transcript_mode="verbatim",
                              timing_method="cross_attention_alignment") if result.text.strip() else []


def _check_verbatim_quality(result: object) -> None:
    """Reject pathological decode loops, while keeping ordinary audible repeats.

    Repetition alone is not an error. Require a long run plus poor timing coverage
    or an implausible output rate; a short stutter never triggers this guard.
    """
    tokens = re.findall(r"\w+", result.text.lower())
    longest = run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    placed = sum(w.start is not None and w.end is not None and w.end > w.start
                 for w in (result.words or []))
    duration = max(0.001, float(result.duration))
    if len(tokens) >= 24 and longest >= 12 and (
        placed < len(tokens) * 0.7 or len(tokens) / duration > 12
    ):
        raise ValueError(f"Rejected verbatim decode loop: {longest} consecutive repeated tokens, "
                         f"{placed}/{len(tokens)} positively timed words")


class AudioEventAnalyzer:
    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    _model: object | None = None
    _processor: object | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        cache_dir: Path,
        window_seconds: float = 2.0,
        hop_seconds: float = 0.5,
        offline: bool = False,
        threshold: float = 0.07,
    ) -> None:
        self.cache_dir = cache_dir
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.threshold = threshold
        self.offline = offline

    def analyze(self, audio_path: Path, duration: float) -> list[TimedLayer]:
        try:
            import soundfile as sf
            import torch
            from transformers import AutoModelForAudioClassification, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the 'events' extra to enable sound-event detection") from exc

        with self._lock:
            if self.__class__._processor is None:
                self.__class__._processor = AutoProcessor.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=str(self.cache_dir),
                    local_files_only=self.offline,
                )
                self.__class__._model = AutoModelForAudioClassification.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=str(self.cache_dir),
                    local_files_only=self.offline,
                )
        processor = self.__class__._processor
        model = self.__class__._model
        audio, sample_rate = sf.read(str(audio_path))
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        expected_rate = int(getattr(processor, "sampling_rate", 16_000))
        if sample_rate != expected_rate:
            audio = _resample_audio(audio, sample_rate, expected_rate)
            sample_rate = expected_rate

        model.eval()
        duration = min(duration, len(audio) / sample_rate)
        detections: list[TimedLayer] = []
        position = 0.0
        while position < duration:
            stop = min(duration, position + self.window_seconds)
            chunk = audio[int(position * sample_rate) : int(stop * sample_rate)]
            if len(chunk) < sample_rate // 4:
                break
            inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                probabilities = model(**inputs).logits[0].sigmoid()
            values, indices = torch.topk(probabilities, 10)
            labels = [(model.config.id2label[int(index)], float(value)) for value, index in zip(values, indices)]
            canonical = _canonical_events(labels, self.threshold)
            for event_type, description, emotion, confidence, raw_labels in canonical:
                # A classifier identifies a candidate window, not its exact onset.
                event_start, event_end = position, stop
                detections.append(
                    TimedLayer(
                        start=event_start,
                        end=event_end,
                        layer=Layer(
                            type=event_type,
                            description=description,
                            emotion=emotion,
                            confidence=confidence,
                            timing_method="classifier_window",
                            review_reasons=["coarse_reaction_timing", "uncalibrated_event_score"],
                            evidence=[Evidence("audio_event", ", ".join(raw_labels), confidence)],
                        ),
                    )
                )
            position += self.hop_seconds
        return _merge_detections(detections, gap=0.05)


REACTION_LABELS = {
    "Screaming": ("scream", "intense / distressed"),
    "Wail, moan": ("prolonged wail or moan", "distressed"),
    "Whimper": ("whimpering vocal reaction", "distressed"),
    "Crying, sobbing": ("crying or sobbing", "sad / distressed"),
    "Gasp": ("audible gasp", "surprised"),
    "Laughter": ("laughter", "amused"),
    "Giggle": ("giggle", "amused"),
    "Sigh": ("audible sigh", "weary / relieved"),
    "Groan": ("groan", "strained / displeased"),
}
def _canonical_events(labels: list[tuple[str, float]], threshold: float = 0.07) -> list[tuple[str, str, str | None, float, list[str]]]:
    # AudioSet is multilabel. Preserve simultaneous labels and do not infer emotion
    # from a sound class. Thresholds are provisional until benchmark calibration.
    return [
        ("vocal_reaction", REACTION_LABELS[label][0], None, score, [label])
        for label, score in labels if label in REACTION_LABELS and score >= threshold
    ]


def _merge_detections(items: list[TimedLayer], gap: float) -> list[TimedLayer]:
    ordered = sorted(items, key=lambda item: ((item.layer.type, item.layer.description or ""), item.start))
    merged: list[TimedLayer] = []
    for item in ordered:
        if merged and (merged[-1].layer.type, merged[-1].layer.description) == (item.layer.type, item.layer.description) and item.start <= merged[-1].end + gap:
            current = merged[-1]
            current.end = max(current.end, item.end)
            if item.layer.confidence > current.layer.confidence:
                current.layer.confidence = item.layer.confidence
                current.layer.evidence = item.layer.evidence
        else:
            merged.append(item)
    return sorted(merged, key=lambda item: (item.start, item.end, item.layer.type))


def _resample_audio(audio: object, source_rate: int, target_rate: int) -> object:
    if source_rate == target_rate or len(audio) == 0:
        return audio
    from scipy.signal import resample_poly
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype("float32")


def separate_vocals(source: Path, output_root: Path, model_name: str = "htdemucs") -> tuple[Path, Path]:
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        model_name,
        "-o",
        str(output_root),
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1800)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    stem_dir = output_root / model_name / source.stem
    return stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"
