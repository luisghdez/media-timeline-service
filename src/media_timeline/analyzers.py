from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from .schema import Evidence, Layer, TimedLayer


class SpeechAnalyzer:
    _models: dict[tuple[str, str, str], object] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        offline: bool = False,
        vad_filter: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.offline = offline
        self.vad_filter = vad_filter

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
        )
        output: list[TimedLayer] = []
        for segment in segments:
            if not segment.text.strip() or segment.no_speech_prob > 0.8:
                continue
            output.extend(_split_segment(segment, self.model_name))
        return output


def _split_segment(segment: object, model_name: str, gap_seconds: float = 0.65) -> list[TimedLayer]:
    words = [word for word in (segment.words or []) if word.probability >= 0.2]
    if not words:
        confidence = max(0.0, min(1.0, 1.0 - float(segment.no_speech_prob)))
        return [
            TimedLayer(
                start=float(segment.start),
                end=float(segment.end),
                layer=Layer(
                    type="speech",
                    text=segment.text.strip(),
                    confidence=confidence,
                    evidence=[Evidence("asr", f"faster-whisper:{model_name}", confidence)],
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
        confidence = sum(float(word.probability) for word in group) / len(group)
        output.append(
            TimedLayer(
                start=float(group[0].start),
                end=float(group[-1].end),
                layer=Layer(
                    type="speech",
                    text=text,
                    confidence=confidence,
                    evidence=[Evidence("asr", f"faster-whisper:{model_name}", confidence)],
                ),
            )
        )
    return output


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
    ) -> None:
        self.cache_dir = cache_dir
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
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
            audio = _linear_resample(audio, sample_rate, expected_rate)
            sample_rate = expected_rate

        detections: list[TimedLayer] = []
        position = 0.0
        while position < duration:
            stop = min(duration, position + self.window_seconds)
            chunk = audio[int(position * sample_rate) : int(stop * sample_rate)]
            if len(chunk) < sample_rate // 4:
                break
            inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                probabilities = model(**inputs).logits[0].softmax(-1)
            values, indices = torch.topk(probabilities, 10)
            labels = [(model.config.id2label[int(index)], float(value)) for value, index in zip(values, indices)]
            canonical = _canonical_events(labels)
            for event_type, description, emotion, confidence, raw_labels in canonical:
                center = position + (stop - position) / 2
                event_start = max(0.0, center - self.hop_seconds / 2)
                event_end = min(duration, center + self.hop_seconds / 2)
                calibrated = _calibrate_confidence(event_type, confidence)
                detections.append(
                    TimedLayer(
                        start=event_start,
                        end=event_end,
                        layer=Layer(
                            type=event_type,
                            description=description,
                            emotion=emotion,
                            confidence=calibrated,
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
def _canonical_events(labels: list[tuple[str, float]]) -> list[tuple[str, str, str | None, float, list[str]]]:
    reactions = [(label, score) for label, score in labels if label in REACTION_LABELS]
    if not reactions:
        return []
    label, score = reactions[0]
    reaction_threshold = 0.07 if label in {"Gasp", "Sigh"} else 0.035
    if score < reaction_threshold:
        return []
    if label in {"Laughter", "Giggle"}:
        description, emotion = "Nonverbal laughter or amused reaction", "amused"
    elif label == "Sigh":
        description, emotion = "Nonverbal sigh", "weary / relieved"
    elif label == "Gasp":
        description, emotion = "Audible gasp", "surprised"
    else:
        description, emotion = "Nonverbal vocal reaction such as a scream, wail, or moan", "distressed / intense"
    return [("vocal_reaction", description, emotion, score, [item[0] for item in reactions[:3]])]


def _merge_detections(items: list[TimedLayer], gap: float) -> list[TimedLayer]:
    ordered = sorted(items, key=lambda item: (repr(item.layer.signature()), item.start))
    merged: list[TimedLayer] = []
    for item in ordered:
        if merged and merged[-1].layer.signature() == item.layer.signature() and item.start <= merged[-1].end + gap:
            current = merged[-1]
            current.end = max(current.end, item.end)
            if item.layer.confidence > current.layer.confidence:
                current.layer.confidence = item.layer.confidence
                current.layer.evidence = item.layer.evidence
        else:
            merged.append(item)
    return sorted(merged, key=lambda item: (item.start, item.end, item.layer.type))


def _calibrate_confidence(event_type: str, raw_score: float) -> float:
    base = 0.48 if event_type == "vocal_reaction" else 0.4
    return min(0.97, base + raw_score * 1.1)


def _linear_resample(audio: object, source_rate: int, target_rate: int) -> object:
    import numpy as np

    if source_rate == target_rate or len(audio) == 0:
        return audio
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype("float32")


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
