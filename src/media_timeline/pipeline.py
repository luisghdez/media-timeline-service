from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .analyzers import AudioEventAnalyzer, SpeechAnalyzer, VerbatimSpeechAnalyzer, separate_vocals
from .delivery import AudioDeliveryAnalyzer, annotate_vocal_delivery
from .timing import ForcedAligner, stamp_timeline, refine_reaction_boundaries
from .fusion import fuse_timeline, prepare_layers, remove_silent_overlaps
from .media import detect_scene_changes, detect_silence, extract_audio, probe
from .schema import OUTPUT_LAYER_TYPES, AnalysisResult, TimedLayer


@dataclass(slots=True)
class PipelineConfig:
    whisper_model: str = "small.en"
    model_cache: Path = Path(".cache/media-timeline")
    use_vocal_separation: bool = True
    use_audio_events: bool = True
    use_scene_detection: bool = True
    offline: bool = False
    silence_threshold_db: int = -42
    scene_threshold: float = 0.24
    event_window_seconds: float = 2.0
    event_hop_seconds: float = 0.5
    event_threshold: float = 0.07
    vad_filter: bool = False
    speech_source: str = "both"
    asr_backend: str = "whisper"
    verbatim_model: str = "nyralabs/CrisperWhisper2.0_large"
    language: str = "en"
    alignment: str = "none"
    delivery: str = "acoustic"
    delivery_model: str = "Qwen/Qwen2.5-Omni-3B"
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name, choices in {
            "speech_source": {"mix", "vocals", "both"},
            "asr_backend": {"whisper", "crisper"},
            "alignment": {"none", "whisperx"},
            "delivery": {"none", "acoustic", "qwen"},
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"Invalid {name}: {getattr(self, name)}")
        if not 0 < self.event_threshold < 1:
            raise ValueError("Event threshold must be between 0 and 1")
        if not 0 < self.event_hop_seconds <= self.event_window_seconds:
            raise ValueError("Event hop must be positive and no greater than window size")


class AnalysisPipeline:
    _inference_lock = threading.Lock()

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.config.model_cache.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        source: str | Path,
        artifacts_dir: str | Path | None = None,
        context_text: str | None = None,
    ) -> AnalysisResult:
        with self._inference_lock:
            return self._analyze(source, artifacts_dir, context_text)

    def _analyze(
        self,
        source: str | Path,
        artifacts_dir: str | Path | None = None,
        context_text: str | None = None,
    ) -> AnalysisResult:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        info = probe(source_path)
        warnings: list[str] = []

        with tempfile.TemporaryDirectory(prefix="media-timeline-") as directory:
            workspace = Path(directory)
            mono_audio = workspace / "audio-16k.wav"
            extract_audio(source_path, mono_audio)
            analysis_audio = mono_audio

            if self.config.use_vocal_separation:
                try:
                    # Separate the normalized waveform so all branches share a clock.
                    vocals, accompaniment = separate_vocals(mono_audio, workspace / "separated")
                    analysis_audio = workspace / "vocals-16k.wav"
                    extract_audio(vocals, analysis_audio)
                    if artifacts_dir:
                        target = Path(artifacts_dir)
                        target.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(vocals, target / "vocals.wav")
                        shutil.copy2(accompaniment, target / "background.wav")
                except Exception as exc:
                    analysis_audio = mono_audio
                    warnings.append(f"Vocal separation unavailable: {exc}")

            layers: list[TimedLayer] = []
            try:
                if self.config.asr_backend == "crisper":
                    speech_analyzer = VerbatimSpeechAnalyzer(
                        self.config.verbatim_model, self.config.model_cache / "crisper",
                        self.config.offline, self.config.language, self.config.device,
                    )
                else:
                    speech_analyzer = SpeechAnalyzer(
                        self.config.whisper_model, self.config.model_cache / "whisper",
                        self.config.offline, self.config.vad_filter, self.config.language,
                    )
                def transcribe(path: Path, audio_source: str) -> list[TimedLayer]:
                    try:
                        return _tag_source(speech_analyzer.analyze(path), audio_source)
                    except Exception as exc:
                        if self.config.asr_backend != "crisper":
                            raise
                        warnings.append(f"Verbatim ASR rejected or unavailable; using Whisper fallback: {exc}")
                        fallback = SpeechAnalyzer(
                            self.config.whisper_model, self.config.model_cache / "whisper",
                            self.config.offline, self.config.vad_filter, self.config.language,
                        ).analyze(path)
                        for item in fallback:
                            item.layer.review_reasons.append("requested_verbatim_asr_fell_back")
                        return _tag_source(fallback, audio_source)
                vocals_ready = analysis_audio != mono_audio
                source = self.config.speech_source
                if source == "vocals" and not vocals_ready:
                    warnings.append("ASR vocals requested but separation failed; using mix")
                    source = "mix"
                if source == "mix":
                    layers.extend(transcribe(mono_audio, "mix"))
                elif source == "vocals":
                    layers.extend(transcribe(analysis_audio, "vocals"))
                else:
                    primary_speech = transcribe(mono_audio, "mix")
                    # Keep successful mix ASR if the secondary branch fails.
                    layers.extend(primary_speech)
                    separated_speech = transcribe(analysis_audio, "vocals") if vocals_ready else []
                    layers.clear()
                    layers.extend(_reconcile_speech(primary_speech, separated_speech))
            except Exception as exc:
                warnings.append(f"Speech analysis unavailable: {exc}")

            if self.config.alignment == "whisperx" and layers:
                try:
                    layers = ForcedAligner(
                        self.config.model_cache / "alignment", self.config.language,
                        self.config.device, self.config.offline,
                    ).align(deepcopy(layers), analysis_audio if self.config.speech_source == "vocals" else mono_audio)
                except Exception as exc:
                    warnings.append(f"Forced alignment unavailable; retaining ASR estimates: {exc}")
                    for item in layers:
                        item.layer.review_reasons.append("requested_alignment_unavailable")

            if self.config.use_audio_events:
                try:
                    layers.extend(
                        AudioEventAnalyzer(
                            self.config.model_cache / "audio-events",
                            self.config.event_window_seconds,
                            self.config.event_hop_seconds,
                            self.config.offline,
                            self.config.event_threshold,
                        ).analyze(mono_audio, info.duration)
                    )
                    if analysis_audio != mono_audio:
                        layers.extend(
                            AudioEventAnalyzer(
                                self.config.model_cache / "audio-events",
                                self.config.event_window_seconds,
                                self.config.event_hop_seconds,
                                self.config.offline,
                                self.config.event_threshold,
                            ).analyze(analysis_audio, info.duration)
                        )
                except Exception as exc:
                    warnings.append(f"Audio event analysis unavailable: {exc}")

            silence = detect_silence(mono_audio, self.config.silence_threshold_db)
            scenes = detect_scene_changes(source_path, self.config.scene_threshold) if self.config.use_scene_detection else []
            try:
                layers = refine_reaction_boundaries(deepcopy(layers), analysis_audio)
            except Exception as exc:
                warnings.append(f"Reaction boundary refinement unavailable: {exc}")
            layers = remove_silent_overlaps(layers, silence)
            layers = prepare_layers(layers, scenes, context_text)
            layers = [item for item in layers if item.layer.type in OUTPUT_LAYER_TYPES]
            if self.config.delivery != "none":
                try:
                    layers = annotate_vocal_delivery(deepcopy(layers), analysis_audio)
                    if self.config.delivery == "qwen":
                        # Listen to the original mix: separation can distort prosody.
                        layers = AudioDeliveryAnalyzer(
                            self.config.model_cache / "delivery", self.config.delivery_model, self.config.offline,
                        ).analyze(deepcopy(layers), mono_audio)
                except Exception as exc:
                    warnings.append(f"Vocal delivery analysis unavailable: {exc}")
                    for item in layers:
                        item.layer.review_reasons.append("requested_delivery_unavailable")

            layers = stamp_timeline(layers, 16000, info.audio_offset, info.duration)
            segments = fuse_timeline(info.duration, layers, silence, scenes)
            event_layers = sorted(layers, key=lambda item: (item.start, item.end, item.layer.type))

            if artifacts_dir:
                target = Path(artifacts_dir)
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mono_audio, target / "audio-16k.wav")

        return AnalysisResult(
            duration=info.duration,
            segments=segments,
            source={
                "filename": source_path.name,
                "audio_codec": info.audio_codec,
                "video_codec": info.video_codec,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "frame_rate": info.frame_rate,
                "context_text": context_text,
                "timeline_origin": "source_playback",
                "analysis_sample_rate": 16000,
                "sample_index_origin": "analysis_waveform",
                "analysis_audio_offset_seconds": info.audio_offset,
                "format_start_time": info.format_start_time,
                "audio_start_time": info.audio_start_time,
                "analysis_config": {key: str(value) if isinstance(value, Path) else value
                                    for key, value in asdict(self.config).items()},
                "confidence_status": "uncalibrated_model_scores; unknown values omitted",
            },
            events=event_layers,
            warnings=warnings,
        )


def _reconcile_speech(primary: list[TimedLayer], secondary: list[TimedLayer]) -> list[TimedLayer]:
    output = list(primary)
    for candidate in secondary:
        duplicate_index: int | None = None
        for index, current in enumerate(output):
            overlap = max(0.0, min(candidate.end, current.end) - max(candidate.start, current.start))
            shorter = max(0.001, min(candidate.end - candidate.start, current.end - current.start))
            text_similarity = SequenceMatcher(
                None,
                (candidate.layer.text or "").lower(),
                (current.layer.text or "").lower(),
            ).ratio()
            if overlap / shorter >= 0.4 and text_similarity >= 0.6:
                duplicate_index = index
                break
        if duplicate_index is None:
            if any(candidate.start < current.end and candidate.end > current.start for current in output):
                candidate.layer.review_reasons.append("overlapping_asr_hypotheses")
                for current in output:
                    if candidate.start < current.end and candidate.end > current.start:
                        current.layer.review_reasons.append("overlapping_asr_hypotheses")
            output.append(candidate)
        else:
            current = output[duplicate_index]
            # Retain the losing transcript and word timings for review. A higher
            # average probability does not prove missing syllables were absent.
            selected, alternative = (candidate, current) if candidate.layer.confidence > current.layer.confidence else (current, candidate)
            selected.layer.alternatives.append({
                "start": alternative.start, "end": alternative.end,
                "text": alternative.layer.verbatim_text or alternative.layer.text,
                "words": [asdict(word) for word in alternative.layer.words],
                "evidence": [asdict(record) for record in alternative.layer.evidence],
                "timebase": "analysis_waveform",
            })
            if (candidate.layer.verbatim_text or candidate.layer.text) != (current.layer.verbatim_text or current.layer.text):
                selected.layer.review_reasons.append("asr_source_disagreement")
            output[duplicate_index] = selected
    return sorted(output, key=lambda item: (item.start, item.end))


def _tag_source(items: list[TimedLayer], source: str) -> list[TimedLayer]:
    from .schema import Evidence
    for item in items:
        item.layer.evidence.append(Evidence("audio_source", source))
    return items
