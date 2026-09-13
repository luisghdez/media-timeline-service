from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .analyzers import AudioEventAnalyzer, SpeechAnalyzer, separate_vocals
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
    vad_filter: bool = False
    speech_source: str = "both"


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
                    vocals, accompaniment = separate_vocals(source_path, workspace / "separated")
                    analysis_audio = vocals
                    if artifacts_dir:
                        target = Path(artifacts_dir)
                        target.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(vocals, target / "vocals.wav")
                        shutil.copy2(accompaniment, target / "background.wav")
                except Exception as exc:
                    warnings.append(f"Vocal separation unavailable: {exc}")

            layers: list[TimedLayer] = []
            try:
                speech_analyzer = SpeechAnalyzer(
                    self.config.whisper_model,
                    self.config.model_cache / "whisper",
                    self.config.offline,
                    self.config.vad_filter,
                )
                vocals_ready = analysis_audio != mono_audio
                source = self.config.speech_source
                if source == "vocals" and not vocals_ready:
                    warnings.append("ASR vocals requested but separation failed; using mix")
                    source = "mix"
                if source == "mix":
                    layers.extend(speech_analyzer.analyze(mono_audio))
                elif source == "vocals":
                    layers.extend(speech_analyzer.analyze(analysis_audio))
                else:
                    primary_speech = speech_analyzer.analyze(mono_audio)
                    separated_speech = speech_analyzer.analyze(analysis_audio) if vocals_ready else []
                    layers.extend(_reconcile_speech(primary_speech, separated_speech))
            except Exception as exc:
                warnings.append(f"Speech analysis unavailable: {exc}")

            if self.config.use_audio_events:
                try:
                    layers.extend(
                        AudioEventAnalyzer(
                            self.config.model_cache / "audio-events",
                            self.config.event_window_seconds,
                            self.config.event_hop_seconds,
                            self.config.offline,
                        ).analyze(mono_audio, info.duration)
                    )
                    if analysis_audio != mono_audio:
                        layers.extend(
                            AudioEventAnalyzer(
                                self.config.model_cache / "audio-events",
                                self.config.event_window_seconds,
                                self.config.event_hop_seconds,
                                self.config.offline,
                            ).analyze(analysis_audio, info.duration)
                        )
                except Exception as exc:
                    warnings.append(f"Audio event analysis unavailable: {exc}")

            silence = detect_silence(mono_audio, self.config.silence_threshold_db)
            scenes = detect_scene_changes(source_path, self.config.scene_threshold) if self.config.use_scene_detection else []
            layers = remove_silent_overlaps(layers, silence)
            layers = prepare_layers(layers, scenes, context_text)
            layers = [item for item in layers if item.layer.type in OUTPUT_LAYER_TYPES]
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
            output.append(candidate)
        elif (
            output[duplicate_index].layer.confidence < 0.75
            and candidate.layer.confidence > output[duplicate_index].layer.confidence
        ):
            output[duplicate_index] = candidate
    return sorted(output, key=lambda item: (item.start, item.end))
