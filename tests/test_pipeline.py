from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from media_timeline import pipeline as module
from media_timeline.pipeline import AnalysisPipeline, PipelineConfig, _reconcile_speech
from media_timeline.schema import Layer, TimedLayer, Word


def test_reconciliation_keeps_disagreeing_word_hypotheses():
    primary = TimedLayer(0, 1, Layer(type="speech", text="oh gosh", confidence=0.9,
                                    words=[Word("oh", 0, 0.3), Word("gosh", 0.4, 1)]))
    secondary = TimedLayer(0, 1, Layer(type="speech", text="o-o-oh gosh", confidence=0.6,
                                      words=[Word("o-o-oh", 0, 0.4), Word("gosh", 0.5, 1)]))
    result = _reconcile_speech([primary], [secondary])
    assert len(result) == 1
    assert result[0].layer.alternatives[0]["words"][0]["text"] == "o-o-oh"
    assert "asr_source_disagreement" in result[0].layer.review_reasons


@pytest.mark.parametrize("config", [dict(event_hop_seconds=0), dict(event_window_seconds=-1),
                                  dict(event_threshold=float("nan")), dict(asr_backend="bad")])
def test_config_rejects_invalid_analysis_parameters(config):
    with pytest.raises(ValueError):
        PipelineConfig(**config)


def test_pipeline_failed_optional_stages_keep_transcript_and_apply_source_clock(monkeypatch, tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(module, "probe", lambda _: SimpleNamespace(duration=2, audio_offset=0.4,
        audio_codec="pcm", video_codec="test", sample_rate=16000, channels=1, frame_rate=30,
        format_start_time=0, audio_start_time=0.4))
    monkeypatch.setattr(module, "extract_audio", lambda _, dest: sf.write(dest, np.zeros(16000), 16000))
    monkeypatch.setattr(module, "detect_silence", lambda *_: [])
    monkeypatch.setattr(module, "detect_scene_changes", lambda *_: [0.6])
    monkeypatch.setattr(module.SpeechAnalyzer, "analyze", lambda *_: [TimedLayer(0.1, 0.5,
        Layer(type="speech", text="oh", confidence=0.9, words=[Word("oh", 0.1, 0.5)]))])
    def failure(*_):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(module.ForcedAligner, "align", failure)
    monkeypatch.setattr(module.AudioDeliveryAnalyzer, "analyze", failure)
    result = AnalysisPipeline(PipelineConfig(model_cache=tmp_path / "cache", use_vocal_separation=False,
        use_audio_events=False, alignment="whisperx", delivery="qwen")).analyze(source).to_dict()
    event = result["events"][0]
    assert event["start"] == 0.5 and event["end"] == 0.9
    assert event["start_sample"] == 1600
    assert event["words"][0]["start"] == 0.5
    assert event["text"] == "oh"
    assert event["review_required"] is True
    assert len(result["warnings"]) == 2
    assert all(s["review_required"] for s in result["segments"])


def test_rejected_verbatim_model_falls_back_with_explicit_warning(monkeypatch, tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(module, "probe", lambda _: SimpleNamespace(duration=1, audio_offset=0,
        audio_codec="pcm", video_codec="test", sample_rate=16000, channels=1, frame_rate=30,
        format_start_time=0, audio_start_time=0))
    monkeypatch.setattr(module, "extract_audio", lambda _, dest: sf.write(dest, np.zeros(16000), 16000))
    monkeypatch.setattr(module, "detect_silence", lambda *_: [])
    monkeypatch.setattr(module.VerbatimSpeechAnalyzer, "analyze", lambda *_: (_ for _ in ()).throw(ValueError("decode loop")))
    monkeypatch.setattr(module.SpeechAnalyzer, "analyze", lambda *_: [TimedLayer(0.1, 0.5,
        Layer(type="speech", text="oh", confidence=0.9))])
    result = AnalysisPipeline(PipelineConfig(model_cache=tmp_path / "cache", asr_backend="crisper",
        use_vocal_separation=False, use_audio_events=False, use_scene_detection=False,
        delivery="none")).analyze(source).to_dict()
    assert result["events"][0]["text"] == "oh"
    assert result["events"][0]["review_required"]
    assert "fallback" in result["warnings"][0]
