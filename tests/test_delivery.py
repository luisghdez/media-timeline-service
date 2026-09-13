import json

import numpy as np
import pytest
import soundfile as sf

from media_timeline.delivery import annotate_vocal_delivery, attach_delivery
from media_timeline.schema import Layer, TimedLayer, Word


def test_energy_pulses_do_not_invent_phonemes_or_panic(tmp_path):
    sr = 16000
    audio = np.zeros(sr, dtype="float32")
    for center in (0.25, 0.38, 0.51, 0.64):
        audio[int(center * sr):int((center + 0.03) * sr)] = 0.8
    path = tmp_path / "vocals.wav"
    sf.write(path, audio, sr)
    item = TimedLayer(0, 1, Layer(type="speech", text="my gosh!", confidence=0.8))
    annotate_vocal_delivery([item], path)
    assert item.layer.text == "my gosh!"
    assert item.layer.emotion is None
    assert item.layer.delivery
    assert item.layer.delivery[0].features == ["repeated_energy_pulses"]
    assert item.layer.delivery[0].perceived_emotion is None


def observation(**kwargs):
    return {"first_word": 0, "last_word": 1, "features": ["rising_pitch"],
            "description": "Pitch rises", "acoustic_evidence": "Audible pitch rises through the phrase",
            "confidence": 0.8, "perceived_emotion": "panicked", "emotion_confidence": 0.65, **kwargs}


def example():
    return TimedLayer(2, 3, Layer(type="speech", text="Oh gosh", words=[Word("Oh", 2, 2.2), Word("gosh", 2.3, 3)]))


def test_delivery_uses_existing_word_times_and_separate_confidences():
    item = example()
    attach_delivery(item, json.dumps({"observations": [observation()]}), {0, 1}, 2, 3, "test")
    span = item.layer.delivery[0]
    assert (span.start, span.end) == (2, 3)
    assert span.confidence == 0.8
    assert span.emotion_confidence == 0.65
    assert item.layer.text == "Oh gosh"
    assert item.layer.needs_review()


@pytest.mark.parametrize("bad", [
    {"first_word": 3}, {"last_word": 0, "first_word": 1}, {"confidence": 1.5},
    {"confidence": float("nan")}, {"start": 2.123}, {"features": ["invented_feature"]},
    {"emotion_confidence": None},
])
def test_invalid_model_output_never_partially_mutates_event(bad):
    item = example()
    payload = json.dumps({"observations": [observation(), observation(**bad)]})
    with pytest.raises(ValueError):
        attach_delivery(item, payload, {0, 1}, 2, 3, "test")
    assert item.layer.delivery == []


def test_qwen_adapter_sends_audio_crops_and_attaches_verified_word_anchors(monkeypatch, tmp_path):
    from media_timeline.delivery import AudioDeliveryAnalyzer
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(64000, dtype="float32"), 16000)
    calls = []
    def infer(self, crop, prompt):
        assert crop.exists()
        assert sf.info(crop).duration == 2
        assert '"index": 0' in prompt and '"index": 1' in prompt
        calls.append(prompt)
        return json.dumps({"observations": [observation()]})
    monkeypatch.setattr(AudioDeliveryAnalyzer, "_infer", infer)
    item = example()
    AudioDeliveryAnalyzer(tmp_path).analyze([item], path)
    assert len(calls) == 1
    assert item.layer.delivery[0].perceived_emotion == "panicked"
    assert item.layer.delivery[0].start == 2
