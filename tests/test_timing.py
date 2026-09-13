from types import SimpleNamespace

import pytest
import numpy as np
import soundfile as sf

from media_timeline.analyzers import _split_segment
from media_timeline.schema import AnalysisResult, DeliverySpan, Layer, TimedLayer, Word
from media_timeline.timing import apply_alignment, refine_reaction_boundaries, stamp_timeline


def test_low_confidence_partial_syllables_are_preserved():
    segment = SimpleNamespace(words=[
        SimpleNamespace(word=" o-", start=0.01, end=0.06, probability=0.1),
        SimpleNamespace(word=" oh", start=0.08, end=0.2, probability=0.9)],
        text="o- oh", start=0.01, end=0.2, no_speech_prob=0)
    event = _split_segment(segment, "test")[0]
    assert event.layer.verbatim_text == "o- oh"
    assert event.layer.words[0].kind == "partial_syllable"
    assert event.layer.words[0].confidence == 0.1
    assert event.layer.needs_review()


def test_sample_indices_and_source_offset_are_distinct():
    layer = Layer(type="speech", words=[Word("hi", 0.0000625, 0.25)],
                  delivery=[DeliverySpan(0.0000625, 0.25)])
    item = TimedLayer(0.0000625, 0.25, layer)
    stamp_timeline([item], 16000, 0.4, 2)
    assert item.start_sample == 1
    assert layer.words[0].start_sample == 1
    assert layer.delivery[0].end_sample == 4000
    assert item.start == pytest.approx(0.4000625)
    result = AnalysisResult(2, [], {}, [item]).to_dict()
    assert result["schema_version"] == "2.0"
    assert result["events"][0]["start"] != 0.4


def test_unalignable_token_stays_in_output_with_original_time():
    item = TimedLayer(0, 1, Layer(type="speech", words=[Word("oh", 0, 0.2), Word("£", 0.3, 1)]))
    apply_alignment(item, [{"chars": [{"char": "o", "start": 0.02, "end": 0.12, "score": 0.8},
                                      {"char": "h", "start": 0.12, "end": 0.24, "score": 0.7}]}, {}])
    assert item.layer.words[0].start == 0.02
    assert item.layer.words[1].start == 0.3
    assert item.layer.words[1].text == "£"
    assert item.layer.timing_method == "mixed_alignment"
    assert item.layer.needs_review()


def test_isolated_reaction_gets_acoustic_bounds_but_overlap_stays_coarse(tmp_path):
    audio = np.zeros(32000, dtype="float32")
    audio[8000:16000] = 0.5
    path = tmp_path / "audio.wav"
    sf.write(path, audio, 16000)
    item = TimedLayer(0, 2, Layer(type="vocal_reaction", timing_method="classifier_window"))
    refine_reaction_boundaries([item], path)
    assert (item.start, item.end) == (0.5, 1)
    assert item.layer.timing_method == "isolated_energy_envelope_10ms"
    candidate = TimedLayer(0, 2, Layer(type="vocal_reaction", timing_method="classifier_window"))
    refine_reaction_boundaries([candidate, TimedLayer(0.4, 1.1, Layer(type="speech"))], path)
    assert (candidate.start, candidate.end) == (0, 2)
