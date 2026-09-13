from types import SimpleNamespace

from media_timeline.analyzers import _canonical_events, _split_segment
from media_timeline.schema import AnalysisResult, Layer, TimedLayer


def test_maps_scream_to_vocal_reaction() -> None:
    events = _canonical_events([("Screaming", 0.72), ("Music", 0.18)])
    kinds = {event[0] for event in events}
    assert kinds == {"vocal_reaction"}


def test_ignores_gunfire_and_other_non_reaction_events() -> None:
    assert _canonical_events([("Gunshot, gunfire", 0.6)]) == []


def test_rejects_weak_gasp_prediction() -> None:
    assert _canonical_events([("Gasp", 0.05)]) == []


def test_asr_segment_splits_on_long_word_gap() -> None:
    words = [
        SimpleNamespace(start=0.5, end=0.8, word=" Hello", probability=0.9),
        SimpleNamespace(start=0.8, end=1.1, word=" there.", probability=0.9),
        SimpleNamespace(start=2.0, end=2.3, word=" Again.", probability=0.9),
    ]
    segment = SimpleNamespace(start=0.5, end=2.3, text="Hello there. Again.", words=words, no_speech_prob=0.01)
    output = _split_segment(segment, "test")
    assert [item.layer.text for item in output] == ["Hello there.", "Again."]


def test_flat_events_flag_low_confidence_for_review() -> None:
    result = AnalysisResult(
        duration=1.0,
        segments=[],
        source={},
        events=[TimedLayer(0.0, 0.5, Layer(type="vocal_reaction", confidence=0.59))],
    )
    assert result.to_dict()["events"][0]["review_required"] is True
