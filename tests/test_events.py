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


def test_multiple_reactions_do_not_imply_emotion():
    events = _canonical_events([("Gasp", 0.8), ("Laughter", 0.7)])
    assert len(events) == 2
    assert all(event[2] is None for event in events)


def test_unplaceable_word_stays_in_verbatim_transcript():
    segment = SimpleNamespace(start=0, end=1, text="oh um gosh", no_speech_prob=0,
        words=[SimpleNamespace(word=" oh", start=0, end=0.2, probability=0.8),
               SimpleNamespace(word=" gosh", start=0.5, end=1, probability=0.8)])
    item = _split_segment(segment, "test")[0]
    assert item.layer.verbatim_text == "oh um gosh"
    assert item.layer.words[1].text == "um"
    assert item.layer.words[1].start is None
    assert item.layer.needs_review()


def test_verbatim_guard_rejects_decode_loop_but_keeps_short_stutter():
    import pytest
    from media_timeline.analyzers import _check_verbatim_quality
    _check_verbatim_quality(SimpleNamespace(text="o o o oh my gosh", duration=2, words=[]))
    with pytest.raises(ValueError, match="decode loop"):
        _check_verbatim_quality(SimpleNamespace(text="oh " * 100, duration=10, words=[]))
