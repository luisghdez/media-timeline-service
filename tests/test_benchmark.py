import pytest

from media_timeline.benchmark import evaluate


def reference(words):
    return {"clips": [{"id": "one", "reviewed": True, "words": words, "reactions": [], "delivery": []}]}


def prediction(words):
    return {"one": {"events": [{"type": "speech", "words": words}]}}


def test_missing_repetition_counts_as_error_and_reduces_timing_coverage():
    words = [{"text": "o-", "start": 0, "end": 0.1, "kind": "partial_syllable"},
             {"text": "oh", "start": 0.2, "end": 0.3},
             {"text": "gosh", "start": 0.4, "end": 0.7}]
    report = evaluate(reference(words), prediction(words[1:]))
    assert report["word_error_rate"] == pytest.approx(1 / 3)
    assert report["word_timing"]["coverage"] == pytest.approx(2 / 3)
    assert report["word_timing"]["within_100ms_on_matched"] == 1
    assert report["word_timing"]["within_100ms_of_all_reference"] == pytest.approx(2 / 3)
    assert report["disfluencies"]["recall"] == 0


def test_unreviewed_generated_reference_cannot_be_scored():
    with pytest.raises(ValueError, match="human-reviewed"):
        evaluate({"clips": [{"id": "one", "reviewed": False}]}, {})


def test_missing_prediction_cannot_silently_improve_score():
    with pytest.raises(ValueError, match="Missing prediction"):
        evaluate(reference([]), {})


def test_timing_metric_detects_large_boundary_error():
    report = evaluate(reference([{"text": "hi", "start": 0.1, "end": 0.4}]),
                      prediction([{"text": "hi", "start": 0.3, "end": 0.7}]))
    assert report["word_timing"]["mean_absolute_error_ms"] == pytest.approx(250)
    assert report["word_timing"]["within_100ms_on_matched"] == 0
