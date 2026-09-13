from media_timeline.fusion import fuse_timeline, prepare_layers, remove_silent_overlaps
from media_timeline.schema import Layer, TimedLayer, Word


def test_fusion_preserves_overlapping_layers():
    layers = [TimedLayer(0.5, 2, Layer(type="speech", text="hello", confidence=0.9)),
              TimedLayer(0, 3, Layer(type="vocal_reaction", description="gasp", confidence=0.8))]
    overlap = next(s for s in fuse_timeline(3, layers, [], []) if s.start == 0.5)
    assert {layer.type for layer in overlap.layers} == {"speech", "vocal_reaction"}


def test_scene_cut_splits_view_without_moving_audio():
    layers = prepare_layers([TimedLayer(7.7, 15.7, Layer(type="vocal_reaction", confidence=0.7))], [7.3, 15.3])
    assert (layers[0].start, layers[0].end) == (7.7, 15.7)
    segments = fuse_timeline(18, layers, [], [7.3, 15.3])
    assert (segments[0].start, segments[-1].end) == (7.7, 15.7)
    assert segments[1].visual_context is not None


def test_spoken_exclamations_and_distinct_reactions_survive():
    layers = [TimedLayer(7, 13, Layer(type="vocal_reaction", description="wail")),
              TimedLayer(8.5, 9, Layer(type="speech", text="No!")),
              TimedLayer(13, 14, Layer(type="vocal_reaction", description="gasp"))]
    prepared = prepare_layers(layers, [7.3], "NOOOOOO #gaming")
    assert len(prepared) == 3
    assert next(x for x in prepared if x.layer.type == "speech").layer.text == "No!"
    assert all(x.layer.text is None for x in prepared if x.layer.type == "vocal_reaction")
    assert prepared[0].layer.context_hint == "NOOOOOO #gaming"
    assert prepared[0].layer.confidence == 0


def test_silence_splits_reactions_and_retains_quiet_speech_for_review():
    layers = [TimedLayer(0, 3, Layer(type="vocal_reaction")),
              TimedLayer(0.5, 2.5, Layer(type="speech", text="quiet words", confidence=0.9))]
    output = remove_silent_overlaps(layers, [(1, 2)])
    assert [(x.start, x.end) for x in output if x.layer.type == "vocal_reaction"] == [(0, 1), (2, 3)]
    speech = next(x for x in output if x.layer.type == "speech")
    assert (speech.start, speech.end) == (0.5, 2.5)
    assert speech.layer.needs_review()
    assert layers[0].end == 3


def test_micro_spans_and_silent_gaps_are_not_coalesced():
    layers = [TimedLayer(0.01, 0.06, Layer(type="speech", text="o-")),
              TimedLayer(0.08, 0.12, Layer(type="speech", text="o-"))]
    assert [(s.start, s.end) for s in fuse_timeline(1, layers, [], [])] == [(0.01, 0.06), (0.08, 0.12)]


def test_word_timings_are_not_lost_when_text_matches():
    layers = [TimedLayer(0, 1, Layer(type="speech", text="oh", words=[Word("oh", 0, 1)])),
              TimedLayer(1, 2, Layer(type="speech", text="oh", words=[Word("oh", 1, 2)]))]
    assert len(fuse_timeline(2, layers, [], [])) == 2


def test_non_vocal_layers_are_omitted():
    layers = [TimedLayer(0, 2, Layer(type="music")), TimedLayer(0.5, 1, Layer(type="speech", text="hi"))]
    assert [x.layer.type for x in prepare_layers(layers, [])] == ["speech"]
    assert fuse_timeline(2, [], [(0, 2)], []) == []
