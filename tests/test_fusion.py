from media_timeline.fusion import fuse_timeline, prepare_layers, remove_silent_overlaps
from media_timeline.schema import Layer, TimedLayer


def test_fusion_preserves_overlapping_layers() -> None:
    layers = [
        TimedLayer(0.5, 2.0, Layer(type="speech", text="hello", confidence=0.9)),
        TimedLayer(0.0, 3.0, Layer(type="vocal_reaction", description="gasp", confidence=0.8)),
    ]
    segments = fuse_timeline(3.0, layers, [], [])
    overlap = next(segment for segment in segments if segment.start == 0.5)
    assert {layer.type for layer in overlap.layers} == {"speech", "vocal_reaction"}


def test_silence_drops_non_speech_gaps() -> None:
    segments = fuse_timeline(2.0, [], [(0.0, 2.0)], [])
    assert segments == []


def test_scene_change_creates_visual_boundary() -> None:
    layers = [TimedLayer(0.0, 2.0, Layer(type="speech", text="hello", confidence=0.8))]
    segments = fuse_timeline(2.0, layers, [], [1.0])
    assert len(segments) == 2
    assert segments[1].visual_context is not None


def test_short_exclamation_is_promoted_into_reaction() -> None:
    layers = [
        TimedLayer(7.0, 12.0, Layer(type="vocal_reaction", description="wail", confidence=0.7)),
        TimedLayer(8.5, 9.0, Layer(type="speech", text="No!", confidence=0.9)),
    ]
    layers = prepare_layers(layers, [7.3])
    segments = fuse_timeline(12.0, layers, [], [7.3])
    reaction_layers = [
        layer for segment in segments for layer in segment.layers if layer.type == "vocal_reaction"
    ]
    assert reaction_layers
    assert all(layer.text == "No!" for layer in reaction_layers)
    assert not any(layer.type == "speech" for segment in segments for layer in segment.layers)


def test_reaction_boundaries_snap_to_nearby_scene_changes() -> None:
    layers = [TimedLayer(7.7, 15.7, Layer(type="vocal_reaction", confidence=0.7))]
    layers = prepare_layers(layers, [7.3, 15.3])
    segments = fuse_timeline(18.0, layers, [], [7.3, 15.3])
    reaction_segments = [segment for segment in segments if segment.layers[0].type == "vocal_reaction"]
    assert reaction_segments[0].start == 7.3
    assert reaction_segments[-1].end == 15.3


def test_context_can_label_long_untranscribed_reaction() -> None:
    layers = [
        TimedLayer(
            7.0,
            13.0,
            Layer(
                type="vocal_reaction",
                description="Nonverbal vocal reaction such as a scream, wail, or moan",
                confidence=0.72,
            ),
        )
    ]
    prepared = prepare_layers(layers, [], "NOOOOOO #gaming")
    assert prepared[0].layer.text == "Noooooo!"
    assert prepared[0].layer.evidence[-1].source == "context"


def test_events_inside_verified_silence_are_removed() -> None:
    layers = [
        TimedLayer(0.0, 1.0, Layer(type="speech", text="hi", confidence=0.7)),
        TimedLayer(2.0, 3.0, Layer(type="vocal_reaction", confidence=0.6)),
    ]
    output = remove_silent_overlaps(layers, [(0.8, 3.0)])
    assert len(output) == 1
    assert output[0].end == 0.8


def test_continuous_reaction_labels_become_one_run() -> None:
    layers = [
        TimedLayer(
            7.0,
            13.0,
            Layer(
                type="vocal_reaction",
                description="Nonverbal vocal reaction such as a scream, wail, or moan",
                confidence=0.75,
            ),
        ),
        TimedLayer(
            13.0,
            14.0,
            Layer(type="vocal_reaction", description="Audible gasp", confidence=0.62),
        ),
    ]
    prepared = prepare_layers(layers, [], "NOOOOOO")
    reactions = [item for item in prepared if item.layer.type == "vocal_reaction"]
    assert len(reactions) == 1
    assert (reactions[0].start, reactions[0].end) == (7.0, 14.0)
    assert reactions[0].layer.text == "Noooooo!"


def test_non_vocal_layers_are_omitted_from_output() -> None:
    layers = [
        TimedLayer(0.0, 2.0, Layer(type="music", confidence=0.9)),
        TimedLayer(0.5, 1.0, Layer(type="speech", text="hi", confidence=0.9)),
        TimedLayer(1.2, 1.6, Layer(type="sound_effect", confidence=0.8)),
    ]
    prepared = prepare_layers(layers, [])
    assert [item.layer.type for item in prepared] == ["speech"]
    segments = fuse_timeline(2.0, prepared, [], [])
    assert all(layer.type in {"speech", "vocal_reaction"} for segment in segments for layer in segment.layers)


def test_tiny_reaction_tail_at_silence_boundary_is_removed() -> None:
    layers = [TimedLayer(16.25, 16.75, Layer(type="vocal_reaction", confidence=0.7))]
    assert remove_silent_overlaps(layers, [(16.427, 19.0)]) == []
