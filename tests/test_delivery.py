from pathlib import Path

import numpy as np
import soundfile as sf

from media_timeline.delivery import annotate_vocal_delivery, _render_stuttered_text
from media_timeline.schema import Layer, TimedLayer


def test_render_stuttered_text_keeps_lexical_tail() -> None:
    assert _render_stuttered_text("my gosh!", 5) == "O-o-o-oh my gosh!"
    assert _render_stuttered_text("oh my gosh", 3) == "O-o-oh my gosh!"
    assert _render_stuttered_text(None, 4) == "O-o-o-oh!"


def test_stuttered_reaction_gets_delivery_from_energy_pulses(tmp_path: Path) -> None:
    sample_rate = 16000
    duration = 1.2
    audio = np.zeros(int(duration * sample_rate), dtype="float32")
    for center in (0.25, 0.38, 0.51, 0.64):
        index = int(center * sample_rate)
        span = int(0.03 * sample_rate)
        audio[index : index + span] = 0.8
    wav = tmp_path / "vocals.wav"
    sf.write(wav, audio, sample_rate)

    layers = [
        TimedLayer(0.45, 0.85, Layer(type="vocal_reaction", text="my gosh!", description="Audible gasp", emotion="surprised", confidence=0.8)),
        TimedLayer(0.0, 0.15, Layer(type="speech", text="hello", confidence=0.9)),
    ]
    annotated = annotate_vocal_delivery(layers, wav)
    reaction = next(item for item in annotated if item.layer.type == "vocal_reaction")
    assert reaction.layer.delivery == "stuttered / panicked"
    assert reaction.layer.emotion == "panicked"
    assert reaction.layer.text == "O-o-o-oh my gosh!"
    assert reaction.layer.evidence[-1].source == "prosody"
