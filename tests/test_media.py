import shutil
import subprocess

import pytest
import soundfile as sf

from media_timeline.media import extract_audio, probe


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_delayed_audio_keeps_source_offset_after_extraction(tmp_path):
    source = tmp_path / "delayed.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=32x32:r=10:d=1",
        "-itsoffset", "0.4", "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=16000:duration=0.5",
        "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source),
    ], check=True)
    metadata = probe(source)
    assert metadata.audio_offset == pytest.approx(0.4)
    audio = tmp_path / "analysis.wav"
    extract_audio(source, audio)
    waveform, sr = sf.read(audio)
    assert sr == 16000
    assert len(waveform) / sr == pytest.approx(0.5, abs=0.01)
    assert max(abs(waveform[:160])) > 0.05
