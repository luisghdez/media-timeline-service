# Media Timeline Service

Produces layered speech and vocal-reaction timelines from video. Schema 2.0 preserves
verbatim hypotheses, individual word times, vocal-delivery spans, model evidence,
and explicit uncertainty. Video scene cuts split the presentation only; they never
move audio events.

## Install and run

Python 3.11+ and `ffmpeg`/`ffprobe` on `PATH` are required. The optional model
packages may impose narrower Python requirements; Python 3.11/3.12 is recommended
when installing the full model stack.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[all,dev]'
.venv/bin/python -m media_timeline.cli input.mp4 --output timeline.json
```

A guarded fallback rejects pathological verbatim decode loops and keeps a Whisper
transcript with an explicit warning. Ordinary short stutters are preserved.

The default retains the lightweight `faster-whisper small.en` baseline and adds
acoustic observations. It preserves the words that model returns, including weak
hypotheses, but cannot recover syllables the model never transcribes. Acoustic
energy pulses are not labeled as stutters or panic.

Optional quality stages are independently selectable:

```bash
.venv/bin/pip install -e '.[verbatim,alignment,delivery]'
.venv/bin/python -m media_timeline.cli input.mp4 \
  --asr-backend crisper --verbatim-model small \
  --alignment whisperx --delivery qwen \
  --asr-source mix --no-separation \
  --output timeline.json --artifacts-dir artifacts
```

Use `--verbatim-model large` to compare the larger model. This is a benchmark
candidate, not a claim of measured accuracy on these clips. CrisperWhisper's
Transformers backend avoids installing its CT2 fork over faster-whisper. Its model
weights have separate terms: the 2.0 standard weights are under a non-commercial
research license; commercial licensing is available from the model author.
See [CrisperWhisper documentation](https://github.com/nyrahealth/CrisperWhisper).

WhisperX aligns existing words against the waveform. Unsupported characters or
failed alignment retain their ASR estimates and require review; they are never
silently interpolated into precise-looking word times. The aligner cannot restore
missing speech. `--language en` selects the ASR/alignment language.

`--delivery qwen` runs Qwen2.5-Omni-3B locally on short audio crops with surrounding
context. It returns acoustic descriptions and perceived emotion, anchored to
existing word indices. It cannot change the transcript or generate timestamps.
The model is substantially heavier than the default pipeline; device placement is
automatic. `--device` controls CrisperWhisper and WhisperX. Model self-reported
confidence remains uncalibrated and reviewable.

`--delivery acoustic` (default) records repeated energy peaks without guessing
phonemes or emotion. `--delivery none` disables delivery analysis.

`--asr-source both` compares mix and separated vocals. Disagreeing hypotheses are
retained as alternatives with their word times and provenance. Separation uses
the same extracted waveform as ASR to preserve a common clock. Optional failures
appear in `warnings`; failed requested alignment/delivery also flag events.

```bash
# Small cached baseline, without separation or the audio classifier
.venv/bin/python -m media_timeline.cli input.mp4 \
  --offline --no-separation --no-events --output timeline.json

# Flat authoritative events rather than the presentation segment array
.venv/bin/python -m media_timeline.cli input.mp4 --events-only
```

`--segments-only` emits the presentation view. `--context` retains a caption/title
as an explicitly unverified hint; it never supplies quoted speech or raises
confidence. `--artifacts-dir` saves the analysis WAV and successful separated stems.
ASR/event/alignment/delivery model caches live under `.cache/media-timeline`.
Use `--offline` after populating those caches. Demucs uses its own Torch cache.

## Output contract (2.0)

- `events` is the authoritative list of independently timed speech and reactions.
- Speech includes `text`, `verbatim_text`, `transcript_mode`, and `words`. Each word
  has text, a kind (word, partial syllable, repetition, filler, or vocalization),
  timing method, and available confidence. `verbatim_text` is the raw hypothesis;
  `transcript_mode=asr` does not imply that every audible disfluency was recovered.
- `delivery` contains timed features, descriptions, acoustic evidence, and optional
  `perceived_emotion`/`emotion_confidence`. Delivery and emotion confidence are
  separate from transcript and timing confidence. Energy-pulse measurements have
  no invented confidence value.
- Times are seconds relative to source playback. `start_sample`/`end_sample` refer
  to the extracted mono 16 kHz waveform. Convert with
  `source_seconds = sample_index / analysis_sample_rate + analysis_audio_offset_seconds`.
  Model estimates are not guaranteed sample-accurate just because sample indices
  are available. Seconds serialize to six decimal places.
- `segments` may split events at scene cuts. Nested words/delivery still carry
  absolute event times and can extend beyond the presentation segment. Use events
  for evaluation, editing, and annotation rather than repeating segment text.
- Reaction classifications are candidate windows, not exact onset/offset labels.
  Isolated sounds bracketed by quiet can receive energy-envelope estimates.
  Ambiguous/overlapping sounds remain coarse and require review.
- `review_required` uses the same rules in both views: explicit review reasons,
  low transcript/word/timing scores, and low delivery/emotion scores. Unknown
  dimension scores are omitted. Legacy `confidence` is retained; zero with an
  unavailable-confidence reason means unknown. Raw scores are not calibrated
  probabilities. Caption text cannot increase them.
- Alternative ASR hypotheses explicitly use the analysis-waveform timebase.

Version 2 removes scene snapping, 120 ms boundary coalescing, speech-to-reaction
promotion, generic reaction-run merging, caption-derived transcriptions, and
silence-based deletion of quiet speech. Silence can split reaction candidates
while preserving both sides. AudioSet scores use sigmoid for independent labels;
`--event-threshold` (default 0.07) is provisional and should be tuned on reviewed
clips. Labels no longer imply emotion.

## Accuracy benchmark

See [benchmarks/README.md](benchmarks/README.md) and the
[implementation validation report](benchmarks/validation.md). `benchmarks/reference.json` lists
the five supplied clips, initially unreviewed. It deliberately contains no
machine-generated ground truth.

```bash
# For a new collection (refuses to overwrite existing annotations)
.venv/bin/python -m media_timeline.benchmark init inputs --output new-reference.json

# After human annotation, place full pipeline outputs at predictions/<clip-id>.json
.venv/bin/python -m media_timeline.benchmark score benchmarks/reference.json predictions
```

The report includes WER, word-boundary MAE/p95, the fraction within 100 ms,
coverage, disfluency precision/recall, reaction timing/detection, delivery/emotion
agreement, and pipeline warnings. Unreviewed clips cannot be scored. Missing
predictions for reviewed clips are errors, not silently excluded examples.

## HTTP service

```bash
uvicorn media_timeline.api:app --host 127.0.0.1 --port 8000
curl -F 'video=@input.mp4' 'http://127.0.0.1:8000/v1/analyze' > timeline.json
```

Use `?events_only=true` or `?segments_only=true`. The default upload limit is 500 MB
(`MEDIA_TIMELINE_MAX_UPLOAD_BYTES`). Configure the optional stages using
`MEDIA_TIMELINE_ASR_BACKEND`, `MEDIA_TIMELINE_VERBATIM_MODEL`,
`MEDIA_TIMELINE_ALIGNMENT`, `MEDIA_TIMELINE_DELIVERY`, `MEDIA_TIMELINE_DELIVERY_MODEL`,
`MEDIA_TIMELINE_LANGUAGE`, and `MEDIA_TIMELINE_DEVICE`. Additional overrides include
`MEDIA_TIMELINE_WHISPER_MODEL`, `MEDIA_TIMELINE_EVENT_THRESHOLD`, `MEDIA_TIMELINE_OFFLINE`,
`MEDIA_TIMELINE_SPEECH_SOURCE`, and the existing scene/separation/event toggles.
See `src/media_timeline/api.py` for all keys.

## Seedance text-to-video

Separate FastAPI module wrapping [Seedance 2.5 on fal.ai](https://fal.ai/models/bytedance/seedance-2.5/text-to-video/api). It does not share the timeline pipeline; compose them later over HTTP/JSON.

```bash
cp .env.example .env   # set FAL_KEY
.venv/bin/pip install -e '.[seedance,dev]'
uvicorn seedance.api:app --host 127.0.0.1 --port 8001
```

Blocking generate (polls fal until the video is ready):

```bash
curl -s http://127.0.0.1:8001/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"An octopus finds a football in the ocean."}'
```

Queue API for long runs (`submit` → `status` → `result`):

```bash
curl -s http://127.0.0.1:8001/v1/jobs \
  -H 'content-type: application/json' \
  -d '{"prompt":"An octopus finds a football in the ocean.","duration":"8","aspect_ratio":"9:16"}'
curl -s 'http://127.0.0.1:8001/v1/jobs/REQUEST_ID?logs=true'
curl -s http://127.0.0.1:8001/v1/jobs/REQUEST_ID/result
```

CLI:

```bash
seedance 'An octopus finds a football in the ocean.' --duration 8 --aspect-ratio 9:16
```

Optional fields match the fal schema: `resolution` (`480p`/`720p`/`1080p`), `duration` (`auto` or 4–30), `aspect_ratio`, `generate_audio`, `bitrate_mode`, `end_user_id`, and `webhook_url` on `/v1/jobs`.

## Wan 3.0 Prime reference-to-video

Separate FastAPI module wrapping [Wan 3.0 Prime reference-to-video on fal.ai](https://fal.ai/models/alibaba/wan-3.0-prime/reference-to-video). Caller input is a single audio file. Duration is taken from that clip (2–30s). Fal caps reference audio at 15s. Output is silent (`audio=false`); mux the original soundtrack later.

```bash
cp .env.example .env   # set FAL_KEY
.venv/bin/pip install -e '.[wan3,dev]'
uvicorn wan3.api:app --host 127.0.0.1 --port 8002
```

Blocking generate (uploads audio, polls fal until the video is ready):

```bash
curl -s http://127.0.0.1:8002/v1/generate \
  -F 'audio=@outputs/arise_artifacts/vocals.wav'
```

Queue API (`submit` → `status` → `result`):

```bash
curl -s http://127.0.0.1:8002/v1/jobs -F 'audio=@outputs/arise_artifacts/vocals.wav'
curl -s 'http://127.0.0.1:8002/v1/jobs/REQUEST_ID?logs=true'
curl -s http://127.0.0.1:8002/v1/jobs/REQUEST_ID/result
```

CLI:

```bash
wan3 outputs/arise_artifacts/vocals.wav -o outputs/wan3/result.json
```

Defaults: `480p`, `16:9`, silent video, and a streamer-reaction prompt. Optional form/CLI fields: `prompt`, `resolution`, `aspect_ratio`, `seed`, and `webhook_url` on `/v1/jobs`.

