# Media Timeline Service

Creates confidence-scored, layered audio timelines from video files. It combines:

- `ffprobe` source verification and exact duration metadata
- `ffmpeg` audio extraction, silence detection, and visual scene boundaries
- `faster-whisper` speech transcription with word timestamps
- AudioSet/AST classification for music, effects, ambience, and nonverbal reactions
- optional Demucs separation so music does not hide dialogue or sustained vocal reactions
- deterministic fusion into overlapping `layers`

The service does not turn uncertain vocalizations into quoted words based on the audio classifier alone. If a model hears a wail but cannot reliably decode it, the result is a `vocal_reaction` with a description. A supplied caption/title may label a matching sustained reaction, but the event's evidence explicitly identifies `context` rather than `asr` as the source.

## Requirements

Python 3.11+ and `ffmpeg`/`ffprobe` on `PATH`.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[all,dev]'
```

Models are downloaded on first use and cached under `.cache/media-timeline`. Use `--offline` or `MEDIA_TIMELINE_OFFLINE=1` after the cache is populated.

## CLI

```bash
media-timeline input.mp4 --output timeline.json --artifacts-dir artifacts
```

`--artifacts-dir` also saves the normalized 16 kHz audio plus separated vocal and background stems when separation succeeds.

To produce the segment array used by the earlier examples:

```bash
media-timeline input.mp4 --segments-only --output timeline.json
```

For the clean, non-repeated event list:

```bash
media-timeline input.mp4 --events-only --output events.json
```

Pass the original post caption/title when available. A repeated exclamation can label a matching sustained vocal reaction, and its evidence explicitly records that the text came from context rather than ASR:

```bash
media-timeline input.mp4 --context 'NOOOOOO #gaming' --events-only
```

Fast mode without separation or audio-event classification:

```bash
media-timeline input.mp4 --no-separation --no-events
```

## HTTP service

```bash
uvicorn media_timeline.api:app --host 127.0.0.1 --port 8000
```

```bash
curl -F 'video=@input.mp4' -F 'context_text=NOOOOOO #gaming' \
  'http://127.0.0.1:8000/v1/analyze' > timeline.json
```

Use `?events_only=true` for a clean event array or `?segments_only=true` for fused timeline segments. The default upload limit is 500 MB and can be changed with `MEDIA_TIMELINE_MAX_UPLOAD_BYTES`.

Container deployment:

```bash
docker build -t media-timeline .
docker run --rm -p 8000:8000 -v media-timeline-models:/models media-timeline
```

The service also accepts environment overrides for the Whisper model, offline mode, separation/events/scenes toggles, silence and scene thresholds, and event window/hop sizes. Their names use the `MEDIA_TIMELINE_` prefix; see `src/media_timeline/api.py` for the exact keys.

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

## Output contract

Each segment has exact `start`/`end` boundaries and one or more simultaneous `layers`. Layers can include `speech`, `vocal_reaction`, `music`, `sound_effect`, `environment_audio`, `silence`, or `unclassified_audio`. Every detected layer includes a confidence value and, when available, model evidence.

The top-level response also contains source metadata, warnings, duration, and a schema version. Flat events and fused segments both expose `review_required`; treat `true` as a request for human or higher-capability model review.
