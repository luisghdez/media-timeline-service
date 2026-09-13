# Validation of schema 2.0 implementation

## Checks performed

- 50 automated tests passed, including the existing API and Seedance tests.
- A real ffmpeg/ffprobe fixture with audio delayed by 400 ms verified that extraction
  keeps a waveform-local clock and exports the corresponding source offset.
- Ran the Arise clip with cached faster-whisper small.en, AST, acoustic delivery,
  and scene detection, without separation. Result: `outputs/arise_timeline_v2.json`.
  The run returned no pipeline warnings, 14 events, 36 words, and 11 acoustic
  delivery spans. Counts are smoke-test observations, not accuracy scores.
- Installed CrisperWhisper 2.0.2 and ran the small checkpoint on the same clip.
  It produced a pathological repetition loop in this environment (245 consecutive
  repeated tokens; 40 of 253 tokens had positive-duration timings). The quality
  guard rejected it and the pipeline returned Whisper output with an explicit
  fallback warning: `outputs/arise_verbatim_v2.json`.
- Qwen response validation and audio-crop/word-anchor integration were tested with
  controlled model responses. Qwen model inference was not run.
- Forced-alignment result handling was tested for unsupported characters and
  incomplete alignment. WhisperX model inference was not run.

## Interpretation

The default is intentionally still Whisper. The new verbatim candidate did not
justify replacing it on this clip. The sample retains “Oh my gosh!” as the decoded
hypothesis; it does not yet recover the full repeated opening reliably. Acoustic
pulse spans deliberately do not claim panic or invent repeated syllables.

No manual reference annotation has been completed, so there is no measured claim
of WER improvement, emotion accuracy, or a 100 ms timing guarantee. The benchmark
manifest remains unreviewed and the scorer refuses to treat it as ground truth.
The next model-selection experiment should compare a larger verbatim model and
an audio-understanding model against manually checked speech/reaction boundaries.

Environment used for the real ASR checks: Python 3.14.7, faster-whisper 1.2.1,
CrisperWhisper 2.0.2, Transformers 5.17.0, Torch 2.14.0, CPU inference. Checkpoint
compatibility and model quality both need further comparison; the loop alone does
not establish its underlying cause.
