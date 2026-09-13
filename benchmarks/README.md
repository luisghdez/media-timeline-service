# Annotating and evaluating the clips

The reference manifest is a starting point for human annotation, not a scored
baseline. Listen to each original clip with a waveform editor. Annotate the
source playback timeline, not a separated stem's local clock. Mark `reviewed`
true only after completing all speech, reaction, and delivery annotations.

Record clear speech, partial syllables, repeated words, fillers, and vocal sounds
separately where their boundaries can be heard. Use the same tokenization across
model comparisons; for a model that returns a combined `o-o-oh` token, do not claim
that its single interval supplies individual syllable boundaries. Add tags such
as `clear_speech`, `music_overlap`, `stutter`, `scream`, and `abrupt_edit`.

Each clip has these arrays (numbers below illustrate the format, not real audio):

```json
{
  "words": [
    {"text": "o-", "start": 1.01, "end": 1.08, "kind": "partial_syllable"},
    {"text": "oh", "start": 1.13, "end": 1.30, "kind": "word"}
  ],
  "reactions": [
    {"label": "audible gasp", "start": 2.00, "end": 2.15}
  ],
  "delivery": [
    {
      "start": 1.01,
      "end": 1.30,
      "features": ["repeated_initial_syllable", "rising_pitch"],
      "perceived_emotion": "panicked"
    }
  ]
}
```

Reaction labels match the descriptions in `REACTION_LABELS` in analyzers.py.
Delivery features match `FEATURES` in delivery.py. When emotion is unclear, omit
it rather than forcing a label. For emotional perception, a second annotator and
agreement review are useful; a score measures human/model agreement, not a
speaker's actual internal state. Do not derive the reference from captions or
copy model timestamps without listening.

Run each configuration on the same inputs, save the full JSON as `<clip-id>.json`
in a separate prediction directory, and score each directory against the same
reviewed reference. Store the reports and model configurations together.

Timing errors are reported on correctly matched words and reactions, alongside
coverage and an all-reference success rate so omissions cannot inflate accuracy.
Reaction/delivery matching requires the same label and temporal IoU >= 0.5.
Repetition/filler metrics use word `kind`; WER also counts omitted or invented
partial syllables. The 100 ms threshold is an evaluation target, not a guarantee.

Suggested initial gate for *clear speech*: at least 95% of all reference word
boundaries within 100 ms. Evaluate stutters, screams, and music overlap separately;
report missed/invented details before tuning thresholds on held-out clips. Five
clips are a smoke benchmark, not enough to calibrate production confidence.
