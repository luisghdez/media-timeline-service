"""Human-reference benchmark. Never treat generated transcripts as ground truth."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _token(text: str) -> str:
    return re.sub(r"[^\w'-]", "", text.lower())


def _word_matches(reference: list[dict], predicted: list[dict]) -> tuple[int, list[tuple[int, int]]]:
    """Levenshtein alignment preserves repetition order and counts all ASR errors."""
    n, m = len(reference), len(predicted)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            equal = _token(reference[i - 1]["text"]) == _token(predicted[j - 1]["text"])
            costs[i][j] = min(costs[i - 1][j] + 1, costs[i][j - 1] + 1, costs[i - 1][j - 1] + (not equal))
    matches = []
    i, j = n, m
    while i or j:
        equal = i > 0 and j > 0 and _token(reference[i - 1]["text"]) == _token(predicted[j - 1]["text"])
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (not equal):
            if equal:
                matches.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return costs[n][m], list(reversed(matches))


def _iou(a: dict, b: dict) -> float:
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return overlap / union if union > 0 else 0.0


def _event_matches(reference: list[dict], predicted: list[dict]) -> list[tuple[int, int]]:
    candidates = sorted(((_iou(a, b), i, j) for i, a in enumerate(reference)
                         for j, b in enumerate(predicted) if a["label"] == b["label"]), reverse=True)
    matches, used_a, used_b = [], set(), set()
    for iou, i, j in candidates:
        if iou >= 0.5 and i not in used_a and j not in used_b:
            matches.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return matches


def _prf(tp: int, reference: int, predicted: int) -> dict:
    return {"true_positive": tp, "reference_count": reference, "prediction_count": predicted,
            "precision": tp / predicted if predicted else None,
            "recall": tp / reference if reference else None,
            "f1": 2 * tp / (reference + predicted) if reference + predicted else None}


def _timing(errors: list[float], expected: int) -> dict:
    ordered = sorted(errors)
    return {"scored_boundaries": len(errors), "reference_boundaries": expected,
            "coverage": len(errors) / expected if expected else None,
            "mean_absolute_error_ms": 1000 * sum(errors) / len(errors) if errors else None,
            "p95_error_ms": 1000 * ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else None,
            "within_100ms_on_matched": sum(e <= 0.1000001 for e in errors) / len(errors) if errors else None,
            "within_100ms_of_all_reference": sum(e <= 0.1000001 for e in errors) / expected if expected else None}


def _labels(spans: list[dict], field: str) -> list[dict]:
    output = []
    for span in spans:
        labels = span.get(field, []) if field == "features" else [span.get(field)]
        for label in labels:
            if label and label != "repeated_energy_pulses":
                output.append({"start": span["start"], "end": span["end"], "label": label})
    return output


def evaluate(reference: dict, predictions: dict[str, dict]) -> dict:
    reviewed = [clip for clip in reference["clips"] if clip.get("reviewed") is True]
    if not reviewed:
        raise ValueError("No human-reviewed reference clips; annotate and set reviewed=true before scoring")
    total_words = edits = 0
    word_errors, reaction_errors = [], []
    counts = {key: [0, 0, 0] for key in ("reactions", "delivery", "emotion", "disfluencies")}
    failures = []
    for clip in reviewed:
        if clip["id"] not in predictions:
            raise ValueError(f"Missing prediction for reviewed clip {clip['id']}")
        result = predictions[clip["id"]]
        if result.get("warnings"):
            failures.append({"id": clip["id"], "warnings": result["warnings"]})
        events = result.get("events", [])
        words = sorted([word for event in events if event["type"] == "speech"
                        for word in event.get("words", [])], key=lambda w: w.get("start", float("inf")))
        expected = clip["words"]
        for word in expected:
            if not _valid_time(word):
                raise ValueError(f"Reference word lacks valid timing in {clip['id']}")
        distance, matched = _word_matches(expected, words)
        edits += distance
        total_words += len(expected)
        for i, j in matched:
            if _valid_time(words[j]):
                word_errors.extend(abs(expected[i][edge] - words[j][edge]) for edge in ("start", "end"))
        ref_dis = sum(w.get("kind", "word") in {"partial_syllable", "repetition", "filler"} for w in expected)
        pred_dis = sum(w.get("kind", "word") in {"partial_syllable", "repetition", "filler"} for w in words)
        tp_dis = sum(expected[i].get("kind", "word") == words[j].get("kind", "word")
                     and expected[i].get("kind", "word") in {"partial_syllable", "repetition", "filler"} for i, j in matched)
        counts["disfluencies"] = [x + y for x, y in zip(counts["disfluencies"], (tp_dis, ref_dis, pred_dis))]
        predicted_reactions = [{"start": e["start"], "end": e["end"], "label": e.get("description", "unknown")}
                               for e in events if e["type"] == "vocal_reaction"]
        spans = [span for e in events for span in e.get("delivery", [])]
        for key, truth, detected in (
            ("reactions", clip.get("reactions", []), predicted_reactions),
            ("delivery", _labels(clip.get("delivery", []), "features"), _labels(spans, "features")),
            ("emotion", _labels(clip.get("delivery", []), "perceived_emotion"), _labels(spans, "perceived_emotion")),
        ):
            if any(not _valid_time(entry) for entry in truth + detected):
                raise ValueError(f"Invalid {key} timing in {clip['id']}")
            matched_events = _event_matches(truth, detected)
            counts[key] = [x + y for x, y in zip(counts[key], (len(matched_events), len(truth), len(detected)))]
            if key == "reactions":
                for i, j in matched_events:
                    reaction_errors.extend(abs(truth[i][edge] - detected[j][edge]) for edge in ("start", "end"))
    return {
        "reviewed_clips": len(reviewed), "unreviewed_clips": len(reference["clips"]) - len(reviewed),
        "word_error_rate": edits / total_words if total_words else None,
        "word_edits": edits, "reference_words": total_words,
        "word_timing": _timing(word_errors, total_words * 2),
        "reaction_timing": _timing(reaction_errors, counts["reactions"][1] * 2),
        **{key: _prf(*value) for key, value in counts.items()},
        "pipeline_warnings": failures,
        "notes": "Timing scored on matched words/events; coverage and all-reference metric include misses. Event matching uses same label and IoU>=0.5. Emotion scores measure agreement with human perception.",
    }


def _valid_time(item: dict) -> bool:
    return all(isinstance(item.get(k), (int, float)) and math.isfinite(item[k]) for k in ("start", "end")) and 0 <= item["start"] < item["end"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("inputs", type=Path)
    init.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("reference", type=Path)
    score.add_argument("predictions", type=Path, help="Directory containing <clip-id>.json outputs")
    score.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        if args.output.exists():
            parser.error("Reference already exists; refusing to overwrite annotations")
        clips = [{"id": f"clip-{i:03d}", "source": str(path.resolve()), "reviewed": False,
                  "tags": [], "words": [], "reactions": [], "delivery": []}
                 for i, path in enumerate(sorted(p for p in args.inputs.rglob("*")
                                                 if p.suffix.lower() in {".mp4", ".wav", ".mov", ".webm"}), 1)]
        result = {"schema_version": "1.0", "timebase": "source_playback_seconds", "clips": clips}
    else:
        reference = json.loads(args.reference.read_text())
        predictions = {p.stem: json.loads(p.read_text()) for p in args.predictions.glob("*.json")}
        try:
            result = evaluate(reference, predictions)
        except ValueError as exc:
            parser.error(str(exc))
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
