"""JSONL artifacts and blank-line segments for the first critic experiment."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from random import Random


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_text(path, text):
    """Commit a complete file; unfinished .tmp files are ignored on restart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def files_fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as file:
            file_digest = hashlib.file_digest(file, "sha256").hexdigest()
        digest.update(file_digest.encode())
    return digest.hexdigest()


def save_config(directory, config):
    path = Path(directory) / "run_config.json"
    if path.exists():
        if read_json(path) != config:
            raise ValueError(f"Settings or input data changed; choose a new output directory: {directory}")
    else:
        write_json(path, config)


def cached_shard(path, expected_ids, key):
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if [row[key] for row in rows] != expected_ids:
        raise ValueError(f"Cached shard has different or missing examples: {path}")
    return rows


def load_corpus(directory):
    directory = Path(directory)
    config = read_json(directory / "run_config.json")
    problems = read_jsonl(directory / "problems.jsonl")
    if fingerprint(problems) != config["problems_sha256"]:
        raise ValueError("Saved problems do not match the collection configuration.")
    paths = sorted((directory / "rollouts").glob("part-*.jsonl"))
    rollouts = [row for path in paths for row in read_jsonl(path)]
    expected = {f"{p['example_id']}:{i}" for p in problems for i in range(config["num_generations"])}
    actual = [row["rollout_id"] for row in rollouts]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Rollout corpus is incomplete or contains duplicate IDs; finish collection first.")
    by_id = {p["example_id"]: p for p in problems}
    if len(by_id) != len(problems):
        raise ValueError("A problem occurs in more than one split.")
    for row in rollouts:
        if row["rollout_id"] != f"{row['example_id']}:{row['sample_index']}":
            raise ValueError("Rollout ID differs from its problem and sample index.")
        if row["split"] != by_id[row["example_id"]]["split"]:
            raise ValueError("A rollout's split differs from its problem's split.")
    digest = files_fingerprint([directory / "run_config.json", directory / "problems.jsonl", *paths])
    return config, by_id, {row["rollout_id"]: row for row in rollouts}, digest


def reasoning_segments(response, rendered_prompt):
    """Return original-text offsets; the opening think tag can be in the prompt."""
    opening = response.find("<think>")
    if opening >= 0:
        start = opening + len("<think>")
    elif rendered_prompt.rfind("<think>") > rendered_prompt.rfind("</think>"):
        start = 0
    else:
        return []
    end = response.find("</think>", start)
    if end < 0:
        return []
    boundaries = list(re.finditer(r"\r?\n(?:[ \t]*\r?\n)+", response[start:end]))
    spans, left = [], start
    for right, next_left in [(start + m.start(), start + m.end()) for m in boundaries] + [(end, end)]:
        text = response[left:right]
        if text.strip():
            spans.append((left + len(text) - len(text.lstrip()), right - len(text) + len(text.rstrip())))
        left = next_left
    return [{"reasoning_start": start, "start": left, "end": right} for left, right in spans]


def merge_paragraphs(response, paragraphs, min_words):
    """Merge consecutive paragraphs, preserving exact source offsets and whitespace."""
    if min_words < 0:
        raise ValueError("min_words must be nonnegative.")
    merged, first, words = [], 0, 0
    for index, paragraph in enumerate(paragraphs):
        words += len(response[paragraph["start"]:paragraph["end"]].split())
        if words >= min_words:
            merged.append({
                **paragraphs[first], "end": paragraph["end"],
                "paragraph_index": first, "paragraph_end_index": index, "word_count": words,
            })
            first, words = index + 1, 0
    if first < len(paragraphs) and merged:
        # Fold a short trailing group into the preceding complete segment.
        merged[-1].update(end=paragraphs[-1]["end"], paragraph_end_index=len(paragraphs) - 1,
                          word_count=merged[-1]["word_count"] + words)
    # If the entire reasoning is too short, there is no segment meeting the minimum.
    return merged


def select_segments(problems, rollouts, segments_per_rollout, seed, min_segment_words=50):
    segments, exclusions = [], Counter()
    for rollout in rollouts.values():
        reason = None
        if rollout["truncated"]:
            reason = "truncated"
        elif rollout["finish_reason"] != "stop":
            reason = "unfinished"
        elif rollout["verifier_error"] is not None:
            reason = "verifier_error"
        problem = problems[rollout["example_id"]]
        paragraphs = reasoning_segments(rollout["response"], problem["rendered_prompt"]) if reason is None else []
        if reason or not paragraphs:
            exclusions[reason or "missing_or_empty_reasoning"] += 1
            continue
        spans = merge_paragraphs(rollout["response"], paragraphs, min_segment_words)
        if not spans:
            exclusions["reasoning_below_min_words"] += 1
            continue
        selection_seed = int(fingerprint([seed, rollout["rollout_id"]])[:8], 16)
        chosen = sorted(Random(selection_seed).sample(range(len(spans)), min(segments_per_rollout, len(spans))))
        for index in chosen:
            span = spans[index]
            first, last = span["paragraph_index"], span["paragraph_end_index"]
            suffix = str(first) if first == last else f"{first}-{last}"
            segments.append({
                "segment_id": f"{rollout['rollout_id']}:{suffix}", "rollout_id": rollout["rollout_id"],
                "example_id": rollout["example_id"], "split": rollout["split"],
                "segment_index": index, "num_segments": len(spans), "num_paragraphs": len(paragraphs),
                "selection_seed": selection_seed, **span,
            })
    return segments, dict(exclusions)


def segment_context(segment, problems, rollouts):
    problem = problems[segment["example_id"]]
    rollout = rollouts[segment["rollout_id"]]
    if rollout["example_id"] != segment["example_id"] or rollout["split"] != segment["split"]:
        raise ValueError("Segment identity or split differs from its rollout.")
    response = rollout["response"]
    start, end = segment["start"], segment["end"]
    if not 0 <= segment["reasoning_start"] <= start < end <= len(response):
        raise ValueError(f"Invalid segment offsets: {segment['segment_id']}")
    return problem, rollout, response[segment["reasoning_start"]:start], response[start:end], response[end:]


def load_segments(directory):
    directory = Path(directory)
    config = read_json(directory / "run_config.json")
    _, problems, rollouts, digest = load_corpus(config["corpus_dir"])
    segments = read_jsonl(directory / "segments.jsonl")
    if digest != config["corpus_sha256"] or fingerprint(segments) != config["segments_sha256"]:
        raise ValueError("Corpus or segment selection changed after target generation.")
    return config, problems, rollouts, segments


def extract_critique(response, truncated=False, require_thinking_end=False):
    """Only the final tagged critique is supervised, never teacher thinking."""
    if truncated:
        return None, "truncated"
    if (require_thinking_end or "<think>" in response) and "</think>" not in response:
        return None, "unfinished_thinking"
    final = response.rsplit("</think>", 1)[-1]
    matches = list(re.finditer(r"<critique>\s*(.*?)\s*</critique>", final, re.DOTALL))
    if len(matches) != 1 or not matches[0].group(1).strip():
        return None, "missing_or_multiple_critiques"
    target = matches[0].group(0)
    if "<think>" in target or "</think>" in target:
        return None, "thinking_in_critique"
    return target, None


def write_preview(path, segments, outputs, problems, rollouts, limit=30):
    """Readable annotations with visible context; raw generations remain in JSONL."""
    parts = []
    for segment, output in list(zip(segments, outputs, strict=True))[:limit]:
        problem, rollout, prefix, current, _ = segment_context(segment, problems, rollouts)
        error = output.get("target_error") or output.get("parse_error")
        response = output.get("target_text") or output.get("critique") or output["response"]
        label = "RAW RESPONSE (rejected)" if error else "CRITIQUE"
        parts.append(
            f"=== {segment['segment_id']} split={segment['split']} rollout_score={rollout['score']} "
            f"status={error or 'accepted'} ===\n\n"
            f"PROBLEM\n{problem['question']}\n\nPREFIX\n{prefix}\n\nSEGMENT\n{current}\n\n"
            f"{label}\n{response}\n\n"
        )
    write_text(path, "".join(parts))
