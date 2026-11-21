"""Split a saved actor corpus into reasoning segments for inspection and annotation."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

from .data import fingerprint, load_corpus, save_config, segment_rollouts, write_json, write_jsonl, write_text


def segment_stats(segments):
    words = [segment["word_count"] for segment in segments]
    word_stats = None
    if words:
        word_stats = {
            "min": min(words),
            "median": median(words),
            "mean": mean(words),
            "max": max(words),
        }
    return {
        "count": len(segments),
        "by_split": dict(Counter(segment["split"] for segment in segments)),
        "words": word_stats,
    }


def write_segment_preview(path, segments, problems):
    """Show each question and its segment boundaries in rollout order."""
    lines = []
    previous_rollout = None
    for segment in segments:
        rollout_id = segment["rollout_id"]
        if rollout_id != previous_rollout:
            lines.extend([
                f"=== Rollout {rollout_id} ({segment['split']}) ===",
                problems[segment["example_id"]]["question"],
                "",
            ])
            previous_rollout = rollout_id
        marker = " [selected]" if segment["selected"] else ""
        lines.extend([
            f"--- {segment['segment_id']} | {segment['word_count']} words{marker} ---",
            segment["text"],
            "",
        ])
    write_text(path, "\n".join(lines))


def main():
    # Choose the boundaries and the sample to annotate for this segmentation.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_segment_words", type=int, default=50,
                        help="Merge paragraphs to at least this many words. Use 0 to keep each paragraph.")
    parser.add_argument("--segments_per_rollout", type=int, default=4,
                        help="Mark up to this many segments per rollout for the teacher.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_segment_words < 0 or args.segments_per_rollout <= 0:
        parser.error("min_segment_words must be nonnegative and segments_per_rollout must be positive.")
    corpus_dir = Path(args.corpus_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Keep all eligible segments so the full partition is available for analysis.
    _, problems, rollouts, corpus_digest = load_corpus(corpus_dir)
    segments, exclusions = segment_rollouts(
        problems, rollouts, args.segments_per_rollout, args.seed, args.min_segment_words,
    )
    save_config(output_dir, {
        "schema_version": 1,
        "stage": "segment",
        "method": "merge_paragraphs",
        "corpus_dir": str(corpus_dir),
        "corpus_sha256": corpus_digest,
        "min_segment_words": args.min_segment_words,
        "segments_per_rollout": args.segments_per_rollout,
        "seed": args.seed,
        "segments_sha256": fingerprint(segments),
    })
    write_jsonl(output_dir / "segments.jsonl", segments)

    # Make the boundaries and the sampled subset inspectable before running a teacher.
    selected = [segment for segment in segments if segment["selected"]]
    write_json(output_dir / "summary.json", {
        "rollouts": len(rollouts),
        "excluded_rollouts": exclusions,
        "all_segments": segment_stats(segments),
        "selected_segments": segment_stats(selected),
    })
    write_segment_preview(output_dir / "preview.txt", segments, problems)
    print(f"Saved {len(segments)} segments to {output_dir}. Selected {len(selected)} for annotation.")


if __name__ == "__main__":
    main()
