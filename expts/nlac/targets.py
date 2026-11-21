"""Generate fixed hindsight-informed critiques for sampled reasoning segments."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

from . import generation
from .data import (
    cached_shard, extract_critique, fingerprint, load_corpus, save_config, segment_context,
    select_segments, write_json, write_jsonl, write_preview,
)
from .prompts import CRITIC_INSTRUCTION, TARGET_INSTRUCTION, target_prompt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--segments_per_rollout", type=int, default=4)
    parser.add_argument(
        "--min_segment_words", type=int, default=50,
        help="Merge paragraphs until this many whitespace-separated words; 0 disables merging.",
    )
    parser.add_argument(
        "--no_thinking", dest="enable_thinking", action="store_false",
        help="Disable thinking in the target generator's chat template.",
    )
    generation.add_arguments(parser, max_new_tokens=2048, temperature=0.6, top_k=0)

    args = parser.parse_args()
    generation.validate(args, parser)
    if args.segments_per_rollout <= 0:
        parser.error("segments_per_rollout must be positive.")
    if args.min_segment_words < 0:
        parser.error("min_segment_words must be nonnegative.")
    args.corpus_dir = str(Path(args.corpus_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    return args


def save_selection(args, segments, exclusions, corpus_digest, num_rollouts):
    output_dir = Path(args.output_dir)
    save_config(output_dir, {
        **vars(args),
        "schema_version": 2,
        "stage": "targets",
        "corpus_sha256": corpus_digest,
        "segments_sha256": fingerprint(segments),
        "critic_instruction": CRITIC_INSTRUCTION,
        "target_instruction": TARGET_INSTRUCTION,
        "privileged_inputs": ["continuation", "verifier_score", "extracted_answer", "reference_answer"],
        "versions": generation.versions(),
    })
    write_jsonl(output_dir / "segments.jsonl", segments)

    # Record excluded rollouts even when the selection is empty.
    word_counts = [segment["word_count"] for segment in segments]
    word_stats = None
    if word_counts:
        word_stats = {
            "mean": mean(word_counts),
            "median": median(word_counts),
            "min": min(word_counts),
            "max": max(word_counts),
        }
    write_json(output_dir / "selection_summary.json", {
        "rollouts": num_rollouts,
        "excluded_rollouts": exclusions,
        "segments_by_split": dict(Counter(segment["split"] for segment in segments)),
        "segment_words": word_stats,
    })


def build_teacher_prompts(segments, problems, rollouts):
    prompts = []
    for segment in segments:
        problem, rollout, prefix, current, continuation = segment_context(segment, problems, rollouts)
        prompts.append(target_prompt(
            question=problem["question"],
            prefix=prefix,
            segment=current,
            continuation=continuation,
            score=rollout["score"],
            model_answer=rollout["model_answer"],
            reference_answer=problem["answer"],
        ))
    return prompts


def annotate_segments(args, segments, problems, rollouts):
    targets_dir = Path(args.output_dir) / "targets"
    teacher = None
    annotations = []
    for batch_index, start in enumerate(range(0, len(segments), args.batch_size)):
        batch = segments[start:start + args.batch_size]
        segment_ids = [segment["segment_id"] for segment in batch]
        shard_path = targets_dir / f"part-{batch_index:05d}.jsonl"
        rows = cached_shard(shard_path, segment_ids, "segment_id")

        # Sample missing batches with a stable seed for each segment.
        if rows is None:
            prompts = build_teacher_prompts(batch, problems, rollouts)
            if teacher is None:
                teacher = generation.load_llm(args)
            seeds = [int(fingerprint([args.seed, segment_id])[:8], 16) for segment_id in segment_ids]
            outputs = generation.generate(teacher, prompts, seeds, args, thinking=args.enable_thinking)

            # Keep every raw response alongside its parsed critique or rejection reason.
            rows = []
            for segment_id, output in zip(segment_ids, outputs, strict=True):
                critique, error = extract_critique(
                    output["response"], output["truncated"], require_thinking_end=args.enable_thinking,
                )
                rows.append({
                    "segment_id": segment_id,
                    **output,
                    "target_text": critique,
                    "target_error": error,
                })
            write_jsonl(shard_path, rows)

        annotations.extend(rows)
        print(f"Annotated/cached {len(annotations)}/{len(segments)} segments", flush=True)
    return annotations


def main():
    # Load the scored corpus and use its actor as the default teacher.
    args = parse_args()
    output_dir = Path(args.output_dir)
    corpus_config, problems, rollouts, corpus_digest = load_corpus(args.corpus_dir)
    generation.resolve_model(args, corpus_config["model_name"], corpus_config["model_revision"])

    # Select segments and save the exact inputs before starting teacher generation.
    segments, exclusions = select_segments(
        problems, rollouts, args.segments_per_rollout, args.seed, args.min_segment_words,
    )
    save_selection(args, segments, exclusions, corpus_digest, num_rollouts=len(rollouts))
    if not segments:
        raise ValueError("No eligible reasoning segments; inspect selection_summary.json.")

    # Generate hindsight-informed critiques, reusing any completed batches.
    annotations = annotate_segments(args, segments, problems, rollouts)

    # Count accepted critiques by split and report the reasons for rejection.
    outcomes = Counter()
    accepted_by_split = Counter()
    for segment, annotation in zip(segments, annotations, strict=True):
        outcomes[annotation["target_error"] or "accepted"] += 1
        if annotation["target_text"]:
            accepted_by_split[segment["split"]] += 1
    write_json(output_dir / "summary.json", {
        "segments": len(segments),
        "outcomes": dict(outcomes),
        "accepted_by_split": dict(accepted_by_split),
    })
    write_preview(output_dir / "preview.txt", segments, annotations, problems, rollouts)


if __name__ == "__main__":
    main()
