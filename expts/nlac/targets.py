"""Generate hindsight-informed critiques for a saved selection of reasoning segments."""

import argparse
from collections import Counter
from pathlib import Path

from . import generation
from .data import (
    cached_shard, extract_critique, fingerprint, load_segments, read_json, save_config,
    segment_context, write_json, write_jsonl, write_preview,
)
from .prompts import CRITIC_INSTRUCTION, TARGET_INSTRUCTION, target_prompt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True, help="Directory containing the saved segments.jsonl.")
    parser.add_argument(
        "--no_thinking", dest="enable_thinking", action="store_false",
        help="Disable thinking in the target generator's chat template.",
    )
    generation.add_arguments(parser, max_new_tokens=2048, temperature=0.6, top_k=0)

    args = parser.parse_args()
    generation.validate(args, parser)
    args.segments_dir = str(Path(args.segments_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    return args


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
    # Load the saved sample and use its actor as the default teacher.
    args = parse_args()
    output_dir = Path(args.output_dir)
    source, problems, rollouts, segments = load_segments(args.segments_dir)
    if not segments:
        raise ValueError(f"No segments selected for annotation in {args.segments_dir}.")
    actor = read_json(Path(source["corpus_dir"]) / "run_config.json")
    generation.resolve_model(args, actor["model_name"], actor["model_revision"])

    # Pin the corpus and segment file used for this teacher run.
    save_config(output_dir, {
        **vars(args),
        "schema_version": 3,
        "stage": "targets",
        "segments_dir": source.get("segments_dir", args.segments_dir),
        "segments_sha256": source["segments_sha256"],
        "corpus_dir": source["corpus_dir"],
        "corpus_sha256": source["corpus_sha256"],
        "critic_instruction": CRITIC_INSTRUCTION,
        "target_instruction": TARGET_INSTRUCTION,
        "privileged_inputs": ["continuation", "verifier_score", "extracted_answer", "reference_answer"],
        "versions": generation.versions(),
    })

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
