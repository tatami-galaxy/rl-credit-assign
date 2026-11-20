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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--segments_per_rollout", type=int, default=4)
    parser.add_argument("--min_segment_words", type=int, default=50,
                        help="Merge paragraphs until this many whitespace-separated words; 0 disables merging.")
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false",
                        help="Disable thinking in the target generator's chat template.")
    generation.add_arguments(parser, max_new_tokens=2048, temperature=0.6, top_k=0)
    args = parser.parse_args()
    generation.validate(args, parser)
    if args.segments_per_rollout <= 0:
        parser.error("segments_per_rollout must be positive.")
    if args.min_segment_words < 0:
        parser.error("min_segment_words must be nonnegative.")
    args.corpus_dir = str(Path(args.corpus_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    source, problems, rollouts, digest = load_corpus(args.corpus_dir)
    generation.resolve_model(args, source["model_name"], source["model_revision"])
    segments, exclusions = select_segments(
        problems, rollouts, args.segments_per_rollout, args.seed, args.min_segment_words,
    )
    config = {
        **vars(args), "schema_version": 2, "stage": "targets", "corpus_sha256": digest,
        "segments_sha256": fingerprint(segments),
        "critic_instruction": CRITIC_INSTRUCTION, "target_instruction": TARGET_INSTRUCTION,
        "privileged_inputs": ["continuation", "verifier_score", "extracted_answer", "reference_answer"],
        "versions": generation.versions(),
    }
    directory = Path(args.output_dir)
    save_config(directory, config)
    write_jsonl(directory / "segments.jsonl", segments)
    word_counts = [s["word_count"] for s in segments]
    write_json(directory / "selection_summary.json", {
        "rollouts": len(rollouts), "excluded_rollouts": exclusions,
        "segments_by_split": dict(Counter(s["split"] for s in segments)),
        "segment_words": {"mean": mean(word_counts), "median": median(word_counts),
                          "min": min(word_counts), "max": max(word_counts)} if word_counts else None,
    })
    if not segments:
        raise ValueError("No eligible reasoning segments; inspect selection_summary.json.")
    llm, all_rows = None, []
    for start in range(0, len(segments), args.batch_size):
        batch = segments[start:start + args.batch_size]
        ids = [s["segment_id"] for s in batch]
        path = directory / "targets" / f"part-{start // args.batch_size:05d}.jsonl"
        rows = cached_shard(path, ids, "segment_id")
        if rows is None:
            messages = []
            for segment in batch:
                problem, rollout, prefix, current, continuation = segment_context(segment, problems, rollouts)
                messages.append(target_prompt(
                    problem["question"], prefix, current, continuation,
                    rollout["score"], rollout["model_answer"], problem["answer"],
                ))
            if llm is None:
                llm = generation.load_llm(args)
            seeds = [int(fingerprint([args.seed, sid])[:8], 16) for sid in ids]
            outputs = generation.generate(llm, messages, seeds, args, thinking=args.enable_thinking)
            rows = []
            for sid, output in zip(ids, outputs, strict=True):
                target, error = extract_critique(
                    output["response"], output["truncated"], require_thinking_end=args.enable_thinking,
                )
                rows.append({"segment_id": sid, **output, "target_text": target, "target_error": error})
            write_jsonl(path, rows)
        all_rows.extend(rows)
        print(f"Annotated/cached {len(all_rows)}/{len(segments)} segments", flush=True)
    write_json(directory / "summary.json", {
        "segments": len(segments), "outcomes": dict(Counter(r["target_error"] or "accepted" for r in all_rows)),
        "accepted_by_split": dict(Counter(s["split"] for s, r in zip(segments, all_rows, strict=True) if r["target_text"])),
    })
    write_preview(directory / "preview.txt", segments, all_rows, problems, rollouts)


if __name__ == "__main__":
    main()
