"""Save baseline or trained critic responses on the same fixed visible contexts."""

import argparse
from collections import Counter
from pathlib import Path

from . import generation
from .data import (
    cached_shard, extract_critique, fingerprint, load_segments, read_json, save_config,
    segment_context, write_json, write_jsonl, write_preview,
)
from .prompts import CRITIC_INSTRUCTION, critic_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets_dir", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--num_examples", type=int)
    generation.add_arguments(parser, max_new_tokens=512, temperature=0.0, top_k=0)
    args = parser.parse_args()
    generation.validate(args, parser)
    if args.num_examples is not None and args.num_examples <= 0:
        parser.error("num_examples must be positive.")
    args.targets_dir = str(Path(args.targets_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    source, problems, rollouts, segments = load_segments(args.targets_dir)
    actor = read_json(Path(source["corpus_dir"]) / "run_config.json")
    generation.resolve_model(args, actor["model_name"], actor["model_revision"])
    segments = [s for s in segments if s["split"] == args.split]
    if args.num_examples is not None:
        segments = segments[:args.num_examples]
    if not segments:
        raise ValueError("No segments in the requested split.")
    directory = Path(args.output_dir)
    save_config(directory, {
        **vars(args), "schema_version": 1, "stage": "predict", "corpus_sha256": source["corpus_sha256"],
        "segments_sha256": fingerprint(segments), "critic_instruction": CRITIC_INSTRUCTION,
        "enable_thinking": False, "versions": generation.versions(),
    })
    llm, all_rows = None, []
    for start in range(0, len(segments), args.batch_size):
        batch = segments[start:start + args.batch_size]
        ids = [s["segment_id"] for s in batch]
        path = directory / "responses" / f"part-{start // args.batch_size:05d}.jsonl"
        rows = cached_shard(path, ids, "segment_id")
        if rows is None:
            messages = []
            for segment in batch:
                problem, _, prefix, current, _ = segment_context(segment, problems, rollouts)
                messages.append(critic_prompt(problem["question"], prefix, current))
            if llm is None:
                llm = generation.load_llm(args)
            seeds = [int(fingerprint([args.seed, sid])[:8], 16) for sid in ids]
            outputs = generation.generate(llm, messages, seeds, args, thinking=False)
            rows = []
            for sid, output in zip(ids, outputs, strict=True):
                critique, error = extract_critique(output["response"], output["truncated"])
                rows.append({"segment_id": sid, **output, "critique": critique, "parse_error": error})
            write_jsonl(path, rows)
        all_rows.extend(rows)
        print(f"Generated/cached {len(all_rows)}/{len(segments)} critiques", flush=True)
    write_json(directory / "summary.json", {
        "segments": len(segments), "outcomes": dict(Counter(r["parse_error"] or "accepted" for r in all_rows)),
    })
    write_preview(directory / "responses.txt", segments, all_rows, problems, rollouts, limit=len(segments))


if __name__ == "__main__":
    main()
