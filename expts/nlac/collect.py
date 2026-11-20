"""Collect a reusable, scored DeepMath corpus from a frozen policy."""

import argparse
from collections import Counter
from pathlib import Path
from shutil import copyfile

from . import generation
from .data import cached_shard, fingerprint, read_jsonl, save_config, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    generation.add_arguments(parser, max_new_tokens=8192, temperature=1.0, top_k=0)
    parser.add_argument("--num_train_problems", type=int, default=200)
    parser.add_argument("--num_val_problems", type=int, default=50)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--split_manifest")

    args = parser.parse_args()
    generation.validate(args, parser)
    if min(args.num_train_problems, args.num_val_problems, args.num_generations) <= 0:
        parser.error("Problem and generation counts must be positive.")
    args.output_dir = str(Path(args.output_dir).resolve())
    generation.resolve_model(args, "openbmb/MiniCPM5-2B-SFT")

    from transformers import AutoTokenizer

    from ..data import deepmath

    data_config = deepmath.DataConfig(split_manifest=args.split_manifest)
    train, _, train_metadata = deepmath.load_training_data(data_config, args.seed)
    validation, _, val_metadata = deepmath.load_eval_data(data_config, args.num_val_problems)

    if args.num_train_problems > len(train):
        parser.error("Requested more training problems than the saved split contains.")
    selected = train.shuffle(seed=args.seed).select(range(args.num_train_problems))
    entries = [{**row["entry"], "split": "train"} for row in selected]
    entries += [{**entry, "split": "validation"} for entry in validation]
    if len({entry["example_id"] for entry in entries}) != len(entries):
        raise ValueError("Training and validation problems overlap.")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
    problems = []
    for entry in entries:
        messages = deepmath.make_prompt(entry["question"])
        ids = generation.prompt_ids(tokenizer, messages, thinking=True)
        problems.append({**entry, "prompt": messages, "prompt_token_ids": ids,
                         "rendered_prompt": tokenizer.decode(ids, skip_special_tokens=False)})
    args.split_manifest = str(Path(data_config.split_manifest).resolve())
    config = {
        **vars(args), "schema_version": 1, "stage": "collect",
        "data": {"train": train_metadata, "validation": val_metadata},
        "problems_sha256": fingerprint(problems), "chat_template": tokenizer.chat_template,
        "enable_thinking": True, "versions": generation.versions(),
    }
    directory = Path(args.output_dir)
    save_config(directory, config)
    path = directory / "problems.jsonl"
    if path.exists() and read_jsonl(path) != problems:
        raise ValueError("The saved problem file changed.")
    if not path.exists():
        write_jsonl(path, problems)
    destination = directory / "split_manifest.json"
    if Path(args.split_manifest).resolve() != destination.resolve():
        copyfile(args.split_manifest, destination)

    requests = [(p, sample) for p in problems for sample in range(args.num_generations)]
    llm, all_rows = None, []
    for start in range(0, len(requests), args.batch_size):
        batch = requests[start:start + args.batch_size]
        ids = [f"{p['example_id']}:{sample}" for p, sample in batch]
        path = directory / "rollouts" / f"part-{start // args.batch_size:05d}.jsonl"
        rows = cached_shard(path, ids, "rollout_id")
        if rows is None:
            if llm is None:
                llm = generation.load_llm(args)
            seeds = [args.seed + p["example_id"] * args.num_generations + sample for p, sample in batch]
            outputs = generation.generate(llm, [p["prompt"] for p, _ in batch], seeds, args, thinking=True)
            rows = [{
                "rollout_id": rollout_id, "example_id": p["example_id"], "sample_index": sample,
                "split": p["split"], **output, **deepmath.score_response(None, output["response"], p),
            } for (p, sample), rollout_id, output in zip(batch, ids, outputs, strict=True)]
            write_jsonl(path, rows)
        all_rows.extend(rows)
        print(f"Collected/cached {len(all_rows)}/{len(requests)} rollouts", flush=True)
    write_json(directory / "summary.json", {
        "rollouts": len(all_rows), "by_split": dict(Counter(r["split"] for r in all_rows)),
        "truncated": sum(r["truncated"] for r in all_rows),
        "verifier_errors": sum(r["verifier_error"] is not None for r in all_rows),
        "full_credit": sum(r["score"] == 1 for r in all_rows),
    })


if __name__ == "__main__":
    main()
