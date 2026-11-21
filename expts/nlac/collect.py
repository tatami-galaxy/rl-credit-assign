"""Collect a reusable, scored DeepMath corpus from a frozen policy."""

import argparse
from collections import Counter
from pathlib import Path
from shutil import copyfile

from . import generation
from .data import cached_shard, fingerprint, read_jsonl, save_config, write_json, write_jsonl


def prepare_corpus_inputs(args, parser):
    from transformers import AutoTokenizer

    from ..data import deepmath

    # Sample training questions by seed and take validation questions in holdout order.
    data_config = deepmath.DataConfig(split_manifest=args.split_manifest)
    train, _, train_metadata = deepmath.load_training_data(data_config, args.seed)
    validation, _, validation_metadata = deepmath.load_eval_data(data_config, args.num_val_problems)
    if args.num_train_problems > len(train):
        parser.error("Requested more training problems than the saved split contains.")
    selected_train = train.shuffle(seed=args.seed).select(range(args.num_train_problems))
    entries = [{**row["entry"], "split": "train"} for row in selected_train]
    entries.extend({**entry, "split": "validation"} for entry in validation)
    if len({entry["example_id"] for entry in entries}) != len(entries):
        raise ValueError("Training and validation problems overlap.")

    # Keep both the token IDs and the rendered text of each thinking-mode prompt.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
    problems = []
    for entry in entries:
        messages = deepmath.make_prompt(entry["question"])
        token_ids = generation.prompt_ids(tokenizer, messages, thinking=True)
        problems.append({
            **entry,
            "prompt": messages,
            "prompt_token_ids": token_ids,
            "rendered_prompt": tokenizer.decode(token_ids, skip_special_tokens=False),
        })

    config = {
        **vars(args),
        "split_manifest": str(Path(data_config.split_manifest).resolve()),
        "schema_version": 1,
        "stage": "collect",
        "data": {"train": train_metadata, "validation": validation_metadata},
        "problems_sha256": fingerprint(problems),
        "chat_template": tokenizer.chat_template,
        "enable_thinking": True,
        "versions": generation.versions(),
    }
    return problems, config


def collect_rollouts(args, problems, output_dir):
    requests = [
        (problem, sample_index)
        for problem in problems
        for sample_index in range(args.num_generations)
    ]
    actor = None
    rollouts = []
    for batch_index, start in enumerate(range(0, len(requests), args.batch_size)):
        batch = requests[start:start + args.batch_size]
        rollout_ids = [f"{problem['example_id']}:{sample_index}" for problem, sample_index in batch]
        shard_path = output_dir / "rollouts" / f"part-{batch_index:05d}.jsonl"

        # Load the actor only when a batch still needs to be generated.
        rows = cached_shard(shard_path, rollout_ids, "rollout_id")
        if rows is None:
            if actor is None:
                actor = generation.load_llm(args)
            rows = generate_scored_batch(actor, args, batch, rollout_ids)
            write_jsonl(shard_path, rows)
        rollouts.extend(rows)
        print(f"Collected/cached {len(rollouts)}/{len(requests)} rollouts", flush=True)
    return rollouts


def generate_scored_batch(actor, args, batch, rollout_ids):
    from ..data import deepmath

    # Tie each seed to the problem and sample index so it is stable across restarts.
    prompts = [problem["prompt"] for problem, _ in batch]
    seeds = [
        args.seed + problem["example_id"] * args.num_generations + sample_index
        for problem, sample_index in batch
    ]
    outputs = generation.generate(actor, prompts, seeds, args, thinking=True)

    rows = []
    for (problem, sample_index), rollout_id, output in zip(batch, rollout_ids, outputs, strict=True):
        score = deepmath.score_response(None, output["response"], problem)
        rows.append({
            "rollout_id": rollout_id,
            "example_id": problem["example_id"],
            "sample_index": sample_index,
            "split": problem["split"],
            **output,
            **score,
        })
    return rows


def main():
    # Validate the collection settings and resolve the frozen actor's revision.
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
    output_dir = Path(args.output_dir).resolve()
    args.output_dir = str(output_dir)
    generation.resolve_model(args, "openbmb/MiniCPM5-2B-SFT")

    # Select the DeepMath problems and render prompts for the actor.
    problems, config = prepare_corpus_inputs(args, parser)

    # Save the inputs before sampling so a restarted run can check them.
    save_config(output_dir, config)
    problems_path = output_dir / "problems.jsonl"
    if problems_path.exists():
        if read_jsonl(problems_path) != problems:
            raise ValueError("The saved problem file changed.")
    else:
        write_jsonl(problems_path, problems)

    source_manifest = Path(config["split_manifest"])
    saved_manifest = output_dir / "split_manifest.json"
    if source_manifest != saved_manifest.resolve():
        copyfile(source_manifest, saved_manifest)

    # Collect or reuse scored rollouts, then summarize the whole corpus.
    rollouts = collect_rollouts(args, problems, output_dir)
    write_json(output_dir / "summary.json", {
        "rollouts": len(rollouts),
        "by_split": dict(Counter(row["split"] for row in rollouts)),
        "truncated": sum(row["truncated"] for row in rollouts),
        "verifier_errors": sum(row["verifier_error"] is not None for row in rollouts),
        "full_credit": sum(row["score"] == 1 for row in rollouts),
    })


if __name__ == "__main__":
    main()
