"""Full-parameter, completion-only SFT of a critic on fixed Option 1 targets."""

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from .data import (
    extract_critique, files_fingerprint, load_segments, read_json, read_jsonl,
    segment_context, write_json,
)
from .generation import prompt_ids
from .prompts import CRITIC_INSTRUCTION, critic_prompt


@dataclass
class RunConfig:
    targets_dir: str
    model_name: str | None = None
    model_revision: str | None = None


@dataclass
class TrainingConfig(SFTConfig):
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    optim: str = "adamw_bnb_8bit"
    gradient_checkpointing: bool = True
    max_length: int | None = 12288
    completion_only_loss: bool | None = True
    packing: bool = False
    dataset_kwargs: dict | str | None = field(default_factory=lambda: {"skip_prepare_dataset": True})
    eval_strategy: str = "epoch"
    prediction_loss_only: bool = True
    logging_steps: float = 10
    save_steps: float = 100
    save_total_limit: int | None = 2
    report_to: None | str | list[str] = "none"


def parse_config(argv=None):
    parser = HfArgumentParser((RunConfig, TrainingConfig), allow_abbrev=False)
    run, args = parser.parse_args_into_dataclasses(args=argv)
    if args.world_size != 1 or args.n_gpu > 1 or args.deepspeed or args.fsdp:
        parser.error("This experiment supports one training process and one visible GPU.")
    if args.packing or args.padding_free or args.completion_only_loss is not True:
        parser.error("Use unpacked, completion-only training for this experiment.")
    if args.max_length is None or args.max_length <= 0:
        parser.error("max_length must be positive.")
    if args.dataset_kwargs != {"skip_prepare_dataset": True}:
        parser.error("Keep dataset_kwargs={'skip_prepare_dataset': true}; labels are built explicitly.")
    run.targets_dir = str(Path(run.targets_dir).resolve())
    if args.run_name is None:
        args.run_name = f"critic-{Path(run.targets_dir).name}-seed{args.seed}"
    if args.output_dir == "trainer_output":
        args.output_dir = f"models/nlac/{args.run_name}"
    return run, args


def prepare_data(directory, tokenizer, max_length):
    directory = Path(directory)
    source, problems, rollouts, segments = load_segments(directory)
    if source["critic_instruction"] != CRITIC_INSTRUCTION:
        raise ValueError("Critic instructions changed; generate a new target dataset version.")
    paths = sorted((directory / "targets").glob("part-*.jsonl"))
    rows = [row for path in paths for row in read_jsonl(path)]
    ids = [row["segment_id"] for row in rows]
    if len(ids) != len(segments) or set(ids) != {s["segment_id"] for s in segments}:
        raise ValueError("Target generation is incomplete or has duplicate IDs.")
    by_id = {s["segment_id"]: s for s in segments}
    datasets = {"train": [], "validation": []}
    report = {"included": [], "excluded": []}
    if tokenizer.eos_token_id is None:
        raise ValueError("Critic training requires an EOS token.")
    for row in rows:
        sid = row["segment_id"]
        segment = by_id[sid]
        if row["target_error"] or not row["target_text"]:
            report["excluded"].append({"segment_id": sid, "reason": row["target_error"] or "empty_target"})
            continue
        target, error = extract_critique(row["target_text"])
        if error or target != row["target_text"]:
            raise ValueError(f"Target includes text outside the final critique: {sid}")
        problem, _, prefix, current, _ = segment_context(segment, problems, rollouts)
        prompt = prompt_ids(tokenizer, critic_prompt(problem["question"], prefix, current), thinking=False)
        completion = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        length = len(prompt) + len(completion)
        if length > max_length:
            report["excluded"].append({"segment_id": sid, "reason": "context_limit", "tokens": length})
            continue
        datasets[segment["split"]].append({
            "input_ids": prompt + completion,
            "labels": [-100] * len(prompt) + completion,
        })
        report["included"].append({"segment_id": sid, "split": segment["split"], "tokens": length})
    for split, examples in datasets.items():
        if not examples:
            raise ValueError(f"No usable {split} targets after filtering: {report['excluded'][:5]}")
    segments_dir = Path(source.get("segments_dir", directory))
    digest = files_fingerprint([directory / "run_config.json", segments_dir / "segments.jsonl", *paths])
    return {split: Dataset.from_list(rows) for split, rows in datasets.items()}, report, digest


def validate_resume(checkpoint, config):
    previous = read_json(Path(checkpoint).parent / "run_config.json")
    for key in ("targets_dir", "model_name", "model_revision", "targets_sha256", "versions",
                "critic_instruction", "chat_template", "enable_thinking"):
        if previous[key] != config[key]:
            raise ValueError(f"Cannot resume with changed {key}.")
    allowed = {"resume_from_checkpoint", "output_dir", "run_name", "report_to", "logging_steps",
               "save_steps", "save_total_limit", "disable_tqdm", "max_steps"}
    old, new = previous["training"], config["training"]
    changed = [key for key in old.keys() | new.keys() if key not in allowed and old.get(key) != new.get(key)]
    if changed:
        raise ValueError(f"Cannot resume with changed training settings: {', '.join(sorted(changed))}")
    if old["max_steps"] != new["max_steps"] and new["lr_scheduler_type"] != "constant":
        raise ValueError("Changing max_steps on resume requires a constant schedule.")


def train(run, args):
    set_seed(args.seed)
    directory = Path(args.output_dir)
    checkpoint = args.resume_from_checkpoint
    if checkpoint is True:
        checkpoint = get_last_checkpoint(str(directory)) if directory.exists() else None
        if checkpoint is None:
            raise ValueError(f"No checkpoint found in {directory}")
    if not checkpoint and (directory / "run_config.json").exists():
        raise FileExistsError("Run already exists; resume or choose a new output directory.")
    source = read_json(Path(run.targets_dir) / "run_config.json")
    actor = read_json(Path(source["corpus_dir"]) / "run_config.json")
    run.model_name = run.model_name or actor["model_name"]
    if run.model_revision is None and run.model_name == actor["model_name"]:
        run.model_revision = actor["model_revision"]
    if not Path(run.model_name).is_dir():
        from huggingface_hub import HfApi
        run.model_revision = HfApi().model_info(run.model_name, revision=run.model_revision).sha
    kwargs = args.model_init_kwargs or {}
    if "revision" in kwargs and kwargs["revision"] != run.model_revision:
        raise ValueError("Set model_revision explicitly; model_init_kwargs has a conflicting revision.")
    args.model_init_kwargs = {**kwargs, "revision": run.model_revision,
                              "dtype": "bfloat16" if args.bf16 else "float32"}
    tokenizer = AutoTokenizer.from_pretrained(run.model_name, revision=run.model_revision, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets, report, digest = prepare_data(run.targets_dir, tokenizer, args.max_length)
    names = ["trl", "transformers", "torch", "datasets", "accelerate"]
    if any(name in args.optim for name in ("bnb", "8bit", "paged")):
        names.append("bitsandbytes")
    config = {
        **asdict(run), "method": "nlac_option1_critic", "targets_sha256": digest,
        "training": args.to_dict(), "versions": {name: version(name) for name in names},
        "critic_instruction": CRITIC_INSTRUCTION, "enable_thinking": False,
        "chat_template": tokenizer.chat_template,
    }
    if checkpoint:
        validate_resume(checkpoint, config)
    write_json(directory / "run_config.json", config)
    write_json(directory / "data_report.json", report)
    trainer = SFTTrainer(
        model=run.model_name, args=args, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
    )
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(directory / "final"))
    write_json(directory / "train_summary.json", result.metrics)
    write_json(directory / "log_history.json", trainer.state.log_history)


def main():
    train(*parse_config())


if __name__ == "__main__":
    main()
