r"""Full-parameter PPO with a separate critic; select one training GPU.

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
        --dataset deepscaler --model_name Qwen/Qwen3-1.7B --max_steps 100

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
        --dataset deepmath --model_name Qwen/Qwen3-1.7B --max_steps 100

Use --dataset reasoning_gym --task leg_counting for procedural tasks.
"""

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from ..config import dataset_module
from .config import parse_config
from .trainer import PPOTrainer


def package_versions(dataset):
    names = ["trl", "transformers", "torch", "datasets", "accelerate", "vllm"]
    names += ["reasoning-gym"] if dataset == "reasoning_gym" else ["math-verify", "latex2sympy2_extended", "sympy"]
    return {name: version(name) for name in names}


def validate_resume(checkpoint, config):
    """Check data identity and settings restored by the optimizer and scheduler."""
    previous = json.loads((Path(checkpoint).parent / "run_config.json").read_text())
    # Before the layout change, RG settings were stored at the top level.
    if "data" not in previous:
        previous["dataset"] = "reasoning_gym"
        previous.setdefault("method", "grpo")
        previous["data"] = {
            key: previous.get(key) for key in ("task", "task_config", "task_sampling")
        }
        from ..data.reasoning_gym import SYSTEM_PROMPTS
        previous["data"].update(system_prompt=SYSTEM_PROMPTS["default"], answer_format="think_then_answer_tags")
    for key in ("method", "model_name", "dataset", "data"):
        if previous.get(key) != config[key]:
            raise ValueError(f"Cannot resume with changed {key}.")
    for name, old_version in previous["versions"].items():
        if config["versions"].get(name) != old_version:
            raise ValueError(f"Cannot resume with changed {name} version.")
    old, new = previous["training"], config["training"]
    fields = (
        "seed", "data_seed", "per_device_train_batch_size", "gradient_accumulation_steps",
        "generation_batch_size", "steps_per_generation", "num_generations", "num_iterations",
        "learning_rate", "critic_learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio",
        "optim", "weight_decay", "max_grad_norm", "critic_max_grad_norm", "critic_warmup_steps",
        "gamma", "lam", "vf_coef", "cliprange_value", "whiten_advantages", "loss_type",
        "epsilon", "epsilon_high", "max_completion_length", "mask_truncated_completions",
        "temperature", "top_p", "top_k", "generation_kwargs", "chat_template_kwargs",
        "bf16", "model_init_kwargs", "shuffle_dataset", "beta", "scale_rewards",
    )
    changed = [key for key in fields if old.get(key) != new.get(key)]
    if changed:
        raise ValueError(f"Cannot resume with changed training settings: {', '.join(changed)}")
    if old["max_steps"] != new["max_steps"] and new["lr_scheduler_type"] != "constant":
        raise ValueError("Changing max_steps on resume requires a constant learning-rate schedule.")


def train(run, data_config, args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint = args.resume_from_checkpoint
    if checkpoint is True:
        checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        if checkpoint is None:
            raise ValueError(f"No checkpoint found in {output_dir}")
    if not checkpoint and (output_dir / "run_config.json").exists():
        raise FileExistsError(f"Run already exists: {output_dir}. Resume or choose a new output_dir.")
    if checkpoint and run.dataset in {"deepscaler", "deepmath"} and data_config.split_manifest is None:
        data_config.split_manifest = str(Path(checkpoint).parent / "split_manifest.json")

    module = dataset_module(run.dataset)
    train_dataset, scorer, metadata = module.load_training_data(data_config, args.seed)
    config = {
        **asdict(run), "method": "ppo", "data": metadata,
        "training": args.to_dict(), "versions": package_versions(run.dataset),
    }
    if checkpoint:
        validate_resume(checkpoint, config)
    tokenizer = AutoTokenizer.from_pretrained(run.model_name, padding_side="left")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        run.model_name, num_labels=1, **args.model_init_kwargs
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id
    trainer = PPOTrainer(
        model=run.model_name, args=args, processing_class=tokenizer,
        train_dataset=train_dataset, reward_funcs=module.make_reward_function(scorer, tokenizer),
        value_model=value_model,
    )
    wandb_run = None
    try:
        if trainer.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            if run.dataset in {"deepscaler", "deepmath"}:
                source, target = Path(data_config.split_manifest), output_dir / "split_manifest.json"
                if source.resolve() != target.resolve():
                    copyfile(source, target)
            (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
            if "wandb" in args.report_to:
                import wandb

                wandb_run = wandb.init(
                    project=run.wandb_project, entity=run.wandb_entity, name=args.run_name,
                    config=config, dir=str(output_dir),
                )
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output_dir / "final"))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main():
    train(*parse_config())


if __name__ == "__main__":
    main()
