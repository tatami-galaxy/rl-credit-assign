r"""Answer-conditioned SDFT with TRL defaults; select one training GPU.

    CUDA_VISIBLE_DEVICES=6 uv run python -m expts.sdft.train \
        --task leg_counting --max_steps 100 --report_to wandb

    CUDA_VISIBLE_DEVICES=6 uv run python -m expts.sdft.train \
        --dataset deepmath --max_steps 100 --use_vllm true
"""

import copy
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile

from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from ..config import dataset_module
from .config import parse_config
from .data import load_training_data
from .trainer import SDFTTrainer


def package_versions(dataset, use_vllm):
    names = ["trl", "transformers", "torch", "datasets", "accelerate"]
    if use_vllm:
        names.append("vllm")
    names += ["reasoning-gym"] if dataset == "reasoning_gym" else ["math-verify", "latex2sympy2_extended", "sympy"]
    return {name: version(name) for name in names}


def validate_resume(checkpoint, config):
    previous = json.loads((Path(checkpoint).parent / "run_config.json").read_text())
    for key in ("method", "model_name", "dataset", "data", "versions"):
        if previous.get(key) != config[key]:
            raise ValueError(f"Cannot resume with changed {key}.")
    old, new = previous["training"], config["training"]
    fields = (
        "seed", "data_seed", "per_device_train_batch_size", "gradient_accumulation_steps",
        "generation_batch_size", "steps_per_generation", "num_generations", "num_iterations",
        "num_train_epochs", "learning_rate", "lr_scheduler_type", "lr_scheduler_kwargs",
        "warmup_steps", "warmup_ratio", "optim", "optim_args", "weight_decay", "max_grad_norm",
        "adam_beta1", "adam_beta2", "adam_epsilon", "max_prompt_length", "max_completion_length",
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "generation_kwargs",
        "chat_template_kwargs", "bf16", "fp16", "model_init_kwargs", "shuffle_dataset",
        "teacher_model_kind", "teacher_update_rate", "teacher_sync_steps", "teacher_prompt_template",
        "generate_from_teacher", "distillation_mode", "distillation_alpha", "distillation_topk",
        "distillation_add_tail", "distillation_is_clip", "num_loss_tokens_to_skip",
        "use_vllm", "vllm_mode", "use_teacher_server", "disable_dropout",
    )
    changed = [name for name in fields if old.get(name) != new.get(name)]
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

    rows, scorer, metadata = load_training_data(run, data_config, args.seed)
    questions_per_batch = args.generation_batch_size // args.num_generations
    if len(rows) < questions_per_batch:
        raise ValueError(f"Training needs at least {questions_per_batch} examples for one generation batch.")
    config = {
        **asdict(run), "method": "sdft", "data": metadata,
        "training": copy.deepcopy(args.to_dict()), "versions": package_versions(run.dataset, args.use_vllm),
    }
    if checkpoint:
        validate_resume(checkpoint, config)
    tokenizer = AutoTokenizer.from_pretrained(
        run.model_name, padding_side="left", revision=args.model_init_kwargs.get("revision"),
        trust_remote_code=args.model_init_kwargs["trust_remote_code"],
    )
    module = dataset_module(run.dataset)
    trainer = SDFTTrainer(
        model=run.model_name, args=args, processing_class=tokenizer, train_dataset=rows,
        score_completions=module.make_reward_function(scorer, tokenizer),
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
