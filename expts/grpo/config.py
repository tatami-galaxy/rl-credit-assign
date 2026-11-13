"""Dataset-independent GRPO training defaults."""

from dataclasses import dataclass, field
from pathlib import Path

from transformers.trainer_utils import SchedulerType
from trl import GRPOConfig

from ..config import parse_options


@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen3-4B"
    dataset: str = "reasoning_gym"
    wandb_project: str | None = None
    wandb_entity: str = "ujan-rl"


@dataclass
class TrainingConfig(GRPOConfig):
    """Our training defaults; all other options inherit TRL's defaults."""

    max_steps: int = 1000
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_generations: int = 8
    max_completion_length: int = 8192
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    learning_rate: float = 1e-5
    lr_scheduler_type: SchedulerType | str = "linear"
    bf16: bool = True
    gradient_checkpointing: bool = True
    loss_type: str = "dapo"
    beta: float = 0.0
    scale_rewards: str = "group"
    mask_truncated_completions: bool = True
    shuffle_dataset: bool | None = None
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    logging_steps: float = 10
    log_completions: bool = False
    report_to: None | str | list[str] = "wandb"
    save_steps: float = 100
    save_total_limit: int | None = 2
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    # Explicitly enable thinking even for templates that default to False.
    chat_template_kwargs: dict | str | None = field(default_factory=lambda: {"enable_thinking": True})
    generation_kwargs: dict | str | None = field(default_factory=lambda: {"min_p": 0.0})


def parse_config(argv=None):
    run, data, args, parser = parse_options(RunConfig, TrainingConfig, argv)
    if run.wandb_project is None:
        run.wandb_project = run.dataset.replace("_", "-")
    if args.max_steps <= 0:
        parser.error("Training requires --max_steps > 0.")
    model_label = Path(run.model_name).name
    label = data.task if run.dataset == "reasoning_gym" else run.dataset
    if args.run_name is None:
        args.run_name = f"grpo-{model_label}-{label}-seed{args.seed}"
    if args.output_dir == "trainer_output":
        args.output_dir = f"/data/ujan/rl_credit/{run.dataset}/{args.run_name}"
    # A finite math dataset should reshuffle every epoch; the RG stream orders itself.
    if args.shuffle_dataset is None:
        args.shuffle_dataset = run.dataset in {"deepscaler", "deepmath"}
    for name in ("chat_template_kwargs", "generation_kwargs"):
        setattr(args, name, {**parser.get_default(name), **(getattr(args, name) or {})})
    if args.chat_template_kwargs["enable_thinking"] is not True:
        parser.error("Only thinking mode is supported; enable_thinking must be true.")
    args.model_init_kwargs = {
        "dtype": "bfloat16" if args.bf16 else "float32", **(args.model_init_kwargs or {})
    }
    return run, data, args
