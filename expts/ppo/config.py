"""Reference-free PPO configuration and training defaults."""

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
class PPOConfig(GRPOConfig):
    max_steps: int = 1000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_generations: int = 8
    max_completion_length: int = 8192
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    learning_rate: float = 1e-6
    lr_scheduler_type: SchedulerType | str = "constant"
    bf16: bool = True
    gradient_checkpointing: bool = True
    loss_type: str = "dapo"
    beta: float = 0.0
    scale_rewards: str = "none"
    mask_truncated_completions: bool = True
    shuffle_dataset: bool | None = None
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    logging_steps: float = 1
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

    gamma: float = 1.0
    lam: float = 1.0
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    critic_learning_rate: float = 1e-5
    critic_max_grad_norm: float = 10.0
    critic_warmup_steps: int = 0
    whiten_advantages: bool = True

    def __post_init__(self):
        self.beta = 0.0
        self.scale_rewards = "none"
        if self.num_generations < 1:
            raise ValueError("PPO requires num_generations >= 1.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("gamma and lam must be in [0, 1].")
        if min(self.vf_coef, self.cliprange_value, self.critic_max_grad_norm, self.critic_warmup_steps) < 0:
            raise ValueError("Value loss, clipping, and critic warmup settings must be nonnegative.")
        if self.critic_learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("critic_learning_rate and max_grad_norm must be positive.")
        if self.loss_type not in {"dapo", "bnpo"}:
            raise ValueError("PPO supports loss_type=dapo or bnpo.")
        if self.deepspeed or self.fsdp:
            raise ValueError("PPO currently supports one training process without DeepSpeed/FSDP.")
        if self.fp16:
            raise ValueError("Use bf16 or float32; the separate critic does not support fp16 scaling.")
        if self.use_vllm and self.vllm_mode == "server" and any(
            name in str(self.optim).lower() for name in ("8bit", "bnb", "paged")
        ):
            raise ValueError("The source PPO experiments found server-mode corruption with bitsandbytes; use adafactor or adamw_torch.")

        # GRPO rejects G=1 even though PPO does not need a within-prompt baseline.
        # Keep its other validation intact. This workaround requires an even rollout batch.
        requested = self.num_generations
        if requested == 1:
            batch_size = self.generation_batch_size or (
                self.per_device_train_batch_size * (self.steps_per_generation or self.gradient_accumulation_steps)
            )
            if batch_size % 2:
                raise ValueError("num_generations=1 currently requires an even generation batch size.")
        self.num_generations = max(2, requested)
        try:
            super().__post_init__()
        finally:
            self.num_generations = requested
        if self.world_size != 1 or self.n_gpu > 1:
            raise ValueError("PPO requires one training process and one visible training GPU.")


def parse_config(argv=None):
    run, data, args, parser = parse_options(RunConfig, PPOConfig, argv)
    if run.wandb_project is None:
        run.wandb_project = run.dataset.replace("_", "-")
    if args.max_steps <= 0:
        parser.error("Training requires --max_steps > 0.")
    model_label = Path(run.model_name).name
    label = data.task if run.dataset == "reasoning_gym" else run.dataset
    if args.run_name is None:
        args.run_name = f"ppo-{model_label}-{label}-seed{args.seed}"
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
