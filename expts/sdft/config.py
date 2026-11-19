"""SDFT experiment defaults, with the repository's thinking-mode prompt contract."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from transformers.trainer_utils import SchedulerType
from transformers.training_args import OptimizerNames
from trl.experimental.sdft import SDFTConfig

from ..config import parse_options


@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen3-4B"
    dataset: str = "reasoning_gym"
    num_train_examples: int | None = None  # RG only; defaults to 10,000 distinct stream indices.
    wandb_project: str | None = None
    wandb_entity: str = "ujan-rl"


@dataclass
class TrainingConfig(SDFTConfig):
    """Explicit experiment defaults; all other options inherit TRL's defaults."""

    _VALID_DICT_FIELDS = SDFTConfig._VALID_DICT_FIELDS + ["chat_template_kwargs", "generation_kwargs"]

    # Training schedule and memory.
    max_steps: int = 1000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    optim: OptimizerNames | str = "adamw_bnb_8bit"
    learning_rate: float = 1e-5
    lr_scheduler_type: SchedulerType | str = "linear"
    bf16: bool | None = None  # TRL enables bf16 unless fp16 is set.
    gradient_checkpointing: bool = True
    disable_dropout: bool = True

    # Teacher and distillation loss.
    teacher_model_kind: str = "base"
    teacher_update_rate: float = 0.05
    teacher_sync_steps: int = 1
    generate_from_teacher: bool = False
    teacher_prompt_template: str = "{prompt}\n\n{privileged_context}"
    distillation_mode: Literal["sampled_token", "full_logits", "topk_logits"] = "topk_logits"
    distillation_alpha: float = 0.5
    distillation_topk: int | None = 100
    distillation_is_clip: float | None = 2.0
    distillation_add_tail: bool = False
    num_loss_tokens_to_skip: int = 0

    # Rollouts and sampling.
    num_generations: int = 8
    num_generations_eval: int | None = None
    num_iterations: int = 1
    generation_batch_size: int | None = None
    steps_per_generation: int | None = None
    max_prompt_length: int | None = 512
    max_completion_length: int | None = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float | None = None
    repetition_penalty: float = 1.0
    generation_kwargs: dict | str | None = None

    # Generation backend.
    use_vllm: bool = True
    use_teacher_server: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_length: int | None = None
    vllm_enable_sleep_mode: bool = False

    # Data, logging, and checkpoints.
    shuffle_dataset: bool = True
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    logging_steps: float = 10
    report_to: None | str | list[str] = "wandb"
    save_steps: float = 500
    save_total_limit: int | None = None
    model_init_kwargs: dict | str | None = None

    # Repository prompt contract: explicitly enable thinking.
    chat_template_kwargs: dict | str | None = field(default_factory=lambda: {"enable_thinking": True})

    def __post_init__(self):
        if self.num_generations < 1 or self.num_iterations < 1:
            raise ValueError("num_generations and num_iterations must be positive.")
        if self.teacher_model_kind not in {"base", "live", "ema"}:
            raise ValueError("teacher_model_kind must be base, live, or ema.")
        if not 0 <= self.distillation_alpha <= 1:
            raise ValueError("distillation_alpha must be in [0, 1].")
        if not 0 < self.teacher_update_rate <= 1 or self.teacher_sync_steps < 1:
            raise ValueError("teacher_update_rate must be in (0, 1] and teacher_sync_steps must be positive.")
        if self.deepspeed or self.fsdp:
            raise ValueError("SDFT currently supports one training process without DeepSpeed/FSDP.")
        super().__post_init__()
        if self.world_size != 1 or self.n_gpu > 1:
            raise ValueError("SDFT requires one training process and one visible training GPU.")


def parse_config(argv=None):
    run, data, args, parser = parse_options(RunConfig, TrainingConfig, argv)
    if run.dataset == "reasoning_gym":
        if run.num_train_examples is None:
            run.num_train_examples = 10_000
        if run.num_train_examples <= 0:
            parser.error("--num_train_examples must be positive.")
    elif run.num_train_examples is not None:
        parser.error("--num_train_examples is only for reasoning_gym; math uses the saved training split.")
    if run.wandb_project is None:
        run.wandb_project = run.dataset.replace("_", "-")
    label = data.task if run.dataset == "reasoning_gym" else run.dataset
    if args.run_name is None:
        args.run_name = f"sdft-{Path(run.model_name).name}-{label}-seed{args.seed}"
    if args.output_dir == "trainer_output":
        args.output_dir = f"/data/ujan/rl_credit/{run.dataset}/{args.run_name}"
    args.chat_template_kwargs = {"enable_thinking": True, **(args.chat_template_kwargs or {})}
    if args.chat_template_kwargs["enable_thinking"] is not True:
        parser.error("Only thinking mode is supported; enable_thinking must be true.")
    if args.remove_unused_columns:
        parser.error("Keep --remove_unused_columns false so task metrics retain the reference entries.")
    args.model_init_kwargs = {
        "dtype": "bfloat16" if args.bf16 else "float32", **(args.model_init_kwargs or {})
    }
    # TRL inserts this into model_init_kwargs during construction. Resolve it
    # before writing/comparing metadata so a fresh process can resume the run.
    args.model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
    return run, data, args
