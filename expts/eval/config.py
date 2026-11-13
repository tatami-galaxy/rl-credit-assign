"""Generation options shared by in-distribution dataset evaluation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import parse_options, run_metadata


@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen3-4B"
    dataset: str = "reasoning_gym"
    wandb_project: str | None = None
    wandb_entity: str = "ujan-rl"


@dataclass
class EvalConfig:
    num_examples: int = 500  # Per task for RG task=all; total for finite math datasets.
    generation_seed: int = 42
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int | None = None
    max_num_seqs: int = 64
    enforce_eager: bool = False
    output_dir: str | None = None


def parse_eval_config(argv=None):
    run, data, args, parser = parse_options(RunConfig, EvalConfig, argv)
    if min(args.num_examples, args.max_new_tokens, args.tensor_parallel_size, args.max_num_seqs) <= 0:
        parser.error("num_examples, max_new_tokens, tensor_parallel_size, and max_num_seqs must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1].")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Sampling requires temperature > 0, 0 < top_p <= 1, and top_k >= 0.")
    if run.dataset in {"deepscaler", "deepmath"} and data.split_manifest is None:
        previous = run_metadata(run.model_name)
        if previous and previous.get("dataset") == run.dataset:
            manifest = Path(run.model_name).parent / "split_manifest.json"
            if not manifest.is_file():
                parser.error(f"Training split manifest is missing: {manifest}")
            data.split_manifest = str(manifest)
    label = data.task if run.dataset == "reasoning_gym" else run.dataset
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    args.output_dir = str(Path(args.output_dir or (
        f"results/{run.dataset}/eval/{Path(run.model_name).name}-{label}-{timestamp}"
    )))
    return run, data, args
