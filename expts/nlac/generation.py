"""Shared vLLM sampling for the three offline generation commands."""

from importlib.metadata import version
from pathlib import Path

from .data import read_json


def add_arguments(parser, *, max_new_tokens, temperature, top_k):
    parser.add_argument("--model_name")
    parser.add_argument("--model_revision")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=top_k)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)


def validate(args, parser):
    if min(args.batch_size, args.max_new_tokens, args.max_model_len) <= 0:
        parser.error("Batch size and token limits must be positive.")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Require temperature >= 0, 0 < top_p <= 1, and top_k >= 0.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1].")


def resolve_model(args, default_name=None, default_revision=None):
    from huggingface_hub import HfApi

    args.model_name = args.model_name or default_name
    if not args.model_name:
        raise ValueError("Specify --model_name.")
    previous = Path(args.output_dir) / "run_config.json"
    saved = read_json(previous) if previous.exists() else {}
    revision = args.model_revision
    if revision is None and saved.get("model_name") == args.model_name:
        revision = saved["model_revision"]
    if revision is None and args.model_name == default_name:
        revision = default_revision
    if not Path(args.model_name).is_dir():
        revision = HfApi().model_info(args.model_name, revision=revision).sha
    args.model_revision = revision


def versions():
    return {name: version(name) for name in ("torch", "transformers", "vllm", "datasets")}


def prompt_ids(tokenizer, messages, thinking):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking,
    )


def load_llm(args):
    from vllm import LLM

    return LLM(
        model=args.model_name, revision=args.model_revision, tokenizer_revision=args.model_revision,
        dtype="bfloat16", seed=args.seed, generation_config="vllm", enforce_eager=True,
        max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def generate(llm, messages, seeds, args, *, thinking):
    from vllm import SamplingParams

    ids = [prompt_ids(llm.get_tokenizer(), prompt, thinking) for prompt in messages]
    if any(len(prompt) + args.max_new_tokens > args.max_model_len for prompt in ids):
        raise ValueError("Prompt plus generation budget exceeds max_model_len; no context was truncated.")
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in ids],
        [SamplingParams(
            n=1, max_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, min_p=0.0, seed=seed, skip_special_tokens=False,
        ) for seed in seeds],
    )
    rows = []
    for output, seed in zip(outputs, seeds, strict=True):
        completion = output.outputs[0]
        rows.append({
            "response": completion.text, "response_token_ids": list(completion.token_ids),
            "response_length": len(completion.token_ids), "generation_seed": seed,
            "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
            "truncated": completion.finish_reason == "length",
        })
    return rows
