r"""Probe whether a base model emits thinking tags with our existing prompts.

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=6 uv run python -m expts.eval.think_tags

The default is 12 rollouts: three seeded leg-counting questions, both existing
system prompts (Reasoning Gym and math), and two completions per prompt. The
math variant uses the same leg-counting questions, not a math dataset holdout.
No thinking tags are added to either prompt. Sampling matches training defaults.
Full prompts, responses, scores, and settings are saved under results/think_tags/.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi

from ..data import math_data, reasoning_gym


def summarize(rows):
    return {
        "num_rollouts": len(rows),
        **{name: sum(row[name] for row in rows) for name in (
            "has_think_open", "has_think_close", "has_think_pair", "truncated",
        )},
        "num_extracted_answers": sum(bool(row["model_answer"]) for row in rows),
        "num_full_credit": sum(row["score"] == 1.0 for row in rows),
        "mean_response_tokens": sum(row["response_length"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--dataset_seed", type=int, default=1_000_000)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    if min(args.num_examples, args.num_generations, args.max_new_tokens) <= 0:
        parser.error("Example count, rollout count, and token budget must be positive.")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Require temperature > 0, 0 < top_p <= 1, and top_k >= 0.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1].")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output_dir = Path(args.output_dir or f"results/think_tags/{Path(args.model_name).name}-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(output_dir)
    entries, scorer, data = reasoning_gym.load_eval_data(
        reasoning_gym.DataConfig(dataset_seed=args.dataset_seed), args.num_examples,
    )
    revision = None if Path(args.model_name).is_dir() else HfApi().model_info(args.model_name).sha
    config = {
        **vars(args), "model_revision": revision, "data": data,
        "prompt_styles": ["reasoning_gym", "math"], "enable_thinking": True,
        "min_p": 0.0, "dtype": "bfloat16", "enforce_eager": True,
        "max_num_seqs": 16, "tensor_parallel_size": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {name: version(name) for name in (
            "torch", "transformers", "vllm", "reasoning-gym", "math-verify",
            "latex2sympy2_extended", "sympy",
        )},
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saving rollouts to {output_dir}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_name, revision=revision, tokenizer_revision=revision,
        dtype="bfloat16", seed=args.generation_seed, tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=16,
        enforce_eager=True, generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    prompts = []
    for style, module in (("reasoning_gym", reasoning_gym), ("math", math_data)):
        for entry in entries:
            messages = module.make_prompt(entry["question"])
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            prompts.append({
                "prompt_id": len(prompts), "prompt_style": style, "entry": entry,
                "messages": messages, "prompt_token_ids": token_ids,
                "rendered_prompt": tokenizer.decode(token_ids, skip_special_tokens=False),
            })
    with (output_dir / "prompts.jsonl").open("w") as file:
        for prompt in prompts:
            file.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    outputs = llm.generate(
        [{"prompt_token_ids": prompt["prompt_token_ids"]} for prompt in prompts],
        [SamplingParams(
            n=args.num_generations, max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            min_p=0.0, seed=args.generation_seed + prompt["prompt_id"],
            skip_special_tokens=False,
        ) for prompt in prompts],
    )
    results = []
    with (output_dir / "rollouts.jsonl").open("w") as json_file, (output_dir / "rollouts.txt").open("w") as text_file:
        for prompt, output in zip(prompts, outputs, strict=True):
            module = reasoning_gym if prompt["prompt_style"] == "reasoning_gym" else math_data
            for completion in output.outputs:
                response = completion.text
                row = {
                    "prompt_id": prompt["prompt_id"], "prompt_style": prompt["prompt_style"],
                    "sample_id": completion.index, "entry": prompt["entry"], "response": response,
                    "has_think_open": "<think>" in response,
                    "has_think_close": "</think>" in response,
                    "has_think_pair": bool(re.search(r"<think>.*?</think>", response, re.DOTALL)),
                    "response_length": len(completion.token_ids),
                    "truncated": completion.finish_reason == "length",
                    "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                    **module.score_response(scorer, response, prompt["entry"]),
                }
                results.append(row)
                json_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                label = f"{row['prompt_style']} question={row['entry']['example_id']} sample={row['sample_id']}"
                status = (f"open={row['has_think_open']} close={row['has_think_close']} "
                          f"tokens={row['response_length']} finish={row['finish_reason']} score={row['score']}")
                text_file.write(f"=== {label} ===\n{status}\n\nPROMPT\n{prompt['rendered_prompt']}\n\nRESPONSE\n{response}\n\n")
                print(f"{label}: {status}", flush=True)

    summary = {
        **summarize(results),
        "by_prompt_style": {style: summarize([row for row in results if row["prompt_style"] == style])
                            for style in ("reasoning_gym", "math")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Full responses: {output_dir / 'rollouts.txt'}")


if __name__ == "__main__":
    main()
