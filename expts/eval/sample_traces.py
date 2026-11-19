r"""Save short DeepMath traces for inspecting natural paragraph boundaries.

    CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 uv run python -m expts.eval.sample_traces \
        --model_name Qwen/Qwen3-4B

Defaults: one response for each of 10 fixed holdout problems, at most 2048 new
tokens. MiniCPM5-2B-SFT uses --temperature 1.0 --top_k 0. The existing math prompt
is used without additional formatting instructions. Original whitespace is
preserved in traces.txt and traces.jsonl; run_config.json records the settings.
"""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--split_manifest")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    if min(args.num_examples, args.max_new_tokens) <= 0:
        parser.error("num_examples and max_new_tokens must be positive.")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Require temperature > 0, 0 < top_p <= 1, and top_k >= 0.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1].")

    from huggingface_hub import HfApi

    from ..data import deepmath

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output_dir = Path(args.output_dir or f"results/traces/{Path(args.model_name).name}-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(output_dir)
    data_config = deepmath.DataConfig(split_manifest=args.split_manifest)
    entries, _, metadata = deepmath.load_eval_data(data_config, args.num_examples)
    copyfile(data_config.split_manifest, output_dir / "split_manifest.json")
    revision = None if Path(args.model_name).is_dir() else HfApi().model_info(args.model_name).sha
    config = {
        **vars(args), "model_revision": revision, "data": metadata,
        "enable_thinking": True, "min_p": 0.0, "dtype": "bfloat16",
        "enforce_eager": True, "max_num_seqs": 10,
        "versions": {name: version(name) for name in ("torch", "transformers", "vllm", "datasets")},
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_name, revision=revision, tokenizer_revision=revision,
        dtype="bfloat16", seed=args.generation_seed, generation_config="vllm",
        gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=10,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    messages = [deepmath.make_prompt(entry["question"]) for entry in entries]
    prompt_ids = [tokenizer.apply_chat_template(
        prompt, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True,
    ) for prompt in messages]
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids],
        [SamplingParams(
            n=1, max_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, min_p=0.0,
            seed=args.generation_seed + entry["example_id"], skip_special_tokens=False,
        ) for entry in entries],
    )
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as json_file, \
            (output_dir / "traces.txt").open("w", encoding="utf-8") as text_file:
        for entry, prompt, ids, output in zip(entries, messages, prompt_ids, outputs, strict=True):
            completion = output.outputs[0]
            row = {
                **entry, "prompt": prompt, "prompt_token_ids": ids,
                "generation_seed": args.generation_seed + entry["example_id"],
                "response": completion.text, "response_token_ids": list(completion.token_ids),
                "response_length": len(completion.token_ids),
                "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                "truncated": completion.finish_reason == "length",
            }
            json_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            label = (f"example={entry['example_id']} tokens={row['response_length']} "
                     f"finish={row['finish_reason']}")
            text_file.write(
                f"=== {label} ===\n\nPROMPT\n{tokenizer.decode(ids, skip_special_tokens=False)}"
                f"\n\nRESPONSE\n{completion.text}\n\n"
            )
            print(label)
    print(f"Saved {len(entries)} traces to {output_dir / 'traces.txt'}")


if __name__ == "__main__":
    main()
