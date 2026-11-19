# SDFT

Full-parameter answer-conditioned self-distillation using
[`trl.experimental.sdft.SDFTTrainer`](https://huggingface.co/docs/trl/en/sdft_trainer).
The student receives the same prompts as PPO and GRPO. The teacher additionally
receives `Reference final answer: <answer>` in the user message, using TRL's
default `{prompt}\n\n{privileged_context}` template. Both models score the same
student-generated completion. Only the student receives gradients.

The main algorithm and training options are explicit in `TrainingConfig` in
`config.py`. The teacher and loss use TRL defaults; the experiment settings
below control training, sampling, and memory. Other options inherit TRL's
defaults. The tested environment uses TRL 1.17.1 / Transformers 5.19.0.

| Option | Default |
| --- | --- |
| Teacher | Frozen initial model (`base`) |
| Distillation | Student top-100 logits, `alpha=0.5`, no tail bucket |
| Generations per question | 8 |
| Generation backend | vLLM colocate (`use_vllm=true`, GPU memory utilization 0.3) |
| Completion / prompt limits | 8,192 / 512 tokens |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20 |
| Batch size / accumulation | 1 / 16 |
| Optimizer | bitsandbytes AdamW (`adamw_bnb_8bit`) |
| Learning rate / schedule | 1e-5 / linear |
| Training duration | 1,000 optimizer steps (`max_steps=1000`, overrides epochs) |
| Reporting | W&B; pass `--report_to none` to disable |

The repository also supplies model/dataset selection, output naming, model
loading dtype, and `enable_thinking=true`.
Reasoning Gym defaults to `leg_counting`; the default model is `Qwen/Qwen3-4B`.
This baseline deliberately uses TRL's defaults rather than the paper's EMA
teacher and full-vocabulary forward-KL settings. All resolved options and
package versions are saved in `run_config.json`.

## Run

Select one available GPU. This implementation supports one training process;
DeepSpeed, FSDP, and multi-GPU training are not supported.

In the current environment, FlashInfer's sampler cannot compile against the
installed CUDA toolkit. Select vLLM's native sampler before launching:

```sh
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

The sampler override keeps vLLM generation and the same top-k/top-p settings;
see [vLLM's environment variables](https://docs.vllm.ai/en/latest/configuration/env_vars/).
For simultaneous independent runs, set a different `MASTER_PORT` for each
process (for example, `29606` and `29607`) to avoid rendezvous port collisions.

```sh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=6 uv run python -m expts.sdft.train \
    --task leg_counting --num_train_examples 10000 \
    --max_steps 100 --run_name sdft-leg-counting --report_to wandb

CUDA_VISIBLE_DEVICES=6 uv run python -m expts.sdft.train \
    --dataset deepmath --max_steps 100 --run_name sdft-deepmath \
    --report_to wandb
```

Use `--dataset deepscaler` for DeepScaleR and `--task all` for the uniform
Reasoning Gym task mixture. Math preparation and holdout manifests are shared
with PPO/GRPO. Only reference answers are used as privileged information; the
math datasets' worked solutions are not loaded.

TRL SDFT does not support iterable datasets. Reasoning Gym therefore uses an
indexed, finite prefix of the existing stream: `num_train_examples` defaults to
10,000 and is recorded in the run config. Entries are generated lazily, with
the same task selection, chunk seeds, and example IDs as the stream. TRL's
default shuffling applies, and subsequent epochs revisit these indices.
`num_train_examples` is only accepted for Reasoning Gym. Keep `PYTHONHASHSEED`
fixed because some upstream task generators iterate sets.

The default `--optim adamw_bnb_8bit` uses bitsandbytes to reduce optimizer-state
memory while retaining full-parameter training and BF16 model weights. Install
`bitsandbytes` in the training environment. Its version is saved in
`run_config.json`. Use `--optim adamw_torch_fused` for standard AdamW, or
`--optim adamw_torch --use_vllm false` for CPU tests. Changing optimizers requires a new run.
See the [bitsandbytes optimizer guide](https://huggingface.co/docs/bitsandbytes/en/optimizers).

Optimizer state is only part of the memory cost: the frozen teacher adds a
model copy, and even top-k distillation constructs full student and teacher
logits. The training microbatch is 1, but TRL generates 16 responses together
by default (batch size times gradient accumulation). `--generation_batch_size
8` reduces this to one question's eight responses per generation call while
keeping 16 accumulated training microbatches per optimizer step.

For long Transformers rollouts, start the process with
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation from
changing allocation sizes. This is especially useful when reserved GPU memory
is much larger than allocated tensor memory. See
[PyTorch's allocator settings](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).

vLLM generation is enabled by default in colocate mode, with
`vllm_gpu_memory_utilization=0.3`. The teacher is still scored locally, and vLLM
adds its own model/cache memory. Use `--use_vllm false` for Transformers generation.

Match token budgets explicitly when comparing algorithms.
Both student and teacher prompts are left-truncated at `max_prompt_length`;
choose a limit large enough for the question plus reference answer. Long
answers can otherwise push the question out of the teacher's context.

## Metrics and evaluation

The existing task scorers log `reward` (`train/reward` in W&B), `full_credit_rate`,
`answer_extraction_rate`, and `thinking_complete_rate` once per generated batch.
Math also reports verifier failures; task mixtures report per-task metrics.
Scores are diagnostics and do not enter the loss. TRL logs distillation loss,
completion lengths, and truncation frequency.

TRL averages distillation loss over tokens within each response, then over
responses. Truncated completions contribute their generated prefixes. These
choices differ from the current PPO/GRPO defaults. Match sampling, question
exposure, and generated-token budgets explicitly for comparisons.

`final/` contains the student and tokenizer. Evaluate with the existing CLI,
which gives the model the ordinary prompt without privileged information:

```sh
CUDA_VISIBLE_DEVICES=7 uv run python -m expts.eval.eval \
    --dataset deepmath --model_name /path/to/sdft-run/final \
    --num_examples 100
```

Pass the same sampling settings to evaluation for every model being compared;
the evaluation defaults differ from TRL SDFT's training defaults.

## Checkpoints and tests

Resume with `--resume_from_checkpoint /path/to/run/checkpoint-N` and the original
experiment settings. The frozen `base` teacher is reloaded from the original
`model_name`; keep those source weights fixed. An optional `--teacher_model_kind
ema` teacher is saved in `teacher_model.pt` and restored in place. A `live`
teacher uses the restored student itself. Buffered rollouts are regenerated on
resume, so restarts during rollout reuse need not be bitwise identical.
Extending `max_steps` requires a constant learning-rate schedule.

The CPU suite exercises prompt isolation, existing split identity, indexed
Reasoning Gym equivalence, diagnostic scoring, and actual tiny-model training
with frozen and EMA teachers, including checkpoint restoration:

```sh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
    uv run python -m unittest discover -s tests -v
```
