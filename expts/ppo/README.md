# PPO

This trainer adapts `/home/ujan/self-distill/train/ppo/train_ppo.py`: TRL's
`GRPOTrainer` handles generation, vLLM weight synchronization, and the clipped
policy loss. A separate, fully trainable critic supplies token-level GAE and a
clipped value loss. There is no reference or reward model; Reasoning Gym's
task scorer supplies the terminal reward, including partial credit. DeepScaleR
and DeepMath use a binary Math-Verify reward through the same trainer; select
`--dataset deepscaler` or `--dataset deepmath`.

`config.py` owns PPO's defaults and CLI resolution; `train.py` owns its setup,
resume checks, logging, and saving. These are independent of `expts/grpo/`.
Data preparation, prompts, and scoring remain shared in `expts/data/`.
PPO still inherits generation and policy-loss machinery from TRL's `GRPOTrainer`.

From the repository root, select one available training GPU:

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
    --model_name Qwen/Qwen3-1.7B --task leg_counting \
    --task_config '{"min_animals":5,"max_animals":5}' \
    --run_name ppo-qwen17-leg5 --max_steps 100
```

The defaults are eight completions per prompt, a microbatch of one and 16
accumulation steps, an 8192-token completion budget, DAPO token normalization,
actor learning rate `1e-6`, critic learning rate `1e-5`, and a constant schedule.
`gamma=lam=1`, so each token's unwhitened advantage is terminal reward minus
the critic's prediction before that token. `vf_coef=0.1`, value clipping is
`0.2`, and advantages are whitened over valid tokens in the generation batch.

Training and evaluation use the same dataset-specific prompts and extraction
as GRPO: a completed `</think>` followed by an `<answer>...</answer>` block for
Reasoning Gym, or a `\boxed{...}` answer for either math dataset. Missing
answers score zero. By default, truncated completions are excluded from both
actor and critic losses. With `--mask_truncated_completions false`, the token
budget is treated as the episode end and the normal task score is used;
there is no extra truncation penalty or continuation-value bootstrap.

- `--num_generations 1` enables one rollout per prompt; the current GRPOConfig
  compatibility workaround requires an even generation batch size.
- `--num_iterations 2` makes two optimization passes over each rollout batch.
- `--critic_warmup_steps N` trains only the critic for the first N optimizer
  steps, included in `max_steps`.
- MiniCPM5-2B-SFT uses `--temperature 1.0 --top_k 0`; choose these explicitly
  for training and evaluation.
- The critic adds model, gradient, optimizer, and activation memory. Reduce
  microbatch size or use `--optim adafactor` and/or `--vllm_mode server` with a
  separately launched server when needed. Only one training process is
  supported; DDP, DeepSpeed, FSDP, and fp16 scaling are not supported.

Match sampling, actor optimizer/rate, task difficulty, rollout count, and token
budget explicitly when comparing PPO with GRPO. PPO's one-rollout option also
changes prompt diversity; it is a separate experimental choice.

Runs record resolved task/training settings and package versions in
`run_config.json`. Default run names start with `ppo-`. Each training checkpoint
includes `value_model.pt` and the shared optimizer's actor/critic state. Resume
with `--resume_from_checkpoint PATH` and the original experiment settings;
extending `max_steps` requires a constant schedule. Buffered rollouts are
regenerated on resume, so a restart during rollout reuse is not bitwise identical
to uninterrupted training.

`final/` contains only the policy and tokenizer. Evaluate it normally:

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
    --model_name /path/to/run/final --task leg_counting \
    --task_config '{"min_animals":5,"max_animals":5}'
```

The implementation targets TRL 1.16.0 and Transformers 5.17.0. The value loss
mirrors that TRL version's DAPO/BNPO accumulation scaling; recheck the trainer
hooks and normalization when upgrading. Run the CPU regression checks with:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 uv run python -m unittest discover -s tests -v
```
