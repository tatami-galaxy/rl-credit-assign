"""TRL SDFT with task metrics and checkpointing for an optional EMA teacher."""

from pathlib import Path

import torch
from trl.experimental.sdft import SDFTTrainer as TRLSDFTTrainer


TEACHER_MODEL_FILE = "teacher_model.pt"


class SDFTTrainer(TRLSDFTTrainer):
    def __init__(self, *args, score_completions, **kwargs):
        self.score_completions = score_completions
        super().__init__(*args, **kwargs)

    def sample_rollouts(self, inputs):
        batch = super().sample_rollouts(inputs)
        mode = "train" if self.model.training else "eval"

        def log_metric(name, value):
            self._metrics[mode][name].append(value)

        # Scores are diagnostics only. The upstream distillation loss is unchanged.
        scores = self.score_completions(
            completion_ids=self._get_completion_ids_list(batch),
            entry=[row["entry"] for row in inputs],
            example_id=[row["example_id"] for row in inputs], log_metric=log_metric,
        )
        log_metric("mean_reward", sum(scores) / len(scores))
        return batch

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save and self.args.teacher_model_kind == "ema":
            directory = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
            torch.save(self.teacher_model.state_dict(), directory / TEACHER_MODEL_FILE)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / TEACHER_MODEL_FILE
        if self.args.teacher_model_kind == "ema" and not path.is_file():
            raise FileNotFoundError(f"Cannot resume SDFT without EMA teacher weights: {path}")
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.args.teacher_model_kind == "ema":
            self.teacher_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        # A restarted run regenerates any buffered rollouts.
        self._step = 0
        self._buffered_inputs = None
        self._last_loaded_step = -1 if self.use_vllm else 0
