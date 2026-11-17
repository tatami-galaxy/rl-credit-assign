"""Student prompts and answer-only teacher context on the existing training data."""

import os
from random import Random

from torch.utils.data import Dataset

from ..config import dataset_module


def privileged_context(entry):
    answer = entry["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("SDFT requires a nonempty reference answer.")
    return f"Reference final answer: {answer}"


class IndexedReasoningGym(Dataset):
    """A finite prefix of the RG stream, generated lazily in any index order.

    Python entries preserve heterogeneous task metadata. TRL's SDFT trainer
    requires indexed data and does not accept the existing HF IterableDataset.
    """

    def __init__(self, config, seed, size):
        from ..data import reasoning_gym as rg

        if size <= 0:
            raise ValueError("Training size must be positive.")
        self.config = config
        self.seed = config.dataset_seed if config.dataset_seed is not None else seed
        self.size = size
        self.scorer = rg.create_task_dataset(config.task, config.task_config, self.seed)
        self._chunk_start = 0
        self._dataset = self.scorer

    def __len__(self):
        return self.size

    def __getitem__(self, example_id):
        from ..data import reasoning_gym as rg

        if not 0 <= example_id < self.size:
            raise IndexError(example_id)
        task = self.config.task
        index = example_id % rg.MIXTURE_CHUNK_SIZE if task == "all" else 0
        chunk_start = example_id - index
        if chunk_start != self._chunk_start:
            self._dataset = rg.create_task_dataset(task, self.config.task_config, self.seed + chunk_start)
            self._chunk_start = chunk_start
        try:
            entry = self._dataset[index]
        except Exception as error:
            name = task
            if task == "all":
                name = Random(self._dataset.seed + index).choices(
                    self._dataset.dataset_names, weights=self._dataset.weights, k=1,
                )[0]
            raise RuntimeError(f"Failed to generate task {name!r} at example {example_id}; no example was skipped.") from error
        return {
            "prompt": rg.make_prompt(entry["question"]), "entry": entry,
            "example_id": example_id, "privileged_context": privileged_context(entry),
        }


def load_training_data(run, config, seed):
    module = dataset_module(run.dataset)
    if run.dataset == "reasoning_gym":
        rows = IndexedReasoningGym(config, seed, run.num_train_examples)
        scorer = rows.scorer
        metadata = {
            **module.data_metadata(config, scorer),
            "training_sampling": "indexed_stream_prefix_v1", "num_train_examples": len(rows),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
        }
    else:
        rows, scorer, metadata = module.load_training_data(config, seed)
        rows = rows.map(lambda row: {"privileged_context": privileged_context(row["entry"])})
    return rows, scorer, {**metadata, "privileged_context": "reference_final_answer_v1"}
