"""Uniform task sampling, chunk replay, grouped rollouts, and reward dispatch."""

import copy
import json
import tempfile
import types
import unittest
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from unittest.mock import patch

from datasets import IterableDataset
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import DATASETS
from trl.trainer.utils import repeat_iterable_dataset

from expts.data.reasoning_gym import (
    create_task_dataset, evaluation_entries, generate_examples, make_reward_function,
)
from expts.grpo.train import validate_resume


@dataclass
class TinyConfig:
    seed: int = 0
    size: int = 500


class TaskA(ProceduralDataset):
    name = "a"
    credit = 1.0

    def __init__(self, config):
        super().__init__(config, config.seed, config.size)

    def __getitem__(self, index):
        return {
            "question": f"{self.name}:{self.seed + index}", "answer": self.name,
            "metadata": {"source_dataset": self.name, self.name: [index]},
        }

    def score_answer(self, answer, entry):
        return self.credit if answer == self.name else 0.0


class TaskB(TaskA):
    name = "b"
    credit = 0.5


class TaskMixtureTest(unittest.TestCase):
    def setUp(self):
        registry = {"a": (TaskA, TinyConfig), "b": (TaskB, TinyConfig), "composite": DATASETS["composite"]}
        patcher = patch.dict(DATASETS, registry, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_uniform_with_replacement(self):
        dataset = create_task_dataset("all", {}, 42, size=10000)
        names = [dataset[i]["metadata"]["source_dataset"] for i in range(10000)]
        counts = Counter(names)
        self.assertEqual(dataset.dataset_names, ["a", "b"])
        self.assertEqual(dataset.weights, [1., 1.])
        self.assertLess(abs(counts["a"] - 5000), 200)
        self.assertTrue(any(a == b for a, b in zip(names, names[1:])))

    def test_chunk_boundaries_match_one_contiguous_seed_range(self):
        with patch("expts.data.reasoning_gym.MIXTURE_CHUNK_SIZE", 4):
            examples = list(islice(generate_examples("all", {}, 42), 12))
            replay = list(islice(generate_examples("all", {}, 42), 12))
        continuous = create_task_dataset("all", {}, 42, size=12)
        self.assertEqual(examples, replay)
        # source_index is local to a chunk; question content uses the same seed+index.
        self.assertEqual([e["entry"]["question"] for e in examples], [continuous[i]["question"] for i in range(12)])
        self.assertEqual([e["example_id"] for e in examples], list(range(12)))
        self.assertEqual(len({e["entry"]["question"] for e in examples}), 12)

    def test_trl_repeats_whole_questions_and_preserves_heterogeneous_metadata(self):
        stream = IterableDataset.from_generator(generate_examples, gen_kwargs={"task": "all", "task_config": {}, "seed": 42})
        repeated = repeat_iterable_dataset(stream, mini_repeat_count=2, batch_size=2, repeat_count=2)
        rows = list(islice(repeated, 8))
        self.assertEqual([r["example_id"] for r in rows], [0, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[2], rows[3])
        self.assertNotEqual(rows[0]["entry"]["metadata"].keys(), rows[2]["entry"]["metadata"].keys())

    def test_reward_dispatch_and_counts_ignore_repeated_completions(self):
        dataset = create_task_dataset("all", {}, 42)
        a, b = dataset.datasets["a"][0], dataset.datasets["b"][0]
        tokenizer = types.SimpleNamespace(decode=lambda text, **kwargs: text)
        reward = make_reward_function(dataset, tokenizer)
        metrics, columns = {}, {}
        scores = reward(
            ["</think><answer>a</answer>", "unfinished", "</think><answer>b</answer>"],
            [a, a, b], example_id=[0, 0, 1], log_metric=metrics.__setitem__, log_extra=columns.__setitem__,
        )
        self.assertEqual(scores, [1., 0., 0.5])
        self.assertEqual(metrics["tasks/a/questions_per_generation"], 1)
        self.assertEqual(metrics["tasks/b/questions_per_generation"], 1)
        self.assertEqual(metrics["tasks/a/mean_reward"], 0.5)
        self.assertEqual(columns["task"], ["a", "a", "b"])
        reward(["unfinished"], [a], example_id=[2], log_metric=metrics.__setitem__)
        self.assertEqual(metrics["tasks/b/questions_per_generation"], 0)

    def test_balanced_evaluation_and_macro_average(self):
        from expts.eval.eval import summarize
        from expts.data.reasoning_gym import DataConfig, load_eval_data

        dataset = create_task_dataset("all", {}, 42, size=3)
        entries = evaluation_entries(dataset)
        self.assertEqual(Counter(e["metadata"]["source_dataset"] for e in entries), {"a": 3, "b": 3})
        prepared, _, metadata = load_eval_data(DataConfig(task="all", dataset_seed=42), 3)
        self.assertEqual([{k: v for k, v in e.items() if k != "example_id"} for e in prepared], entries)
        self.assertEqual(metadata["evaluation_sampling"], "balanced_per_task")
        results = [
            {"metadata": {"source_dataset": task}, "score": score, "model_answer": "x",
             "response_length": 2, "truncated": False, "thinking_complete": True}
            for task, score in [("a", 1.), ("a", 1.), ("b", 0.)]
        ]
        summary = summarize(results, by_task=True)
        self.assertEqual(summary["full_credit_rate"], 2 / 3)
        self.assertEqual(summary["macro_full_credit_rate"], 0.5)
        self.assertEqual(summary["macro_mean_reward"], 0.5)

    def test_resume_rejects_changed_mixture(self):
        dataset = create_task_dataset("all", {}, 42)
        from expts.data.reasoning_gym import DataConfig, data_metadata

        config = {"dataset": "reasoning_gym", "model_name": "test", "method": "grpo",
                  "data": data_metadata(DataConfig(task="all"), dataset),
                  "training": {"max_steps": 1, "lr_scheduler_type": "constant"},
                  "versions": {"reasoning-gym": "test"}}
        self.assertEqual(config["data"]["task_config"]["datasets"][0]["config"]["seed"], 43)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "run_config.json").write_text(json.dumps(config))
            checkpoint = Path(root) / "checkpoint-1"
            validate_resume(checkpoint, config)
            for key, value in (("task", "a"), ("task_config", {}), ("task_sampling", {})):
                changed = copy.deepcopy(config)
                changed["data"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_resume(checkpoint, changed)
            # Metadata written before the reorganization is still understood.
            legacy = {key: value for key, value in config.items() if key not in ("data", "dataset", "method")}
            legacy.update({key: config["data"][key] for key in ("task", "task_config", "task_sampling")})
            (Path(root) / "run_config.json").write_text(json.dumps(legacy))
            validate_resume(checkpoint, config)
            changed = copy.deepcopy(config)
            changed["versions"]["reasoning-gym"] = "other"
            with self.assertRaisesRegex(ValueError, "version"):
                validate_resume(checkpoint, changed)

    def test_failure_is_named_and_not_skipped(self):
        with patch.object(TaskA, "__getitem__", side_effect=ValueError("broken")):
            with self.assertRaisesRegex(RuntimeError, "task 'b'.*example 0"):
                next(generate_examples("all", {}, 42))

    def test_all_uses_defaults(self):
        with self.assertRaisesRegex(ValueError, "defaults"):
            create_task_dataset("all", {"max_animals": 5}, 42)


if __name__ == "__main__":
    unittest.main()
