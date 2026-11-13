"""Math grading, dataset isolation, fixed holdouts, and checkpoint identity."""

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset
from math_verify.errors import TimeoutException

from expts.grpo.config import parse_config as parse_grpo_config
from expts.data import deepscaler as ds, math_data
from expts.eval.config import parse_eval_config
from expts.ppo.config import parse_config as parse_ppo_config
from expts.ppo.train import validate_resume


CPU_ARGS = ["--use_cpu", "true", "--bf16", "false", "--report_to", "none", "--use_vllm", "false"]


class MathRewardTest(unittest.TestCase):
    def test_equivalence_and_incorrect_answers(self):
        for gold, prediction in [(r"\frac{1}{2}", "0.5"), (r"$\sqrt{4}$", "2"),
                                 ("35", "035"), ("x+1", "1+x"),
                                 (r"\{1,2\}", r"\{2,1\}"), ("2", "x=2")]:
            with self.subTest(gold=gold):
                result = math_data.score_response(None, "</think>\\boxed{" + prediction + "}", {"answer": gold})
                self.assertEqual(result["score"], 1.0)
                self.assertIsNone(result["verifier_error"])
        self.assertEqual(math_data.score_response(None, r"</think>\boxed{3}", {"answer": "2"})["score"], 0.)

    def test_only_last_box_after_thinking_counts(self):
        for text in [r"\boxed{2}", r"\boxed{2}</think>unfinished", r"</think>\boxed{}",
                     r"</think>\boxed{2", r"</think>\boxed{2} then \boxed{3}"]:
            with self.subTest(text=text):
                self.assertEqual(math_data.score_response(None, text, {"answer": "2"})["score"], 0.)
        answer, done = math_data.extract_final_answer(r"</think>\boxed{\frac{1}{2}}")
        self.assertEqual(answer, r"\frac{1}{2}")
        self.assertTrue(done)

    def test_empty_gold_is_never_rewarded_and_timeouts_are_logged(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            math_data.score_response(None, r"</think>\boxed{1}", {"answer": ""})
        with patch.object(math_data, "verify", side_effect=TimeoutException("slow")):
            result = math_data.score_response(None, r"</think>\boxed{1}", {"answer": "1"})
        self.assertEqual(result["score"], 0.)
        self.assertEqual(result["verifier_error"], "timeout")
        with patch.object(math_data, "parse_math", return_value=[]):
            result = math_data.score_response(None, r"</think>\boxed{1}", {"answer": "1"})
        self.assertEqual(result["verifier_error"], "prediction_parse_failure")

    def test_training_callback_matches_evaluation(self):
        tokenizer = types.SimpleNamespace(decode=lambda text, **kwargs: text)
        texts = [r"</think>\boxed{0.5}", r"</think>\boxed{9}", "unfinished"]
        entries = [{"answer": r"\frac{1}{2}"}] * 3
        metrics, columns = {}, {}
        rewards = math_data.make_reward_function(None, tokenizer)(
            texts, entries, log_metric=metrics.__setitem__, log_extra=columns.__setitem__,
        )
        self.assertEqual(rewards, [math_data.score_response(None, t, e)["score"] for t, e in zip(texts, entries)])
        self.assertEqual(rewards, [1., 0., 0.])
        self.assertEqual(metrics["answer_extraction_rate"], 2 / 3)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.rows = Dataset.from_list([
            *[{"problem": f"Question {i}", "answer": str(i), "solution": "not a prompt"} for i in range(8)],
            {"problem": " Question   0 ", "answer": "0", "solution": ""},
            {"problem": "empty", "answer": "", "solution": ""},
            {"problem": "conflict", "answer": "1", "solution": ""},
            {"problem": "conflict", "answer": "2", "solution": ""},
        ])
        self.settings = {
            "dataset_id": ds.DATASET_ID, "dataset_revision": ds.DEFAULT_REVISION,
            "source_split": "train", "holdout_size": 3, "split_seed": 42,
            "schema_version": 1, "verifier": math_data.verifier_metadata(),
        }
        self.manifest = math_data.build_manifest(self.rows, self.settings)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "split.json"
        self.path.write_text(json.dumps(self.manifest))

    def test_reproducible_deduplicated_partition(self):
        self.assertEqual(self.manifest, math_data.build_manifest(self.rows, self.settings))
        self.assertFalse(set(self.manifest["train_ids"]) & set(self.manifest["validation_ids"]))
        self.assertEqual(len(self.manifest["train_ids"]), 5)
        self.assertEqual({r["reason"] for r in self.manifest["excluded"]},
                         {"duplicate_problem", "empty_answer", "conflicting_answers"})

    def test_subset_size_does_not_change_training_or_holdout(self):
        config = ds.DataConfig(split_manifest=str(self.path))
        with patch.object(ds, "load_dataset", return_value=self.rows):
            training, _, _ = ds.load_training_data(config, 99)
            small, _, _ = ds.load_eval_data(config, 1)
            large, _, _ = ds.load_eval_data(config, 3)
            with self.assertRaisesRegex(ValueError, "holdout"):
                ds.load_eval_data(config, 4)
        self.assertEqual(small, large[:1])
        self.assertEqual(list(training["example_id"]), self.manifest["train_ids"])
        self.assertNotIn("solution", str(training[0]["prompt"]))
        self.assertEqual(training[0]["entry"]["answer"], self.rows[training[0]["example_id"]]["answer"])

    def test_manifest_conflicts_and_corruption_fail(self):
        with patch.object(ds, "load_dataset", return_value=self.rows):
            with self.assertRaisesRegex(ValueError, "settings"):
                ds.load_eval_data(ds.DataConfig(split_manifest=str(self.path), split_seed=10), 1)
            bad = copy.deepcopy(self.manifest)
            bad["train_ids"][0] = bad["validation_ids"][0]
            self.path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "row IDs"):
                ds.load_eval_data(ds.DataConfig(split_manifest=str(self.path)), 1)
            bad = copy.deepcopy(self.manifest)
            bad["content_sha256"] = "changed"
            self.path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "content"):
                ds.load_eval_data(ds.DataConfig(split_manifest=str(self.path)), 1)

    def test_checkpoint_eval_inherits_manifest(self):
        root = Path(self.temp.name)
        (root / "run_config.json").write_text(json.dumps({"dataset": "deepscaler"}))
        (root / "split_manifest.json").write_text(json.dumps(self.manifest))
        _, data, args = parse_eval_config([
            "--dataset", "deepscaler", "--model_name", str(root / "final"), "--num_examples", "2",
        ])
        self.assertEqual(data.split_manifest, str(root / "split_manifest.json"))
        self.assertEqual(args.num_examples, 2)


class ConfigTest(unittest.TestCase):
    def test_method_and_dataset_configs_are_independent(self):
        for parser in (parse_grpo_config, parse_ppo_config):
            _, math, training = parser(CPU_ARGS + ["--dataset", "deepscaler", "--holdout_size", "20"])
            self.assertTrue(training.shuffle_dataset)
            self.assertEqual(math.holdout_size, 20)
            self.assertFalse(hasattr(math, "task"))
            _, rg, training = parser(CPU_ARGS + ["--dataset", "reasoning_gym", "--task", "all"])
            self.assertFalse(training.shuffle_dataset)
            self.assertEqual(rg.task, "all")
            with self.assertRaises(ValueError):
                parser(CPU_ARGS + ["--dataset", "deepscaler", "--task", "leg_counting"])
            with self.assertRaises(ValueError):
                parser(CPU_ARGS + ["--dataset", "reasoning_gym", "--holdout_size", "20"])

    def test_resume_rejects_changed_data_and_accepts_original(self):
        config = {"method": "ppo", "dataset": "deepscaler", "model_name": "test",
                  "data": {"manifest_sha256": "original"}, "versions": {},
                  "training": {"max_steps": 2, "lr_scheduler_type": "constant"}}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run_config.json").write_text(json.dumps(config))
            checkpoint = Path(directory) / "checkpoint-1"
            validate_resume(checkpoint, config)
            config["training"]["max_steps"] = 3
            validate_resume(checkpoint, config)
            config["data"]["manifest_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "data"):
                validate_resume(checkpoint, config)


if __name__ == "__main__":
    unittest.main()
