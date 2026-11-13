"""DeepMath schema mapping, metadata, dataset isolation, and trainer configuration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from expts.grpo.config import parse_config as parse_grpo_config
from expts.data import deepmath, deepscaler, math_data
from expts.eval.config import parse_eval_config
from expts.ppo.config import parse_config as parse_ppo_config
from expts.ppo.train import validate_resume


CPU_ARGS = ["--use_cpu", "true", "--bf16", "false", "--report_to", "none", "--use_vllm", "false"]


class DeepMathTest(unittest.TestCase):
    def setUp(self):
        self.raw = Dataset.from_list([
            {"question": f"Question {i}", "final_answer": str(i), "difficulty": 4.5 + i / 2,
             "topic": "Mathematics -> Algebra", "r1_solution_1": "secret solution",
             "r1_solution_2": "another solution", "r1_solution_3": "third solution"}
            for i in range(6)
        ] + [
            {"question": " Question   0 ", "final_answer": "0", "difficulty": 4.5,
             "topic": "Mathematics -> Algebra"},
            {"question": "No answer", "final_answer": "", "difficulty": 9., "topic": "Other"},
        ])
        loader = patch.object(deepmath, "load_dataset", side_effect=self.project_columns)
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        self.settings = {
            "dataset_id": deepmath.DATASET_ID, "dataset_revision": deepmath.DEFAULT_REVISION,
            "source_split": "train", "holdout_size": 2, "split_seed": 42,
            "schema_version": 1, "verifier": math_data.verifier_metadata(),
        }
        canonical = deepmath.load_rows(deepmath.DEFAULT_REVISION, "train")
        self.manifest = math_data.build_manifest(canonical, self.settings)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "split_manifest.json"
        self.path.write_text(json.dumps(self.manifest))

    def project_columns(self, dataset_id, *, revision, split, columns):
        self.assertEqual(dataset_id, deepmath.DATASET_ID)
        self.assertEqual(revision, deepmath.DEFAULT_REVISION)
        self.assertEqual(split, "train")
        self.assertEqual(columns, ["question", "final_answer", "difficulty", "topic"])
        return self.raw

    def test_mapping_retains_metadata_and_excludes_solutions(self):
        config = deepmath.DataConfig(split_manifest=str(self.path))
        training, _, metadata = deepmath.load_training_data(config, 99)
        self.assertEqual(len(training), 4)
        self.assertEqual(metadata["dataset_id"], deepmath.DATASET_ID)
        self.assertEqual(metadata["dataset_revision"], deepmath.DEFAULT_REVISION)
        for row in training:
            original = self.raw[row["example_id"]]
            self.assertEqual(row["prompt"][-1]["content"], original["question"])
            self.assertEqual(row["entry"]["answer"], original["final_answer"])
            self.assertEqual(row["entry"]["metadata"], {
                "difficulty": original["difficulty"], "topic": original["topic"],
                "source_dataset": "deepmath", "row_id": row["example_id"],
            })
            self.assertNotIn("r1_solution", json.dumps(row))
            self.assertNotIn("secret solution", json.dumps(row))

    def test_fixed_subsets_are_disjoint_and_score_with_same_reward(self):
        config = deepmath.DataConfig(split_manifest=str(self.path))
        small, _, _ = deepmath.load_eval_data(config, 1)
        full, _, metadata = deepmath.load_eval_data(config, 2)
        self.assertEqual(small, full[:1])
        self.assertEqual(metadata["partition"], "validation")
        self.assertFalse(set(e["example_id"] for e in full) & set(self.manifest["train_ids"]))
        self.assertEqual({e["reason"] for e in self.manifest["excluded"]}, {"duplicate_problem", "empty_answer"})
        self.assertIs(deepmath.score_response, deepscaler.score_response)
        for entry in full:
            result = deepmath.score_response(None, "</think>\\boxed{" + entry["answer"] + "}", entry)
            self.assertEqual(result["score"], 1.)
        with self.assertRaisesRegex(ValueError, "holdout"):
            deepmath.load_eval_data(config, 3)

    def test_foreign_manifests_are_rejected_before_loading(self):
        wrong = {**self.manifest, "dataset_id": deepscaler.DATASET_ID,
                 "dataset_revision": deepscaler.DEFAULT_REVISION}
        self.path.write_text(json.dumps(wrong))
        self.loader.reset_mock()
        with self.assertRaisesRegex(ValueError, "settings"):
            deepmath.load_eval_data(deepmath.DataConfig(split_manifest=str(self.path)), 1)
        self.loader.assert_not_called()
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "settings"):
            deepscaler.load_eval_data(deepscaler.DataConfig(split_manifest=str(self.path)), 1)

    def test_checkpoint_eval_resolves_deepmath_manifest(self):
        root = Path(self.temp.name)
        (root / "run_config.json").write_text(json.dumps({"dataset": "deepmath"}))
        for directory in ("final", "checkpoint-2"):
            run, config, args = parse_eval_config([
                "--dataset", "deepmath", "--model_name", str(root / directory), "--num_examples", "1",
            ])
            self.assertEqual(config.split_manifest, str(self.path))
            entries, _, _ = deepmath.load_eval_data(config, args.num_examples)
            self.assertEqual(len(entries), 1)
            self.assertIn("results/deepmath/eval", args.output_dir)

    def test_both_training_methods_resolve_deepmath_defaults(self):
        for parser, method in ((parse_grpo_config, "grpo"), (parse_ppo_config, "ppo")):
            run, data, args = parser(CPU_ARGS + ["--dataset", "deepmath", "--holdout_size", "2"])
            self.assertEqual(run.dataset, "deepmath")
            self.assertEqual(run.wandb_project, "deepmath")
            self.assertEqual(data.holdout_size, 2)
            self.assertTrue(args.shuffle_dataset)
            self.assertTrue(args.run_name.startswith(method + "-"))
            self.assertIn("deepmath", args.run_name)
            self.assertIn("/deepmath/", args.output_dir)
            with self.assertRaises(ValueError):
                parser(CPU_ARGS + ["--dataset", "deepmath", "--task", "leg_counting"])

    def test_resume_checks_manifest_identity(self):
        config = deepmath.DataConfig(split_manifest=str(self.path))
        _, _, metadata = deepmath.load_training_data(config, 42)
        previous = {"dataset": "deepmath", "model_name": "test", "method": "ppo", "data": metadata,
                    "training": {"max_steps": 1, "lr_scheduler_type": "constant"}, "versions": {}}
        root = Path(self.temp.name)
        (root / "run_config.json").write_text(json.dumps(previous))
        checkpoint = root / "checkpoint-1"
        validate_resume(checkpoint, previous)
        changed = {**previous, "data": {**metadata, "manifest_sha256": "other"}}
        with self.assertRaisesRegex(ValueError, "data"):
            validate_resume(checkpoint, changed)


if __name__ == "__main__":
    unittest.main()
