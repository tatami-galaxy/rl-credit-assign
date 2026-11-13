"""Independent algorithm configuration, training setup, and resume contracts."""

import copy
import json
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from expts.grpo import train as grpo
from expts.ppo import train as ppo


METHODS = (("grpo", grpo), ("ppo", ppo))
CPU_ARGS = ["--use_cpu", "true", "--bf16", "false", "--report_to", "none", "--use_vllm", "false"]


class TrainingConfigTest(unittest.TestCase):
    def test_defaults_and_explicit_overrides(self):
        for method, module in METHODS:
            with self.subTest(method=method):
                run, _, args = module.parse_config(CPU_ARGS)
                self.assertEqual(run.model_name, "Qwen/Qwen3-4B")
                self.assertEqual(run.wandb_project, "reasoning-gym")
                self.assertEqual(args.run_name, f"{method}-Qwen3-4B-leg_counting-seed42")
                self.assertEqual(args.output_dir, f"/data/ujan/rl_credit/reasoning_gym/{args.run_name}")
                expected = {
                    "learning_rate": 1e-5 if method == "grpo" else 1e-6,
                    "lr_scheduler_type": "linear" if method == "grpo" else "constant",
                    "per_device_train_batch_size": 8 if method == "grpo" else 1,
                    "gradient_accumulation_steps": 2 if method == "grpo" else 16,
                    "scale_rewards": "group" if method == "grpo" else "none",
                    "num_generations": 8, "max_completion_length": 8192, "max_steps": 1000,
                    "beta": 0., "loss_type": "dapo", "mask_truncated_completions": True,
                    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "shuffle_dataset": False,
                }
                self.assertEqual({key: getattr(args, key) for key in expected}, expected)
                _, _, overridden = module.parse_config(CPU_ARGS + [
                    "--run_name", "custom", "--output_dir", "/tmp/custom-run",
                    "--learning_rate", "0.000003", "--shuffle_dataset", "true",
                    "--chat_template_kwargs", '{"extra_option":true}',
                    "--generation_kwargs", '{"repetition_penalty":1.1}',
                    "--model_init_kwargs", '{"dtype":"bfloat16"}',
                ])
                self.assertEqual(overridden.run_name, "custom")
                self.assertEqual(overridden.output_dir, "/tmp/custom-run")
                self.assertEqual(overridden.learning_rate, 3e-6)
                self.assertTrue(overridden.shuffle_dataset)
                self.assertEqual(overridden.chat_template_kwargs, {"enable_thinking": True, "extra_option": True})
                self.assertEqual(overridden.generation_kwargs, {"min_p": 0., "repetition_penalty": 1.1})
                self.assertEqual(overridden.model_init_kwargs, {"dtype": "bfloat16"})


class TrainingFlowTest(unittest.TestCase):
    def test_setup_artifacts_resume_and_logging_cleanup(self):
        for method, module in METHODS:
            for dataset in ("reasoning_gym", "deepscaler", "deepmath"):
                with (
                    self.subTest(method=method, dataset=dataset),
                    tempfile.TemporaryDirectory() as root,
                    ExitStack() as stack,
                ):
                    output = Path(root) / "run"
                    run, data, args = module.parse_config(CPU_ARGS + [
                        "--dataset", dataset, "--output_dir", str(output), "--max_steps", "1",
                    ])
                    args.report_to = ["wandb"]
                    if dataset != "reasoning_gym":
                        source = Path(root) / "source-split.json"
                        source.write_text('{"split":"fixed"}\n')
                        data.split_manifest = str(source)
                    rows, scorer, reward = object(), object(), object()
                    metadata = {"dataset": dataset}
                    adapter = types.SimpleNamespace(
                        load_training_data=Mock(return_value=(rows, scorer, metadata)),
                        make_reward_function=Mock(return_value=reward),
                    )
                    stack.enter_context(patch.object(module, "dataset_module", return_value=adapter))
                    stack.enter_context(patch.object(module, "package_versions", return_value={"trl": "test"}))
                    tokenizer = types.SimpleNamespace(pad_token_id=7)
                    load_tokenizer = stack.enter_context(patch.object(
                        module.AutoTokenizer, "from_pretrained", return_value=tokenizer,
                    ))
                    trainer = Mock(accelerator=types.SimpleNamespace(is_main_process=True))
                    factory = stack.enter_context(patch.object(
                        module, "GRPOTrainer" if method == "grpo" else "PPOTrainer", return_value=trainer,
                    ))
                    if method == "ppo":
                        critic = Mock()
                        load_critic = stack.enter_context(patch.object(
                            module.AutoModelForSequenceClassification, "from_pretrained", return_value=critic,
                        ))
                    wandb_run = Mock()
                    wandb = types.SimpleNamespace(init=Mock(return_value=wandb_run))
                    stack.enter_context(patch.dict("sys.modules", {"wandb": wandb}))

                    module.train(run, data, args)
                    load_tokenizer.assert_called_once_with(run.model_name, padding_side="left")
                    adapter.load_training_data.assert_called_once_with(data, args.seed)
                    adapter.make_reward_function.assert_called_once_with(scorer, tokenizer)
                    self.assertIs(factory.call_args.kwargs["train_dataset"], rows)
                    self.assertIs(factory.call_args.kwargs["reward_funcs"], reward)
                    if method == "ppo":
                        load_critic.assert_called_once_with(run.model_name, num_labels=1, **args.model_init_kwargs)
                        self.assertIs(factory.call_args.kwargs["value_model"], critic)
                        self.assertEqual(critic.config.pad_token_id, 7)
                    else:
                        self.assertNotIn("value_model", factory.call_args.kwargs)
                    trainer.train.assert_called_once_with(resume_from_checkpoint=None)
                    trainer.save_model.assert_called_once_with(str(output / "final"))
                    saved = json.loads((output / "run_config.json").read_text())
                    self.assertEqual(saved["method"], method)
                    self.assertEqual(saved["data"], metadata)
                    self.assertEqual(saved["training"], args.to_dict())
                    wandb_run.finish.assert_called_once_with()
                    if dataset != "reasoning_gym":
                        self.assertEqual((output / "split_manifest.json").read_bytes(), source.read_bytes())
                    with self.assertRaises(FileExistsError):
                        module.train(run, data, args)

                    checkpoint = str(output / "checkpoint-1")
                    args.resume_from_checkpoint = True
                    if dataset != "reasoning_gym":
                        data.split_manifest = None
                    stack.enter_context(patch.object(module, "get_last_checkpoint", return_value=checkpoint))
                    module.train(run, data, args)
                    trainer.train.assert_called_with(resume_from_checkpoint=checkpoint)
                    if dataset != "reasoning_gym":
                        self.assertEqual(data.split_manifest, str(output / "split_manifest.json"))

                    trainer.train.side_effect = RuntimeError("training failed")
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        module.train(run, data, args)
                    self.assertEqual(wandb.init.call_count, 3)
                    self.assertEqual(wandb_run.finish.call_count, 3)
                    self.assertEqual(trainer.save_model.call_count, 2)

    def test_resume_rejects_changed_training_and_cross_algorithm_checkpoints(self):
        for method, module in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                _, _, args = module.parse_config(CPU_ARGS + ["--lr_scheduler_type", "constant"])
                config = {"method": method, "model_name": "model", "dataset": "reasoning_gym",
                          "data": {}, "versions": {"trl": "test"}, "training": args.to_dict()}
                path = Path(root) / "run_config.json"
                path.write_text(json.dumps(config))
                checkpoint = Path(root) / "checkpoint-1"
                module.validate_resume(checkpoint, config)
                changes = {"learning_rate": 0.1, "num_generations": 4, "temperature": 1.}
                if method == "ppo":
                    changes.update(critic_learning_rate=0.1, critic_warmup_steps=5, lam=0.5)
                for field, value in changes.items():
                    changed = copy.deepcopy(config)
                    changed["training"][field] = value
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                        module.validate_resume(checkpoint, changed)
                changed = copy.deepcopy(config)
                changed["method"] = "ppo" if method == "grpo" else "grpo"
                with self.assertRaisesRegex(ValueError, "method"):
                    module.validate_resume(checkpoint, changed)
                changed = copy.deepcopy(config)
                changed["training"]["max_steps"] += 1
                module.validate_resume(checkpoint, changed)
                config["training"]["lr_scheduler_type"] = "linear"
                changed["training"]["lr_scheduler_type"] = "linear"
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "constant"):
                    module.validate_resume(checkpoint, changed)


if __name__ == "__main__":
    unittest.main()
