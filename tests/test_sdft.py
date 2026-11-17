"""SDFT defaults, data isolation, rollout scoring, and real CPU checkpoint tests."""

import copy
import io
import json
import tempfile
import types
import unittest
from collections import defaultdict
from contextlib import redirect_stderr
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from datasets import Dataset
from reasoning_gym.dataset import ProceduralDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl.experimental.sdft import SDFTConfig, SDFTTrainer as TRLSDFTTrainer
from trl.experimental.sdft.sdft_trainer import DemonstrationTeacherContextBuilder

from expts.data import reasoning_gym as rg, deepscaler, deepmath, math_data
from expts.sdft.config import RunConfig, TrainingConfig, parse_config
from expts.sdft.data import IndexedReasoningGym, load_training_data, privileged_context
from expts.sdft import train as sdft
from expts.sdft.trainer import SDFTTrainer, TEACHER_MODEL_FILE


CPU_ARGS = ["--use_cpu", "true", "--bf16", "false", "--report_to", "none"]


class ConfigTest(unittest.TestCase):
    def test_matches_trl_defaults_except_experiment_overrides(self):
        base = SDFTConfig(use_cpu=True, bf16=False, report_to="none").to_dict()
        ours = TrainingConfig(use_cpu=True, bf16=False, report_to="none").to_dict()
        ours.pop("chat_template_kwargs")
        base.pop("chat_template_kwargs")
        self.assertEqual(ours.pop("max_steps"), 1000)
        base.pop("max_steps")
        self.assertEqual(ours, base)
        run, data, args = parse_config(CPU_ARGS)
        self.assertEqual(run.num_train_examples, 10_000)
        self.assertEqual(data.task, "leg_counting")
        self.assertTrue(args.run_name.startswith("sdft-"))
        self.assertEqual(args.teacher_model_kind, "base")
        self.assertEqual((args.distillation_mode, args.distillation_alpha, args.distillation_topk),
                         ("topk_logits", 0.5, 100))
        self.assertEqual(args.num_generations, 8)
        self.assertFalse(args.use_vllm)

    def test_overrides_and_dataset_specific_options(self):
        _, _, args = parse_config(CPU_ARGS + [
            "--teacher_model_kind", "ema", "--distillation_mode", "full_logits",
            "--distillation_alpha", "0", "--num_generations", "1",
            "--chat_template_kwargs", '{"extra":true}',
            "--generation_kwargs", '{"min_p":0.0}',
            "--model_init_kwargs", '{"attn_implementation":"eager"}',
        ])
        self.assertEqual(args.teacher_model_kind, "ema")
        self.assertEqual(args.distillation_alpha, 0)
        self.assertEqual(args.chat_template_kwargs, {"enable_thinking": True, "extra": True})
        self.assertEqual(args.generation_kwargs, {"min_p": 0.0})
        self.assertEqual(args.model_init_kwargs, {
            "dtype": "float32", "attn_implementation": "eager", "trust_remote_code": False,
        })
        for dataset in ("deepscaler", "deepmath"):
            run, _, args = parse_config(CPU_ARGS + ["--dataset", dataset])
            self.assertIsNone(run.num_train_examples)
            self.assertIn(dataset, args.run_name)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(CPU_ARGS + ["--dataset", dataset, "--num_train_examples", "10"])
        for flags in (["--num_train_examples", "0"], ["--remove_unused_columns", "true"],
                      ["--chat_template_kwargs", '{"enable_thinking":false}']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(CPU_ARGS + flags)


@dataclass
class TinyConfig:
    seed: int = 0
    size: int = 1


class TinyTask(ProceduralDataset):
    def __init__(self, config):
        super().__init__(config, config.seed, config.size)

    def __getitem__(self, index):
        return {"question": f"Question {self.seed + index}", "answer": "correct",
                "metadata": {"seed": self.seed, "source_dataset": "tiny"}}


class DataTest(unittest.TestCase):
    def test_indexed_rg_matches_stream_in_arbitrary_order(self):
        registry = {"tiny": (TinyTask, TinyConfig), "composite": rg.DATASETS["composite"]}
        with patch.dict(rg.DATASETS, registry, clear=True), patch.object(rg, "MIXTURE_CHUNK_SIZE", 4):
            for task in ("tiny", "all"):
                rows = IndexedReasoningGym(rg.DataConfig(task=task, dataset_seed=77), seed=42, size=12)
                expected = list(islice(rg.generate_examples(task, {}, 77), 12))
                for index in (8, 0, 3, 11, 4, 7, 4):
                    row = rows[index]
                    self.assertEqual({k: row[k] for k in expected[index]}, expected[index])
                    self.assertEqual(row["privileged_context"], "Reference final answer: correct")
                with self.assertRaises(IndexError):
                    rows[12]
                with self.assertRaises(IndexError):
                    rows[-1]

    def test_teacher_context_is_separate_from_student_prompt(self):
        row = IndexedReasoningGym(rg.DataConfig(), 42, 1)[0]
        original = copy.deepcopy(row["prompt"])
        builder = DemonstrationTeacherContextBuilder(types.SimpleNamespace(
            args=types.SimpleNamespace(teacher_prompt_template="{prompt}\n\n{privileged_context}"),
        ))
        teacher = builder.select_generation_prompts([row["prompt"]], [row["privileged_context"]])[0]
        self.assertEqual(row["prompt"], original)
        self.assertEqual(teacher[0], original[0])
        self.assertEqual(teacher[-1]["content"], original[-1]["content"] + "\n\n" + row["privileged_context"])
        self.assertNotIn("Reference final answer:", original[-1]["content"])
        with self.assertRaises(ValueError):
            privileged_context({"answer": ""})

    def test_math_split_and_entries_are_preserved(self):
        source = Dataset.from_list([{"problem": f"Problem {i}", "answer": str(i)} for i in range(6)])
        for name, module in (("deepscaler", deepscaler), ("deepmath", deepmath)):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as root:
                settings = {"dataset_id": module.DATASET_ID, "dataset_revision": module.DEFAULT_REVISION,
                            "source_split": "train", "holdout_size": 2, "split_seed": 42,
                            "schema_version": 1, "verifier": math_data.verifier_metadata()}
                manifest = math_data.build_manifest(source, settings)
                path = Path(root) / "split.json"
                path.write_text(json.dumps(manifest))
                config = module.DataConfig(split_manifest=str(path))
                with patch.object(module, "load_rows", return_value=source):
                    original, _, original_metadata = module.load_training_data(config, 42)
                    rows, _, metadata = load_training_data(RunConfig(dataset=name), config, 42)
                    heldout, _, _ = module.load_eval_data(config, 2)
                self.assertEqual(list(rows["example_id"]), list(original["example_id"]))
                self.assertEqual(metadata["manifest_sha256"], original_metadata["manifest_sha256"])
                self.assertFalse(set(rows["example_id"]) & {e["example_id"] for e in heldout})
                for before, after in zip(original, rows, strict=True):
                    self.assertEqual({k: after[k] for k in before}, before)
                    self.assertEqual(after["privileged_context"], privileged_context(before["entry"]))

    def test_generation_failure_names_task_and_index(self):
        rows = IndexedReasoningGym(rg.DataConfig(), 42, 1)
        with patch.object(type(rows.scorer), "__getitem__", side_effect=ValueError("broken")):
            with self.assertRaisesRegex(RuntimeError, "leg_counting.*example 0"):
                rows[0]


class MetricsTest(unittest.TestCase):
    def test_task_scores_do_not_change_training_batch(self):
        trainer = object.__new__(SDFTTrainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        row = IndexedReasoningGym(rg.DataConfig(), 42, 1)[0]
        tokenizer = types.SimpleNamespace(decode=lambda text, **kwargs: text)
        trainer.score_completions = rg.make_reward_function(
            rg.create_task_dataset("leg_counting", {}, 42), tokenizer,
        )
        texts = [f"</think><answer>{row['entry']['answer']}</answer>", "unfinished"]
        trainer._get_completion_ids_list = lambda batch: texts
        batch = {"completion_ids": torch.tensor([[1], [2]]), "completion_mask": torch.ones(2, 1)}
        with patch.object(TRLSDFTTrainer, "sample_rollouts", return_value=batch):
            self.assertIs(trainer.sample_rollouts([row, row]), batch)
        self.assertEqual(set(batch), {"completion_ids", "completion_mask"})
        self.assertEqual(trainer._metrics["train"]["mean_reward"], [0.5])
        self.assertEqual(trainer._metrics["train"]["thinking_complete_rate"], [0.5])


class ResumeTest(unittest.TestCase):
    def test_resume_checks_data_and_sdft_settings(self):
        _, _, args = parse_config(CPU_ARGS + ["--lr_scheduler_type", "constant", "--max_steps", "1"])
        config = {"method": "sdft", "model_name": "model", "dataset": "reasoning_gym",
                  "data": {"num_train_examples": 20}, "versions": {"trl": "test"},
                  "training": args.to_dict()}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "run_config.json").write_text(json.dumps(config))
            checkpoint = Path(root) / "checkpoint-1"
            sdft.validate_resume(checkpoint, config)
            for name, value in (("teacher_model_kind", "ema"), ("distillation_alpha", 0),
                                ("teacher_prompt_template", "changed"), ("max_prompt_length", 1024),
                                ("num_generations", 1), ("generate_from_teacher", True)):
                changed = copy.deepcopy(config)
                changed["training"][name] = value
                with self.subTest(field=name), self.assertRaisesRegex(ValueError, name):
                    sdft.validate_resume(checkpoint, changed)
            for name, value in (("method", "ppo"), ("data", {}), ("versions", {})):
                changed = {**config, name: value}
                with self.assertRaisesRegex(ValueError, name):
                    sdft.validate_resume(checkpoint, changed)
            changed = copy.deepcopy(config)
            changed["training"]["max_steps"] = 2
            sdft.validate_resume(checkpoint, changed)
            config["training"]["lr_scheduler_type"] = "linear"
            changed["training"]["lr_scheduler_type"] = "linear"
            (Path(root) / "run_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "constant"):
                sdft.validate_resume(checkpoint, changed)


def tiny_model(directory):
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "Question": 3, "answer": 4}
    vocab.update({f"token{i}": i for i in range(5, 128)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
        padding_side="left",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
    tokenizer.save_pretrained(directory)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
    ))
    model.save_pretrained(directory)
    return tokenizer


class TrainingTest(unittest.TestCase):
    def test_real_entrypoint_can_resume_with_fresh_cli_arguments(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as root:
            model_dir, output = Path(root) / "model", Path(root) / "run"
            tiny_model(model_dir)
            flags = CPU_ARGS + [
                "--model_name", str(model_dir), "--output_dir", str(output),
                "--num_train_examples", "2", "--num_generations", "1",
                "--per_device_train_batch_size", "1", "--max_completion_length", "4",
                "--save_steps", "1", "--lr_scheduler_type", "constant", "--disable_tqdm", "true",
            ]
            sdft.train(*parse_config(flags + ["--max_steps", "1"]))
            sdft.train(*parse_config(flags + [
                "--max_steps", "2", "--resume_from_checkpoint", str(output / "checkpoint-1"),
            ]))
            state = json.loads((output / "checkpoint-2" / "trainer_state.json").read_text())
            self.assertEqual(state["global_step"], 2)

    def test_real_student_update_and_base_or_ema_teacher_resume(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "model"
            tokenizer = tiny_model(model_dir)
            rows = Dataset.from_list([
                {"prompt": "Question", "privileged_context": "Reference answer: token5",
                 "entry": {"answer": "token5"}, "example_id": i} for i in range(2)
            ])
            for kind in ("base", "ema"):
                with self.subTest(teacher=kind):
                    args = TrainingConfig(
                        output_dir=str(Path(root) / kind), use_cpu=True, bf16=False, report_to="none",
                        teacher_model_kind=kind, max_steps=1, save_steps=1,
                        num_generations=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                        max_completion_length=4, learning_rate=0.01, lr_scheduler_type="constant",
                        optim="adamw_torch", disable_tqdm=True,
                    )
                    def build():
                        return SDFTTrainer(
                            model=str(model_dir), args=args, train_dataset=rows, processing_class=tokenizer,
                            score_completions=lambda completion_ids, **kwargs: [0.] * len(completion_ids),
                        )
                    trainer = build()
                    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
                    teacher_before = {n: p.detach().clone() for n, p in trainer.teacher_model.named_parameters()}
                    trainer.train()
                    self.assertTrue(any(not torch.equal(before[n], p) for n, p in trainer.model.named_parameters()))
                    for name, p in trainer.teacher_model.named_parameters():
                        self.assertFalse(p.requires_grad)
                        self.assertIsNone(p.grad)
                        expected = teacher_before[name]
                        if kind == "ema":
                            expected = expected * (1 - args.teacher_update_rate) + dict(trainer.model.named_parameters())[name].detach() * args.teacher_update_rate
                        torch.testing.assert_close(p, expected)
                    checkpoint = Path(args.output_dir) / "checkpoint-1"
                    self.assertEqual((checkpoint / TEACHER_MODEL_FILE).is_file(), kind == "ema")
                    restored = build()
                    restored._load_from_checkpoint(str(checkpoint))
                    for original, loaded in zip(trainer.teacher_model.parameters(), restored.teacher_model.parameters(), strict=True):
                        torch.testing.assert_close(original, loaded)
                    args.max_steps = 2
                    restored.train(resume_from_checkpoint=str(checkpoint))
                    self.assertEqual(restored.state.global_step, 2)
                    if kind == "ema":
                        with self.assertRaisesRegex(FileNotFoundError, "EMA teacher"):
                            restored._load_from_checkpoint(str(Path(root) / "missing"))

    def test_entrypoint_writes_metadata_and_cleans_up_logging(self):
        with tempfile.TemporaryDirectory() as root:
            run, data, args = parse_config(CPU_ARGS + ["--output_dir", root, "--max_steps", "1"])
            args.report_to = ["wandb"]
            trainer = Mock(accelerator=types.SimpleNamespace(is_main_process=True))
            wandb_run = Mock()
            wandb = types.SimpleNamespace(init=Mock(return_value=wandb_run))
            with (
                patch.object(sdft, "SDFTTrainer", return_value=trainer),
                patch.object(sdft.AutoTokenizer, "from_pretrained", return_value=object()),
                patch.object(sdft, "package_versions", return_value={"trl": "test"}),
                patch.dict("sys.modules", {"wandb": wandb}),
            ):
                sdft.train(run, data, args)
                saved = json.loads((Path(root) / "run_config.json").read_text())
                self.assertEqual(saved["method"], "sdft")
                self.assertEqual(saved["data"]["privileged_context"], "reference_final_answer_v1")
                self.assertEqual(saved["training"], args.to_dict())
                trainer.save_model.assert_called_once_with(str(Path(root) / "final"))
                wandb_run.finish.assert_called_once()
                with self.assertRaises(FileExistsError):
                    sdft.train(run, data, args)
                args.resume_from_checkpoint = str(Path(root) / "checkpoint-1")
                trainer.train.side_effect = RuntimeError("failed")
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    sdft.train(run, data, args)
                self.assertEqual(wandb_run.finish.call_count, 2)


if __name__ == "__main__":
    unittest.main()
