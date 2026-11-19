"""Offline critic data isolation, caching, and real tiny-model SFT checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from expts.data import deepmath
from expts.nlac import collect, generation, predict, targets
from expts.nlac.data import (
    extract_critique, load_corpus, read_json, read_jsonl, reasoning_segments,
    segment_context, select_segments, write_json, write_jsonl,
)
from expts.nlac.prompts import critic_prompt, target_prompt
from expts.nlac.train import parse_config, prepare_data, train, validate_resume


class SegmentTest(unittest.TestCase):
    def test_offsets_preserve_reasoning_and_exclude_final_answer(self):
        reasoning = "  First line.\nAn equation.\n \nSecond paragraph.\r\n\t\r\nLast paragraph.  "
        for response, prompt in ((f"<think>{reasoning}</think>FINAL", "prompt"),
                                 (f"{reasoning}</think>FINAL", "prompt<think>\n")):
            spans = reasoning_segments(response, prompt)
            self.assertEqual([response[s["start"]:s["end"]] for s in spans],
                             ["First line.\nAn equation.", "Second paragraph.", "Last paragraph."])
            self.assertTrue(all(s["end"] <= response.index("</think>") for s in spans))
        self.assertEqual(reasoning_segments("text without closing", "<think>"), [])
        self.assertEqual(reasoning_segments("text</think>", "<think></think>"), [])

    def test_selection_is_reproducible_and_excludes_unusable_rollouts(self):
        problems = {1: {"rendered_prompt": "<think>"}}
        base = {"example_id": 1, "split": "train", "truncated": False, "verifier_error": None, "finish_reason": "stop",
                "response": "\n\n".join(f"Paragraph {i}" for i in range(12)) + "</think>FINAL"}
        rollouts = {str(i): {**base, "rollout_id": str(i)} for i in range(4)}
        rollouts["1"]["truncated"] = True
        rollouts["2"]["verifier_error"] = "timeout"
        rollouts["3"]["response"] = "no thinking"
        segments, excluded = select_segments(problems, rollouts, 4, 42)
        self.assertEqual((segments, excluded), select_segments(problems, rollouts, 4, 42))
        self.assertEqual(len(segments), 4)
        self.assertEqual(len({s["paragraph_index"] for s in segments}), 4)
        self.assertEqual(excluded, {"truncated": 1, "verifier_error": 1, "missing_or_empty_reasoning": 1})

    def test_privileged_information_is_only_in_target_prompt(self):
        visible = critic_prompt("QUESTION", "PREFIX", "CURRENT")
        privileged = target_prompt("QUESTION", "PREFIX", "CURRENT", "FUTURE_SECRET", 0, "PREDICTED", "GOLD_SECRET")
        self.assertNotIn("FUTURE_SECRET", json.dumps(visible))
        self.assertNotIn("GOLD_SECRET", json.dumps(visible))
        self.assertIn("FUTURE_SECRET", json.dumps(privileged))
        self.assertIn("GOLD_SECRET", json.dumps(privileged))
        self.assertEqual(visible, critic_prompt("QUESTION", "PREFIX", "CURRENT"))

    def test_only_final_critique_is_extracted(self):
        raw = "<think><critique>Private draft</critique> Secret analysis</think>\n<critique>Final judgment.</critique>"
        self.assertEqual(extract_critique(raw, require_thinking_end=True), ("<critique>Final judgment.</critique>", None))
        self.assertEqual(extract_critique(raw, truncated=True)[1], "truncated")
        self.assertEqual(extract_critique("<critique>Draft</critique>", require_thinking_end=True)[1], "unfinished_thinking")
        self.assertIsNone(extract_critique("<critique>A</critique><critique>B</critique>")[0])
        self.assertIsNone(extract_critique("<think>Unfinished")[0])


def tiny_model(directory):
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "Useful": 3, "step": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}{% if enable_thinking %}<think>\n{% else %}<think>\n</think>\n{% endif %}{% endif %}"
    tokenizer.save_pretrained(directory)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=4096, pad_token_id=0, eos_token_id=1,
    )).save_pretrained(directory)
    return tokenizer


class PipelineTest(unittest.TestCase):
    def test_cached_pipeline_and_real_training_resume(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model, corpus, labels = root / "model", root / "corpus", root / "labels"
            tokenizer = tiny_model(model)
            manifest = root / "split.json"
            write_json(manifest, {"train_ids": [1], "validation_ids": [2]})
            entries = [{"example_id": i, "question": f"Compute {i} + 1", "answer": "2", "metadata": {}} for i in (1, 2)]

            def training_data(config, seed):
                config.split_manifest = str(manifest)
                return Dataset.from_list([{"entry": entries[0]}]), None, {"partition": "train"}

            def generate(llm, prompts, seeds, args, thinking):
                if "PRIVILEGED REMAINDER" in prompts[0][1]["content"]:
                    response = "<critique>Useful step with a justified calculation.</critique>"
                    if thinking:
                        response = "Private teacher reasoning</think>\n" + response
                elif not thinking:
                    response = "<critique>Useful step.</critique>"
                else:
                    response = "First calculation.\n\nSecond calculation.\n\nLast calculation.</think>\n\\boxed{2}"
                return [{"response": response, "response_token_ids": tokenizer.encode(response),
                         "response_length": len(tokenizer.encode(response)), "generation_seed": seed,
                         "finish_reason": "stop", "stop_reason": None, "truncated": False} for seed in seeds]

            with patch.object(deepmath, "load_training_data", side_effect=training_data), \
                    patch.object(deepmath, "load_eval_data", return_value=([entries[1]], None, {"partition": "validation"})), \
                    patch.object(generation, "load_llm", return_value=object()), \
                    patch.object(generation, "generate", side_effect=generate) as sampler:
                collect_flags = ["collect", "--model_name", str(model), "--output_dir", str(corpus),
                                 "--num_train_problems", "1", "--num_val_problems", "1", "--num_generations", "1", "--batch_size", "1"]
                with patch.object(sys, "argv", collect_flags):
                    collect.main()
                    calls = sampler.call_count
                    collect.main()
                    self.assertEqual(sampler.call_count, calls)
                with patch.object(sys, "argv", collect_flags + ["--temperature", "0.5"]):
                    with self.assertRaisesRegex(ValueError, "Settings"):
                        collect.main()
                target_flags = ["targets", "--corpus_dir", str(corpus), "--output_dir", str(labels), "--segments_per_rollout", "2"]
                with patch.object(sys, "argv", target_flags):
                    targets.main()
                    calls = sampler.call_count
                    targets.main()
                    self.assertEqual(sampler.call_count, calls)
                with patch.object(sys, "argv", target_flags + ["--no_thinking"]):
                    with self.assertRaisesRegex(ValueError, "Settings"):
                        targets.main()
                direct_labels = root / "labels-no-thinking"
                with patch.object(sys, "argv", ["targets", "--corpus_dir", str(corpus),
                                              "--output_dir", str(direct_labels), "--no_thinking",
                                              "--segments_per_rollout", "2"]):
                    targets.main()
                    self.assertFalse(sampler.call_args.kwargs["thinking"])
                    self.assertFalse(read_json(direct_labels / "run_config.json")["enable_thinking"])
                    direct_rows = read_jsonl(direct_labels / "targets/part-00000.jsonl")
                    self.assertEqual(len(direct_rows), 4)
                    self.assertTrue(all(r["target_error"] is None and "</think>" not in r["response"] for r in direct_rows))
                    calls = sampler.call_count
                    targets.main()
                    self.assertEqual(sampler.call_count, calls)
                with patch.object(sys, "argv", ["predict", "--targets_dir", str(labels), "--output_dir", str(root / "predictions")]):
                    predict.main()
                prediction = read_jsonl(root / "predictions/responses/part-00000.jsonl")
                self.assertEqual(len(prediction), 2)
                self.assertTrue(all(row["segment_id"].startswith("2:") for row in prediction))

            datasets, report, _ = prepare_data(labels, tokenizer, 2048)
            self.assertEqual((len(datasets["train"]), len(datasets["validation"])), (2, 2))
            source, problems, rollouts, _ = load_corpus(corpus)
            segments = {s["segment_id"]: s for s in read_jsonl(labels / "segments.jsonl")}
            for split in datasets:
                for included, row in zip([r for r in report["included"] if r["split"] == split], datasets[split], strict=True):
                    problem, _, prefix, current, _ = segment_context(segments[included["segment_id"]], problems, rollouts)
                    prompt = generation.prompt_ids(tokenizer, critic_prompt(problem["question"], prefix, current), False)
                    self.assertEqual(row["labels"][:len(prompt)], [-100] * len(prompt))
                    self.assertEqual(row["input_ids"][:len(prompt)], prompt)
                    self.assertEqual(row["labels"][len(prompt):], row["input_ids"][len(prompt):])
                    self.assertEqual(row["labels"][-1], tokenizer.eos_token_id)
            with self.assertRaisesRegex(ValueError, "context_limit"):
                prepare_data(labels, tokenizer, 2)

            flags = ["--targets_dir", str(labels), "--output_dir", str(root / "training"),
                     "--use_cpu", "true", "--bf16", "false", "--optim", "adamw_torch",
                     "--gradient_checkpointing", "false", "--gradient_accumulation_steps", "1",
                     "--max_length", "2048", "--save_steps", "1", "--disable_tqdm", "true",
                     "--learning_rate", "0.01"]
            train(*parse_config(flags + ["--max_steps", "1"]))
            original = Qwen3ForCausalLM.from_pretrained(model)
            updated = Qwen3ForCausalLM.from_pretrained(root / "training/final")
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(original.parameters(), updated.parameters(), strict=True)))
            checkpoint = root / "training/checkpoint-1"
            train(*parse_config(flags + ["--max_steps", "2", "--resume_from_checkpoint", str(checkpoint)]))
            self.assertEqual(read_json(root / "training/checkpoint-2/trainer_state.json")["global_step"], 2)
            config = read_json(root / "training/run_config.json")
            config["targets_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "targets_sha256"):
                validate_resume(checkpoint, config)
            rows = read_jsonl(corpus / "rollouts/part-00000.jsonl")
            rows[0]["response"] += "changed"
            write_jsonl(corpus / "rollouts/part-00000.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Corpus"):
                prepare_data(labels, tokenizer, 2048)


if __name__ == "__main__":
    unittest.main()
