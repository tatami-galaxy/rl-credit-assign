"""Shared finite-math data preparation, fixed holdouts, and Math-Verify rewards."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from random import Random

from datasets import Dataset
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from latex2sympy2_extended import NormalizationConfig


SYSTEM_PROMPT = r"Solve the problem step by step. Think carefully before you answer. Put your thinking between <think> and </think>. Put your final answer inside \boxed{...}."
VERIFIER_TIMEOUT = 3
GOLD_EXTRACTION = LatexExtractionConfig()
PRED_EXTRACTION = LatexExtractionConfig(normalization_config=NormalizationConfig(
    basic_latex=True, units=True, malformed_operators=False, nits=False,
    boxed="all", equations=False,
))


@dataclass
class DataConfig:
    dataset_revision: str | None = None  # Pinned default, or inherited from a manifest.
    source_split: str | None = None
    holdout_size: int | None = None
    split_seed: int | None = None
    split_manifest: str | None = None


def validate_config(config, parser):
    if config.holdout_size is not None and config.holdout_size <= 0:
        parser.error("--holdout_size must be positive.")
    if config.dataset_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", config.dataset_revision):
        parser.error("--dataset_revision must be a full commit SHA so data remains reproducible.")


def make_prompt(question):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def extract_final_answer(response):
    """Extract the last boxed answer after thinking, including nested LaTeX braces."""
    if "</think>" not in response:
        return None, False
    final = response.rsplit("</think>", 1)[1]
    matches = list(re.finditer(r"\\boxed\s*\{", final))
    if not matches:
        return None, True
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(final)):
        depth += (final[index] == "{") - (final[index] == "}")
        if depth == 0:
            return final[start:index].strip() or None, True
    return None, True


def parse_math(answer, gold=False):
    # Reference answers contain both bare LaTeX and already delimited math.
    answer = answer.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if answer.startswith(left) and answer.endswith(right):
            answer = answer[len(left):-len(right)].strip()
            break
    if not answer:
        return []
    return parse(
        "$" + answer + "$", extraction_config=[GOLD_EXTRACTION if gold else PRED_EXTRACTION],
        fallback_mode="no_fallback", extraction_mode="first_match",
        parsing_timeout=VERIFIER_TIMEOUT, raise_on_error=True,
    )


@lru_cache(maxsize=65536)
def parse_reference(answer):
    return parse_math(answer, gold=True)


def score_response(dataset, response, entry):
    answer, thinking_done = extract_final_answer(response)
    result = {"model_answer": answer, "score": 0.0, "thinking_complete": thinking_done,
              "verifier_error": None}
    if not answer:
        return result
    # References were audited during preparation; a failure here is a data error.
    gold = parse_reference(entry["answer"])
    if not gold:
        raise ValueError(f"Unparseable reference for row {entry.get('example_id')}")
    try:
        prediction = parse_math(answer)
        if not prediction:
            result["verifier_error"] = "prediction_parse_failure"
        else:
            result["score"] = float(verify(
                gold, prediction, timeout_seconds=VERIFIER_TIMEOUT, raise_on_error=True,
            ))
    except TimeoutException:
        result["verifier_error"] = "timeout"
    except Exception as error:
        result["verifier_error"] = type(error).__name__
    return result


def make_reward_function(dataset, tokenizer):
    def math_reward(completion_ids, entry, log_metric=None, log_extra=None, **kwargs):
        results = [score_response(dataset, tokenizer.decode(ids, skip_special_tokens=False), example)
                   for ids, example in zip(completion_ids, entry, strict=True)]
        if log_metric:
            for name, values in (
                ("full_credit_rate", [r["score"] == 1 for r in results]),
                ("answer_extraction_rate", [bool(r["model_answer"]) for r in results]),
                ("thinking_complete_rate", [r["thinking_complete"] for r in results]),
                ("verifier_failure_rate", [r["verifier_error"] is not None for r in results]),
            ):
                log_metric(name, sum(values) / len(values))
        if log_extra:
            log_extra("extracted_answer", [r["model_answer"] for r in results])
            log_extra("reference_answer", [e["answer"] for e in entry])
            log_extra("verifier_error", [r["verifier_error"] for r in results])
        return [r["score"] for r in results]
    return math_reward


def verifier_metadata():
    return {
        "math-verify": version("math-verify"),
        "latex2sympy2_extended": version("latex2sympy2_extended"),
        "sympy": version("sympy"),
        "timeout_seconds": VERIFIER_TIMEOUT,
        "policy": "boxed_after_think_binary_v1",
    }


def content_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["problem"], row["answer"]], ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(rows, settings):
    """Deduplicate before partitioning and record every excluded source row."""
    groups = defaultdict(list)
    excluded = []
    for index, row in enumerate(rows):
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            excluded.append({"row_id": index, "reason": "empty_problem"})
        elif not isinstance(row["answer"], str) or not row["answer"].strip():
            excluded.append({"row_id": index, "reason": "empty_answer"})
        else:
            groups[" ".join(row["problem"].split())].append(index)
    retained = []
    for ids in groups.values():
        # Conservatively exclude conflicting labels rather than choosing one.
        if len({rows[i]["answer"].strip() for i in ids}) > 1:
            excluded.extend({"row_id": i, "reason": "conflicting_answers"} for i in ids)
            continue
        index = ids[0]
        try:
            valid = bool(parse_reference(rows[index]["answer"]))
        except (Exception, TimeoutException):
            valid = False
        if not valid:
            excluded.extend({"row_id": i, "reason": "reference_parse_failure"} for i in ids)
            continue
        retained.append(index)
        excluded.extend({"row_id": i, "reason": "duplicate_problem"} for i in ids[1:])
    if not 0 < settings["holdout_size"] < len(retained):
        raise ValueError(f"holdout_size must leave training examples; only {len(retained)} valid unique rows.")
    Random(settings["split_seed"]).shuffle(retained)
    size = settings["holdout_size"]
    return {
        **settings, "source_rows": len(rows), "content_sha256": content_hash(rows),
        "validation_ids": retained[:size], "train_ids": retained[size:],
        "excluded": sorted(excluded, key=lambda row: row["row_id"]),
    }


def load_data(config, *, dataset_name, dataset_id, default_revision, load_rows, num_examples=None):
    """Load canonical problem/answer rows; extra columns become example metadata."""
    partition = "train" if num_examples is None else "validation"
    manifest = None
    if config.split_manifest:
        manifest = json.loads(Path(config.split_manifest).read_text())
    defaults = {"dataset_revision": default_revision, "source_split": "train",
                "holdout_size": 500, "split_seed": 42}
    settings = {"dataset_id": dataset_id, "schema_version": 1, "verifier": verifier_metadata()}
    for name, default in defaults.items():
        value = getattr(config, name)
        settings[name] = value if value is not None else (manifest[name] if manifest else default)
    if manifest and any(manifest.get(key) != value for key, value in settings.items()):
        raise ValueError("Split manifest does not match dataset, split, or verifier settings.")
    if not re.fullmatch(r"[0-9a-f]{40}", settings["dataset_revision"]):
        raise ValueError("The dataset revision must be a full commit SHA.")
    if not manifest:
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        path = Path(f"data/{dataset_name}/split-{key}.json")
        if path.is_file():
            manifest = json.loads(path.read_text())
        config.split_manifest = str(path.resolve())
    rows = load_rows(settings["dataset_revision"], settings["source_split"])
    if manifest is None:
        print(f"Auditing {len(rows)} {dataset_name} references and preparing a fixed holdout...", flush=True)
        manifest = build_manifest(rows, settings)
        path = Path(config.split_manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement keeps concurrent processes from reading partial JSON.
        import os
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)
    if any(manifest.get(key) != value for key, value in settings.items()):
        raise ValueError("Cached split manifest settings do not match.")
    if manifest["source_rows"] != len(rows) or manifest["content_sha256"] != content_hash(rows):
        raise ValueError("Dataset content differs from the split manifest.")
    all_ids = manifest["train_ids"] + manifest["validation_ids"] + [r["row_id"] for r in manifest["excluded"]]
    if sorted(all_ids) != list(range(len(rows))) or len(manifest["validation_ids"]) != settings["holdout_size"]:
        raise ValueError("Split manifest has overlapping, missing, or invalid row IDs.")
    print(f"{dataset_name}: {len(manifest['train_ids'])} train, {len(manifest['validation_ids'])} held out; "
          f"exclusions={dict(Counter(r['reason'] for r in manifest['excluded']))}", flush=True)
    ids = manifest[f"{partition}_ids"]
    if num_examples is not None:
        if not 0 < num_examples <= len(ids):
            raise ValueError(f"Requested {num_examples} evaluation examples, but holdout contains {len(ids)}.")
        ids = ids[:num_examples]
    entries = []
    for index in ids:
        row = rows[index]
        extra = {key: value for key, value in row.items() if key not in ("problem", "answer")}
        entries.append({
            "example_id": index, "question": row["problem"], "answer": row["answer"],
            "metadata": {**extra, "source_dataset": dataset_name, "row_id": index},
        })
    metadata = {**settings, "system_prompt": SYSTEM_PROMPT, "partition": partition,
                "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
    if partition == "train":
        entries = Dataset.from_list([{"prompt": make_prompt(e["question"]), "entry": e,
                                      "example_id": e["example_id"]} for e in entries])
    return entries, None, metadata
