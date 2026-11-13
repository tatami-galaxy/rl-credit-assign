"""Compose run, dataset, and method options without accepting unrelated flags."""

import argparse
import json
from pathlib import Path

from transformers import HfArgumentParser


def dataset_module(name):
    if name == "reasoning_gym":
        from .data import reasoning_gym
        return reasoning_gym
    if name == "deepscaler":
        from .data import deepscaler
        return deepscaler
    if name == "deepmath":
        from .data import deepmath
        return deepmath
    raise ValueError(f"Unknown dataset: {name}")


def parse_options(run_config_class, config_class, argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--dataset", choices=("reasoning_gym", "deepscaler", "deepmath"), default=run_config_class.dataset,
    )
    selected, _ = selector.parse_known_args(argv)
    module = dataset_module(selected.dataset)
    parser = HfArgumentParser((run_config_class, module.DataConfig, config_class), allow_abbrev=False)
    run, data, args = parser.parse_args_into_dataclasses(args=argv)
    module.validate_config(data, parser)
    return run, data, args, parser


def run_metadata(model_name):
    """Find the run config beside a local final/ or checkpoint-N/ directory."""
    path = Path(model_name).parent / "run_config.json"
    return json.loads(path.read_text()) if path.is_file() else None
