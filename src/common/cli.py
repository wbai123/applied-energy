"""Validate shared command-line identities, numeric domains, and result paths."""

import argparse
import json
import math
import pathlib
import re
import sys
import tempfile

from common.outputs import write_day_test, write_hour_test
from common.rewards import CHOICES, defaults
from feeder.loads import dataset_dir, dataset_feeder

MAX_SEED = 4294967295
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class ArgumentParser(argparse.ArgumentParser):
    """Print concise command-line errors to stderr and exit with status two."""

    def error(self, message):
        print(message, file=sys.stderr)
        raise SystemExit(2)


def dataset_name(value):
    """Validate and return one dataset name."""
    try:
        dataset_dir(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def run_name(value):
    """Validate and return one run name."""
    if RUN_NAME.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"invalid run name: {value!r}")
    return value


def seed_value(value):
    """Validate and return one random seed."""
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be an integer in 0..{MAX_SEED}")
    return seed


def positive_integer(value):
    """Validate and return one positive integer."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer of at least 1") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be an integer of at least 1")
    return number


def beta_value(value):
    """Validate and return one finite non-negative beta."""
    try:
        beta = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("beta must be a finite non-negative float") from error
    if not math.isfinite(beta) or beta < 0:
        raise argparse.ArgumentTypeError("beta must be a finite non-negative float")
    return beta


def nonnegative_float(value):
    """Validate and return one finite non-negative float."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a finite non-negative float") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative float")
    return number


def add_reward_arguments(parser, name):
    """Add one flag per parameter of a registered reward, with its signature defaults."""
    for parameter, default in defaults(name).items():
        flag = "--" + parameter.replace("_", "-")
        if isinstance(default, float):
            parser.add_argument(
                flag, type=nonnegative_float, default=default, help=f"default: {default}"
            )
        elif isinstance(default, str):
            parser.add_argument(
                flag,
                choices=CHOICES.get(parameter),
                default=default,
                help=f"default: {default}",
            )
        else:
            raise TypeError(f"unsupported reward parameter type: {parameter}")


def result_directory(root, dataset, algorithm, name=None):
    """Resolve a result directory without allowing a run name to escape its parent."""
    base = (pathlib.Path(root) / "results" / dataset / algorithm).resolve()
    if name is None:
        return base
    checked = run_name(name)
    target = (base / checked).resolve()
    if target.parent != base:
        raise ValueError(f"unsafe result directory: {target}")
    return target


def display_path(path, root):
    """Return a repository-relative path for command output."""
    return pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())


def evaluation_source(train_dataset, dataset, algorithm, name):
    """Validate dataset identities and describe a cross-dataset evaluation source."""
    dataset_name(train_dataset)
    dataset_name(dataset)
    if train_dataset == dataset:
        return None
    if dataset_feeder(train_dataset) != dataset_feeder(dataset):
        raise ValueError("training and evaluation datasets must name the same feeder")
    return {"train_dataset": train_dataset, "algorithm": algorithm, "run": name}


def validate_evaluation_source(directory, expected):
    """Require a readable source record matching the selected training run."""
    path = pathlib.Path(directory) / "source.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"missing or malformed evaluation source: {path}") from error
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"evaluation source does not match the selected training run: {path}")


def guard_evaluation_directory(directory, *, cross_dataset):
    """Prevent foreign evaluations from mixing with a destination's training artifacts."""
    directory = pathlib.Path(directory)
    if cross_dataset:
        if (directory / "run.json").exists() or (directory / "checkpoints").exists():
            raise ValueError("cross-dataset evaluation cannot write into a training directory")
    elif (directory / "source.json").exists():
        try:
            record = json.loads((directory / "source.json").read_text())
            training = (
                record.get("train_dataset", "unknown") if isinstance(record, dict) else "unknown"
            )
        except OSError, ValueError:
            training = "unreadable source.json"
        raise ValueError(f"default evaluation refuses source.json from train_dataset {training!r}")


def write_cross_dataset_results(rows, hour_rows, action_count, output, hour_output, source, root):
    """Stage both result CSVs and provenance before replacing the complete output triple."""
    scratch = pathlib.Path(root) / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=scratch) as temporary:
        staging = pathlib.Path(temporary)
        write_day_test(rows, staging / output.name)
        write_hour_test(hour_rows, staging / hour_output.name, action_count)
        (staging / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        for destination in (output, hour_output, output.parent / "source.json"):
            (staging / destination.name).replace(destination)
