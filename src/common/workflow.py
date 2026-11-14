"""Validate and run shared one-shot or sequential notebook workflows.

The workflow separates execution from presentation. ``train`` mode trains and
evaluates, ``evaluate`` mode uses saved checkpoints, and ``view`` mode only reads
saved results. Preflight checks finish before any subprocess starts. Train and
evaluate modes always reuse complete compatible reference outputs, including
when overwrite replaces learned outputs. Run a reference evaluator directly to
deliberately recompute compatible reference outputs.
"""

import argparse
import csv
import dataclasses
import json
import math
import pathlib
import sys
import time

from common import commands
from common.cli import (
    MAX_SEED,
    dataset_name,
    evaluation_source,
    guard_evaluation_directory,
    run_name,
    validate_evaluation_source,
)
from common.observations import day_observation_size, voltage_observation_size
from common.outputs import DAY_COLUMNS, hour_columns, static_columns
from common.rewards import REWARDS, reward_parameters
from common.runs import resolve_run_settings
from feeder.loads import dataset_feeder, flatten_hours, read_days
from feeder.simulator import FeederSimulator

PROTOCOLS = {
    "one_shot": {"reference_evaluator": "src/reference/test.py"},
    "sequential": {"reference_evaluator": "src/seq_reference/test.py"},
}
LEARNERS = {
    ("one_shot", "ddpg"): {
        "result_algorithm": "ddpg",
        "observation": "voltage_observation",
        "trainer": "src/ddpg/train.py",
        "evaluator": "src/ddpg/test.py",
        "checkpoints": ("Actor_ddpg", "Critic_ddpg"),
        "label": "DDPG",
    },
    ("sequential", "ddpg"): {
        "result_algorithm": "seq_ddpg",
        "observation": "day_observation",
        "trainer": "src/seq_ddpg/train.py",
        "evaluator": "src/seq_ddpg/test.py",
        "checkpoints": ("Actor_ddpg", "Critic_ddpg"),
        "label": "DDPG",
    },
    ("sequential", "td3"): {
        "result_algorithm": "seq_td3",
        "observation": "day_observation",
        "trainer": "src/seq_td3/train.py",
        "evaluator": "src/seq_td3/test.py",
        "checkpoints": ("Actor_td3", "Critic_td3"),
        "label": "TD3",
    },
}
REFERENCE_LABELS = {"neutral": "Neutral", "regcontrol": "RegControl"}


@dataclasses.dataclass(frozen=True)
class ExperimentResults:
    """Hold one workflow's validated selections and aligned saved evidence.

    ``settings`` contains the current notebook selection. ``run_tasks`` and
    ``layer_sizes`` come from each saved run record. Result dictionaries use
    display labels and retain matched input order across learned and reference
    controllers.
    """

    settings: dict
    root: pathlib.Path
    simulator: FeederSimulator
    action_count: int
    evaluation_days: tuple
    evaluation_scenarios: tuple
    run_names: tuple
    algorithm_label: str
    learned_labels: tuple
    method_labels: tuple
    run_directories: dict
    reference_directories: dict
    run_tasks: dict
    layer_sizes: dict
    training_records: dict
    training_counts: dict
    update_counts: dict
    result_rows: dict
    daily_rows: dict
    hourly_rows: dict
    durations: dict


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, not {value!r}.")
    return value


def _number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number, not {value!r}.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number, not {value!r}.")
    return number


def _required(settings, name):
    if name not in settings:
        raise ValueError(f"Missing workflow setting: {name}.")
    return settings[name]


def _checked_run_name(value):
    try:
        return run_name(value)
    except (TypeError, argparse.ArgumentTypeError) as error:
        raise ValueError(str(error)) from error


def _validate_settings(settings):
    """Validate notebook selections and derive protocol-specific run names.

    The returned dictionary normalizes paths, tuples, numeric values, beta text,
    and optional settings without changing the caller's dictionary.
    """
    if not isinstance(settings, dict):
        raise ValueError("Workflow settings must be a dictionary.")
    protocol = _required(settings, "protocol")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol!r}.")
    algorithm = _required(settings, "algorithm")
    if (protocol, algorithm) not in LEARNERS:
        if algorithm == "td3" and protocol == "one_shot":
            raise ValueError("The TD3 algorithm supports the sequential protocol only.")
        raise ValueError(f"Unsupported algorithm {algorithm!r} for {protocol}.")
    mode = _required(settings, "mode")
    if mode not in {"train", "evaluate", "view"}:
        raise ValueError("MODE must be 'train', 'evaluate', or 'view'.")
    overwrite = _required(settings, "overwrite")
    if not isinstance(overwrite, bool):
        raise ValueError("OVERWRITE must be a Boolean value.")
    split = _required(settings, "split")
    if split not in {"validation", "test"}:
        raise ValueError("SPLIT must be validation or test.")
    root = pathlib.Path(_required(settings, "root")).resolve()
    feeder = _required(settings, "feeder")
    if not isinstance(feeder, str) or not feeder:
        raise ValueError("FEEDER must be a non-empty name.")
    dataset = _required(settings, "dataset")
    train_dataset = settings.get("train_dataset", dataset)
    try:
        dataset_name(dataset)
        dataset_name(train_dataset)
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    try:
        recorded_feeder = dataset_feeder(dataset)
        if recorded_feeder != feeder:
            raise ValueError(
                f"Dataset {dataset} names feeder {recorded_feeder}, expected {feeder}."
            )
        if train_dataset != dataset:
            if protocol != "sequential" or mode not in {"evaluate", "view"}:
                raise ValueError(
                    "A differing train_dataset requires sequential evaluate or view mode."
                )
            if dataset_feeder(train_dataset) != feeder:
                raise ValueError("Training and evaluation datasets must name the same feeder.")
    except (OSError, ValueError) as error:
        raise ValueError(str(error)) from error
    seeds = _required(settings, "seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("SEEDS must contain at least one seed.")
    seeds = tuple(_integer("Each seed", seed, 0) for seed in seeds)
    if any(seed > MAX_SEED for seed in seeds):
        raise ValueError(f"Each seed must be no greater than {MAX_SEED}.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("SEEDS must contain distinct values.")
    evaluation_days = settings.get("evaluation_days")
    if evaluation_days is not None:
        evaluation_days = _integer("EVALUATION_DAYS", evaluation_days, 1)
    reward = _required(settings, "reward")
    if reward not in REWARDS:
        raise ValueError(f"REWARD must be one of {tuple(REWARDS)}.")
    overrides = _required(settings, "reward_parameters")
    try:
        parameters = reward_parameters(reward, overrides)
    except ValueError as error:
        raise ValueError(f"REWARD_PARAMETERS: {error}.") from error
    if protocol == "one_shot" and parameters.get("beta", 0.0) != 0:
        raise ValueError("REWARD_PARAMETERS: one-shot training needs beta 0.")
    run_name_setting = _required(settings, "run_name")
    if not isinstance(run_name_setting, str) or not run_name_setting:
        raise ValueError("RUN_NAME must be a non-empty run name.")
    _checked_run_name(run_name_setting)

    normalized = dict(settings)
    normalized.update(
        protocol=protocol,
        algorithm=algorithm,
        mode=mode,
        overwrite=overwrite,
        split=split,
        root=root,
        feeder=feeder,
        dataset=dataset,
        train_dataset=train_dataset,
        seeds=seeds,
        evaluation_days=evaluation_days,
        reward=reward,
        reward_parameters=parameters,
        run_name=run_name_setting,
    )
    if protocol == "one_shot":
        normalized["episodes"] = _integer("EPISODES", _required(settings, "episodes"), 1)
        normalized.update(beta=None, beta_text=None, plot_day=None, show_hourly_details=False)
    else:
        normalized["training_days"] = _integer(
            "TRAINING_DAYS", _required(settings, "training_days"), 1
        )
        beta = _number("BETA", _required(settings, "beta"))
        beta_text = str(beta)
        normalized["plot_day"] = _integer("PLOT_DAY", _required(settings, "plot_day"), 0)
        details = _required(settings, "show_hourly_details")
        if not isinstance(details, bool):
            raise ValueError("SHOW_HOURLY_DETAILS must be a Boolean value.")
        normalized.update(beta=beta, beta_text=beta_text, show_hourly_details=details)
    normalized["run_names"] = tuple(
        _checked_run_name(f"{run_name_setting}_seed{seed}") for seed in seeds
    )
    return normalized


def _paths(settings):
    """Derive learned and reference directories for the selected protocol."""
    protocol = settings["protocol"]
    root = settings["root"]
    dataset = settings["dataset"]
    algorithm = LEARNERS[(protocol, settings["algorithm"])]["result_algorithm"]
    run_directories = {
        name: root / "results" / dataset / algorithm / name for name in settings["run_names"]
    }
    if protocol == "one_shot":
        reference_names = {controller: controller for controller in REFERENCE_LABELS}
        reference_algorithm = "reference"
    else:
        reference_names = {
            controller: _checked_run_name(f"{controller}_b{settings['beta_text']}")
            for controller in REFERENCE_LABELS
        }
        reference_algorithm = "seq_reference"
    reference_directories = {
        controller: root / "results" / dataset / reference_algorithm / name
        for controller, name in reference_names.items()
    }
    return algorithm, run_directories, reference_directories


def _result_paths(settings, run_directories, reference_directories):
    """List every evaluation output owned by the selected split and protocol."""
    split = settings["split"]
    sequential = settings["protocol"] == "sequential"

    def outputs(directory):
        paths = [directory / f"{split}.csv"]
        if sequential:
            paths.append(directory / f"{split}_hourly.csv")
        return tuple(paths)

    learned = {name: outputs(path) for name, path in run_directories.items()}
    if settings["train_dataset"] != settings["dataset"]:
        learned = {
            name: (*paths, run_directories[name] / "source.json") for name, paths in learned.items()
        }
    references = {controller: outputs(path) for controller, path in reference_directories.items()}
    return learned, references


def _training_directories(settings, algorithm):
    """Locate saved training artifacts independently of evaluation destinations."""
    return {
        name: settings["root"] / "results" / settings["train_dataset"] / algorithm / name
        for name in settings["run_names"]
    }


def _validate_sources(settings, algorithm, run_directories):
    """Validate provenance for each selected cross-dataset result triple."""
    if settings["train_dataset"] == settings["dataset"]:
        return
    for name, directory in run_directories.items():
        expected = evaluation_source(
            settings["train_dataset"], settings["dataset"], algorithm, name
        )
        for filename in (f"{settings['split']}.csv", f"{settings['split']}_hourly.csv"):
            if not (directory / filename).is_file():
                raise ValueError(f"missing evaluation result: {directory / filename}")
        validate_evaluation_source(directory, expected)


def _resolve_one(settings, simulator, algorithm, run_dir):
    """Resolve one saved run against the currently selected physical task."""
    sequential = settings["protocol"] == "sequential"
    observation = "day_observation" if sequential else "voltage_observation"
    size = day_observation_size(simulator) if sequential else voltage_observation_size(simulator)
    return resolve_run_settings(
        run_dir,
        feeder=settings["feeder"],
        dataset=settings["train_dataset"],
        algorithm=algorithm,
        observation_name=observation,
        observation_size=size,
        action_count=simulator.action_count,
    )


def _validated_output_rows(settings, simulator, days, label, paths):
    """Read and validate one method's result files against selected inputs."""
    if settings["protocol"] == "one_shot":
        rows = _read_csv(paths[0])
        expected = [scenario.id for scenario in flatten_hours(days)]
        if not rows or tuple(rows[0]) != static_columns(simulator.action_count):
            raise RuntimeError(f"{label} has an unexpected result schema.")
        if [row["scenario"] for row in rows] != expected:
            raise RuntimeError(f"{label} does not contain the expected scenarios in order.")
        return rows, None

    daily = _read_csv(paths[0])
    hourly = _read_csv(paths[1])
    day_ids = [day.id for day in days]
    expected_hours = [(day.id, hour) for day in days for hour in range(24)]
    if not daily or tuple(daily[0]) != DAY_COLUMNS:
        raise RuntimeError(f"{label} has an unexpected daily result schema.")
    if not hourly or tuple(hourly[0]) != hour_columns(simulator.action_count):
        raise RuntimeError(f"{label} has an unexpected hourly result schema.")
    if [row["day"] for row in daily] != day_ids:
        raise RuntimeError(f"{label} does not contain the expected days in order.")
    actual_hours = [(row["day"], int(row["hour"])) for row in hourly]
    if actual_hours != expected_hours:
        raise RuntimeError(f"{label} does not contain the expected hours in order.")
    if any(float(row["beta"]) != settings["beta"] for row in [*daily, *hourly]):
        raise RuntimeError(f"{label} does not contain the selected beta.")
    by_day = {}
    for row in hourly:
        by_day.setdefault(row["day"], []).append(row)
    for day_id, day_row in zip(day_ids, daily, strict=True):
        hours = by_day[day_id]
        checks = (
            (int(day_row["tap_steps"]), sum(int(row["tap_steps"]) for row in hours)),
            (int(day_row["feasible_hours"]), sum(int(row["feasible"]) for row in hours)),
        )
        if any(left != right for left, right in checks):
            raise RuntimeError(f"{label} has inconsistent counts for {day_id}.")
        voltage = sum(float(row["voltage_cost"]) for row in hours)
        combined = sum(float(row["hourly_cost"]) for row in hours)
        if abs(float(day_row["voltage_cost"]) - voltage) > 1e-12:
            raise RuntimeError(f"{label} has inconsistent voltage cost for {day_id}.")
        if abs(float(day_row["daily_cost"]) - combined) > 1e-12:
            raise RuntimeError(f"{label} has inconsistent daily cost for {day_id}.")
        mean_mad = sum(float(row["mad"]) for row in hours) / len(hours)
        if abs(float(day_row["mean_mad"]) - mean_mad) > 1e-12:
            raise RuntimeError(f"{label} has inconsistent mean absolute deviation for {day_id}.")
    return daily, hourly


def _preflight(
    settings,
    simulator,
    days,
    algorithm,
    run_directories,
    reference_directories,
):
    """Check all inputs, saved settings, checkpoints, and output boundaries.

    Training mode requires its input split and protects existing run directories.
    Evaluate mode requires compatible run settings and both checkpoints. View mode
    requires all selected result files. Without overwrite, writing modes reject
    existing learned output. Compatible reference outputs are always reused.
    Missing references are scheduled, while malformed or partial references
    require overwrite before their controller is scheduled. All detected
    problems are reported together before command execution.
    """
    errors = []
    resolved_settings = {}
    learned_outputs, reference_outputs = _result_paths(
        settings, run_directories, reference_directories
    )
    all_outputs = [path for paths in learned_outputs.values() for path in paths]
    all_outputs.extend(path for paths in reference_outputs.values() for path in paths)
    mode = settings["mode"]
    cross_dataset = settings["train_dataset"] != settings["dataset"]
    if mode == "evaluate" or (mode == "view" and not cross_dataset):
        for directory in run_directories.values():
            guard_evaluation_directory(directory, cross_dataset=cross_dataset)
    learner = LEARNERS[(settings["protocol"], settings["algorithm"])]
    if mode == "train":
        try:
            read_days(settings["dataset"], "train", simulator.load_names, settings["feeder"])
        except (OSError, ValueError) as error:
            errors.append(f"training split: {error}")
        if not settings["overwrite"]:
            for name, path in run_directories.items():
                if path.exists():
                    errors.append(
                        f"existing training run {name}: {path.relative_to(settings['root'])}"
                    )
    else:
        for name, path in _training_directories(settings, algorithm).items():
            if mode == "evaluate":
                for checkpoint in learner["checkpoints"]:
                    candidate = path / "checkpoints" / checkpoint
                    if not candidate.is_file():
                        relative = candidate.relative_to(settings["root"])
                        errors.append(f"missing checkpoint for run {name}: {relative}")
            if settings["protocol"] == "sequential" and not (path / "train.csv").is_file():
                errors.append(
                    f"missing training record for run {name}: "
                    f"{(path / 'train.csv').relative_to(settings['root'])}"
                )
            try:
                task, layers = _resolve_one(settings, simulator, algorithm, path)
                resolved_settings[name] = (task, layers)
                requested = {
                    "name": settings["reward"],
                    "parameters": settings["reward_parameters"],
                }
                if task["reward"] != requested:
                    errors.append(
                        f"reward mismatch for run {name}: recorded {task['reward']}, "
                        f"requested {requested}"
                    )
                if settings["protocol"] == "sequential":
                    benchmark = task["evaluation"].get("beta")
                    if benchmark != settings["beta"]:
                        errors.append(
                            f"benchmark beta mismatch for run {name}: recorded {benchmark}, "
                            f"requested {settings['beta_text']}"
                        )
            except (OSError, ValueError) as error:
                errors.append(f"saved settings for run {name}: {error}")

    if mode in {"train", "evaluate"} and not settings["overwrite"]:
        for paths in learned_outputs.values():
            existing = [path for path in paths if path.exists()]
            if existing:
                rendered = ", ".join(str(path.relative_to(settings["root"])) for path in paths)
                description = "protected output pair" if len(paths) == 2 else "existing output"
                errors.append(f"{description}: {rendered}")
    reference_controllers = []
    if mode in {"train", "evaluate"}:
        for controller, paths in reference_outputs.items():
            present = [path.is_file() for path in paths]
            if not any(present):
                reference_controllers.append(controller)
                continue
            if not all(present):
                if settings["overwrite"]:
                    reference_controllers.append(controller)
                else:
                    rendered = ", ".join(str(path.relative_to(settings["root"])) for path in paths)
                    errors.append(f"partial reference output pair: {rendered}")
                continue
            try:
                _validated_output_rows(
                    settings,
                    simulator,
                    days,
                    REFERENCE_LABELS[controller],
                    paths,
                )
            except (OSError, RuntimeError, ValueError) as error:
                if settings["overwrite"]:
                    reference_controllers.append(controller)
                else:
                    errors.append(f"incompatible reference {controller}: {error}")
    if mode == "view":
        for path in all_outputs:
            if not path.is_file():
                errors.append(f"missing result: {path.relative_to(settings['root'])}")
        try:
            _validate_sources(settings, algorithm, run_directories)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    if errors:
        if mode == "train":
            action = "Choose new run names or set OVERWRITE = True after checking the paths."
        elif mode == "evaluate":
            action = (
                "Set MODE = 'train' for missing run artifacts, or set OVERWRITE = True "
                "to replace existing evaluation outputs."
            )
        else:
            action = "Set MODE = 'evaluate' to create missing results, or correct the run settings."
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors) + f"\n{action}")
    return resolved_settings, tuple(reference_controllers)


def _print_settings(settings, evaluation_days, run_directories, reference_directories):
    """Print the selected dataset, evaluation scope, run mapping, and mode."""
    scenarios = len(evaluation_days) * 24
    print(f"Dataset: {settings['dataset']} (data/scenarios/{settings['dataset']})")
    print(f"Split: {settings['split']}")
    print(f"Evaluated: {len(evaluation_days)} days, {scenarios} hours")
    print(f"Seeds: {list(settings['seeds'])}")
    if settings["protocol"] == "sequential":
        if settings["mode"] == "train":
            print(f"Training days per seed: {settings['training_days']}")
        else:
            print("Training days per seed: recorded in saved training records")
        print(f"Switching-cost weight beta: {settings['beta_text']}")
    print(f"Reward: {settings['reward']} {settings['reward_parameters']}")
    print("Display names and run directories:")
    algorithm_label = LEARNERS[(settings["protocol"], settings["algorithm"])]["label"]
    for seed, name in zip(settings["seeds"], settings["run_names"], strict=True):
        path = run_directories[name].relative_to(settings["root"])
        print(f"  {algorithm_label} seed {seed}: {path}/")
    for controller, label in REFERENCE_LABELS.items():
        path = reference_directories[controller].relative_to(settings["root"])
        print(f"  {label}: {path}/")
    print(f"Mode: {settings['mode']}")
    print(f"Overwrite: {settings['overwrite']}")
    if settings["evaluation_days"] is not None:
        print(
            f"*** LIMITED EVALUATION: first {len(evaluation_days)} days of {settings['split']} ***"
        )


def _print_cell(settings, run_directories):
    """Print one line naming a matrix cell's reward and run directories."""
    paths = ", ".join(
        f"{run_directories[name].relative_to(settings['root'])}/" for name in settings["run_names"]
    )
    print(
        f"Cell {settings['run_name']}: reward {settings['reward']} "
        f"{settings['reward_parameters']}; runs {paths}"
    )


def _timed_command(arguments, label, root, durations, *, log=print):
    """Run one child command, record elapsed wall time, and return its output."""
    started = time.monotonic()
    lines = commands.run_command(arguments, root, log=log)
    duration = time.monotonic() - started
    durations[label] = duration
    print(f"{label} wall time: {duration:.3f} seconds")
    return lines


def _training_arguments(settings, seed, name):
    """Build the trainer argument list for one selected seed and run name."""
    protocol = settings["protocol"]
    learner = LEARNERS[(protocol, settings["algorithm"])]
    arguments = [
        sys.executable,
        str(settings["root"] / learner["trainer"]),
        "--dataset",
        settings["dataset"],
        "--seed",
        str(seed),
    ]
    if protocol == "one_shot":
        arguments.extend(["--episodes", str(settings["episodes"])])
    else:
        arguments.extend(["--days", str(settings["training_days"])])
    arguments.extend(["--reward", settings["reward"]])
    for parameter, value in settings["reward_parameters"].items():
        arguments.extend(["--" + parameter.replace("_", "-"), str(value)])
    if protocol == "sequential":
        arguments.extend(["--benchmark-beta", settings["beta_text"]])
    arguments.extend(["--run", name])
    if settings["overwrite"]:
        arguments.append("--overwrite")
    return arguments


def _evaluation_arguments(settings, name):
    """Build the learned-controller evaluation arguments for one saved run."""
    learner = LEARNERS[(settings["protocol"], settings["algorithm"])]
    arguments = [
        sys.executable,
        str(settings["root"] / learner["evaluator"]),
        "--dataset",
        settings["dataset"],
        "--run",
        name,
        "--split",
        settings["split"],
    ]
    if settings["train_dataset"] != settings["dataset"]:
        arguments.extend(["--train-dataset", settings["train_dataset"]])
    if settings["evaluation_days"] is not None:
        arguments.extend(["--days", str(settings["evaluation_days"])])
    if settings["overwrite"]:
        arguments.append("--overwrite")
    return arguments


def _reference_arguments(settings, controllers):
    """Build reference evaluation arguments for the selected protocol."""
    arguments = [
        sys.executable,
        str(settings["root"] / PROTOCOLS[settings["protocol"]]["reference_evaluator"]),
        "--dataset",
        settings["dataset"],
        "--controller",
    ]
    if settings["protocol"] == "one_shot":
        arguments.extend(controllers)
    else:
        arguments.extend([*controllers, "--beta", settings["beta_text"]])
    arguments.extend(["--split", settings["split"]])
    if settings["evaluation_days"] is not None:
        arguments.extend(["--days", str(settings["evaluation_days"])])
    if settings["overwrite"]:
        arguments.append("--overwrite")
    return arguments


def _evaluation_log(line):
    print(line.replace(", feasible ", ", feasible at [0.95, 1.05] "))


def _run_commands(settings, durations, reference_controllers):
    """Execute only the training and evaluation stages selected by ``mode``.

    ``view`` launches nothing. ``evaluate`` skips training. ``train`` completes
    every seed's training before evaluating learned and reference controllers.
    Child failures and interruptions propagate from ``common.commands``.
    """
    root = settings["root"]
    algorithm_label = LEARNERS[(settings["protocol"], settings["algorithm"])]["label"]
    if settings["mode"] == "train":
        for seed, name in zip(settings["seeds"], settings["run_names"], strict=True):
            print(f"Training {name}")
            _timed_command(
                _training_arguments(settings, seed, name),
                f"Training {name}",
                root,
                durations,
            )
    if settings["mode"] in {"train", "evaluate"}:
        for seed, name in zip(settings["seeds"], settings["run_names"], strict=True):
            label = f"{algorithm_label} seed {seed}"
            print(f"Evaluating {label}")
            _timed_command(
                _evaluation_arguments(settings, name),
                f"Evaluation {label}",
                root,
                durations,
                log=_evaluation_log,
            )
        if settings["protocol"] == "one_shot":
            for controller in reference_controllers:
                label = REFERENCE_LABELS[controller]
                print(f"Evaluating {label}")
                _timed_command(
                    _reference_arguments(settings, (controller,)),
                    f"Evaluation {label}",
                    root,
                    durations,
                    log=_evaluation_log,
                )
        elif reference_controllers:
            label = " and ".join(REFERENCE_LABELS[name] for name in reference_controllers)
            print(f"Evaluating {label}")
            _timed_command(
                _reference_arguments(settings, reference_controllers),
                f"Evaluation {label}",
                root,
                durations,
                log=_evaluation_log,
            )


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _read_settings_and_training(
    settings, simulator, algorithm, run_directories, labels, resolved_settings
):
    """Read each saved task, network widths, and optional training history."""
    tasks = {}
    layer_sizes = {}
    records = {}
    counts = {}
    update_counts = {}
    for name, label in zip(settings["run_names"], labels, strict=True):
        directory = run_directories[name]
        task, layers = resolved_settings.get(name) or _resolve_one(
            settings, simulator, algorithm, directory
        )
        tasks[label] = task
        layer_sizes[label] = tuple(layers)
        training_path = directory / "train.csv"
        if training_path.is_file():
            records[label] = _read_csv(training_path)
            counts[label] = len(records[label])
        else:
            records[label] = None
            counts[label] = None
        update_path = directory / "counts.json"
        update_counts[label] = _read_json(update_path) if update_path.is_file() else None
    return tasks, layer_sizes, records, counts, update_counts


def _read_one_shot(settings, simulator, days, run_directories, reference_directories, labels):
    """Read static outputs and require identical ordered scenario identities.

    Alignment pairs every learned and reference row on the same selected hourly
    input before summaries or exact paired counts are calculated.
    """
    rows = {}
    for name, label in zip(settings["run_names"], labels, strict=True):
        paths = (run_directories[name] / f"{settings['split']}.csv",)
        rows[label], _unused = _validated_output_rows(settings, simulator, days, label, paths)
    for controller, label in REFERENCE_LABELS.items():
        paths = (reference_directories[controller] / f"{settings['split']}.csv",)
        rows[label], _unused = _validated_output_rows(settings, simulator, days, label, paths)
    print(
        f"Scenario alignment passed for {len(tuple(flatten_hours(days)))} hours "
        f"and {len(rows)} methods."
    )
    return rows


def _read_sequential(settings, simulator, days, run_directories, reference_directories, labels):
    """Read and validate aligned daily and hourly outputs for every method.

    Checks cover schemas, day-hour order, selected benchmark beta, and
    reconstruction of each daily row from its hourly rows. Counts must match
    exactly; voltage cost, daily cost, and mean absolute deviation must agree
    within an absolute 1e-12.
    """
    daily = {}
    hourly = {}
    for name, label in zip(settings["run_names"], labels, strict=True):
        directory = run_directories[name]
        paths = (
            directory / f"{settings['split']}.csv",
            directory / f"{settings['split']}_hourly.csv",
        )
        daily[label], hourly[label] = _validated_output_rows(
            settings, simulator, days, label, paths
        )
    for controller, label in REFERENCE_LABELS.items():
        directory = reference_directories[controller]
        paths = (
            directory / f"{settings['split']}.csv",
            directory / f"{settings['split']}_hourly.csv",
        )
        daily[label], hourly[label] = _validated_output_rows(
            settings, simulator, days, label, paths
        )
    day_ids = [day.id for day in days]
    expected_hours = [(day.id, hour) for day in days for hour in range(24)]
    print(
        f"Day and hour alignment passed for {len(day_ids)} days, "
        f"{len(expected_hours)} hours, and {len(daily)} methods."
    )
    return daily, hourly


def run_experiment(settings, *, verbose=True):
    """Validate, execute the selected mode, and return aligned saved results.

    Current selections determine the feeder, dataset, protocol, split, seeds,
    output locations, and requested benchmark. Saved run records determine each
    learned controller's recorded reward, action semantics, and network widths.
    Preflight completes before ``train`` or ``evaluate`` launches commands.
    ``view`` launches none. After execution, the function verifies that all
    controller rows describe matching inputs and returns them for report-only
    functions that never retrain or reevaluate. With ``verbose=False`` one line
    replaces the settings block.
    """
    selected = _validate_settings(settings)
    simulator = FeederSimulator(selected["feeder"])
    days = read_days(
        selected["dataset"], selected["split"], simulator.load_names, selected["feeder"]
    )
    if selected["evaluation_days"] is not None:
        days = days[: selected["evaluation_days"]]
    if selected["protocol"] == "sequential" and selected["plot_day"] >= len(days):
        raise ValueError(f"PLOT_DAY must be less than the {len(days)} selected evaluation days.")
    algorithm, run_directories, reference_directories = _paths(selected)
    algorithm_label = LEARNERS[(selected["protocol"], selected["algorithm"])]["label"]
    labels = tuple(f"{algorithm_label} seed {seed}" for seed in selected["seeds"])
    method_labels = (*labels, *REFERENCE_LABELS.values())
    if verbose:
        _print_settings(selected, days, run_directories, reference_directories)
    else:
        _print_cell(selected, run_directories)
    resolved_settings, reference_controllers = _preflight(
        selected,
        simulator,
        days,
        algorithm,
        run_directories,
        reference_directories,
    )
    print("Preflight passed. No subprocess has started before this point.")
    durations = {}
    if selected["mode"] == "view":
        print("Using existing saved results. View mode launches no subprocess.")
    _run_commands(selected, durations, reference_controllers)
    _validate_sources(selected, algorithm, run_directories)
    tasks, layers, training_records, training_counts, update_counts = _read_settings_and_training(
        selected,
        simulator,
        algorithm,
        _training_directories(selected, algorithm),
        labels,
        resolved_settings,
    )
    scenarios = tuple(flatten_hours(days)) if selected["protocol"] == "one_shot" else ()
    result_rows = {}
    daily_rows = {}
    hourly_rows = {}
    if selected["protocol"] == "one_shot":
        result_rows = _read_one_shot(
            selected, simulator, days, run_directories, reference_directories, labels
        )
    else:
        daily_rows, hourly_rows = _read_sequential(
            selected, simulator, days, run_directories, reference_directories, labels
        )
    return ExperimentResults(
        settings=selected,
        root=selected["root"],
        simulator=simulator,
        action_count=simulator.action_count,
        evaluation_days=tuple(days),
        evaluation_scenarios=scenarios,
        run_names=selected["run_names"],
        algorithm_label=algorithm_label,
        learned_labels=labels,
        method_labels=method_labels,
        run_directories=run_directories,
        reference_directories=reference_directories,
        run_tasks=tasks,
        layer_sizes=layers,
        training_records=training_records,
        training_counts=training_counts,
        update_counts=update_counts,
        result_rows=result_rows,
        daily_rows=daily_rows,
        hourly_rows=hourly_rows,
        durations=durations,
    )


def check_matrix_identity(control, cells):
    """Require every matrix result to share the control's evaluation identity.

    Results must agree in dataset, split, protocol, algorithm, seeds, benchmark
    beta, and the ordered evaluation day identities.
    """
    keys = ("dataset", "split", "protocol", "algorithm", "seeds", "beta")

    def identity(results):
        selected = {key: results.settings[key] for key in keys}
        selected["days"] = tuple(day.id for day in results.evaluation_days)
        return selected

    expected = identity(control)
    for key, results in cells.items():
        actual = identity(results)
        if actual != expected:
            differing = sorted(name for name in expected if actual[name] != expected[name])
            raise ValueError(f"matrix cell {key} differs from the control in {differing}")


def run_experiment_matrix(base, cells, *, rate):
    """Read the alpha-zero control, run every listed cell, and return all results.

    ``base`` is a complete settings dictionary without ``reward_parameters``; its
    ``run_name`` names the control. ``cells`` lists ``(deviation, alpha, run_name)``
    triples. The control always runs in ``view`` mode with only the sequential
    benchmark beta as a reward override, so the matrix never trains or replaces
    it. Each cell runs in ``base["mode"]`` under its own run name. The function
    returns ``(control, cells)`` with ``cells`` keyed by ``(deviation, alpha)`` in
    the given order.
    """
    if not isinstance(base, dict) or "reward_parameters" in base:
        raise ValueError("The matrix base settings must not contain reward_parameters.")
    keys = [(deviation, alpha) for deviation, alpha, _name in cells]
    names = [name for _deviation, _alpha, name in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("CELLS must not repeat a (deviation, alpha) pair.")
    if len(set(names)) != len(names):
        raise ValueError("CELLS must not repeat a run name.")
    sequential = base.get("protocol") == "sequential"
    control_parameters = {"beta": base.get("beta")} if sequential else {}
    try:
        control = run_experiment(
            {**base, "mode": "view", "reward_parameters": control_parameters}, verbose=False
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}\nThe control runs are missing or incompatible. "
            "Run the control notebook first; the matrix only reads them."
        ) from error
    results = {}
    for deviation, alpha, name in cells:
        parameters = {
            **control_parameters,
            "deviation": deviation,
            "alpha": alpha,
            "rate": rate,
        }
        results[(deviation, alpha)] = run_experiment(
            {**base, "run_name": name, "reward_parameters": parameters}, verbose=False
        )
    check_matrix_identity(control, results)
    return control, results
