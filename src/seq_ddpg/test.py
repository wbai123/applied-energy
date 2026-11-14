"""Evaluate a saved sequential DDPG actor and write daily and hourly metrics for complete days."""

import pathlib
import sys

from common.cli import (
    ArgumentParser,
    beta_value,
    dataset_name,
    display_path,
    evaluation_source,
    guard_evaluation_directory,
    positive_integer,
    result_directory,
    run_name,
    write_cross_dataset_results,
)
from common.evaluation import run_day_test
from common.observations import day_observation, day_observation_size
from common.outputs import day_summary, write_day_test, write_hour_test
from ddpg.checkpoints import load_agent_for_run
from feeder.envs import DayEnv
from feeder.loads import dataset_feeder, read_days
from feeder.simulator import FeederSimulator
from feeder.taps import PHYSICAL_RANGE, project

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args():
    """Parse the saved run, split, day limit, benchmark override, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--train-dataset", type=dataset_name)
    parser.add_argument("--run", type=run_name, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--days", type=positive_integer)
    parser.add_argument("--benchmark-beta", type=beta_value, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Load a compatible actor, evaluate complete days, and write paired CSVs.

    The deterministic actor acts on each next-hour observation while executed
    taps persist through the day. The saved benchmark beta is the default; an
    override changes output filenames and evaluation costs, not the saved reward.
    Both output files are protected as a pair unless ``--overwrite`` is present.
    """
    args = parse_args()
    try:
        train_dataset = args.train_dataset or args.dataset
        if train_dataset != args.dataset and args.benchmark_beta is not None:
            raise ValueError("--benchmark-beta cannot be combined with a differing --train-dataset")
        source = evaluation_source(train_dataset, args.dataset, "seq_ddpg", args.run)
        run_dir = result_directory(ROOT, train_dataset, "seq_ddpg", args.run)
        output_dir = result_directory(ROOT, args.dataset, "seq_ddpg", args.run)
        guard_evaluation_directory(output_dir, cross_dataset=source is not None)
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        selected_days = read_days(args.dataset, args.split, simulator.load_names, simulator.feeder)
        if args.days is not None:
            selected_days = selected_days[: args.days]
        suffix = ""
        if source is None and args.benchmark_beta is not None:
            suffix = f"_b{str(args.benchmark_beta)}"
        output = output_dir / f"{args.split}{suffix}.csv"
        hour_output = output_dir / f"{args.split}{suffix}_hourly.csv"
        destinations = (output, hour_output)
        if source is not None:
            destinations += (output_dir / "source.json",)
        if not args.overwrite:
            for destination in destinations:
                if destination.exists():
                    raise ValueError(
                        f"{display_path(destination, ROOT)} exists; use --overwrite to replace it"
                    )
        env = DayEnv(simulator)
        agent, task = load_agent_for_run(
            run_dir,
            feeder=simulator.feeder,
            dataset=train_dataset,
            algorithm="seq_ddpg",
            observation_name="day_observation",
            observation_size=day_observation_size(simulator),
            action_count=env.action_count,
            action_bounds=PHYSICAL_RANGE,
        )
        beta = task["evaluation"]["beta"] if args.benchmark_beta is None else args.benchmark_beta
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"days: {len(selected_days)}")
    print(f"benchmark beta: {beta}")

    def controller(state):
        positions, _ratios = project(agent.choose_action_deterministic(day_observation(state)))
        return positions

    rows, hour_rows = run_day_test(env, controller, selected_days, beta)
    if source is None:
        write_day_test(rows, output)
        write_hour_test(hour_rows, hour_output, env.action_count)
    else:
        write_cross_dataset_results(
            rows, hour_rows, env.action_count, output, hour_output, source, ROOT
        )
    print(f"wrote {display_path(output, ROOT)}")
    print(f"wrote {display_path(hour_output, ROOT)}")
    print(day_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
