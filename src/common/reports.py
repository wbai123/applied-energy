"""Display notebook reports from validated, aligned experiment results.

These functions read the supplied ``ExperimentResults`` and feeder configuration.
They do not train, evaluate, change saved outputs, or claim performance beyond
the selected split.
"""

import json
import statistics

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from common.display import show_table
from common.outputs import daily_summary, paired_counts, static_summary
from common.plots import (
    SEED_MEAN_CAPTION,
    hourly_detail_rows,
    plot_cost_feasibility,
    plot_day_ratios,
    plot_hourly_steps,
    plot_hourly_voltage,
    plot_static_taps,
    plot_training,
    tap_ratio_fields,
)
from common.workflow import check_matrix_identity
from feeder.config import read_config
from feeder.taps import N_TAPS, ratio_of_index

FIXED_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
REFERENCE_COLORS = {"Neutral": "#7f7f7f", "RegControl": "#d62728"}
DEVIATION_COLORS = {"abs": "#1f77b4", "sq": "#ff7f0e", "exp": "#2ca02c"}
TRAINING_COST_CAPTION = (
    "Cost uses the fixed evaluation metric on training episodes and includes exploration. "
    "It is not held-out performance or proof of convergence."
)
# Rows are (metric, format, paired, higher_is_better). The first paired row is the
# voltage cost used by the all-seeds count.
MATRIX_METRICS = {
    "one_shot": (
        ("Mean hourly voltage cost", ".3e", True, False),
        ("Mean absolute deviation (p.u.)", ".4e", True, False),
        ("Feasible scenarios (%)", ".1f", False, True),
        ("Mean violating voltage measurements/scenario", ".2f", False, False),
    ),
    "sequential": (
        ("Mean daily combined cost", ".5e", False, False),
        ("Mean daily voltage cost", ".5e", True, False),
        ("Mean absolute deviation (p.u.)", ".4e", True, False),
        ("Mean tap steps/day", ".3f", True, False),
        ("Feasible hours (%)", ".1f", False, True),
    ),
}


def _seed_label(results):
    """Keep the single-seed label or name the selected seeds' arithmetic mean."""
    if len(results.learned_labels) == 1:
        return results.learned_labels[0]
    return f"{results.algorithm_label} seed mean"


def _colors(results):
    """Assign one learned color and the established reference colors."""
    seed = results.settings["seeds"][0]
    return {_seed_label(results): FIXED_COLORS.get(seed, "#1f77b4"), **REFERENCE_COLORS}


def _linestyles(results):
    """Give the plotted learned and reference series distinct line styles."""
    return {_seed_label(results): "-", "Neutral": "--", "RegControl": "-."}


def _mean_series(results, values, references=("Neutral", "RegControl")):
    """Average already aligned numeric seed values and retain selected references."""
    return {
        _seed_label(results): np.mean([values[label] for label in results.learned_labels], axis=0),
        **{label: values[label] for label in references},
    }


def _numeric_series(results, rows, fields, *, ratios=False, references=("Neutral", "RegControl")):
    """Prepare field-by-point arrays, converting positions before averaging ratios."""
    values = {
        label: np.asarray([[float(row[field]) for row in rows[label]] for field in fields])
        for label in (*results.learned_labels, *references)
    }
    if ratios:
        values = {label: ratio_of_index(value) for label, value in values.items()}
    return _mean_series(results, values, references)


def _summary_series(results):
    """Average saved per-run metric summaries for the cost and feasibility bars."""
    summaries = _summaries(results)
    columns = tuple(summaries[results.learned_labels[0]])
    values = {
        label: [summary[column] for column in columns] for label, summary in summaries.items()
    }
    return {
        label: dict(zip(columns, values, strict=True))
        for label, values in _mean_series(results, values).items()
    }


def training_series(results, *, column=None):
    """Align equal episode indices, average the common prefix, and report unequal lengths."""
    histories = [results.training_records[label] for label in results.learned_labels]
    if any(not rows for rows in histories):
        print(f"{_seed_label(results)}: training curve unavailable")
        return None
    lengths = [len(rows) for rows in histories]
    count = min(lengths)
    if len(set(lengths)) > 1:
        print(f"{_seed_label(results)}: record lengths {lengths}; using common prefix {count}.")
    episodes = np.asarray([int(row["episode"]) for row in histories[0][:count]])
    for rows in histories[1:]:
        if not np.array_equal(episodes, [int(row["episode"]) for row in rows[:count]]):
            raise ValueError("training curves require matching episode indices")
    if column is None:
        column = "daily_cost" if results.settings["protocol"] == "sequential" else "voltage_cost"
    costs = np.mean([[float(row[column]) for row in rows[:count]] for rows in histories], axis=0)
    return episodes, costs


def _mean_caption(results):
    """Persist the ratio and movement qualification as notebook Markdown output."""
    if len(results.learned_labels) > 1:
        display(Markdown(SEED_MEAN_CAPTION))


def record_rows(results):
    """Build rows from saved reward, benchmark, architecture, and history data."""
    unit = "days" if results.settings["protocol"] == "sequential" else "episodes"
    rows = []
    for label in results.learned_labels:
        task = results.run_tasks[label]
        rows.append(
            {
                "Run": label,
                "Reward": task["reward"]["name"],
                "Reward parameters": task["reward"]["parameters"],
                "Benchmark beta": task["evaluation"].get("beta"),
                "Layer sizes": results.layer_sizes[label],
                f"Completed training {unit}": (
                    results.training_counts[label]
                    if results.training_counts[label] is not None
                    else "unavailable"
                ),
                "Update counts": results.update_counts[label] or "unavailable",
            }
        )
    return rows


def show_training_summary(results):
    """Display recorded run settings, completed counts, and physical cost curves.

    Sequential episode counts represent complete days and are also reported as
    24 interactions per completed day. Curves include exploratory training data;
    they are not held-out performance or evidence of convergence.
    """
    sequential = results.settings["protocol"] == "sequential"
    title = (
        "Recorded reward and benchmark beta by run"
        if sequential
        else "Recorded training settings by run"
    )
    show_table(record_rows(results), title=title)
    completed = dict(
        zip(
            results.run_names,
            (results.training_counts[label] for label in results.learned_labels),
            strict=True,
        )
    )
    if sequential:
        for name, count in completed.items():
            if count is not None:
                print(f"{name}: {count} completed episodes, {24 * count} environment interactions")
    else:
        print("Completed training episodes:", completed)
    return show_training_cost(results)


def show_training_cost(results):
    """Display the physical training cost curve and its interpretation without adding a table."""
    series = training_series(results)
    if series is None:
        return None
    sequential = results.settings["protocol"] == "sequential"
    figure, axis = plot_training(
        {_seed_label(results): series}, sequential, _colors(results), _linestyles(results)
    )
    plt.show()
    display(Markdown(TRAINING_COST_CAPTION))
    return figure, axis


def _summaries(results):
    """Calculate one aggregate physical summary per aligned controller result."""
    if results.settings["protocol"] == "one_shot":
        return {
            label: static_summary(results.result_rows[label]) for label in results.method_labels
        }
    beta = results.settings["beta"]
    return {
        label: daily_summary(results.daily_rows[label], beta) for label in results.method_labels
    }


def _metric_format(name):
    """Return scientific notation for cost metrics and general notation otherwise."""
    return ".5e" if "cost" in name else ".6g"


def _seed_statistics(results, summaries, columns, formats):
    """Display mean and sample standard deviation across trained seed runs.

    These values describe variation across run-level aggregates. They are not
    across-day variation and do not form confidence intervals.
    """
    if len(results.learned_labels) < 2:
        print("Across-seed summary skipped because one seed is selected.")
        return
    rows = []
    for name, function in (
        ("Across-seed mean", statistics.fmean),
        ("Sample standard deviation", statistics.stdev),
    ):
        row = {f"{results.algorithm_label} statistic": name}
        for column, format_spec in zip(columns, formats, strict=True):
            value = function(summaries[label][column] for label in results.learned_labels)
            row[column] = format(value, format_spec)
        rows.append(row)
    show_table(rows, title=f"Across-seed {results.algorithm_label} summary")


def _controller_explanation(results, *, sequential):
    """Display feeder-specific measurement and operational-cost definitions."""
    count = len(results.simulator.measurement_names)
    convention = read_config(results.settings["feeder"]).voltage_convention
    references = (
        "Neutral holds every regulator ratio at 1.0. RegControl uses the feeder's native "
        "regulator control."
    )
    if sequential:
        definition = (
            "Hourly voltage cost is the mean squared excursion outside [0.95, 1.05] p.u. "
            f"over {count} monitored {convention} voltage measurements; in-band measurements "
            "contribute zero. Daily voltage cost sums 24 hourly costs, and the table averages "
            "these daily totals across evaluation days. Lower cost is better. Beta weights tap "
            "movement; it is not a monetary price."
        )
    else:
        definition = (
            "Hourly voltage cost is the mean squared excursion outside [0.95, 1.05] p.u. "
            f"over {count} monitored {convention} voltage measurements; in-band measurements "
            "contribute zero. Lower cost is better."
        )
    display(Markdown(references + "\n\n" + definition))


def _show_controller_comparison_oneshot(results):
    """Display static aggregates, seed statistics, and cost-feasibility plots."""
    summaries = _summaries(results)
    _controller_explanation(results, sequential=False)
    rows = [
        {
            "Method": label,
            "Mean hourly voltage cost": f"{values['Mean hourly voltage cost']:.3e}",
            "Mean absolute deviation (p.u.)": f"{values['Mean absolute deviation (p.u.)']:.4e}",
            "Feasible scenarios (%)": f"{values['Feasible scenarios (%)']:.1f}",
            "Mean violating voltage measurements/scenario": (
                f"{values['Mean violating voltage measurements/scenario']:.2f}"
            ),
            "Scenarios": values["Scenarios"],
        }
        for label, values in summaries.items()
    ]
    show_table(rows, title=f"Overall results on {results.settings['split']}")
    columns = (
        "Mean hourly voltage cost",
        "Mean absolute deviation (p.u.)",
        "Feasible scenarios (%)",
        "Mean violating voltage measurements/scenario",
    )
    _seed_statistics(results, summaries, columns, (".3e", ".4e", ".1f", ".2f"))
    _figure, axes = plot_cost_feasibility(
        _summary_series(results),
        results.settings["split"],
        _colors(results),
        _linestyles(results),
    )
    scale = axes[0].get_yscale()
    plt.show()
    if scale == "log":
        display(
            Markdown(
                "The voltage-cost panel uses a log scale because every displayed cost "
                "is positive. Log spacing shows multiplicative differences."
            )
        )
    else:
        display(
            Markdown(
                "The voltage-cost panel uses a linear scale because at least one "
                "displayed cost is zero."
            )
        )
    return _figure, axes


def _show_controller_comparison_seq(results):
    """Display daily aggregates and seed statistics under the selected beta."""
    summaries = _summaries(results)
    _controller_explanation(results, sequential=True)
    rows = [
        {
            "Method": label,
            "Mean daily combined cost": f"{values['Mean daily combined cost']:.5e}",
            "Mean daily voltage cost": f"{values['Mean daily voltage cost']:.5e}",
            "Mean daily switching cost": f"{values['Mean daily switching cost']:.5e}",
            "Mean absolute deviation (p.u.)": f"{values['Mean absolute deviation (p.u.)']:.4e}",
            "Mean tap steps/day": f"{values['Mean tap steps/day']:.3f}",
            "Feasible hours (%)": f"{values['Feasible hours (%)']:.1f}",
            "Days": values["Days"],
        }
        for label, values in summaries.items()
    ]
    show_table(rows, title=f"Overall results on {results.settings['split']}")
    columns = (
        "Mean daily combined cost",
        "Mean daily voltage cost",
        "Mean daily switching cost",
        "Mean absolute deviation (p.u.)",
        "Mean tap steps/day",
        "Feasible hours (%)",
    )
    _seed_statistics(results, summaries, columns, (".5e", ".5e", ".5e", ".4e", ".3f", ".1f"))


def show_controller_comparison(results):
    """Display physical summaries and across-seed variation for all methods.

    One-shot summaries average hourly scenarios. Sequential summaries first sum
    each complete day, then average those daily totals across evaluated days.
    """
    if results.settings["protocol"] == "one_shot":
        return _show_controller_comparison_oneshot(results)
    return _show_controller_comparison_seq(results)


def show_paired_counts(results):
    """Display exact paired lower, equal, and higher counts against references.

    Static rows pair matching scenarios; sequential rows pair matching days.
    Equality is exact and the counts do not express practical equivalence or
    effect size.
    """
    sequential = results.settings["protocol"] == "sequential"
    source = results.daily_rows if sequential else results.result_rows
    column = "daily_cost" if sequential else "cost"
    rows = []
    for label in results.learned_labels:
        for reference in ("Neutral", "RegControl"):
            lower, equal, higher = paired_counts(source[label], source[reference], column)
            suffix = "days" if sequential else "scenarios"
            rows.append(
                {
                    f"{results.algorithm_label} run": label,
                    "Reference": reference,
                    f"Lower-cost {suffix}": lower,
                    f"Equal-cost {suffix}": equal,
                    f"Higher-cost {suffix}": higher,
                }
            )
    title = (
        "Paired combined-cost counts on matching days"
        if sequential
        else "Paired voltage-cost counts on matching scenarios"
    )
    show_table(rows, title=title)


def _comparison_metrics(results, label):
    """Return per-run daily metrics and an explicit feasible-hour count."""
    rows = results.daily_rows[label]
    summary = daily_summary(rows, results.settings["beta"])
    feasible = sum(int(row["feasible_hours"]) for row in rows)
    return {
        "Mean daily voltage cost": summary["Mean daily voltage cost"],
        "Mean tap steps/day": summary["Mean tap steps/day"],
        "Mean daily switching cost": summary["Mean daily switching cost"],
        "Mean daily combined cost": summary["Mean daily combined cost"],
        "Feasible hours": feasible,
    }


def show_algorithm_comparison(left, right):
    """Compare two sequential learners on matching saved days and seeds.

    The comparison uses common physical costs only. Paired differences match
    equal seed numbers and day identities. Matching exogenous loads and neutral
    initial taps do not make the learners' observation trajectories identical.
    """
    if left.settings["protocol"] != "sequential" or right.settings["protocol"] != "sequential":
        raise ValueError("algorithm comparison requires sequential results")
    checks = (
        (left.settings["dataset"], right.settings["dataset"], "dataset"),
        (left.settings["split"], right.settings["split"], "split"),
        (left.settings["beta"], right.settings["beta"], "beta"),
        (left.settings["seeds"], right.settings["seeds"], "seeds"),
        (
            tuple(day.id for day in left.evaluation_days),
            tuple(day.id for day in right.evaluation_days),
            "days",
        ),
    )
    for left_value, right_value, name in checks:
        if left_value != right_value:
            raise ValueError(f"algorithm comparison requires matching {name}")
    if left.algorithm_label == right.algorithm_label:
        raise ValueError("algorithm comparison requires different algorithm labels")

    metric_names = (
        "Mean daily voltage cost",
        "Mean tap steps/day",
        "Mean daily switching cost",
        "Mean daily combined cost",
    )
    denominator = len(left.evaluation_days) * 24
    per_algorithm = {}
    rows = []
    paired_rows = []
    for results in (left, right):
        per_algorithm[results.algorithm_label] = []
        for seed, label in zip(results.settings["seeds"], results.learned_labels, strict=True):
            values = _comparison_metrics(results, label)
            per_algorithm[results.algorithm_label].append(values)
            rows.append(
                {
                    "Algorithm": results.algorithm_label,
                    "Seed": seed,
                    **{name: format(values[name], _metric_format(name)) for name in metric_names},
                    "Feasible hours": f"{values['Feasible hours']} / {denominator}",
                }
            )
    show_table(rows, title="Per-seed sequential algorithm comparison")

    aggregate_rows = []
    for algorithm, values in per_algorithm.items():
        row = {"Algorithm": algorithm}
        for name in (*metric_names, "Feasible hours"):
            selected = [entry[name] for entry in values]
            mean = statistics.fmean(selected)
            spread = statistics.stdev(selected) if len(selected) > 1 else None
            spec = _metric_format(name)
            text = (
                f"{mean:{spec}} ± {spread:{spec}}" if spread is not None else f"{mean:{spec}} ± n/a"
            )
            row[name] = f"{text} / {denominator}" if name == "Feasible hours" else text
        aggregate_rows.append(row)
    show_table(aggregate_rows, title="Across-seed mean ± sample standard deviation")

    for seed, left_label, right_label in zip(
        left.settings["seeds"], left.learned_labels, right.learned_labels, strict=True
    ):
        left_rows = left.daily_rows[left_label]
        right_rows = right.daily_rows[right_label]
        left_costs = [float(row["daily_cost"]) for row in left_rows]
        right_costs = [float(row["daily_cost"]) for row in right_rows]
        lower = sum(a < b for a, b in zip(left_costs, right_costs, strict=True))
        equal = sum(a == b for a, b in zip(left_costs, right_costs, strict=True))
        higher = sum(a > b for a, b in zip(left_costs, right_costs, strict=True))
        paired_rows.append(
            {
                "Seed": seed,
                f"{left.algorithm_label} lower": lower,
                "Equal": equal,
                f"{left.algorithm_label} higher": higher,
                f"Mean {left.algorithm_label} − {right.algorithm_label}": (
                    f"{statistics.fmean(a - b for a, b in zip(left_costs, right_costs)):.5e}"
                ),
            }
        )
    show_table(paired_rows, title="Paired daily combined-cost counts")
    display(
        Markdown(
            "Pairs share exogenous loads and neutral initial taps. Their observation "
            "trajectories can differ after controllers execute different taps."
        )
    )


def show_recipe_comparison(recipes):
    """Display ordered saved recipe values with recorded provenance."""
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError("recipes must be a non-empty ordered mapping")
    setting_names = []
    for recipe in recipes.values():
        if not isinstance(recipe, dict):
            raise ValueError("each recipe must be a settings mapping")
        for name in recipe:
            if name not in setting_names:
                setting_names.append(name)
    rows = []
    for name in setting_names:
        row = {"Setting": name}
        for label, recipe in recipes.items():
            if name not in recipe:
                row[label] = "not recorded"
                continue
            entry = recipe[name]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"value", "provenance"}
                or entry["provenance"] not in {"recorded", "reconstructed"}
            ):
                raise ValueError(f"invalid recipe entry for {label}/{name}")
            value = entry["value"]
            rendered = (
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            )
            row[label] = f"{rendered} ({entry['provenance']})"
        rows.append(row)
    show_table(rows, title="Recorded and reconstructed learner recipes")


def show_tap_sample(results):
    """Display five static loading ranks and the complete executed-ratio figure.

    Sorted multipliers order independent scenarios for presentation. Connecting
    lines do not represent a trajectory or continuous executed action.
    """
    if results.settings["protocol"] != "one_shot":
        raise ValueError("show_tap_sample requires one-shot results")
    ordered = sorted(
        enumerate(results.evaluation_scenarios),
        key=lambda item: (float(item[1].factors[0]), item[0]),
    )
    ranked = [ordered[int(value * (len(ordered) - 1))][1] for value in (0, 0.25, 0.5, 0.75, 1)]
    sampled = [
        next(
            scenario
            for scenario in results.evaluation_scenarios
            if float(scenario.factors[0]) == float(rank.factors[0])
        )
        for rank in ranked
    ]
    lookups = {
        label: {row["scenario"]: row for row in rows} for label, rows in results.result_rows.items()
    }
    rows = []
    for scenario in sampled:
        for label in results.method_labels:
            source = lookups[label][scenario.id]
            indices = [int(source[f"k{number}"]) for number in range(1, results.action_count + 1)]
            rows.append(
                {
                    "Scenario": scenario.id,
                    "Loading multiplier": float(scenario.factors[0]),
                    "Method": label,
                    **tap_ratio_fields(indices),
                }
            )
    show_table(rows, title="Executed ratios at five loading-multiplier ranks", max_height=520)
    ratios = _numeric_series(
        results,
        results.result_rows,
        [f"k{index}" for index in range(1, results.action_count + 1)],
        ratios=True,
        references=("RegControl",),
    )
    figure, axes = plot_static_taps(
        [float(scenario.factors[0]) for scenario in results.evaluation_scenarios],
        ratios,
        _colors(results),
        _linestyles(results),
        seed_mean=len(results.learned_labels) > 1,
    )
    plt.show()
    _mean_caption(results)
    display(
        Markdown(
            "The x-axis orders independent scenarios by loading multiplier, not time. "
            "Connecting lines aid reading; they are not trajectories or continuous actions."
        )
    )
    return figure, axes


def show_day(results, day, hourly_details):
    """Display one selected sequential day without changing saved results.

    Figures use the saved hourly rows for every method. Changing the selected day
    or detail-table switch changes presentation only and launches no evaluation.
    """
    if results.settings["protocol"] != "sequential":
        raise ValueError("show_day requires sequential results")
    if (
        isinstance(day, bool)
        or not isinstance(day, int)
        or not 0 <= day < len(results.evaluation_days)
    ):
        raise ValueError(f"day must be an integer from 0 to {len(results.evaluation_days) - 1}")
    if not isinstance(hourly_details, bool):
        raise ValueError("hourly_details must be a Boolean value")
    selected_day = results.evaluation_days[day]
    selected_day_id = selected_day.id
    print(f"Selected day: {selected_day_id}")
    multipliers = np.asarray([float(scenario.factors[0]) for scenario in selected_day.hours])
    selected = {
        label: [row for row in results.hourly_rows[label] if row["day"] == selected_day_id]
        for label in results.method_labels
    }
    colors = _colors(results)
    styles = _linestyles(results)
    mean = len(results.learned_labels) > 1
    ratios = _numeric_series(
        results, selected, [f"k{i}" for i in range(1, results.action_count + 1)], ratios=True
    )
    ratio_plot = plot_day_ratios(
        selected_day_id, multipliers, ratios, colors, styles, seed_mean=mean
    )
    plt.show()
    _mean_caption(results)
    metrics = _numeric_series(results, selected, ("tap_steps", "voltage_cost"))
    step_plot = plot_hourly_steps(
        selected_day_id,
        {label: values[0] for label, values in metrics.items()},
        colors,
        styles,
        seed_mean=mean,
    )
    plt.show()
    _mean_caption(results)
    display(Markdown(f"Switching cost is tap steps times beta ({results.settings['beta_text']})."))
    voltage_plot = plot_hourly_voltage(
        selected_day_id,
        {label: values[1] for label, values in metrics.items()},
        colors,
        styles,
        focused=(_seed_label(results), "RegControl"),
    )
    plt.show()
    display(
        Markdown(
            "The upper panel keeps every method. The lower panel repeats the controller "
            "series without Neutral so their smaller differences remain readable."
        )
    )
    details = hourly_detail_rows(selected, results.method_labels, results.action_count)
    if hourly_details:
        display(
            Markdown(
                "Tap indices run from 0 to 32. Convert an index with "
                "`ratio = 0.90 + 0.00625 × index`."
            )
        )
        show_table(
            details,
            title=f"Hourly tap settings and costs on {selected_day_id}",
            max_height=520,
        )
    else:
        print("Hourly detail table hidden. Set SHOW_HOURLY_DETAILS = True to display it.")
    return ratio_plot, step_plot, voltage_plot


def _relation(left, right):
    if left < right:
        return "lower than"
    if left > right:
        return "higher than"
    return "equal to"


def _budget_text(results):
    """Describe recorded completed training episodes or their absence by run."""
    values = [results.training_counts[label] for label in results.learned_labels]
    sequential = results.settings["protocol"] == "sequential"
    if all(value is not None for value in values) and len(set(values)) == 1:
        unit = "complete training days" if sequential else "training episodes"
        return f"{values[0]} {unit} per seed"
    unit = " days" if sequential else ""
    return ", ".join(
        f"{label}: {value if value is not None else 'unavailable'}{unit}"
        for label, value in zip(results.learned_labels, values, strict=True)
    )


def _show_conclusions_oneshot(results):
    """Describe one-shot learned aggregates against both static references."""
    summaries = _summaries(results)
    split = results.settings["split"]
    budget = _budget_text(results)
    costs = [summaries[label]["Mean hourly voltage cost"] for label in results.learned_labels]
    feasible = [summaries[label]["Feasible scenarios (%)"] for label in results.learned_labels]
    mean_cost = statistics.fmean(costs)
    mean_feasible = statistics.fmean(feasible)
    if len(results.learned_labels) >= 2:
        seed_result = (
            f"Across {len(results.learned_labels)} independent seeds, "
            f"{results.algorithm_label} mean hourly "
            f"voltage cost is {mean_cost:.3e} with sample standard deviation "
            f"{statistics.stdev(costs):.3e}."
        )
        feasibility = (
            f"Across {len(results.learned_labels)} independent seeds, "
            f"{results.algorithm_label} feasible "
            f"scenarios average {mean_feasible:.1f}% with sample standard deviation "
            f"{statistics.stdev(feasible):.1f} percentage points"
        )
    else:
        seed_result = (
            f"The selected {results.algorithm_label} run has mean hourly voltage cost "
            f"{mean_cost:.3e}."
        )
        feasibility = (
            f"The selected {results.algorithm_label} run has "
            f"{mean_feasible:.1f}% feasible scenarios"
        )
    neutral = summaries["Neutral"]
    regcontrol = summaries["RegControl"]
    lines = [
        f"The {split} evaluation covers {len(results.evaluation_scenarios)} independent "
        f"scenarios for {len(results.learned_labels)} {results.algorithm_label} seed"
        f"{'s' if len(results.learned_labels) != 1 else ''} and {budget}.",
        seed_result
        + " This is "
        + f"{_relation(mean_cost, neutral['Mean hourly voltage cost'])} Neutral "
        + f"({neutral['Mean hourly voltage cost']:.3e}) and "
        + f"{_relation(mean_cost, regcontrol['Mean hourly voltage cost'])} RegControl "
        + f"({regcontrol['Mean hourly voltage cost']:.3e}).",
        feasibility
        + "; this is "
        + f"{_relation(mean_feasible, neutral['Feasible scenarios (%)'])} Neutral "
        + f"({neutral['Feasible scenarios (%)']:.1f}%) and "
        + f"{_relation(mean_feasible, regcontrol['Feasible scenarios (%)'])} RegControl "
        + f"({regcontrol['Feasible scenarios (%)']:.1f}%).",
        f"These results are limited to the {split} split, this synthetic process, "
        "and the recorded training budget.",
    ]
    display(Markdown("\n\n".join(lines)))


def _show_conclusions_seq(results):
    """Describe sequential learned cost, movement, and feasibility aggregates."""
    summaries = _summaries(results)
    split = results.settings["split"]
    budget = _budget_text(results)
    combined = [summaries[label]["Mean daily combined cost"] for label in results.learned_labels]
    voltage = [summaries[label]["Mean daily voltage cost"] for label in results.learned_labels]
    steps = [summaries[label]["Mean tap steps/day"] for label in results.learned_labels]
    feasible = [summaries[label]["Feasible hours (%)"] for label in results.learned_labels]
    deviation = [
        summaries[label]["Mean absolute deviation (p.u.)"] for label in results.learned_labels
    ]
    mean_combined = statistics.fmean(combined)
    mean_voltage = statistics.fmean(voltage)
    mean_deviation = statistics.fmean(deviation)
    mean_steps = statistics.fmean(steps)
    mean_feasible = statistics.fmean(feasible)
    if len(results.learned_labels) >= 2:
        seed_result = (
            f"Across {len(results.learned_labels)} independent seeds, "
            f"{results.algorithm_label} mean daily combined "
            f"cost is {mean_combined:.5e} with sample standard deviation "
            f"{statistics.stdev(combined):.5e}."
        )
    else:
        seed_result = (
            f"The selected {results.algorithm_label} run has mean daily combined cost "
            f"{mean_combined:.5e}."
        )
    neutral = summaries["Neutral"]
    regcontrol = summaries["RegControl"]
    boundary_shares = []
    for label in results.learned_labels:
        actions = [
            int(row[f"k{index}"])
            for row in results.hourly_rows[label]
            for index in range(1, results.action_count + 1)
        ]
        boundary_shares.append(sum(action in (0, N_TAPS - 1) for action in actions) / len(actions))
    saturation = ""
    if all(share >= 0.95 for share in boundary_shares):
        formatted = ", ".join(
            f"{label} {share:.1%}"
            for label, share in zip(results.learned_labels, boundary_shares, strict=True)
        )
        saturation = (
            f"The saved {results.algorithm_label} policies are saturated at the tap "
            f"limits: the boundary share of executed regulator-hour actions is {formatted}."
        )
    lines = [
        f"The {split} evaluation covers {len(results.evaluation_days)} days for "
        f"{len(results.learned_labels)} {results.algorithm_label} seed"
        f"{'s' if len(results.learned_labels) != 1 else ''}, beta "
        f"{results.settings['beta_text']}, and {budget}.",
        seed_result
        + " This is "
        + f"{_relation(mean_combined, neutral['Mean daily combined cost'])} Neutral "
        + f"({neutral['Mean daily combined cost']:.5e}) and "
        + f"{_relation(mean_combined, regcontrol['Mean daily combined cost'])} RegControl "
        + f"({regcontrol['Mean daily combined cost']:.5e}).",
        f"{results.algorithm_label} averages voltage cost {mean_voltage:.5e}, "
        f"mean absolute deviation {mean_deviation:.4e} p.u., "
        f"{mean_steps:.3f} tap steps/day, "
        f"and {mean_feasible:.1f}% feasible hours. Neutral uses "
        f"{neutral['Mean tap steps/day']:.3f} steps/day with voltage cost "
        f"{neutral['Mean daily voltage cost']:.5e}, mean absolute deviation "
        f"{neutral['Mean absolute deviation (p.u.)']:.4e} p.u., and "
        f"{neutral['Feasible hours (%)']:.1f}% feasible hours; RegControl uses "
        f"{regcontrol['Mean tap steps/day']:.3f} steps/day with voltage cost "
        f"{regcontrol['Mean daily voltage cost']:.5e}, mean absolute deviation "
        f"{regcontrol['Mean absolute deviation (p.u.)']:.4e} p.u., and "
        f"{regcontrol['Feasible hours (%)']:.1f}% feasible hours.",
        f"Against RegControl, {results.algorithm_label}'s voltage cost is "
        f"{_relation(mean_voltage, regcontrol['Mean daily voltage cost'])} RegControl's, "
        f"its tap steps are {_relation(mean_steps, regcontrol['Mean tap steps/day'])} "
        f"RegControl's, and its feasible hours are "
        f"{_relation(mean_feasible, regcontrol['Feasible hours (%)'])} RegControl's.",
        saturation,
        f"These results are limited to the {split} split, these dataset inputs, "
        "and the recorded training budget.",
    ]
    display(Markdown("\n\n".join(line for line in lines if line)))


def show_conclusions(results):
    """Display protocol-specific conclusions limited to the supplied results.

    Conclusions compare run-level learned means with references and report selected
    physical metrics. They retain the split, dataset, seed, and recorded
    training-budget limits; they do not assert convergence or general performance.
    """
    if results.settings["protocol"] == "one_shot":
        return _show_conclusions_oneshot(results)
    return _show_conclusions_seq(results)


def _matrix_cost(protocol):
    """Return the voltage cost metric: the first paired row of the protocol."""
    return next(row[0] for row in MATRIX_METRICS[protocol] if row[2])


def run_values(results, metric):
    """Return one per-run metric value for each learned seed, in seed order."""
    summaries = _summaries(results)
    return [summaries[label][metric] for label in results.learned_labels]


def spread_text(values, spec):
    """Format a seed mean with its sample standard deviation, or n/a for one seed."""
    mean = statistics.fmean(values)
    spread = f"{statistics.stdev(values):{spec}}" if len(values) > 1 else "n/a"
    return f"{mean:{spec}} ± {spread}"


def _cell_label(key):
    """Name one deviation kind and its declared alpha."""
    kind, alpha = key
    return f"{kind} alpha {alpha:g}"


def _matrix_entries(cells, control):
    """Keep the supplied cell order and append the alpha-zero baseline."""
    return {_cell_label(key): results for key, results in cells.items()} | {"alpha = 0": control}


def _matrix_styles(cells):
    """Use deviation colors, solid cell lines, and a dashed black baseline."""
    colors = {_cell_label(key): DEVIATION_COLORS[key[0]] for key in cells}
    colors["alpha = 0"] = "#000000"
    styles = {label: "-" for label in colors}
    styles["alpha = 0"] = "--"
    return colors, styles


def _matrix_table_rows(cells, control, metric, spec):
    """Build one metric row per cell and one baseline row without references."""
    return [
        {"Cell": label, metric: spread_text(run_values(results, metric), spec)}
        for label, results in _matrix_entries(cells, control).items()
    ]


def _paired_rows(cells, control, metric, spec):
    """Build one row per cell from differences paired by seed with the baseline."""
    baseline = run_values(control, metric)
    rows = []
    for key, results in cells.items():
        differences = [
            cell - base for cell, base in zip(run_values(results, metric), baseline, strict=True)
        ]
        rows.append({"Cell": _cell_label(key), metric: spread_text(differences, spec)})
    return rows


def show_reward_matrix_training(cells, control):
    """Display per-run training records and one mean physical cost curve per cell and baseline."""
    if not cells:
        raise ValueError("the reward matrix needs at least one cell")
    check_matrix_identity(control, cells)
    series = {}
    for label, results in _matrix_entries(cells, control).items():
        show_table(record_rows(results), title=f"Recorded training settings: {label}")
        values = training_series(results)
        if values is not None:
            series[label] = values
    if not series:
        return None
    colors, styles = _matrix_styles(cells)
    plot = plot_training(series, control.settings["protocol"] == "sequential", colors, styles)
    plt.show()
    display(Markdown(TRAINING_COST_CAPTION))
    return plot


def matrix_numeric(entries, rows, fields, *, ratios=False):
    """Prepare one aligned numeric seed-mean array for each matrix display label."""
    return {
        label: _numeric_series(results, rows[label], fields, ratios=ratios, references=())[
            _seed_label(results)
        ]
        for label, results in entries.items()
    }


def _show_matrix_oneshot(cells, control):
    """Reuse static bars and ratio panels for cells and the baseline."""
    entries = _matrix_entries(cells, control)
    colors, styles = _matrix_styles(cells)
    summaries = {
        label: _summary_series(results)[_seed_label(results)] for label, results in entries.items()
    }
    bars = plot_cost_feasibility(summaries, control.settings["split"], colors, styles)
    plt.show()
    ratios = matrix_numeric(
        entries,
        {label: results.result_rows for label, results in entries.items()},
        [f"k{i}" for i in range(1, control.action_count + 1)],
        ratios=True,
    )
    taps = plot_static_taps(
        [float(s.factors[0]) for s in control.evaluation_scenarios],
        ratios,
        colors,
        styles,
        seed_mean=len(control.learned_labels) > 1,
    )
    plt.show()
    _mean_caption(control)
    return {"cost_feasibility": bars, "ratios": taps}


def _show_matrix_seq(cells, control):
    """Reuse day ratios, hourly steps, and one voltage-cost panel for the matrix."""
    entries = _matrix_entries(cells, control)
    colors, styles = _matrix_styles(cells)
    day = control.evaluation_days[control.settings["plot_day"]]
    rows = {
        label: {
            seed: [row for row in results.hourly_rows[seed] if row["day"] == day.id]
            for seed in results.learned_labels
        }
        for label, results in entries.items()
    }
    ratios = matrix_numeric(
        entries, rows, [f"k{i}" for i in range(1, control.action_count + 1)], ratios=True
    )
    mean = len(control.learned_labels) > 1
    taps = plot_day_ratios(
        day.id, [float(s.factors[0]) for s in day.hours], ratios, colors, styles, seed_mean=mean
    )
    plt.show()
    _mean_caption(control)
    metrics = matrix_numeric(entries, rows, ("tap_steps", "voltage_cost"))
    steps = plot_hourly_steps(
        day.id,
        {label: values[0] for label, values in metrics.items()},
        colors,
        styles,
        seed_mean=mean,
    )
    plt.show()
    _mean_caption(control)
    voltage = plot_hourly_voltage(
        day.id, {label: values[1] for label, values in metrics.items()}, colors, styles
    )
    plt.show()
    return {"ratios": taps, "tap_steps": steps, "voltage_cost": voltage}


def show_reward_matrix(cells, control, *, include_training=True):
    """Display cell and baseline metrics, paired differences, and shared seed-mean figures.

    Set ``include_training=False`` when the notebook displays training records
    separately with ``show_reward_matrix_training``. Reference data remain
    validated by the workflow but do not appear in this report.
    """
    if not cells:
        raise ValueError("the reward matrix needs at least one cell")
    check_matrix_identity(control, cells)
    protocol = control.settings["protocol"]
    split = control.settings["split"]
    plots = {}
    if include_training:
        plots["training"] = show_reward_matrix_training(cells, control)
    for metric, spec, _paired, _higher in MATRIX_METRICS[protocol]:
        show_table(
            _matrix_table_rows(cells, control, metric, spec),
            title=f"{metric} on {split}: seed mean ± sample standard deviation",
        )
    for metric, spec, paired, _higher in MATRIX_METRICS[protocol]:
        if paired:
            show_table(
                _paired_rows(cells, control, metric, spec),
                title=(
                    f"Paired difference, cell minus control, in {metric} on matching seeds; "
                    "negative means lower than the control"
                ),
            )
    plots.update(
        _show_matrix_oneshot(cells, control)
        if protocol == "one_shot"
        else _show_matrix_seq(cells, control)
    )
    return plots


def show_reward_matrix_conclusions(cells, control):
    """State the control value and best cell for every metric, as facts only."""
    if not cells:
        raise ValueError("the reward matrix needs at least one cell")
    check_matrix_identity(control, cells)
    protocol = control.settings["protocol"]
    split = control.settings["split"]
    lines = []
    for metric, spec, _paired, higher_is_better in MATRIX_METRICS[protocol]:
        control_mean = statistics.fmean(run_values(control, metric))
        means = {
            key: statistics.fmean(run_values(results, metric)) for key, results in cells.items()
        }
        pick, word = (max, "highest") if higher_is_better else (min, "lowest")
        best = pick(means, key=means.get)
        lines.append(
            f"{metric}: the alpha = 0 control has seed mean {control_mean:{spec}}; the {word} "
            f"seed mean is {means[best]:{spec}} for deviation {best[0]}, alpha {best[1]}."
        )
    cost = _matrix_cost(protocol)
    baseline = run_values(control, cost)
    below = sum(
        all(value < base for value, base in zip(run_values(results, cost), baseline, strict=True))
        for results in cells.values()
    )
    lines.append(
        f"{below} of {len(cells)} cells have a {cost.lower()} below the control on all "
        f"{len(baseline)} seeds."
    )
    lines.append(
        f"These statements describe the saved {split} split and these dataset inputs only."
    )
    display(Markdown("\n\n".join(lines)))
