"""Run and report sequential switching-cost cells on matched validation inputs."""

import statistics

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from common import workflow
from common.display import show_table
from common.plots import (
    SEED_MEAN_CAPTION,
    plot_beta_distributions,
    plot_beta_tradeoff,
    plot_beta_training,
    plot_day_ratios,
    plot_hourly_steps,
    plot_hourly_voltage,
)
from common.reports import (
    TRAINING_COST_CAPTION,
    matrix_numeric,
    record_rows,
    run_values,
    spread_text,
    training_series,
)

BETA_COLORS = {1e-3: "#1f77b4", 1e-5: "#ff7f0e", 1e-7: "#2ca02c"}
PHYSICAL_METRICS = (
    ("Mean daily voltage cost", ".5e"),
    ("Mean absolute deviation (p.u.)", ".4e"),
    ("Mean tap steps/day", ".3f"),
    ("Feasible hours (%)", ".1f"),
)


def run_beta_matrix(base, cells):
    """Run the supplied beta cells quietly, preserving their display order."""
    if not isinstance(base, dict) or "reward_parameters" in base or "beta" in base:
        raise ValueError("base must omit reward_parameters and beta")
    if base.get("protocol") != "sequential":
        raise ValueError("beta matrices require a sequential protocol")
    betas = [beta for beta, _name in cells]
    names = [name for _beta, name in cells]
    if len(set(betas)) != len(betas):
        raise ValueError("cells must not repeat a beta")
    if len(set(names)) != len(names):
        raise ValueError("cells must not repeat a run name")
    if not cells:
        raise ValueError("the beta matrix needs at least one cell")
    selected = dict(base)
    parameters = {key: selected.pop(key) for key in ("deviation", "alpha", "rate")}
    results = {
        float(beta): workflow.run_experiment(
            {
                **selected,
                "beta": beta,
                "run_name": name,
                "reward_parameters": {**parameters, "beta": beta},
            },
            verbose=False,
        )
        for beta, name in cells
    }
    check_beta_matrix_identity(results)
    print(
        "Beta matrix identity passed: dataset, split, protocol, algorithm, seeds, and days match."
    )
    return results


def check_beta_matrix_identity(cells):
    """Require matched evaluation identities and each cell's declared beta."""
    if not cells:
        raise ValueError("the beta matrix needs at least one cell")
    keys = ("dataset", "split", "protocol", "algorithm", "seeds")

    def identity(result):
        return {key: result.settings[key] for key in keys} | {
            "days": tuple(day.id for day in result.evaluation_days)
        }

    expected = identity(next(iter(cells.values())))
    for beta, result in cells.items():
        actual = identity(result)
        differing = [key for key in expected if actual[key] != expected[key]]
        if result.settings["beta"] != beta:
            differing.append("beta")
        if differing:
            raise ValueError(f"beta matrix cell {beta} differs in {differing}")


def _styles(cells, benchmark):
    colors = {f"beta {beta}": BETA_COLORS[beta] for beta in cells}
    styles = {f"beta {beta}": "--" if beta == benchmark else "-" for beta in cells}
    return colors, styles


def _benchmark(cells, benchmark):
    if benchmark not in cells:
        raise ValueError(f"benchmark beta {benchmark} must be a cell")
    return cells[benchmark]


def show_beta_matrix_training(cells):
    """Show recorded settings and exploratory voltage and movement training curves."""
    check_beta_matrix_identity(cells)
    series = {"voltage_cost": {}, "tap_steps": {}}
    for beta, result in cells.items():
        label = f"beta {beta}"
        show_table(record_rows(result), title=f"Recorded training settings: {label}")
        for column, values in series.items():
            curve = training_series(result, column=column)
            if curve is not None:
                values[label] = curve
    colors, styles = _styles(cells, 1e-5)
    plot = plot_beta_training(series, colors, styles)
    plt.show()
    display(Markdown(TRAINING_COST_CAPTION))
    return plot


def show_beta_matrix(cells, benchmark):
    """Show physical metrics, paired differences, selected-day figures, and daily distributions."""
    check_beta_matrix_identity(cells)
    baseline = _benchmark(cells, benchmark)
    for metric, spec in PHYSICAL_METRICS:
        show_table(
            [
                {"Cell": f"beta {beta}", metric: spread_text(run_values(result, metric), spec)}
                for beta, result in cells.items()
            ],
            title=(
                f"{metric} on {baseline.settings['split']}: seed mean ± sample standard deviation"
            ),
        )
    show_table(
        [
            {"Cell": f"beta {beta}"}
            | {
                metric: spread_text(run_values(result, metric), ".5e")
                for metric in ("Mean daily combined cost", "Mean daily switching cost")
            }
            for beta, result in cells.items()
        ],
        title="Mean daily combined cost at each cell's own beta (not comparable across cells)",
    )
    for metric, spec in PHYSICAL_METRICS[:3]:
        values = run_values(baseline, metric)
        show_table(
            [
                {
                    "Cell": f"beta {beta}",
                    metric: spread_text(
                        [a - b for a, b in zip(run_values(result, metric), values, strict=True)],
                        spec,
                    ),
                }
                for beta, result in cells.items()
                if beta != benchmark
            ],
            title=(
                f"Paired difference, cell minus beta {benchmark} cell, in {metric} "
                "on matching seeds; negative means lower than the benchmark"
            ),
        )
    plots = _show_day(cells, baseline, benchmark)
    colors, _styles_by_label = _styles(cells, benchmark)
    points = {
        f"beta {beta}": (
            run_values(result, "Mean tap steps/day"),
            run_values(result, "Mean absolute deviation (p.u.)"),
        )
        for beta, result in cells.items()
    }
    plots["tradeoff"] = plot_beta_tradeoff(points, colors)
    plt.show()
    distributions = {
        f"beta {beta}": [row for seed in result.learned_labels for row in result.daily_rows[seed]]
        for beta, result in cells.items()
    }
    plots["distributions"] = plot_beta_distributions(distributions, colors)
    plt.show()
    return plots


def _show_day(cells, baseline, benchmark):
    entries = {f"beta {beta}": result for beta, result in cells.items()}
    colors, styles = _styles(cells, benchmark)
    day = baseline.evaluation_days[baseline.settings["plot_day"]]
    print(f"Selected day: {day.id}")
    rows = {
        label: {
            seed: [row for row in result.hourly_rows[seed] if row["day"] == day.id]
            for seed in result.learned_labels
        }
        for label, result in entries.items()
    }
    ratios = matrix_numeric(
        entries, rows, [f"k{i}" for i in range(1, baseline.action_count + 1)], ratios=True
    )
    mean = len(baseline.learned_labels) > 1
    taps = plot_day_ratios(
        day.id,
        [float(hour.factors[0]) for hour in day.hours],
        ratios,
        colors,
        styles,
        seed_mean=mean,
    )
    plt.show()
    metrics = matrix_numeric(entries, rows, ("tap_steps", "voltage_cost"))
    steps = plot_hourly_steps(
        day.id,
        {label: values[0] for label, values in metrics.items()},
        colors,
        styles,
        seed_mean=mean,
    )
    plt.show()
    voltage = plot_hourly_voltage(
        day.id, {label: values[1] for label, values in metrics.items()}, colors, styles
    )
    plt.show()
    if mean:
        display(Markdown(SEED_MEAN_CAPTION))
    return {"ratios": taps, "tap_steps": steps, "voltage_cost": voltage}


def show_beta_matrix_conclusions(cells, benchmark):
    """State saved physical extrema and strict beta trends without recommendations."""
    check_beta_matrix_identity(cells)
    baseline = _benchmark(cells, benchmark)
    lines = []
    for metric, spec in PHYSICAL_METRICS:
        means = {
            beta: statistics.fmean(run_values(result, metric)) for beta, result in cells.items()
        }
        pick, word = (max, "highest") if metric == "Feasible hours (%)" else (min, "lowest")
        best = pick(means, key=means.get)
        lines.append(
            f"{metric}: the beta {benchmark} benchmark cell has seed mean "
            f"{means[benchmark]:{spec}}; "
            f"the {word} seed mean is {means[best]:{spec}} for beta {best}."
        )
    for metric, direction in (
        ("Mean tap steps/day", "decrease"),
        ("Mean absolute deviation (p.u.)", "increase"),
    ):
        values = [run_values(cells[beta], metric) for beta in sorted(cells)]
        sign = -1 if direction == "decrease" else 1
        means = [statistics.fmean(row) for row in values]
        mean_trend = all(sign * (b - a) > 0 for a, b in zip(means, means[1:]))
        seed_trend = all(
            sign * (b - a) > 0
            for left, right in zip(values, values[1:])
            for a, b in zip(left, right, strict=True)
        )
        lines.append(
            f"{metric}: seed means strictly {direction} as beta increases: "
            f"{'yes' if mean_trend else 'no'}. "
            f"This holds on every seed individually: {'yes' if seed_trend else 'no'}."
        )
    lines.append(
        f"These statements describe the saved {baseline.settings['split']} split "
        "and these dataset inputs only."
    )
    display(Markdown("\n\n".join(lines)))
