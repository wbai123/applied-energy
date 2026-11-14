"""Build the paper figures that compare sequential DDPG with RegControl.

Every function reads validated ``ExperimentResults`` in view mode. Nothing here
trains, evaluates, or changes saved outputs. Numeric helpers return plain
values so tests can check them against hand calculations; plot functions only
draw those values.
"""

import re
import statistics

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from matplotlib.lines import Line2D

from common import workflow
from common.display import show_table
from feeder.bands import PRIMARY_BAND
from feeder.taps import N_TAPS

REFERENCE = "RegControl"
FEEDER_LABELS = {"ieee13": "IEEE 13-bus", "ieee37mod": "Modified IEEE 37-bus"}
LOADING_LABELS = {"random": "Random", "ercot": "ERCOT"}
# Rows are (display name, daily column, display scale, axis unit suffix).
METRICS = (
    ("Daily combined cost", "daily_cost", 1e3, " (×10⁻³)"),
    ("Daily voltage-excursion cost", "voltage_cost", 1e3, " (×10⁻³)"),
    ("Daily tap steps", "tap_steps", 1.0, ""),
)
DDPG_COLOR = "#1f5fa8"
REFERENCE_COLOR = "#c8443f"
SETTING_COLORS = ("#1f5fa8", "#1f9d8f", "#8e5bbf", "#c47a12")
SETTING_MARKERS = ("D", "o", "o", "^")
BAND_COLOR = "#e3efe0"
INK = "#22313f"
MUTED = "#5d6b78"
GRID = "#e6e9ec"


def load_case(base, feeder, dataset, run_name, beta, reward_parameters, training_days):
    """Read one saved feeder, dataset, and setting through the view-mode workflow."""
    return workflow.run_experiment(
        {
            **base,
            "protocol": "sequential",
            "algorithm": "ddpg",
            "mode": "view",
            "overwrite": False,
            "feeder": feeder,
            "dataset": dataset,
            "run_name": run_name,
            "beta": beta,
            "reward": "band",
            "reward_parameters": {**reward_parameters, "beta": beta},
            "training_days": training_days,
            "evaluation_days": None,
            "plot_day": 0,
            "show_hourly_details": False,
        },
        verbose=False,
    )


def case_label(key):
    """Name one (feeder, loading) case for axes and tables."""
    feeder, loading = key
    return f"{FEEDER_LABELS[feeder]} / {LOADING_LABELS[loading]}"


def _column(rows, column):
    return np.array([float(row[column]) for row in rows])


def seed_days(results, column):
    """Return one row of daily values per training seed, in seed order."""
    return np.array(
        [_column(results.daily_rows[label], column) for label in results.learned_labels]
    )


def reference_days(results, column):
    """Return the RegControl daily values on the same days."""
    return _column(results.daily_rows[REFERENCE], column)


def seed_spread(values):
    """Return the mean and sample standard deviation; the spread is None for one value."""
    values = [float(value) for value in values]
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None


def metric_summary(results, column):
    """Summarize one daily metric: per-seed day means, their mean and spread, and RegControl.

    Each seed first averages its own days. The mean and sample standard
    deviation then describe variation across training runs, not across days.
    """
    seeds = seed_days(results, column).mean(axis=1)
    mean, spread = seed_spread(seeds)
    return {
        "seeds": [float(value) for value in seeds],
        "mean": mean,
        "sd": spread,
        "reference": float(reference_days(results, column).mean()),
    }


def change_label(ratio):
    """Describe a DDPG to RegControl ratio in words that state its direction."""
    if ratio == 1:
        return "equal to RegControl"
    if ratio < 1:
        return f"{100 * (1 - ratio):.1f}% lower"
    if ratio < 2:
        return f"{100 * (ratio - 1):.1f}% higher"
    return f"{ratio:.1f}x as high"


def relative_change(summary):
    """Return the ratio of mean daily metrics to RegControl, or None for a zero reference.

    The ratio divides the seed mean by the RegControl mean. Its spread is the
    sample standard deviation of the per-seed ratios.
    """
    reference = summary["reference"]
    if reference == 0:
        return None
    ratio, spread = seed_spread([value / reference for value in summary["seeds"]])
    return {"ratio": ratio, "sd": spread, "label": change_label(ratio)}


def seed_mean_days(results, column="daily_cost"):
    """Return the per-day mean across seeds and its sample standard deviation."""
    values = seed_days(results, column)
    spread = values.std(axis=0, ddof=1) if len(values) > 1 else None
    return values.mean(axis=0), spread


def paired_days(results):
    """Count days by the seed-mean DDPG combined cost against RegControl."""
    mean, _spread = seed_mean_days(results)
    reference = reference_days(results, "daily_cost")
    return {
        "lower": int((mean < reference).sum()),
        "equal": int((mean == reference).sum()),
        "higher": int((mean > reference).sum()),
        "days": len(reference),
    }


def daily_savings(results):
    """Return RegControl cost minus seed-mean DDPG cost per day, with the seed spread."""
    mean, spread = seed_mean_days(results)
    return reference_days(results, "daily_cost") - mean, spread


def box_summary(values):
    """Return the quartiles, full range, and mean of the daily values."""
    values = np.asarray(values, dtype=float)
    low, median, high = np.percentile(values, [25, 50, 75])
    return {
        "min": float(values.min()),
        "q1": float(low),
        "median": float(median),
        "q3": float(high),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def seed_label(results, seed):
    """Return the result label of one selected training seed."""
    seeds = list(results.settings["seeds"])
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not among the loaded seeds {seeds}")
    return results.learned_labels[seeds.index(seed)]


def day_operation(results, label, day_id):
    """Return one controller's executed day: positions, cumulative steps, and voltage range.

    ``positions`` holds one row of 24 integer tap positions per regulator.
    Cumulative steps include the first movement from neutral.
    """
    rows = [row for row in results.hourly_rows[label] if row["day"] == day_id]
    if len(rows) != 24:
        raise ValueError(f"{label} has {len(rows)} hourly rows for {day_id}; expected 24")
    rows.sort(key=lambda row: int(row["hour"]))
    return {
        "positions": np.array(
            [
                [int(row[f"k{index}"]) for row in rows]
                for index in range(1, results.action_count + 1)
            ]
        ),
        "cumulative_steps": np.cumsum([int(row["tap_steps"]) for row in rows]),
        "vmin": _column(rows, "vmin"),
        "vmax": _column(rows, "vmax"),
    }


def preference_point(results):
    """Return seed mean and spread of daily tap steps and mean absolute deviation."""
    steps, steps_sd = seed_spread(seed_days(results, "tap_steps").mean(axis=1))
    deviation, deviation_sd = seed_spread(seed_days(results, "mean_mad").mean(axis=1))
    return {
        "tap_steps": steps,
        "tap_steps_sd": steps_sd,
        "deviation": deviation,
        "deviation_sd": deviation_sd,
    }


def _seed_note(results):
    count = len(results.learned_labels)
    if count == 1:
        return "DDPG: one training seed, no seed spread"
    return f"DDPG: mean across {count} training seeds; error bars: ±1 seed SD"


def _first(cases):
    return next(iter(cases.values()))


def _keys(cases, loading=None, feeder=None):
    return [
        key
        for key in cases
        if (loading is None or key[1] == loading) and (feeder is None or key[0] == feeder)
    ]


def _ordered(cases, position):
    return list(dict.fromkeys(key[position] for key in cases))


def _style(axis, *, grid="x"):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(axis=grid, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _titles(figure, title, subtitle, notes=(), *, header=0.58):
    """Place the title, subtitle, and footnotes at fixed distances from the figure edges."""
    height = figure.get_figheight()
    figure.suptitle(
        title,
        x=0.01,
        y=1 - 0.06 / height,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    figure.text(0.01, 1 - 0.36 / height, subtitle, ha="left", va="top", fontsize=8, color=MUTED)
    for index, note in enumerate(reversed(notes)):
        figure.text(0.01, (0.07 + 0.17 * index) / height, note, fontsize=7.5, color=MUTED)
    bottom = (0.12 + 0.17 * len(notes)) / height
    figure.get_layout_engine().set(rect=(0, bottom, 1, 1 - header / height - bottom))


def _controller_handles():
    return [
        Line2D([], [], marker="D", color=DDPG_COLOR, linestyle="", markersize=6, label="DDPG"),
        Line2D(
            [], [], marker="o", color=REFERENCE_COLOR, linestyle="", markersize=6, label=REFERENCE
        ),
    ]


def plot_overall(cases):
    """Figure 1: absolute daily metrics for DDPG and RegControl, grouped by loading."""
    loadings = _ordered(cases, 1)
    figure, axes = plt.subplots(
        len(METRICS), len(loadings), figsize=(11, 6.6), squeeze=False, layout="constrained"
    )
    for column, loading in enumerate(loadings):
        keys = _keys(cases, loading=loading)
        days = len(cases[keys[0]].evaluation_days)
        split = cases[keys[0]].settings["split"]
        for row, (name, field, scale, unit) in enumerate(METRICS):
            axis = axes[row][column]
            for position, key in enumerate(keys):
                summary = metric_summary(cases[key], field)
                spread = None if summary["sd"] is None else summary["sd"] * scale
                axis.plot(
                    [summary["mean"] * scale, summary["reference"] * scale],
                    [position, position],
                    color="#b9c6d6",
                    linewidth=1.5,
                    zorder=1,
                )
                axis.errorbar(
                    summary["mean"] * scale,
                    position,
                    xerr=spread,
                    fmt="D",
                    color=DDPG_COLOR,
                    markersize=6,
                    capsize=3,
                    zorder=3,
                )
                axis.plot(
                    summary["reference"] * scale, position, "o", color=REFERENCE_COLOR, zorder=2
                )
            axis.set_yticks(range(len(keys)), [FEEDER_LABELS[key[0]] for key in keys])
            axis.set_ylim(len(keys) - 0.5, -0.5)
            axis.set_xlim(left=0)
            axis.set_xlabel(name + unit, fontsize=8, color=INK)
            _style(axis)
            if row == 0:
                axis.set_title(
                    f"{LOADING_LABELS[loading]} | {days} {split} days",
                    fontsize=9,
                    fontweight="bold",
                    color=INK,
                )
    figure.legend(
        handles=_controller_handles(),
        loc="outside upper right",
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    beta = _first(cases).settings["beta"]
    _titles(
        figure,
        "Figure 1 | Overall performance in physical cost components",
        f"All {_first(cases).settings['split']} days | {_seed_note(_first(cases))}",
        (
            f"Fixed evaluation: combined cost = voltage-excursion cost + {beta:g} × tap steps. "
            "Lower is better; panel scales differ.",
        ),
    )
    return figure


def plot_relative(cases):
    """Figure 2: DDPG as a share of RegControl in the same feeder and loading case."""
    keys = list(cases)
    figure, axes = plt.subplots(
        1, len(METRICS), figsize=(11, 3.9), sharey=True, layout="constrained"
    )
    for axis, (name, field, _scale, _unit) in zip(axes, METRICS, strict=True):
        changes = [relative_change(metric_summary(cases[key], field)) for key in keys]
        ratios = [100 * change["ratio"] for change in changes if change is not None]
        # Axis limits cover the error-bar endpoints as well as the means.
        ends = [
            100 * (change["ratio"] + sign * (change["sd"] or 0.0))
            for change in changes
            if change is not None
            for sign in (-1, 1)
        ]
        logarithmic = bool(ratios) and max(ratios) > 200
        for position, change in enumerate(changes):
            if change is None:
                axis.text(
                    0.5,
                    position,
                    "RegControl is zero: no ratio",
                    transform=axis.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=MUTED,
                )
                continue
            spread = None if change["sd"] is None else 100 * change["sd"]
            axis.errorbar(
                100 * change["ratio"],
                position,
                xerr=spread,
                fmt="D",
                color=DDPG_COLOR,
                markersize=6,
                capsize=3,
            )
            axis.annotate(
                change["label"],
                (100 * change["ratio"], position),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
        axis.axvline(100, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2)
        if logarithmic:
            axis.set_xscale("log")
            lowest = min(end for end in ends if end > 0)  # A log axis shows positive ends only.
            axis.set_xlim(min(1, lowest / 3), max(ends) * 4)
            axis.xaxis.set_major_formatter(lambda value, _position: f"{value:,.0f}%")
            axis.set_xlabel("DDPG / RegControl (log scale)", fontsize=8, color=INK)
        else:
            axis.set_xlim(0, max(115, *(end * 1.15 for end in ends)) if ends else 115)
            axis.xaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
            axis.set_xlabel("DDPG / RegControl", fontsize=8, color=INK)
        axis.set_yticks(range(len(keys)), [case_label(key) for key in keys])
        axis.set_ylim(len(keys) - 0.5, -0.7)
        axis.set_title(name, fontsize=9, fontweight="bold", color=INK)
        _style(axis)
    _titles(
        figure,
        "Figure 2 | Change relative to RegControl",
        "Each label compares DDPG with RegControl in the same feeder and loading case. "
        f"{_seed_note(_first(cases))}. Red dashed line: RegControl = 100%.",
        (
            "Ratios use mean daily metrics, not averages of daily percentages. "
            "A higher voltage-excursion cost can coexist with a lower combined cost.",
        ),
    )
    return figure


def plot_day_operation(cases, loading, day_index, seed):
    """Figure 3: one real day of load, cumulative tap steps, and executed tap positions."""
    keys = _keys(cases, loading=loading)
    figure, axes = plt.subplots(
        4,
        len(keys),
        figsize=(11, 8),
        squeeze=False,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1, 1, 1.3, 1.3]},
    )
    hours = np.arange(24)
    image = None
    for column, key in enumerate(keys):
        results = cases[key]
        day = results.evaluation_days[day_index]
        controllers = (
            ("DDPG", seed_label(results, seed), DDPG_COLOR),
            (REFERENCE, REFERENCE, REFERENCE_COLOR),
        )
        load, steps = axes[0][column], axes[1][column]
        load.plot(hours, [float(hour.factors[0]) for hour in day.hours], color=INK, linewidth=1.6)
        load.set_title(FEEDER_LABELS[key[0]], fontsize=9, fontweight="bold", color=INK)
        load.set_ylabel("Load multiplier", fontsize=8, color=INK)
        _style(load, grid="y")
        for row, (name, label, color) in enumerate(controllers):
            operation = day_operation(results, label, day.id)
            steps.step(
                hours, operation["cumulative_steps"], where="post", color=color, linewidth=1.6
            )
            axis = axes[2 + row][column]
            image = axis.imshow(
                operation["positions"],
                aspect="auto",
                cmap="cividis",
                vmin=0,
                vmax=N_TAPS - 1,
                extent=(-0.5, 23.5, results.action_count - 0.5, -0.5),
            )
            for regulator, positions in enumerate(operation["positions"]):
                for hour, position in enumerate(positions):
                    axis.text(
                        hour,
                        regulator,
                        str(position),
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if position < 20 else "black",
                    )
            axis.set_yticks(
                range(results.action_count),
                [f"R{index}" for index in range(1, results.action_count + 1)],
            )
            axis.set_title(
                f"{name} executed tap positions", fontsize=8.5, fontweight="bold", color=color
            )
            axis.tick_params(colors=MUTED, labelsize=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
        steps.set_ylabel("Cumulative\ntap steps", fontsize=8, color=INK)
        steps.set_ylim(bottom=0)
        _style(steps, grid="y")
        axes[3][column].set_xlabel("Hour of day", fontsize=8, color=INK)
        axes[3][column].set_xticks([0, 4, 8, 12, 16, 20, 23])
        for axis in axes[:, column]:
            axis.tick_params(labelbottom=True)
    bar = figure.colorbar(image, ax=axes[2:, :], fraction=0.025, pad=0.01)
    bar.set_label(f"Executed tap position (neutral = {(N_TAPS - 1) // 2})", fontsize=8)
    bar.set_ticks([0, (N_TAPS - 1) // 2, N_TAPS - 1])
    figure.legend(
        handles=[
            Line2D([], [], color=DDPG_COLOR, linewidth=1.6, label="DDPG"),
            Line2D([], [], color=REFERENCE_COLOR, linewidth=1.6, label=REFERENCE),
        ],
        loc="outside upper right",
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    day_id = cases[keys[0]].evaluation_days[day_index].id
    _titles(
        figure,
        "Figure 3 | One real day: loading, movement, and executed taps",
        f"{LOADING_LABELS[loading]} load | {day_id} | DDPG seed {seed} | "
        "identical load profile and neutral start for both controllers",
        (
            "Both controllers start at neutral. Movement includes the first move from neutral; "
            "heatmaps show executed integer positions.",
            "Illustrative run, not an average or a claim of typical performance. "
            "Figure 4 shows the voltages of this same run.",
        ),
    )
    return figure


def plot_day_voltage(cases, loading, day_index, seed):
    """Figure 4: hourly minimum and maximum monitored voltage against the target band."""
    keys = _keys(cases, loading=loading)
    figure, axes = plt.subplots(
        len(keys),
        2,
        figsize=(11, 5.6),
        squeeze=False,
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    low, high = PRIMARY_BAND
    hours = np.arange(24)
    for row, key in enumerate(keys):
        results = cases[key]
        day = results.evaluation_days[day_index]
        controllers = (
            ("DDPG", seed_label(results, seed), DDPG_COLOR),
            (REFERENCE, REFERENCE, REFERENCE_COLOR),
        )
        extremes = [low, high]
        for column, (name, label, color) in enumerate(controllers):
            operation = day_operation(results, label, day.id)
            axis = axes[row][column]
            axis.axhspan(low, high, color=BAND_COLOR, zorder=0)
            axis.plot(hours, operation["vmin"], color=color, linewidth=1.6)
            axis.plot(hours, operation["vmax"], color=color, linewidth=1.6, linestyle="--")
            axis.fill_between(
                hours,
                operation["vmin"],
                low,
                where=operation["vmin"] < low,
                interpolate=True,
                color=color,
                alpha=0.3,
            )
            axis.fill_between(
                hours,
                operation["vmax"],
                high,
                where=operation["vmax"] > high,
                interpolate=True,
                color=color,
                alpha=0.3,
            )
            _style(axis, grid="y")
            axis.set_xticks([0, 4, 8, 12, 16, 20, 23])
            if row == 0:
                axis.set_title(name, fontsize=9, fontweight="bold", color=color)
            if column == 0:
                axis.set_ylabel(
                    f"{FEEDER_LABELS[key[0]]}\nPost-action voltage (p.u.)", fontsize=8, color=INK
                )
            if row == len(keys) - 1:
                axis.set_xlabel("Hour of day", fontsize=8, color=INK)
            extremes.extend((operation["vmin"].min(), operation["vmax"].max()))
        axes[row][0].set_ylim(min(extremes) - 0.005, max(extremes) + 0.005)
    figure.legend(
        handles=[
            Line2D([], [], color=INK, linewidth=1.6, label="Minimum monitored voltage"),
            Line2D(
                [], [], color=INK, linewidth=1.6, linestyle="--", label="Maximum monitored voltage"
            ),
        ],
        loc="outside upper right",
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    day_id = cases[keys[0]].evaluation_days[day_index].id
    _titles(
        figure,
        "Figure 4 | Actual voltage ranges and excursions",
        f"Same day and run as Figure 3: {LOADING_LABELS[loading]} {day_id}, DDPG seed {seed}. "
        f"Green target band: {low:.2f}–{high:.2f} p.u.",
        (
            "The objective penalizes excursions outside the target band; "
            "the band is not enforced as a hard constraint.",
            "Shading marks excursions of the extreme voltages. "
            "Its area is not the numerical voltage-excursion cost.",
        ),
    )
    return figure


def plot_paired_days(cases):
    """Figure 5: RegControl against seed-mean DDPG daily combined cost, one point per day."""
    loadings, feeders = _ordered(cases, 1), _ordered(cases, 0)
    figure, axes = plt.subplots(
        len(loadings), len(feeders), figsize=(11, 8.4), squeeze=False, layout="constrained"
    )
    scale = 1e3
    for row, loading in enumerate(loadings):
        for column, feeder in enumerate(feeders):
            axis = axes[row][column]
            results = cases[(feeder, loading)]
            mean, _spread = seed_mean_days(results)
            reference = reference_days(results, "daily_cost")
            counts = paired_days(results)
            limit = 1.05 * scale * max(mean.max(), reference.max())
            axis.plot([0, limit], [0, limit], color=MUTED, linestyle="--", linewidth=1)
            axis.fill_between([0, limit], [0, limit], 0, color=DDPG_COLOR, alpha=0.06, linewidth=0)
            axis.scatter(
                reference * scale, mean * scale, s=9, color=DDPG_COLOR, alpha=0.45, linewidths=0
            )
            axis.set_xlim(0, limit)
            axis.set_ylim(0, limit)
            axis.set_aspect("equal")
            axis.text(
                0.03,
                0.97,
                "Shaded region below the line:\nLower cost with DDPG\n\n"
                f"Seed-mean DDPG lower:\n{counts['lower']}/{counts['days']} days "
                f"({100 * counts['lower'] / counts['days']:.1f}%)",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                color=INK,
            )
            axis.set_title(case_label((feeder, loading)), fontsize=9, fontweight="bold", color=INK)
            axis.set_xlabel("RegControl daily combined cost (×10⁻³)", fontsize=8, color=INK)
            axis.set_ylabel("DDPG daily combined cost (×10⁻³)", fontsize=8, color=INK)
            _style(axis, grid="both")
    count = len(_first(cases).learned_labels)
    _titles(
        figure,
        "Figure 5 | Paired daily combined costs",
        f"One point = one matched {_first(cases).settings['split']} day. "
        f"DDPG cost is averaged across {count} training seeds.",
        (
            "Counts apply to the mean across seeds, not to every individual seed. "
            "Panel scales differ; the dashed line always means equal cost.",
        ),
    )
    return figure


def plot_daily_savings(cases):
    """Figure 6: daily savings against RegControl with a boxplot of the same days."""
    loadings, feeders = _ordered(cases, 1), _ordered(cases, 0)
    figure, axes = plt.subplots(
        len(loadings),
        2 * len(feeders),
        figsize=(11, 6.6),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": [6, 1] * len(feeders)},
    )
    scale = 1e3
    for row, loading in enumerate(loadings):
        for column, feeder in enumerate(feeders):
            series, box = axes[row][2 * column], axes[row][2 * column + 1]
            results = cases[(feeder, loading)]
            saving, spread = daily_savings(results)
            saving = saving * scale
            if loading == "ercot":
                positions = np.array(
                    [day.id for day in results.evaluation_days], dtype="datetime64[D]"
                )
                series.set_xlabel("Validation date", fontsize=8, color=INK)
                series.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=(1, 7)))
                series.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
            else:
                positions = np.arange(len(saving))
                series.set_xlabel("Validation-day index", fontsize=8, color=INK)
            if spread is not None:
                series.fill_between(
                    positions,
                    saving - spread * scale,
                    saving + spread * scale,
                    color=DDPG_COLOR,
                    alpha=0.18,
                    linewidth=0,
                )
            series.plot(positions, saving, color=DDPG_COLOR, linewidth=0.8)
            series.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
            series.set_title(
                case_label((feeder, loading)), fontsize=9, fontweight="bold", color=INK
            )
            series.set_ylabel("Daily combined-cost saving (×10⁻³)", fontsize=8, color=INK)
            series.text(
                0.01,
                0.97,
                "↑ DDPG lower cost",
                transform=series.transAxes,
                va="top",
                fontsize=7.5,
                color=MUTED,
            )
            _style(series, grid="y")
            summary = box_summary(saving)
            box.boxplot(
                saving,
                whis=(0, 100),
                widths=0.6,
                patch_artist=True,
                boxprops={"facecolor": "#b9cde6", "edgecolor": DDPG_COLOR},
                medianprops={"color": INK, "linewidth": 1.4},
                whiskerprops={"color": MUTED},
                capprops={"color": MUTED},
            )
            box.plot(1, summary["mean"], "D", color=DDPG_COLOR, markersize=5)
            box.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
            box.set_ylim(series.get_ylim())
            box.set_xticks([1], ["Days"])
            box.tick_params(labelleft=False)
            _style(box, grid="y")
    count = len(_first(cases).learned_labels)
    figure.legend(
        handles=[
            Line2D(
                [], [], color=DDPG_COLOR, linewidth=1, label=f"Mean saving across {count} seeds"
            ),
            Line2D([], [], color=DDPG_COLOR, linewidth=6, alpha=0.18, label="±1 seed SD"),
            Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Equal cost"),
        ],
        loc="outside upper right",
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    _titles(
        figure,
        "Figure 6 | Daily savings and their distribution",
        "Saving = RegControl cost − seed-mean DDPG cost. Positive values: DDPG lower cost.",
        (
            "Boxplots summarize the daily seed-mean savings: box = middle 50% of days; "
            "line = median; diamond = mean; whiskers = full range.",
            "Shaded bands describe variation across training seeds, not confidence intervals. "
            "All days are retained without smoothing.",
        ),
    )
    return figure


def _math_powers(text):
    """Render Unicode powers of ten such as 10⁻³ with mathtext, so all exponents share one font."""
    plain = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
    return re.sub(
        r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda match: f"$10^{{{match.group(1).translate(plain)}}}$",
        text,
    )


def plot_preferences(settings, descriptions):
    """Figure 7: voltage deviation against tap movement for separately trained settings.

    ``settings`` maps a letter to ``{feeder: results}``; ``descriptions`` maps
    the same letter to its legend text.
    """
    feeders = list(next(iter(settings.values())))
    figure, axes = plt.subplots(
        1, len(feeders), figsize=(11, 4.9), squeeze=False, layout="constrained"
    )
    for axis, feeder in zip(axes[0], feeders, strict=True):
        for index, (letter, by_feeder) in enumerate(settings.items()):
            point = preference_point(by_feeder[feeder])
            axis.errorbar(
                point["tap_steps"],
                point["deviation"],
                xerr=point["tap_steps_sd"],
                yerr=point["deviation_sd"],
                fmt=SETTING_MARKERS[index],
                color=SETTING_COLORS[index],
                markersize=7,
                capsize=3,
            )
            axis.annotate(
                letter,
                (point["tap_steps"], point["deviation"]),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
        axis.set_xlim(left=0)
        axis.set_title(FEEDER_LABELS[feeder], fontsize=9, fontweight="bold", color=INK)
        axis.set_xlabel(
            "Mean executed tap steps per day\n← Fewer operations", fontsize=8, color=INK
        )
        _style(axis, grid="both")
    axes[0][0].set_ylabel(
        "Mean absolute voltage deviation from 1.0 p.u.\n↓ Closer to nominal voltage",
        fontsize=8,
        color=INK,
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=SETTING_MARKERS[index],
                color=SETTING_COLORS[index],
                linestyle="",
                markersize=7,
                label=f"{letter}  {_math_powers(descriptions[letter])}",
            )
            for index, letter in enumerate(settings)
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1 - 0.5 / figure.get_figheight()),
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    sample = next(iter(next(iter(settings.values())).values()))
    count = len(sample.learned_labels)
    _titles(
        figure,
        "Figure 7 | Adjusting operating preferences with α and β",
        f"{len(sample.evaluation_days)} matched {sample.settings['split']} days | "
        f"Points: {count}-seed means | Error bars: ±1 seed SD",
        (
            "Voltage deviation measures distance from 1.0 p.u., not excursions outside "
            "0.95–1.05 p.u. Axis scales differ between panels.",
            "Each setting is a separately trained policy. "
            "These observed operating points do not establish a Pareto frontier.",
        ),
        header=1.05,
    )
    return figure


def _number(value, scale, digits):
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def _digits(scale):
    """Show scaled costs with four decimals and tap steps with two."""
    return 2 if scale == 1 else 4


def show_summary_table(cases):
    """Show the numbers behind Figures 1 and 2 for every case and metric."""
    rows = []
    for key, results in cases.items():
        for name, field, scale, unit in METRICS:
            summary = metric_summary(results, field)
            change = relative_change(summary)
            rows.append(
                {
                    "Case": case_label(key),
                    "Metric": name + unit,
                    "DDPG seed mean": _number(summary["mean"], scale, _digits(scale)),
                    "±1 seed SD": _number(summary["sd"], scale, _digits(scale)),
                    "RegControl": _number(summary["reference"], scale, _digits(scale)),
                    "DDPG against RegControl": "no ratio: RegControl is zero"
                    if change is None
                    else change["label"],
                }
            )
    show_table(rows, title="Mean daily metrics per case")


def show_seed_table(cases):
    """Show each training seed's mean daily metrics as supplementary material."""
    for key, results in cases.items():
        summaries = {
            field: metric_summary(results, field) for _name, field, _scale, _unit in METRICS
        }
        rows = [
            {"Seed": str(seed)}
            | {
                name + unit: _number(summaries[field]["seeds"][index], scale, _digits(scale))
                for name, field, scale, unit in METRICS
            }
            for index, seed in enumerate(results.settings["seeds"])
        ]
        rows.append(
            {"Seed": REFERENCE}
            | {
                name + unit: _number(summaries[field]["reference"], scale, _digits(scale))
                for name, field, scale, unit in METRICS
            }
        )
        show_table(rows, title=f"{case_label(key)}: mean daily metrics per training seed")


def show_conclusions(cases):
    """State the saved comparisons per case without claims beyond the selected split."""
    first = _first(cases)
    lines = []
    for key, results in cases.items():
        parts = []
        for name, field, _scale, _unit in METRICS:
            change = relative_change(metric_summary(results, field))
            parts.append(
                f"no ratio for {name.lower()} (RegControl is zero)"
                if change is None
                else f"{name.lower()} {change['label']}"
            )
        counts = paired_days(results)
        lines.append(
            f"**{case_label(key)}.** Against RegControl, DDPG has {', '.join(parts)}. "
            f"The seed-mean DDPG combined cost is lower on {counts['lower']} of "
            f"{counts['days']} days, equal on {counts['equal']}, and higher on {counts['higher']}."
        )
    lines.append(
        f"These statements describe {len(first.learned_labels)} training seeds on the saved "
        f"{first.settings['split']} split. They use means across seeds and do not state that "
        "every seed has a lower cost."
    )
    display(Markdown("\n\n".join(lines)))


def _direction(after, before):
    return "lower" if after < before else "higher" if after > before else "equal"


def show_preference_summary(settings, descriptions, steps):
    """Show each setting's operating point and the direction of each named change.

    ``steps`` lists (from letter, to letter, description) pairs to compare.
    """
    feeders = list(next(iter(settings.values())))
    rows = []
    for letter, by_feeder in settings.items():
        for feeder in feeders:
            point = preference_point(by_feeder[feeder])
            rows.append(
                {
                    "Setting": f"{letter}: {descriptions[letter]}",
                    "Feeder": FEEDER_LABELS[feeder],
                    "Mean tap steps per day": f"{point['tap_steps']:.2f}",
                    "±1 seed SD (steps)": _number(point["tap_steps_sd"], 1, 2),
                    "Mean absolute deviation (p.u.)": f"{point['deviation']:.5f}",
                    "±1 seed SD (p.u.)": _number(point["deviation_sd"], 1, 5),
                }
            )
    show_table(rows, title="Operating points behind Figure 7")
    lines = []
    for before, after, description in steps:
        for feeder in feeders:
            old = preference_point(settings[before][feeder])
            new = preference_point(settings[after][feeder])
            lines.append(
                f"{before} → {after} ({description}), {FEEDER_LABELS[feeder]}: "
                f"mean voltage deviation is {_direction(new['deviation'], old['deviation'])} "
                f"and tap movement is {_direction(new['tap_steps'], old['tap_steps'])}."
            )
    lines.append(
        "These separately trained policies illustrate alternative operating compromises. "
        "They do not establish a Pareto frontier."
    )
    display(Markdown("\n\n".join(lines)))
