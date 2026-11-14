"""Build feeder-independent notebook figures from aligned saved results."""

import matplotlib.pyplot as plt
import numpy as np

from feeder.taps import ratio_of_index


def tap_ratio_fields(indices):
    """Convert regulator-ordered positions to numbered physical p.u. ratios."""
    ratios = ratio_of_index(indices)
    return {f"Ratio {index}": float(value) for index, value in enumerate(ratios, start=1)}


def hourly_detail_rows(selected_hourly, method_labels, action_count):
    """Build a 24-hour executed-position, ratio, movement, and cost table."""
    rows = []
    for hour in range(24):
        for label in method_labels:
            source = selected_hourly[label][hour]
            indices = [int(source[f"k{index}"]) for index in range(1, action_count + 1)]
            row = {"Hour": hour, "Method": label}
            row.update((f"Index {index}", value) for index, value in enumerate(indices, 1))
            row.update(tap_ratio_fields(indices))
            row.update(
                {
                    "Tap steps": int(source["tap_steps"]),
                    "Voltage cost": float(source["voltage_cost"]),
                    "Switching cost": float(source["switching_cost"]),
                }
            )
            rows.append(row)
    return rows


def _coincident_caption(sequences, labels):
    """Name labels with identical plotted values on every regulator."""
    groups = []
    for label in labels:
        sequence = sequences[label]
        for group in groups:
            if np.array_equal(sequences[group[0]], sequence):
                group.append(label)
                break
        else:
            groups.append([label])
    return " ".join(
        f"Curves for {', '.join(group)} coincide." for group in groups if len(group) > 1
    )


SEED_MEAN_CAPTION = (
    "The learned series is the arithmetic mean over the selected seeds "
    "and is not an executable tap position."
)


def _ratio_caption(series, seed_mean):
    """Combine the mean qualification with coincidence on the plotted values."""
    return " ".join(
        text
        for text in (
            SEED_MEAN_CAPTION if seed_mean else "",
            _coincident_caption(series, series),
        )
        if text
    )


def plot_cost_feasibility(summaries, split, colors, linestyles):
    """Plot static mean cost and feasibility with a zero-safe cost scale."""
    labels = tuple(summaries)
    mean_costs = [summaries[label]["Mean hourly voltage cost"] for label in labels]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    axes[0].bar(positions, mean_costs, color=[colors[label] for label in labels])
    scale = "log" if all(cost > 0 for cost in mean_costs) else "linear"
    if scale == "log":
        axes[0].set_yscale("log")
    axes[0].set_ylabel(f"Mean hourly voltage cost ({scale} scale)")
    axes[0].set_title(f"Mean voltage cost on {split}")
    axes[1].bar(
        positions,
        [summaries[label]["Feasible scenarios (%)"] for label in labels],
        color=[colors[label] for label in labels],
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Feasible scenarios (%)")
    axes[1].set_title(f"Scenarios inside [0.95, 1.05] p.u. on {split}")
    for axis in axes:
        for bar, label in zip(axis.patches, labels, strict=True):
            bar.set_linestyle(linestyles[label])
            bar.set_edgecolor(colors[label])
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_static_taps(multipliers, ratios, colors, linestyles, *, seed_mean=False):
    """Plot executed ratios against independently sorted static multipliers."""
    multipliers = np.asarray(multipliers)
    labels = tuple(ratios)
    action_count = len(ratios[labels[0]])
    order = np.argsort(multipliers, kind="stable")
    fig, axes = plt.subplots(
        1, action_count, figsize=(4 * action_count, 3.6), dpi=100, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    caption = _ratio_caption(ratios, seed_mean)
    for regulator, axis in enumerate(axes, start=1):
        axis.axhline(1.0, color="#b0b0b0", linewidth=0.7, zorder=0)
        for bound in (0.90, 1.10):
            axis.axhline(
                bound,
                color="#b0b0b0",
                linewidth=0.7,
                linestyle="--",
                zorder=0,
                label="legal bounds" if axis is axes[-1] and bound == 0.90 else None,
            )
        for label in labels:
            values = np.asarray(ratios[label][regulator - 1])
            axis.plot(
                multipliers[order],
                values[order],
                color=colors[label],
                linewidth=0.7,
                alpha=0.75,
                linestyle=linestyles[label],
                label=label,
            )
        axis.set_title(f"Regulator {regulator}")
        axis.set_xlabel("Loading multiplier")
        axis.set_ylim(0.88, 1.12)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Executed ratio (p.u.)")
    axes[-1].legend(fontsize=7)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", color="#666666", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1) if caption else None)
    return fig, axes


def plot_day_ratios(day_id, multipliers, ratios, colors, linestyles, *, seed_mean=False):
    """Plot one day's hourly multiplier and persistent executed tap ratios."""
    hours = np.arange(24)
    labels = tuple(ratios)
    action_count = len(ratios[labels[0]])
    fig, axes = plt.subplots(
        action_count + 1,
        1,
        figsize=(10, 2.1 * (action_count + 1)),
        dpi=100,
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, *([1] * action_count)]},
    )
    axes[0].plot(hours, multipliers, color="#4c4c4c", marker="o", linewidth=0.8)
    axes[0].set_title(f"Loading and executed regulator ratios on {day_id}")
    axes[0].set_ylabel("Loading\nmultiplier")
    axes[0].grid(alpha=0.25)
    caption = _ratio_caption(ratios, seed_mean)
    markers = ("o", "s", "^", "D", "v")
    for regulator, axis in enumerate(axes[1:], start=1):
        axis.axhline(1.0, color="#b0b0b0", linewidth=0.7, zorder=0)
        for bound in (0.90, 1.10):
            axis.axhline(
                bound,
                color="#b0b0b0",
                linewidth=0.7,
                linestyle="--",
                zorder=0,
                label="legal bounds" if regulator == 1 and bound == 0.90 else None,
            )
        for label_index, label in enumerate(labels):
            values = ratios[label][regulator - 1]
            axis.step(
                hours,
                values,
                where="post",
                marker=markers[label_index % len(markers)],
                markersize=2.5,
                color=colors[label],
                linestyle=linestyles[label],
                label=label,
            )
        axis.set_ylabel(f"Regulator {regulator}\nratio (p.u.)")
        axis.set_ylim(0.88, 1.12)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Hour (count)")
    axes[-1].set_xticks(hours)
    axes[1].legend(fontsize=7, ncol=3)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", color="#666666", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1) if caption else None)
    return fig, axes


def plot_training(series, sequential, colors, linestyles):
    """Smooth physical cost series after their caller aligns and averages the seeds."""
    figure, axis = plt.subplots(figsize=(8, 4), dpi=100)
    positive = True
    for label, (episodes, costs) in series.items():
        window = min(50 if sequential else 200, len(costs))
        moving = np.convolve(costs, np.ones(window) / window, mode="valid")
        positive = positive and bool(np.all(moving > 0))
        axis.plot(
            episodes[window - 1 :],
            moving,
            color=colors[label],
            linestyle=linestyles[label],
            label=label,
        )
    metric = "combined cost per day" if sequential else "voltage cost per scenario"
    axis.set_title(f"Training {metric}")
    axis.set_xlabel(
        "Episode / complete day (count)"
        if sequential
        else "Episode / environment interaction (count)"
    )
    scale = "log" if positive else "linear"
    axis.set_yscale(scale)
    axis.set_ylabel(f"Training {metric} ({scale} scale)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis


def plot_beta_training(series, colors, linestyles):
    """Smooth seed-mean daily voltage and movement curves with the sequential window."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    for axis, column, title in zip(
        axes,
        ("voltage_cost", "tap_steps"),
        ("Training voltage cost per day", "Training tap steps per day"),
        strict=True,
    ):
        positive = True
        for label, (episodes, values) in series[column].items():
            window = min(50, len(values))
            moving = np.convolve(values, np.ones(window) / window, mode="valid")
            positive = positive and bool(np.all(moving > 0))
            axis.plot(
                episodes[window - 1 :],
                moving,
                color=colors[label],
                linestyle=linestyles[label],
                label=label,
            )
        axis.set_yscale("log" if positive and series[column] else "linear")
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.set_xlabel("Episode / complete day (count)")
        axis.grid(alpha=0.25)
        if series[column]:
            axis.legend()
    figure.tight_layout()
    return figure, axes


def plot_beta_tradeoff(points, colors):
    """Show individual runs and larger cell means in physical tap-deviation coordinates."""
    figure, axis = plt.subplots(figsize=(8, 4), dpi=100)
    for label, (steps, deviations) in points.items():
        axis.scatter(steps, deviations, s=20, color=colors[label], alpha=0.65)
        axis.scatter(
            [np.mean(steps)],
            [np.mean(deviations)],
            s=110,
            color=colors[label],
            edgecolors="black",
            label=label,
        )
    axis.set_xlabel("Mean tap steps/day")
    axis.set_ylabel("Mean absolute deviation (p.u.)")
    axis.set_title("Tap movement and voltage deviation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.text(
        0.5,
        0.01,
        "Small markers: individual runs. Large markers: cell seed means.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure, axis


def plot_beta_distributions(rows, colors):
    """Pool saved days across seeds into cell-specific voltage and movement boxes."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    labels = tuple(rows)
    for axis, column, title in zip(
        axes, ("voltage_cost", "tap_steps"), ("Daily voltage cost", "Daily tap steps"), strict=True
    ):
        values = [[float(row[column]) for row in rows[label]] for label in labels]
        boxes = axis.boxplot(values, tick_labels=labels, patch_artist=True)
        for patch, label in zip(boxes["boxes"], labels, strict=True):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.65)
        if all(value > 0 for cell in values for value in cell):
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.5,
        0.01,
        "Each box pools all evaluated days across the selected seeds.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure, axes


def _hourly_axis(axis, series, colors, linestyles, title, ylabel):
    """Draw ready numeric series at the 24 hourly positions."""
    hours = np.arange(24)
    for label, values in series.items():
        axis.step(
            hours,
            values,
            where="mid",
            marker="o",
            markersize=2.5,
            color=colors[label],
            linestyle=linestyles[label],
            label=label,
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(hours)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=3)


def plot_hourly_steps(day_id, series, colors, linestyles, *, seed_mean=False):
    """Plot mean or individual hourly tap-step counts without rounding."""
    figure, axis = plt.subplots(figsize=(10, 3.4), dpi=100)
    _hourly_axis(
        axis,
        series,
        colors,
        linestyles,
        f"Hourly tap movement on {day_id}",
        "Tap steps in the hour (count)",
    )
    axis.set_xlabel("Hour (count)")
    if seed_mean:
        figure.text(0.5, 0.01, SEED_MEAN_CAPTION, ha="center", fontsize=8, color="#666666")
    figure.tight_layout(rect=(0, 0.08, 1, 1) if seed_mean else None)
    return figure, axis


def plot_hourly_voltage(day_id, series, colors, linestyles, *, focused=None):
    """Plot hourly voltage costs with an optional repeated focused panel."""
    count = 2 if focused else 1
    figure, axes = plt.subplots(count, 1, figsize=(10, 3.1 * count), dpi=100, sharex=True)
    axes = np.atleast_1d(axes)
    _hourly_axis(
        axes[0], series, colors, linestyles, f"Hourly voltage cost on {day_id}", "Voltage cost"
    )
    if focused:
        _hourly_axis(
            axes[1],
            {label: series[label] for label in focused},
            colors,
            linestyles,
            "Focused view: " + " and ".join(focused),
            "Voltage cost",
        )
    axes[-1].set_xlabel("Hour (count)")
    figure.tight_layout()
    return figure, axes
