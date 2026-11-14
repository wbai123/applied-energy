"""Define learning rewards from executed physical outcomes.

These formulas affect replay and training only. Evaluation uses fixed operational
metrics from ``common.metrics`` even when a selected reward changes.

``REWARDS`` maps a name to a function. A new reward is one function, one dict
entry, and, when its parameters differ from the band's, one flag line per
trainer. The movement weight of a reward, when it has one, is named ``beta``.
"""

import inspect
import math

import numpy as np

DEVIATIONS = {
    "abs": lambda deviation, rate: deviation,
    "sq": lambda deviation, rate: deviation**2,
    "exp": lambda deviation, rate: np.exp(rate * deviation) - 1.0,
}


def band_reward(
    voltages,
    tap_steps,
    *,
    beta=0.0,
    deviation="abs",
    alpha=0.0,
    rate=20.0,
    voltage_min=0.95,
    voltage_max=1.05,
    scale=1000.0,
):
    """Return the negative scaled band, deviation, and movement cost.

    The reward is ``-scale * (C + alpha * D + beta * M)``. ``C`` is the mean,
    over monitored p.u. voltages ``v``, of the squared distance below
    ``voltage_min`` or above ``voltage_max``; in-band values contribute zero.
    ``D`` is the mean of one deviation kind applied per voltage to
    ``d = |v - 1|``: ``abs`` uses ``d``, ``sq`` uses ``d**2``, and ``exp`` uses
    ``exp(rate * d) - 1``. ``M`` is ``tap_steps``, the executed tap distance
    summed over regulators, including the first move from neutral. The function
    skips ``D`` when ``alpha`` is zero and ``M`` when ``beta`` is zero.
    """
    values = np.asarray(voltages, dtype=float)
    excursion = np.maximum(voltage_min - values, 0.0) + np.maximum(values - voltage_max, 0.0)
    total = float((excursion**2).mean())
    if alpha != 0:
        total += alpha * float(DEVIATIONS[deviation](np.abs(values - 1.0), rate).mean())
    if beta != 0:
        total += beta * tap_steps
    return -scale * total


# Reward name to reward function; trainers and notebooks select rewards by name.
REWARDS = {"band": band_reward}

# Allowed values of string reward parameters.
CHOICES = {"deviation": tuple(DEVIATIONS)}


def defaults(name):
    """Return the keyword-only parameters of one reward and their defaults."""
    if name not in REWARDS:
        raise ValueError(f"unknown reward: {name!r}")
    return {
        parameter.name: parameter.default
        for parameter in inspect.signature(REWARDS[name]).parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }


def reward_parameters(name, overrides):
    """Return the complete validated parameter dict of one reward.

    The result holds the signature defaults updated by ``overrides``. Every value
    keeps its default's type. Floats are finite and non-negative, ``rate`` is
    positive, and strings lie in ``CHOICES``.
    """
    parameters = defaults(name)
    if not isinstance(overrides, dict):
        raise ValueError(f"reward parameters for {name!r} must be a dictionary")
    for parameter, value in overrides.items():
        if parameter not in parameters:
            raise ValueError(f"unknown parameter {parameter!r} for reward {name!r}")
        default = parameters[parameter]
        if type(value) is not type(default):
            raise ValueError(
                f"parameter {parameter!r} must be {type(default).__name__}, not {value!r}"
            )
        if isinstance(value, float) and (not math.isfinite(value) or value < 0):
            raise ValueError(f"parameter {parameter!r} must be finite and non-negative")
        if parameter == "rate" and value == 0:
            raise ValueError("parameter 'rate' must be positive")
        if parameter in CHOICES and value not in CHOICES[parameter]:
            raise ValueError(
                f"parameter {parameter!r} must be one of {CHOICES[parameter]}, not {value!r}"
            )
        parameters[parameter] = value
    return parameters
