# Rewards, observations, and run records

The physical simulator returns voltages in p.u., tap positions, tap ratios, and convergence.
`OneShotEnv` and `DayEnv` advance the physical experiment and return `PhysicalState` and `HourOutcome`.
They do not choose a learning reward.

## Change a learning reward

Edit or add a named reward function in `src/common/rewards.py`.
`REWARDS` in `src/common/rewards.py` is a dict from reward name to reward function.
Each function states its parameter defaults in its signature; `CHOICES` lists the allowed values of string parameters.
Each trainer generates one flag per default and passes one complete parameter dictionary to training and `run.json`.
A new reward is one function and one dict entry; a reward with other parameters also adds one flag line per trainer.
The movement weight of a reward, when it has one, is named `beta`.
The `band` reward adds the band cost, `alpha` times a mean voltage deviation, and `beta` times executed tap steps.
Its flags are `--beta`, `--deviation` (`abs`, `sq`, or `exp`), `--alpha`, `--rate`, `--voltage-min`, `--voltage-max`, and `--scale`.
The selected function receives the executed outcome; it does not change the physical solve.

```python
from common.rewards import REWARDS, reward_parameters

parameters = reward_parameters("band", {"beta": 1e-5})
reward = REWARDS["band"](outcome.result.voltages, outcome.tap_steps, **parameters)
```

The default `scale` of 1000 multiplies the cost in the reward.
The default `alpha` is zero, so the deviation term is off by default.
One-shot training requires `beta = 0`; sequential runs pass `--beta 1e-5`.
`--run` is required on every trainer; the notebooks state each run name.

## Change an observation

Edit `voltage_observation` or `day_observation` in `src/common/observations.py`.
Size networks with `voltage_observation_size(simulator)` or `day_observation_size(simulator)` for the encoding in use.
When an encoding changes, update its size function in `common.observations`.
The physical environments contain no observation-size logic.
The training runner calls the selected function after each environment reset and step.

```python
from common.observations import voltage_observation

state = env.reset(scenario)
observation = voltage_observation(state)
```

Physical voltages are p.u.; voltage network features are `(V - 1) / 0.1` in float32.
Day features append normalized held tap positions, then sine and cosine of the current hour.
The action proposal is a physical tap ratio in p.u.
Execution projects it to an integer tap position; replay stores the executed ratio.

## Compare runs

`src/common/metrics.py` computes fixed operational voltage cost, mean absolute deviation from 1 p.u., violations, and feasibility.
`src/common/evaluation.py` uses these metrics for learned and reference controllers.
The daily benchmark adds `beta` times executed tap steps to the voltage cost.
Static and hourly CSVs carry a `mad` column; day CSVs carry `mean_mad`, the mean of the 24 hourly values.
These comparisons do not depend on the selected training reward.
Sequential training accepts `--benchmark-beta`; it defaults to the reward beta and records the effective value.
`--run` is always explicit.
Sequential testing uses the recorded benchmark beta by default and writes `<split>.csv` and `<split>_hourly.csv`.
Its `--benchmark-beta` override writes `<split>_b<beta>.csv` and `<split>_b<beta>_hourly.csv`.
The `<beta>` text follows Python's `str(beta)` spelling.

New `run.json` files have `task` and `agent` groups.
`task` records feeder, dataset, algorithm, observation, action representation, reward, and evaluation settings.
`agent.layer_sizes` records the widths needed to reconstruct a checkpoint.
`agent.settings` is optional for algorithms whose records predate complete recipes.
Sequential TD3 requires it and records random streams, optimizer rates, replay behavior, update schedule, noise in normalized and physical units, action mapping, initialization, training budget, and inference-only checkpoint support.
`src/ddpg/checkpoints.py` validates the record and tensor shapes before evaluation writes output.
`src/td3/checkpoints.py` applies the corresponding checks to `seq_td3` records and twin-critic checkpoints.
`common.runs.resolve_run_settings` is the shared source for saved settings.
Record validation requires a registered reward name and its complete parameter dictionary.
It resolves task metadata and network widths without loading checkpoints or importing PyTorch.
