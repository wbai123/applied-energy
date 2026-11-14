# Reproducibility

This file defines the environment, commands, inputs, and output directories.

## Environment

Use Linux, macOS, or WSL2 with Python 3.14.
`uv.lock` pins every package and the DSS-Extensions engine.

```bash
uv sync
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

The fixed thread count preserves seeded reduction order.
`PYTHONUNBUFFERED=1` displays progress immediately.

## Commands

```bash
uv run src/datasets/generate.py --feeder ieee13 --dataset ieee13-random
uv run src/datasets/ercot.py archive --feeder ieee13 --dataset ieee13-ercot
uv run src/datasets/ercot.py report --feeder ieee13 --dataset ieee13-ercotsup
uv run src/datasets/ercot.py archive --feeder ieee37mod --dataset ieee37mod-ercot
uv run src/datasets/ercot.py report --feeder ieee37mod --dataset ieee37mod-ercotsup
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000 --reward band --run band_seed0
uv run src/ddpg/test.py --dataset ieee13-random --run band_seed0 --split validation
uv run src/ddpg/test.py --dataset ieee13-random --run band_seed0 --split test
uv run src/reference/test.py --dataset ieee13-random --controller neutral --split test
uv run src/reference/test.py --dataset ieee13-random --controller regcontrol --split test
uv run src/seq_reference/test.py --dataset ieee13-random --controller neutral regcontrol --beta 1e-5 --split test
uv run src/seq_ddpg/train.py --dataset ieee13-random --seed 0 --days 1000 --reward band --beta 1e-5 --run band_b1e-05_seed0
uv run src/seq_ddpg/test.py --dataset ieee13-random --run band_b1e-05_seed0 --split validation
uv run src/seq_ddpg/test.py --dataset ieee13-random --run band_b1e-05_seed0 --split test
uv run src/seq_td3/train.py --dataset ieee13-random --seed 0 --days 1000 --reward band --beta 1e-5 --run band_b1e-05_seed0
uv run src/seq_td3/test.py --dataset ieee13-random --run band_b1e-05_seed0 --split validation
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000 --reward band --deviation sq --alpha 0.01 --run band_sq_a0.01_seed0
uv run src/ddpg/test.py --dataset ieee13-random --run band_sq_a0.01_seed0 --split validation
uv run src/seq_ddpg/train.py --dataset ieee13-random --seed 0 --days 1000 --reward band --beta 1e-5 --deviation exp --alpha 0.1 --run band_b1e-05_exp_a0.1_seed0
uv run src/seq_ddpg/test.py --dataset ieee13-random --run band_b1e-05_exp_a0.1_seed0 --split validation
uv run src/seq_ddpg/train.py --dataset ieee13-ercot --seed 0 --days 7985 --reward band --beta 1e-5 --run band_b1e-05_seed0
uv run src/seq_ddpg/test.py --dataset ieee13-ercot --run band_b1e-05_seed0 --split validation
```

The reward matrix compares `abs` alpha `0.1`, `sq` alpha `0.001`, and `exp` alpha `1e-05`.
One-shot prefixes are `band_abs_a0.1`, `band_sq_a0.001`, and `band_exp_a1e-05`; sequential prefixes add `b1e-05_` after `band_`.
Append `_seed<seed>` to each prefix.
Matrix notebooks read validation results and compare these cells with `alpha = 0`; their tables and figures omit reference controllers.
Figures average saved seed values before plotting; tables retain sample standard deviations.
The exponential cell uses these commands for each dataset and seed:

```bash
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000 --reward band --deviation exp --alpha 1e-05 --run band_exp_a1e-05_seed0
uv run src/ddpg/test.py --dataset ieee13-random --run band_exp_a1e-05_seed0 --split validation
uv run src/seq_ddpg/train.py --dataset ieee13-random --seed 0 --days 1000 --reward band --beta 1e-5 --deviation exp --alpha 1e-05 --run band_b1e-05_exp_a1e-05_seed0
uv run src/seq_ddpg/test.py --dataset ieee13-random --run band_b1e-05_exp_a1e-05_seed0 --split validation
```

Replace `ieee13` with `ieee37mod` for the second feeder.
The default seed set is 0, 1, and 2.
Run each training and learned test command once per seed.

## Inputs

The three feeder inputs live under `data/<feeder>/` beside `feeder.toml`.
Named inputs live under `data/scenarios/<dataset>/`.
Each dataset uses one uniform multiplier per hourly row, repeated across its loads.
Each CSV header starts with `day_id,hour,` and continues with the compiled load order.
`generation.json` records the feeder and generation settings.
Each split uses `np.random.default_rng([seed, k])`, with a distinct `k`.
Training order uses `np.random.default_rng([seed, 1])`.
Each shuffled pass visits every available item once.
The requested training budget may stop inside a shuffled pass.
Regenerated inputs require researchers to retrain controllers and revalidate outputs.

`loads.py` converts multiplier rows to absolute kW and kvar.
`FeederSimulator` receives absolute load arrays in its frozen `load_names` order.

ERCOT sources live in `data/sources/ercot/` and `data/sources/ercotsup/` with original bytes.
`ieee13-ercot` uses measured system load from January 2002 through September 2026.
Each hourly multiplier equals load divided by its calendar-year peak.
Peaks include finite source values before daylight-saving and Uri days are dropped.
Training covers 2002–2023; validation covers 2024–September 2025; the reserved test split covers October 2025–September 2026.
`ieee13-ercotsup` contains October 8–9, 2026, scaled by the 2026 archive peak.
The report converter fills at most one missing actual per complete day from its current-day forecast.
Both converter modes record source checksums and refuse existing datasets without `--overwrite`.
See [ERCOT dataset notes](../data/scenarios/ieee13-ercot/README.md) for details.
The ieee37 ERCOT datasets share the ieee13 multipliers, archive splits, dropped days, and supervisor benchmark dates.

## Outputs

All outputs live under `results/<dataset>/<algorithm>/<run>/`.
Trainers write `train.csv`, `run.json`, and `checkpoints/`.
One-shot training columns are `episode,scenario,steps,score,voltage_cost,mad`; physical metrics use the fixed evaluation definitions.
Sequential training columns are `episode,day,steps,score,voltage_cost,mean_mad,tap_steps,feasible_hours,daily_cost`; daily cost uses the recorded benchmark beta.
Sequential TD3 also writes `counts.json` with environment interactions and critic, actor, and target update counts.
Its inference-only checkpoints are `checkpoints/Actor_td3` and `checkpoints/Critic_td3`.
New `run.json` files record task and agent settings, including reward, observation, action units, and network widths.
Static evaluators write `validation.csv` or `test.csv`.
Sequential evaluators also write `validation_hourly.csv` or `test_hourly.csv`.
Sequential evaluate and view modes reject a requested beta that differs from a saved benchmark beta.

Sequential evaluators accept `--train-dataset` to load checkpoints and settings from another named dataset with the same feeder.
The option defaults to `--dataset`; an explicitly equal value keeps the existing output layout.
Cross-dataset results live under `results/<dataset>/<algorithm>/<run>/`, with daily and hourly CSVs plus `source.json`.
The source record names the training dataset, algorithm, and run; evaluation leaves the training directory unchanged.
Any existing member of this output triple requires `--overwrite` to replace the whole triple.

```bash
uv run src/seq_ddpg/test.py --dataset ieee13-ercotsup --train-dataset ieee13-ercot --run band_b1e-05_seed0 --split test
```

Static evaluation columns are `scenario,k1..kn,vmin,vmax,cost,mad,viol_cnt,feasible`.
Day evaluation columns are `day,beta,voltage_cost,mean_mad,tap_steps,daily_cost,feasible_hours`.
Hourly evaluation columns are:

`day,hour,beta,k1_before..kn_before,k1..kn,tap_steps,voltage_cost,mad,switching_cost,hourly_cost,vmin,vmax,viol_cnt,feasible,converged`

`mad` is the mean absolute deviation from 1 p.u.; `mean_mad` averages the 24 hourly values.

Do not overwrite existing output silently.
Trainers refuse existing runs unless `--overwrite` is passed.
Trainers do not resume runs.

## History-free solves

Each fixed-tap solve writes every load and tap before solving.
Snapshot mode removes dependence on earlier solution paths.
Native static and native sequential operations retain their distinct initialization conventions.
