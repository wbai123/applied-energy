# Learned regulator control on three feeders

This repository studies `ieee13` and `ieee37mod`. The `ieee37std` case and datasets stay in the tree without current results.
It contains one-shot DDPG, sequential DDPG, sequential TD3, and two reference controllers.

## Setup

```bash
uv sync
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
uv run src/datasets/generate.py --feeder ieee13 --dataset ieee13-random
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

`uv.lock` pins the packages and DSS-Extensions engine.
Run every command from the repository root.

## Structure

| Location | Contents |
|---|---|
| `src/datasets/` | Deterministic named dataset generator. |
| `src/feeder/` | Named simulator, physical rules, load readers, and environments. |
| `src/common/` | Rewards, observations, metrics, shared notebook workflow and reports, evaluation, and run records. |
| `src/ddpg/` | One-shot DDPG experiment. |
| `src/reference/` | Neutral and native RegControl static tests. |
| `src/seq_ddpg/` | Sequential DDPG pilot on complete days. |
| `src/td3/` | Canonical continuous-action TD3 learner and checkpoint loader. |
| `src/seq_td3/` | Sequential TD3 training and evaluation on complete days. |
| `src/seq_reference/` | Neutral and native RegControl day tests. |
| `data/<feeder>/` | Three feeder cases with `feeder.toml` settings. |
| `data/scenarios/` | Named datasets with three splits and generation settings. |
| `data/sources/` | Original ERCOT hourly archives and supervisor load report. |
| `experiments/<feeder>/` | One-shot and sequential notebooks. |
| `experiments/paper_figures.ipynb` | Cross-feeder DDPG and RegControl paper figures. |
| `docs/` | Reproducibility instructions. |
| `papers/` | Research sources managed separately. |
| `tmp/` | Ignored scratch files. |

`FeederSimulator(feeder)` owns one DSS-Extensions context.
`solve` takes absolute loads and one position per controlled transformer.
`solve_native` takes absolute loads and the previous positions.
`solve_native_from_case` takes absolute loads only and lets the compiled case's controls choose.
All three operations return an independent `SolveResult` copy.

`OneShotEnv` makes one decision after a neutral reset.
`DayEnv` makes 24 decisions while loads advance and taps persist.
Both return physical states and outcomes; training applies the reward registry in `src/common/rewards.py` and observation functions from `src/common/`.
See [rewards and observations](docs/rewards_and_observations.md) for formulas, units, and run records.

## Dataset

Each named random dataset uses one multiplier per hourly row, repeated across its feeder's loads.
The generator creates deterministic train, validation, and test splits.

```bash
uv run src/datasets/generate.py --feeder ieee13 --dataset ieee13-random
uv run src/datasets/generate.py --feeder ieee37mod --dataset ieee37mod-random
```

Regenerating inputs requires researchers to retrain controllers and revalidate outputs.

The ERCOT converter builds measured daily shapes and a separate two-day supervisor benchmark for IEEE13.
See [ERCOT inputs](data/scenarios/ieee13-ercot/README.md) for scaling, exclusions, and chronological splits.

```bash
uv run src/datasets/ercot.py archive --feeder ieee13 --dataset ieee13-ercot
uv run src/datasets/ercot.py report --feeder ieee13 --dataset ieee13-ercotsup
uv run src/datasets/ercot.py archive --feeder ieee37mod --dataset ieee37mod-ercot
uv run src/datasets/ercot.py report --feeder ieee37mod --dataset ieee37mod-ercotsup
```

## Commands

```bash
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000 --reward band --run band_seed0
uv run src/ddpg/test.py --dataset ieee13-random --run band_seed0 --split validation
uv run src/ddpg/test.py --dataset ieee13-random --run band_seed0 --split test
uv run src/reference/test.py --dataset ieee13-random --controller neutral --split test
uv run src/reference/test.py --dataset ieee13-random --controller regcontrol --split test
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
uv run src/seq_ddpg/test.py --dataset ieee13-ercotsup --train-dataset ieee13-ercot --run band_b1e-05_seed0 --split test
uv run src/seq_reference/test.py --dataset ieee13-random --controller neutral regcontrol --beta 1e-5 --split test
```

Learned tests require compatible checkpoints.
The reward matrix compares `abs` alpha `0.1`, `sq` alpha `0.001`, and `exp` alpha `1e-05`.
Replace `ieee13` with `ieee37mod` for the second feeder.
The default seed set is 0, 1, and 2; run each training command once per seed.

Run outputs live under `results/<dataset>/<algorithm>/<run>/`.
Do not overwrite existing output silently.
Every training command names its run with `--run`; the notebooks state the same names.
Trainers refuse existing output unless `--overwrite` is passed.
"# applied-energy" 
