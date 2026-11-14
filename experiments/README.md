# Experiment notebooks

Start each kernel in its notebook's own directory.
Use the project environment or an existing kernel with the pinned dependencies.
The notebooks import project modules from `src/`; they install no packages.

`ieee13/` and `ieee37mod/` each contain one-shot DDPG, sequential DDPG, and sequential TD3 notebooks.
The sequential TD3 notebook is `sq_random_td3.ipynb` in each feeder directory.
`ieee13/sq_ercot_ddpg.ipynb` views the measured ERCOT validation runs.
`ieee13/sq_ercotsup_ddpg.ipynb` views sequential DDPG and references on the two report days, with hourly details.
`ieee37mod/sq_ercot_ddpg.ipynb` views that feeder’s measured ERCOT validation runs.
`ieee37mod/sq_ercotsup_ddpg.ipynb` views that feeder’s two-day report benchmark.
Each feeder directory also holds `os_random_ddpg_reward_matrix.ipynb`, `sq_random_ddpg_reward_matrix.ipynb`, and `sq_ercot_ddpg_reward_matrix.ipynb` for the three-cell deviation reward matrix.
Each matrix notebook only reads the `alpha = 0` control runs of `os_random_ddpg.ipynb`, `sq_random_ddpg.ipynb`, or `sq_ercot_ddpg.ipynb`.
Run the control notebook before its matrix notebook.
In a matrix notebook, `CELLS` lists each cell's deviation, alpha, and run name.
The three cells are `abs` alpha `0.1`, `sq` alpha `0.001`, and `exp` alpha `1e-05`.
One-shot run prefixes are `band_abs_a0.1`, `band_sq_a0.001`, and `band_exp_a1e-05`.
Sequential run prefixes are `band_b1e-05_abs_a0.1`, `band_b1e-05_sq_a0.001`, and `band_b1e-05_exp_a1e-05`.
Each prefix gains `_seed0`, `_seed1`, or `_seed2`.
Matrix tables show each cell and the `alpha = 0` baseline, without reference controllers.
Matrix figures reuse the standard training, cost, and ratio plots with one arithmetic seed mean per cell and baseline.
Sequential matrices use `PLOT_DAY` for ratios, hourly tap steps, and a single hourly voltage-cost panel.
Training records cover every cell and the baseline.
Training figures plot smoothed seed means of fixed evaluation costs on exploratory episodes, using daily combined cost or one-shot voltage cost.
Each feeder directory also holds `sq_random_ddpg_beta_matrix.ipynb` and `sq_ercot_ddpg_beta_matrix.ipynb` for the switching-cost beta matrix.
The beta cells are `1e-3`, `1e-5`, and `1e-7` with the `abs` alpha `0.1` reward.
Run prefixes are `band_b0.001_abs_a0.1`, `band_b1e-05_abs_a0.1`, and `band_b1e-07_abs_a0.1`.
The beta `1e-5` cell is the benchmark row.
Each cell's combined cost uses its own beta and is not comparable across cells.
Set `BENCHMARK_BETA` and `CELLS` in the parameter cell.
The ERCOT control and matrix notebooks use 7,985 training days per run, one full pass over the archive training split.
The report benchmark loads policies from `ieee13-ercot` through `train_dataset` and keeps its saved mode at `view`.
Each notebook names its feeder and dataset in its parameter cell.

`paper_figures.ipynb` in this directory draws seven figures that compare sequential DDPG with RegControl across both feeders and both loading settings.
Start its kernel in `experiments/`.
It reads saved validation results in `view` mode and never trains or evaluates.
Its first section lists the assumptions behind the figures by number.
`SEEDS` defaults to seeds 0 to 9 for this notebook only; the other notebooks keep seeds 0, 1, and 2.
`PLOT_DAY` and `PLOT_SEED` select the single run shown in the two selected-day figures.
`SETTINGS` lists the four reward settings of the supplementary preference figure.
Make sure every listed seed has saved validation results before running it.

Edit the parameter cell, restart the kernel, and run all cells.
`MODE = "train"` trains and evaluates fresh runs.
`MODE = "evaluate"` evaluates existing checkpoints, while `MODE = "view"` reads saved results without launching subprocesses.
`OVERWRITE` permits writing over selected outputs in `train` and `evaluate` modes.
The former `TRAIN = True` maps to `MODE = "train"`.
The former `TRAIN = False` with evaluation maps to `MODE = "evaluate"`; use `view` to display saved results.
The default seed set is `[0, 1, 2]`, and `SPLIT` defaults to `validation`.
`EVALUATION_DAYS = None` evaluates the full split.
One-shot notebooks expose `REWARD` and `REWARD_PARAMETERS` in the parameter cell; the former deviation weight parameter is removed.
Sequential notebooks expose `REWARD`, `REWARD_PARAMETERS`, and `BETA`, with 1,000 days and beta `1e-5` by default.
`REWARD_PARAMETERS` overrides the defaults of the registered reward; one-shot runs keep `beta` at zero.
`RUN_NAME` names the run directories; the workflow appends `_seed<seed>`.
The workflow selects sequential DDPG with `"algorithm": "ddpg"` and sequential TD3 with `"algorithm": "td3"`.
TD3 is unavailable for the one-shot protocol.
The optional `"train_dataset"` setting defaults to `"dataset"`.
A differing value supports only sequential `evaluate` and `view` modes and requires the same feeder.
The workflow reads checkpoints, settings, and training records from `train_dataset`, while evaluation inputs and outputs use `dataset`.
Cross-dataset `view` requires both CSVs and a matching `source.json` naming the training dataset, algorithm, and run.
Cross-dataset `evaluate` protects the entire output triple; `OVERWRITE = True` replaces it and repairs missing or invalid provenance.
The workflow validates provenance after evaluation and reuses compatible reference outputs on the evaluation dataset.
Neither mode requires the evaluation dataset's train split.
The selected reward applies during training; evaluation uses fixed operational metrics.
`SHOW_HOURLY_DETAILS` controls the optional selected-day table.

Sequential workflow preflight validates each reference output pair before launching a command.
It reuses every complete compatible Neutral or RegControl pair, even when `OVERWRITE = True` replaces learned outputs.
It evaluates only missing reference controllers.
Run the sequential reference evaluator directly with `--overwrite` to deliberately recompute compatible references.

Make sure existing outputs are protected after a run by setting `OVERWRITE = False`.

## Writing a new notebook

- Use notebooks as research control tools and feedback loops. Keep stable implementation in source modules.
- Reuse existing simulation, algorithm, reward, observation, execution, and reporting code.
- Keep substantial experiment-specific code in one notebook when only that notebook uses it. Do not extract code for length alone.
- Move behavior into a descriptive source function when two notebooks need it. Update both callers instead of copying cells.
- Parameters, short explanations, and short calls may repeat. Refactor behavior that would otherwise need synchronized fixes.
- Separate execution from presentation. Plot edits and example selection must not retrain or reevaluate.
- Select feeder, algorithm, protocol, and settings through the workflow. Keep algorithm training in its algorithm directory.
- Keep one-off scientific choices readable and editable. Avoid generalizing them before another use appears.
- Follow the established layout, kernel setup, naming, and concise style. Do not install packages inside notebooks.
- Keep reproducible outputs embedded in executed notebooks. Store durable CSV and checkpoint outputs in the results layout, without HTML exports.
- Split a report function with a large one-shot block and a large sequential block into `_oneshot` and `_seq` implementations behind a short dispatcher.
- Keep small protocol differences followed by shared work in one function.

A one-shot notebook can call the shared workflow directly:

```python
from common.workflow import run_experiment

results = run_experiment({
    "protocol": "one_shot",
    "algorithm": "ddpg",
    "feeder": "ieee13",
    "dataset": "ieee13-random",
    "seeds": [0, 1, 2],
    "mode": "view",
    "overwrite": False,
    "episodes": 20000,
    "split": "validation",
    "evaluation_days": None,
    "reward": "band",
    "reward_parameters": {},
    "run_name": "band",
    "root": ROOT,
})
```
