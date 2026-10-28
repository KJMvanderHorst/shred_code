# get-shredded

How does sparse-sensor reconstruction degrade under sensor faults, and which
methods make it robust?

The flow is reconstructed from a few probes: SHRED (a recurrent decoder over
a window of sensor readings), its static ablation SDN, the attention-based
RobustSHRED v1/v2 and SenseiverSDN, which can ignore flagged probes, and
gappy POD as a linear baseline. Probes fail during training and evaluation:
they go missing or emit garbage, with the fault either flagged (oracle) or
undetected (silent).

## Data

- `data/cylinder_nektar_wake.mat`: Nektar cylinder wake. Each probe reads u and v. This is the research dataset.
- `data/wave_1d.npz`: synthetic 1D wave (`get_shredded.wave_data.generate_wave_dataset`). Only for quick checks that a new method or fault runs.

`data/` is not in git.

## Setup and quickstart

```
uv sync
uv run pytest
uv run python scripts/run.py +experiment=dropout_baseline data=wave train.epochs=20 'eval.seeds=[0]'
```

The last command trains every method of the baseline experiment briefly on the
wave data. It prints one JSON line per run and writes
`outputs/dropout_baseline/wave/summary.json`. Runs are resumable: a completed
run with the same config is skipped.

## Layout

```
configs/config.yaml       defaults: sensors, noise, training, evaluation
configs/data/             nektar_uv, wave
configs/method/           one file per method with its hyperparameters
configs/corruption/       clean, noise, missing, garbage
configs/experiment/       one file per experiment: its runs and conditions
experiments/LOG.md        lab notebook: every experiment and feature, newest first
experiments/<name>/       README per experiment: question, run set, results
docs/protocol.md          every protocol choice, its default and caveats
scripts/run.py            the only entry point
src/get_shredded/         data, corruption, models, runner, plots
notebooks/                analysis of outputs/<name>/<data>/summary.json
```

The pre-restructure code (CYLINDER_ALL pipeline, meeting runner and notebooks)
is in the git tag `pre-restructure`.
