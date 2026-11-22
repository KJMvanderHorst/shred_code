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

## Options for tuning

Off by default; see `docs/protocol.md`.

- `eval.split=valid`: evaluate on the validation windows instead of the test windows (for tuning; scores are
  optimistic because validation also drives early stopping). Results record `eval_split`.
- `methods.<m>.train_options: {...}`: extra keyword arguments for `fit` (only new options added to `model.py`).
- `sensors.placement: random` or `explicit` (with `sensors.locations: [...]`): other probe placements than block QR.
- A method can be defined inside an experiment file, without touching `configs/config.yaml`:

```yaml
# @package _global_
defaults:
  - /method@methods.shred_wide: shred
name: my_experiment
methods:
  shred_wide: {hidden_size: 128}
runs:
  - {method: shred_wide, train: missing}
```

## Overnight mode

`scripts/autoresearch.py` tunes overnight on the validation split, in a separate git worktree, with short
stateless Claude calls planning each batch and plain `run.py` runs (6 CPU workers) doing the training. Its rules
for the agent are in `AUTORESEARCH.md`; you write the ideas (template: `docs/ideas_template.md`).

```
uv run python scripts/autoresearch.py init --ideas my_ideas.md        # -> ../get-shredded-auto, branch auto/<date>
cd ../get-shredded-auto
uv run python scripts/autoresearch.py run --until 07:00 --max-cost-usd 20
uv run python scripts/autoresearch.py status                          # any time, from another terminal
```

In the morning read `auto/report.md` (and `auto/results.tsv`, `auto/journal.md`). Numbers are valid-split and
optimistic; promote a candidate by putting it next to its incumbent in a daytime experiment and running it on the
test split. Ctrl-C once finishes the current batch and writes the report; `report` reruns only the report.
`init --dry-run` + `run --dry-run --max-steps 3` tests the wiring on the wave data in `../get-shredded-auto-dry`.

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
scripts/run.py            the only entry point for experiments
scripts/autoresearch.py   overnight tuning driver (AUTORESEARCH.md)
src/get_shredded/         data, corruption, models, runner, plots
notebooks/                analysis of outputs/<name>/<data>/summary.json
```

The pre-restructure code (CYLINDER_ALL pipeline, meeting runner and notebooks)
is in the git tag `pre-restructure`.
