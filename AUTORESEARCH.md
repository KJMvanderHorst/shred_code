# AUTORESEARCH: rules for the overnight tuning agent

You are tuning, not doing research. The user owns research decisions, the protocol and the test set.
Each step: record the trials that finished, plan exactly one next batch, exit. The driver
(`scripts/autoresearch.py`) runs everything; you only read files and edit the files under "You may edit".
Every protocol rule in `CLAUDE.md` and `docs/protocol.md` still holds.

## Read each step (keep context small)

`auto/plan.md` · the last 5 entries of `auto/journal.md` · `auto/results.tsv` · `auto/decisions.tsv` ·
the method and corruption configs you build on (`configs/method/`, `configs/corruption/`, `configs/config.yaml`
for defaults) · a trial's `outputs_auto/<exp>/<data>/summary.json` or `seed<i>/<run>/result.json` only if
results.tsv is not enough. Never `arrays.npz` unless one specific question needs it. Never the main checkout
(`../get-shredded`) and never any `outputs/` folder.

`results.tsv` columns: `value_mean`, `value_std` (population std over seeds) and `per_seed` are the objective;
`epochs_used` is the number of validation checks per seed (one at epoch 1, then every 20); `problem_id`
identifies the reconstruction problem (data, lags, sensors, placement, noise, conditions, draws).

## Intake (first call of the night)

Turn each idea in `auto/ideas.md` into a numbered entry in `auto/plan.md`, in the user's priority order:

- id (lowercase letters and digits only, e.g. `lr`, `wd2`), title, and your interpretation if the idea is
  vague (write it down explicitly);
- a testable hypothesis;
- the reference (incumbent): an existing method and training corruption, with its problem settings;
- the objective: summary.json keys plus aggregation (`mean` default, or `max`); for robustness ideas name
  the conditions and k-range (keys look like `clean_err`, `missing_oracle_k2_err`, `garbage_silent_p0.3_err_u`);
- a budget: by default at most 4 single-seed screens and 1 three-seed confirmation;
- status: `planned` or `daytime-only`. Daytime-only means the idea needs `data.split` changes, new fault
  *types*, new metrics, new data, or edits to locked files. Do not run those.

## Trials

- One experiment file per trial: `configs/experiment/auto_<idea>_<trial>.yaml` (trial ids `t1`, `t2`, ...,
  numbered per idea and never reused). The first line is `# @package _global_`, `name:` equals the file
  name, and `runs` has exactly one entry.
- You may set: new `auto_*` methods (inline, or `- /method@methods.auto_x: <existing method>` in `defaults`
  plus field overrides under `methods.auto_x`), new `auto_*` corruptions (same pattern with
  `/corruption@corruptions.auto_x`), `runs` (`method`, `train`, `early_stopping`),
  `train.{epochs,batch_size,lr,patience}`, `eval.conditions`, `sensors.{num,rank,placement,locations}`,
  `noise_frac`, `data.lags`. Anything else is rejected. The driver sets seeds, evaluation split
  (validation), device (CPU) and outputs itself.
- The first trial of an idea re-measures the incumbent unless `results.tsv` already has it for the same
  `problem_id` and objective. Changing sensors, placement, noise_frac, data.lags or conditions makes a new
  problem: measure the incumbent under the same settings.
- Change one thing per trial unless the idea says otherwise. Screen on seed `[0]`; confirm on `[0, 1, 2]`,
  and confirm the incumbent on `[0, 1, 2]` too.
- Fill the job slots the prompt gives you (one job = one trial × one seed). Use the driver's time
  estimate; a batch that does not fit before the deadline is refused.

Example (a method variant built on an existing one):

```yaml
# @package _global_
defaults:
  - /method@methods.auto_lr_t2: flow_trunk_shred_w256_cos
name: auto_lr_t2
methods:
  auto_lr_t2: {lr: 5.0e-3}
runs:
  - {method: auto_lr_t2, train: clean}
eval:
  conditions:
    - {corruption: clean, flags: oracle, ps: [0.0]}
```

## Gate

A candidate is *better* only if its three-seed mean beats the incumbent's (same `problem_id`, same
objective) by more than the larger of the two seed stds AND it is lower on every seed (seeds share split and
placement, so compare per seed). Otherwise it is a *tie*: write it down and move on. Never call a
single-seed result better. The driver re-checks every `keep` and downgrades it to `tie` if the gate fails.
Lower is always better (every metric is an error or a distance).

## Budget discipline

When an idea's budget is spent, close it with a verdict: better / tie / worse / inconclusive / failed.
Never rewrite an idea to make it succeed, and do not drift into ideas that are not listed. Write
"Follow-up suggestions" in the journal for the user; never run them.

## Code options

Only in `src/get_shredded/model.py`, only as new off-by-default arguments: to `fit` (set per method through
`methods.<m>.train_options: {name: value}`), or to a model class (set through a method key the runner already
passes). Each option gets a test in `tests/test_auto_<name>.py`. Never change existing defaults, losses,
metrics or evaluation. The driver runs pytest and an exact-reproduction canary of the default methods; if
either fails, the whole step is reverted.

## You may edit

`configs/experiment/auto_*.yaml` · `auto/plan.md` · `auto/journal.md` · `auto/report.md` (report call only) ·
`src/get_shredded/model.py` · `tests/test_auto_*.py`. Nothing else. You have no shell, no git and no network.
There are no test-set metrics in this worktree; do not look for them.

## Journal entry

```
### <idea>/<trial> · <HH:MM>
Hypothesis: … · Changed: … (vs <incumbent experiment>)
Result: <copied from results.tsv: value_mean ± value_std, per_seed>
Decision: keep | discard | tie, because … · Next: …
```

## Report (`auto/report.md`, last call)

Most important first, at most one screen per idea:

- per idea: status (better / tie / worse / inconclusive / failed / not reached / daytime-only), the numbers
  (incumbent vs best candidate, per seed, objective), and the exact config to promote (file and method block);
- **Promote**: for each `better` idea, the daytime steps: copy the candidate method (and any `auto_*`
  corruption) into a new `configs/experiment/<name>.yaml` next to the incumbent run, with the same problem
  settings and both runs, then `uv run python scripts/run.py +experiment=<name>`. That evaluates both on the
  test split on CPU with 6 workers (the default). Never compare with old MPS results;
- failures, guard violations and gate overrides; cost; throughput per hour (copy the driver's numbers).
