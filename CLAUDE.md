# CLAUDE.md — get-shredded

## Project

Research code for one question: how does sparse-sensor reconstruction degrade
under sensor faults, and which methods make it robust? Probes on the Nektar
cylinder wake (u, v) go missing or emit garbage, flagged (oracle) or
undetected (silent). Methods: SHRED, SDN, RobustSHRED v1/v2, gappy POD;
SenseiverSDN is expensive and opt-in (final experiments only). The synthetic
1D wave is only for quick checks that a new method or fault runs at all.

All results so far are tentative. See README.md for layout and
`docs/protocol.md` for every protocol choice.

Stack: Python >= 3.10, uv, Hydra, NumPy, SciPy, PyTorch, scikit-learn, matplotlib.

## Protocol

- Preserve experimental semantics and reproducibility.
- Do not change seeds, splits, placement, architectures, metrics, defaults or
  any row of `docs/protocol.md` unless the user asks. When a choice is unclear,
  ask instead of assuming.
- A protocol change updates `docs/protocol.md` in the same PR.
- Do not change defaults just to make a run faster.

## Workflow: idea to PR

1. Branch: `exp/<name>` (an experiment), `feat/<name>` (a core change) or `fix/<name>`.
2. Experiment: add `configs/experiment/<name>.yaml` (explicit `runs` and
   `eval.conditions`) and `experiments/<name>/README.md` from `_template`.
3. Core change: small, in the existing modules, with a test.
4. Check, in order: `uv run pytest`; a wave smoke run for a new method or fault
   (`data=wave train.epochs=20 'eval.seeds=[0]'`); a Nektar smoke run
   (`train.epochs=20 'eval.seeds=[0]'`); then the full run, only when the user
   asks for it.
5. Fill in the README results from `summary.json`, open a PR into main.

## Running experiments

Agent context is scarce.

- `scripts/run.py` is the only entry point. It is quiet (one JSON line per
  run) and resumable; never stream epoch output or add verbose logging to sweeps.
- Run exactly the requested run set. Do not add seeds, sensor counts, methods
  or conditions, and do not design follow-up experiments on your own.
- Do not rerun a completed configuration unless its config changed or the user asks.
- Launch once and let it finish; do not poll.
- Read `outputs/<name>/<data>/summary.json` first; open per-run
  `result.json` or `arrays.npz` only when needed. Never print arrays, logs
  or datasets; report metrics and paths concisely.
- No per-run plots during sweeps; aggregate plots via `get_shredded.plots`
  in notebooks.

## Code

- Small, localized changes; keep the Hydra config pattern (no parallel config mechanisms).
- Every experiment loop has an explicit, inspectable run set.
- Keep output formats unless the task requires a change.
- Do not rewrite working scientific code for style.
