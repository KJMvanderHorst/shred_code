---
applyTo: "scripts/**/*.py,configs/**/*.yaml,src/get_shredded/experiment.py,src/get_shredded/model.py,src/get_shredded/plotting.py"
---

# Experiment-code instructions

- Experiment code must distinguish scientific computation from agent-facing output.
- Long-running experiments must support a non-verbose execution mode.
- Automated sweeps should not emit epoch-by-epoch progress to stdout.
- Preserve detailed scientific results on disk while keeping stdout concise.
- Prefer structured JSON/NPZ result artifacts over large textual output.
- Experiment runners should be resumable where practical.
- Completed configurations should not be rerun unnecessarily.
- Do not silently change scientific protocol to reduce runtime.
- Do not automatically enable per-run plots, GIFs, or other large artifacts during sweeps.
- Do not add autonomous follow-up experiments.
- Any new experiment loop should have an explicit and inspectable run set.