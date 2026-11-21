# <name>

**Status:** planned | running | tentative | settled
**Code:** branch `exp/<name>`, PR #<n> (merge commit `<sha>` once merged)

## Question

What this experiment answers, in one or two sentences.

## Protocol deviations

Rows of `docs/protocol.md` this experiment changes, or "none".

## Run set

`configs/experiment/<name>.yaml`: runs x conditions x seeds, written out.

```
uv run python scripts/run.py +experiment=<name>
```

Device: `cpu`, 6 workers (the default since speed_benchmark), unless pinned. Keep one device for the whole experiment.

## Results

Copied from `outputs/<name>/<data>/summary.json` (mean ± std over seeds).

## Conclusion
