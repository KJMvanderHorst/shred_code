# <name>

**Status:** planned | running | tentative | settled

## Question

What this experiment answers, in one or two sentences.

## Protocol deviations

Rows of `docs/protocol.md` this experiment changes, or "none".

## Run set

`configs/experiment/<name>.yaml`: runs x conditions x seeds, written out.

```
uv run python scripts/run.py +experiment=<name>
```

## Results

Copied from `outputs/<name>/<data>/summary.json` (mean ± std over seeds).

## Conclusion
