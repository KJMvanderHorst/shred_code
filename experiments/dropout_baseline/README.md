# dropout_baseline

**Status:** tentative

## Question

How much does reconstruction degrade when probes drop out (p = 0.3), and which
methods stay robust when trained with dropout?

## Protocol deviations

None.

## Run set

`configs/experiment/dropout_baseline.yaml` on `nektar_uv`, seeds 0, 1, 2:
`shred`, `sdn` (each trained with `noise` and with `missing`), `senseiver_sdn`,
`robust_shred_v1`, `robust_shred_v2` (trained with `missing`), and `gappy_pod`.
Conditions: `missing` with `oracle` flags at p = 0 and p = 0.3.

```
uv run python scripts/run.py +experiment=dropout_baseline
```

## Results

From the original meeting run (2026-10-14, `pre-restructure` tag,
`meeting_runner.py`). The restructured runner reproduces that pipeline
bit for bit (checked at 20 epochs, seed 0), so these numbers stand for this
experiment. Relative L2 error, mean ± std over 3 seeds. `shred_mask__missing`
is not in the run set; it was run in the meeting as `SHRED-dropout+mask`.

| Run | Clean | p = 0 (noise) | p = 0.3 |
|---|---:|---:|---:|
| `shred__noise` | 0.0040 ± 0.0006 | 0.0047 ± 0.0006 | 0.1589 ± 0.0062 |
| `shred__missing` | 0.0102 ± 0.0027 | 0.0108 ± 0.0025 | 0.0145 ± 0.0035 |
| `shred_mask__missing` | 0.0099 ± 0.0013 | 0.0102 ± 0.0013 | 0.0113 ± 0.0018 |
| `sdn__noise` | 0.0040 ± 0.0004 | 0.0071 ± 0.0006 | 0.1803 ± 0.0061 |
| `sdn__missing` | 0.0101 ± 0.0012 | 0.0127 ± 0.0014 | 0.0243 ± 0.0017 |
| `senseiver_sdn__missing` | 0.0152 ± 0.0017 | 0.0169 ± 0.0017 | 0.0283 ± 0.0046 |
| `robust_shred_v1__missing` | 0.0108 ± 0.0015 | 0.0115 ± 0.0014 | 0.0150 ± 0.0011 |
| `robust_shred_v2__missing` | 0.0094 ± 0.0009 | 0.0103 ± 0.0012 | 0.0151 ± 0.0019 |
| `gappy_pod__clean` | 0.0007 ± 0.0000 | 0.0139 ± 0.0003 | 0.0916 ± 0.0089 |

## Conclusion

Not written yet.
