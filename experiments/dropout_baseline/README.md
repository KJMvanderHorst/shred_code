# dropout_baseline

**Status:** tentative
**Code:** branch `exp/rerun-baselines` (rerun on `runner.py`); original: `meeting_runner.py` (PR #7, tag `pre-restructure`)

## Question

How much does reconstruction degrade when probes drop out, and which methods
stay robust when trained with dropout?

## Protocol deviations

None. Differences from the 2026-10-14 meeting run:
- `senseiver_sdn__missing` dropped (expensive, opt-in for final experiments).
- `shred_mask__missing` added (it was in the meeting as `SHRED-dropout+mask`).
- Exactly-k evaluation (`ks: all`) added next to the meeting's per-probe p.

## Run set

`configs/experiment/dropout_baseline.yaml` on `nektar_uv`, seeds 0, 1, 2, 10 test draws per window:
`shred`, `sdn` (each trained with `noise` and with `missing`), `shred_mask`,
`robust_shred_v1`, `robust_shred_v2` (trained with `missing`), and `gappy_pod`.
Conditions: clean; `missing` with `oracle` flags at p = 0 and p = 0.3, and at exactly k = 0..7.

```
uv run python scripts/run.py +experiment=dropout_baseline
```

Figures: `notebooks/dropout_baseline.ipynb` (saved to `outputs/dropout_baseline/nektar_uv/figures/`).

## Results

Rerun 2026-11-07, `outputs/dropout_baseline/nektar_uv/summary.json`. Relative L2 error,
mean ± std over 3 seeds. The p = 0.3 column of the meeting run is shown for comparison.

| Run | Clean | p = 0 (noise) | p = 0.3 | Meeting p = 0.3 |
|---|---:|---:|---:|---:|
| `shred__noise` | 0.0040 ± 0.0006 | 0.0047 ± 0.0006 | 0.1589 ± 0.0062 | 0.1589 ± 0.0062 |
| `shred__missing` | 0.0102 ± 0.0027 | 0.0108 ± 0.0025 | 0.0145 ± 0.0035 | 0.0145 ± 0.0035 |
| `shred_mask__missing` | 0.0099 ± 0.0013 | 0.0102 ± 0.0013 | 0.0113 ± 0.0018 | 0.0113 ± 0.0018 |
| `sdn__noise` | 0.0040 ± 0.0004 | 0.0071 ± 0.0006 | 0.1803 ± 0.0061 | 0.1803 ± 0.0061 |
| `sdn__missing` | 0.0101 ± 0.0012 | 0.0127 ± 0.0014 | 0.0243 ± 0.0017 | 0.0243 ± 0.0017 |
| `robust_shred_v1__missing` | 0.0101 ± 0.0008 | 0.0110 ± 0.0008 | 0.0146 ± 0.0012 | 0.0150 ± 0.0011 |
| `robust_shred_v2__missing` | 0.0098 ± 0.0007 | 0.0107 ± 0.0011 | 0.0143 ± 0.0016 | 0.0151 ± 0.0019 |
| `gappy_pod__clean` | 0.0007 ± 0.0000 | 0.0139 ± 0.0003 | 0.0916 ± 0.0089 | 0.0916 ± 0.0089 |

Verification: SHRED, SHRED+mask, SDN and gappy POD reproduce the meeting bit for
bit. RobustSHREDv1/v2 differ per seed (e.g. v1 seed 1 clean 0.0129 → 0.0090), within
the seed spread; same model code and device (MPS), and the new pipeline is
deterministic (identical models in `random_readings`). The restructure check
covered 20 epochs only; the cause of the divergence in full training is not identified.

Exactly k of 8 probes missing (oracle), mean over 3 seeds:

| Run | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shred__noise` | 0.0047 | 0.0806 | 0.1340 | 0.1754 | 0.2165 | 0.2718 | 0.3083 | 0.3458 |
| `shred__missing` | 0.0108 | 0.0111 | 0.0121 | 0.0137 | 0.0170 | 0.0284 | 0.0662 | 0.1534 |
| `shred_mask__missing` | 0.0102 | 0.0092 | 0.0095 | 0.0096 | 0.0116 | 0.0190 | 0.0511 | 0.1512 |
| `sdn__noise` | 0.0071 | 0.0917 | 0.1484 | 0.1975 | 0.2450 | 0.2931 | 0.3324 | 0.3713 |
| `sdn__missing` | 0.0127 | 0.0141 | 0.0165 | 0.0197 | 0.0258 | 0.0471 | 0.1060 | 0.2071 |
| `robust_shred_v1__missing` | 0.0110 | 0.0115 | 0.0125 | 0.0146 | 0.0187 | 0.0274 | 0.0613 | 0.1258 |
| `robust_shred_v2__missing` | 0.0107 | 0.0110 | 0.0114 | 0.0129 | 0.0165 | 0.0266 | 0.0700 | 0.1459 |
| `gappy_pod__clean` | 0.0139 | 0.0704 | 0.0706 | 0.0962 | 0.1009 | 0.1547 | 0.1961 | 0.3541 |

## Conclusion

Not written yet.
