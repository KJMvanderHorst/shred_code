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

Tentative (3 seeds, one flow).

1. **Fault-agnostic training is brittle.** Noise-trained SHRED/SDN go from 0.004 clean
   to 0.16–0.18 at p = 0.3; a single missing probe (k = 1) already costs 0.08–0.09, 20× clean.
2. **Dropout training fixes it, at a clean cost.** Every dropout-trained model stays at
   0.011–0.024 at p = 0.3, but clean error rises ≈ 2.5× (0.004 → 0.010).
3. **No architecture ranking is supported.** SHRED+mask 0.0113 ± 0.002 is marginally best;
   SHRED 0.0145, RobustSHREDv1 0.0146, v2 0.0143 tie, and their order flips per seed
   (p = 0.3 by seed 0/1/2: SHRED 0.0097/0.0159/0.0180, v1 0.0162/0.0135/0.0140,
   v2 0.0166/0.0133/0.0131). v1 is only slightly better at k = 7 (0.126 vs 0.153; one
   level, 3 seeds). Possible reason (not tested): a missing probe reads a constant 0 at
   every lag, so it is self-flagging and plain SHRED can infer the mask.
4. **Time history matters.** SDN (last lag only) is ≈ 2× worse than SHRED (0.024 vs 0.0145).
5. **Graceful, then a cliff.** Dropout-trained models stay at 0.01–0.03 up to k = 5 of 8,
   then 0.05–0.11 at k = 6 and 0.13–0.21 at k = 7.
6. **Gappy POD has no redundancy.** Best clean (0.0007), but 0.014 with noise and 0.07 at
   k = 1: 8 probes × 2 channels = 16 measurements for 16 modes, so losing one probe makes
   the least-squares problem underdetermined.
7. **v dominates the error.** err_v ≈ 5× err_u for every model (e.g. SHRED+mask at
   p = 0.3: u 0.006, v 0.034).

**Validity.** What these runs can and cannot show:
- *Data vs parameters.* 200 snapshots, 10 lags → ≈ 191 windows, ≈ 153 train / 19 val /
  19 test. Every model has ≈ 4.2 M parameters, of which the decoder (64 → 350 → 400 → 10 000,
  4.17 M) is identical across methods; the encoders that define the "methods" are small
  (SHRED GRU ≈ 41 k, RobustSHREDv1 ≈ 51 k, v2 ≈ 100 k; 1–2 % of the total).
- *What is learned.* The Re = 100 wake is periodic and the split is interleaved (each test
  window has training windows one step away), so the decoder can in effect memorise the
  training snapshots and the task is close to recovering the shedding phase from the
  probes (interpretation, not measured). The experiment compares how robustly the
  encoders recover it; absolute errors are interpolation errors, not reconstruction of
  unseen flow.
- *Noise sources.* Hyperparameters (lr, hidden 64, early stopping) were set for SHRED and
  not tuned per method; early stopping selects on 19 windows with one corruption draw;
  a seed changes split, placement, initialisation and faults at once.
- *What stands.* Large (≥ 3×) or structural effects: noise-only vs dropout training,
  gappy POD's lack of redundancy. Architecture differences among dropout-trained models do not.

Other caveats: the zero fill is an easy fault to detect; the v1/v2 divergence from the
meeting run is unexplained.
