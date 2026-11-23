# residual_flags

**Status:** tentative
**Code:** branch `exp/residual-flags`, PR #<n>

## Question

Can physics-free, residual-based fault detection replace oracle flags? A bad reading should disagree with the
field the model predicts at that probe's own location: flag probes with a large residual, then feed the flags into
the existing mask. Test time only. The question is whether this works at all, not which model is best.

## Protocol deviations

None for existing runs. New rows in `docs/protocol.md`: *Residual flags* (test-time detection) and *Partial flags*
(`flag_frac`, used only to train the one new checkpoint).

- Faults are unchanged from `random_readings` (garbage, whole window, both channels, drawn per test window and
  draw). They have no time axis, so each window has one residual per probe: its last lag. The flags cover the
  whole window (v1's and shred_mask's masks are per window per probe), and the soft weight applies at every lag.
- Pass 1 runs the model with no flags (all probes in). Score d_j = sqrt((r_u/s_u)² + (r_v/s_v)²), with s the
  residual std on the 19 clean validation windows plus the fixed validation noise. Flag if d_j > τ_j = quantile q
  of d_j on the same windows.

## Run set

Checkpoints (none retrained; all MPS, 3 seeds, standard training):

| name here | run | experiment | training corruption |
|---|---|---|---|
| v1-missing | `robust_shred_v1__missing` | random_readings | missing p = 0.3, oracle flags |
| shred_mask | `shred_mask__missing` | random_readings | missing p = 0.3, oracle flags |
| v1-garbage | `robust_shred_v1__garbage` | random_readings | garbage p = 0.3, no flags |
| v1-mixed | `robust_shred_v1__garbage_mixed` | **residual_flags_train** (new) | garbage p = 0.3, each fault flagged with prob. 0.5 |

```
uv run python scripts/run.py +experiment=residual_flags_train   # trains v1-mixed (3 runs, ~1.5 min)
uv run python scripts/run.py +experiment=residual_flags         # test-time evaluation (~30 s)
```

`configs/experiment/residual_flags.yaml`, evaluated on the garbage conditions of `random_readings` (exactly k of 8
probes faulty, k = 0..7, 10 draws per test window, same masks and garbage values) plus noise-free clean:

- step 0, self-influence and the full 16 × 16 response matrix (offset one reading at the last lag by 1 and 3 std;
  clean test windows; all four models);
- per model: oracle flags, no flags, residual flags at q = 0.95 / 0.99 / 0.999, soft (v1 models), so 6 variants for
  each v1 model and 2 for shred_mask (oracle, residual q = 0.99): 20 variants × 3 seeds;
- follow-up: `topk` for v1-missing and v1-mixed (22 variants in total).

Device: `mps` (the checkpoints' device), one process.

## Results

From `outputs/residual_flags/nektar_uv/summary.json`, mean ± std over 3 seeds; full tables and plots in
`notebooks/residual_flags.ipynb`. Every oracle and no-flag row reproduces its source experiment's cached errors
exactly (difference 0.0, all seeds).

**Step 0** (clean test windows, offset of 1 std at the last lag, response in std units, mean |·| over probes):

| model | self | other probes, same channel | other channel | max off-diagonal |
|---|---|---|---|---|
| v1-missing | 0.023 | 0.021 | 0.010 | 0.077 |
| shred_mask | 0.015 | 0.007 | 0.004 | 0.022 |
| v1-garbage | 0.012 | 0.008 | 0.006 | 0.022 |
| v1-mixed | 0.015 | 0.008 | 0.004 | 0.027 |

Self-influence s_j ≤ 0.08 on every probe and channel for every model (δ = 1 and 3 std give the same picture).

**Field error** (`err`, garbage, k of 8 probes faulty; k = 0 = noise only):

| model | flags | k = 0 | k = 1 | k = 2 | k = 4 |
|---|---|---|---|---|---|
| v1-missing | oracle | 0.0110 ± 0.0008 | 0.0115 ± 0.0009 | 0.0125 ± 0.0011 | 0.0187 ± 0.0025 |
| v1-missing | none | 0.0110 ± 0.0008 | 0.0715 ± 0.0129 | 0.1231 ± 0.0268 | 0.2153 ± 0.0259 |
| v1-missing | residual q = 0.99 | 0.0120 ± 0.0011 | 0.1333 ± 0.0166 | 0.2225 ± 0.0345 | 0.3162 ± 0.0405 |
| v1-missing | soft | 0.0129 ± 0.0017 | 0.1103 ± 0.0124 | 0.1902 ± 0.0363 | 0.2915 ± 0.0287 |
| shred_mask | oracle | 0.0102 ± 0.0013 | 0.0446 ± 0.0009 | 0.0804 ± 0.0013 | 0.2114 ± 0.0078 |
| shred_mask | residual q = 0.99 | 0.0324 ± 0.0015 | 0.1125 ± 0.0095 | 0.1449 ± 0.0185 | 0.2669 ± 0.0259 |
| v1-garbage | none (as trained) | 0.0319 ± 0.0025 | 0.0405 ± 0.0038 | 0.0551 ± 0.0072 | 0.1110 ± 0.0131 |
| v1-garbage | oracle | 0.0319 ± 0.0025 | 0.0452 ± 0.0085 | 0.0642 ± 0.0167 | 0.1290 ± 0.0187 |
| v1-garbage | residual q = 0.99 | 0.0515 ± 0.0210 | 0.0924 ± 0.0181 | 0.1501 ± 0.0283 | 0.2360 ± 0.0187 |
| v1-garbage | soft | 0.0924 ± 0.0279 | 0.1240 ± 0.0191 | 0.1626 ± 0.0220 | 0.2301 ± 0.0112 |
| v1-mixed | oracle | 0.0170 ± 0.0047 | 0.0178 ± 0.0048 | 0.0194 ± 0.0050 | 0.0304 ± 0.0046 |
| v1-mixed | none | 0.0170 ± 0.0047 | 0.0220 ± 0.0025 | 0.0301 ± 0.0023 | 0.0678 ± 0.0078 |
| v1-mixed | residual q = 0.99 | 0.0170 ± 0.0036 | 0.0312 ± 0.0072 | 0.0690 ± 0.0114 | 0.1752 ± 0.0158 |
| v1-mixed | soft | 0.0185 ± 0.0028 | 0.0318 ± 0.0024 | 0.0716 ± 0.0132 | 0.1818 ± 0.0142 |

The threshold matters little: relative to q = 0.99, q = 0.95 is 0–23 % worse and q = 0.999 0–7 % better, for every
v1 model and k.

**Detection** at q = 0.99 (per window, draw and probe; false alarm = flagged healthy / healthy):

| model | recall (k = 1..7) | false alarm, clean | k = 0 | k = 1 | k = 2 | k = 4 |
|---|---|---|---|---|---|---|
| v1-missing | 0.991–0.996 | 0.015 | 0.093 | 0.432 | 0.656 | 0.869 |
| shred_mask | 0.993–1.000 | 0.004 | 0.079 | 0.549 | 0.780 | 0.944 |
| v1-garbage | 0.973–0.986 | 0.055 | 0.075 | 0.148 | 0.277 | 0.482 |
| v1-mixed | 0.988–0.995 | 0.031 | 0.087 | 0.157 | 0.263 | 0.490 |

### Follow-up: does the score separate faulty from healthy probes?

Pass-1 scores and labels per (window, draw, probe) are saved in `arrays.npz` of every scored variant (residual,
soft, topk); the residual q = 0.99 arrays are used here (the score does not depend on q). Adding them recomputed
the existing scored variants; every metric is identical to before (max difference 0.0). New variant `topk`: flag
the k highest scores, k = the true number of faults (an upper bound for any ranking rule), for v1-missing and
v1-mixed. Notebook section 7.

**Rank accuracy** (the k faulty probes are exactly the k highest scores; chance = 1 / C(8, k): 0.125 at k = 1,
0.036 at k = 2, 0.014 at k = 4):

| model | k = 1 | k = 2 | k = 3 | k = 4 | k = 7 |
|---|---|---|---|---|---|
| v1-missing | 0.932 ± 0.011 | 0.768 ± 0.045 | 0.614 ± 0.044 | 0.421 ± 0.035 | 0.275 ± 0.026 |
| v1-mixed | 0.988 ± 0.009 | 0.963 ± 0.011 | 0.932 ± 0.004 | 0.825 ± 0.062 | 0.393 ± 0.035 |
| shred_mask | 0.972 ± 0.011 | 0.889 ± 0.030 | 0.718 ± 0.033 | 0.475 ± 0.017 | 0.214 ± 0.017 |

**Score gap** (median faulty score / 95th percentile of the healthy scores in the same cases):

| model | k = 1 | k = 2 | k = 3 | k = 4 | k = 7 |
|---|---|---|---|---|---|
| v1-missing | 2.23 ± 0.50 | 1.41 ± 0.30 | 1.05 ± 0.14 | 0.87 ± 0.09 | 0.64 ± 0.04 |
| v1-mixed | 6.80 ± 0.12 | 4.92 ± 0.28 | 4.00 ± 0.37 | 2.56 ± 0.52 | 0.65 ± 0.01 |
| shred_mask | 4.27 ± 0.44 | 2.65 ± 0.12 | 1.74 ± 0.13 | 1.31 ± 0.05 | 0.74 ± 0.01 |

**Flags per case at q = 0.99** (share of cases): k = 1: v1-missing flags exactly 1 probe in 22 % of cases and ≥ 6 in 36 %; v1-mixed exactly 1 in
41 %, ≤ 3 in 88 %; shred_mask exactly 1 in 6 %. k = 2: exactly 2 in 8 % / 31 % / 1 %, and ≥ 7 in 53 % / 6 % / 63 %.
Per-snapshot error grows with the flag count (v1-missing at k = 1: 0.010 with 1 flag, 0.022 with 4, 0.367 with all
8 flagged); full table in the notebook.

**Top-k** (`err`):

| method | k = 1 | k = 2 | k = 4 | k = 7 |
|---|---|---|---|---|
| v1-missing, oracle | 0.0115 ± 0.0009 | 0.0125 ± 0.0011 | 0.0187 ± 0.0025 | 0.1258 ± 0.0145 |
| v1-missing, top-k | 0.0562 ± 0.0084 | 0.0948 ± 0.0230 | 0.1873 ± 0.0344 | 0.3491 ± 0.0254 |
| v1-missing, no flags | 0.0715 ± 0.0129 | 0.1231 ± 0.0268 | 0.2153 ± 0.0259 | 0.3415 ± 0.0306 |
| v1-mixed, oracle | 0.0178 ± 0.0048 | 0.0194 ± 0.0050 | 0.0304 ± 0.0046 | 0.1275 ± 0.0100 |
| v1-mixed, top-k | 0.0180 ± 0.0044 | 0.0203 ± 0.0051 | 0.0490 ± 0.0148 | 0.3137 ± 0.0095 |
| v1-mixed, no flags | 0.0220 ± 0.0025 | 0.0301 ± 0.0023 | 0.0678 ± 0.0078 | 0.2925 ± 0.0053 |

v1-missing top-k: cases where the top k are the true faults have oracle-level error (rms 0.0115 at k = 1, 0.0123
at k = 2); the 7 % / 23 % misses average 0.166 / 0.145 and set the pooled error.

## Conclusion

Tentative: one flow, 3 seeds, 19 test windows per seed, untuned flag fraction for v1-mixed.

1. **Residual flags with a fixed threshold do not replace oracle flags.** They are worse than no flags for every
   model and every q. Recall is ≈ 0.99, but garbage raises the scores of the healthy probes in the same window, so
   a threshold set on clean data flags them too (15–78 % of healthy probes at k = 1–2). Each false alarm removes a
   healthy probe, and the error grows with the number of flags (v1-mixed, k = 2: 0.016 with the right count of 2,
   0.094 with 7, 0.402 with all 8 flagged).
2. **Correct flags still help; wrong flags cost more than they gain.** Oracle flags beat no flags for v1-missing
   and v1-mixed, as in `dropout_baseline` and `random_readings`, and top-k flags (almost always correct for
   v1-mixed) are almost as good (0.020 vs 0.019 at k = 2). This refines the earlier conclusion ("flags help; the
   flag regime must match training"): flags help only when they are mostly correct, and only for a model trained
   with a mask. v1-garbage, never trained with one, gains nothing even from oracle flags (0.055 → 0.064).
3. **The score itself separates faulty from healthy probes well; the threshold is the weak part.** The faulty
   probe is the top score in 93–99 % of k = 1 cases (chance 12.5 %), and v1-mixed ranks the faulty probes exactly
   in 96 % / 83 % of the cases at k = 2 / 4. Top-k is an upper bound, since it is told the true number of faults.
4. **Soft weights are an intervention, not information.** log w is added to v1's attention logits, so the model
   never reads the residual: w multiplies the probe's attention weight. w = exp(−0.5 d²) falls below 0.1 at
   d ≈ 2.2, where the hard thresholds lie, so soft ≈ hard exclusion (0.072 vs 0.069, v1-mixed, k = 2). When garbage
   lifts every score, attention flows to the unbiased null token. On clean data, a model never trained with biases
   is distorted (v1-garbage 0.032 → 0.090).
5. **Training matters more than test-time detection.** Without oracle flags, the best model is v1-mixed (trained
   on garbage with half the faults flagged) with no flags: 0.022 / 0.030 / 0.068 at k = 1 / 2 / 4, against
   0.041 / 0.055 / 0.111 for v1-garbage and 0.072 / 0.123 / 0.215 for v1-missing, consistent on every seed. It also
   gives the cleanest scores (gap 6.8 vs 2.2 for v1-missing at k = 1). The price is a higher clean error (0.017 vs
   0.011). Why it also beats v1-garbage is not explained by these runs.
6. **Step 0 did not predict the failure.** Self-influence ≤ 0.08 and a small linear influence on the other probes
   (≤ 0.08 std per std) said detection should work. Garbage is far outside that small-offset regime and lifts the
   healthy probes' scores much more.

Leads (not tested): a rule relative to the other probes in the same window instead of a clean-data threshold
(top-k shows the room); the score as a trained input feature instead of an attention bias; tuning v1-mixed's flag
fraction; thresholds from more calibration data than 19 windows and one noise draw.
