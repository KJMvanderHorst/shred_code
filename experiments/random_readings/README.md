# random_readings

**Status:** tentative
**Code:** branch `exp/rerun-baselines`; original: `random_readings.py` (PR #7, tag `pre-restructure`)

## Question

If faulty probes emit garbage instead of dropping out, how do the models cope
when the fault is flagged (oracle) and when it is undetected (silent)? And does
training on silent garbage help?

## Protocol deviations

None. Differences from the 2026-10-15 meeting follow-up:
- Garbage-trained models on 3 seeds (meeting: seed 0 only), trained by the runner
  instead of `train_random_readings`.
- `senseiver_sdn__missing` dropped (expensive, opt-in).
- Exactly-k evaluation (`ks: all`) added next to the meeting's p = 0.3.
- Models are trained in this experiment, not loaded from `dropout_baseline`; with
  the same seeds they are identical to the `dropout_baseline` models.

## Run set

`configs/experiment/random_readings.yaml` on `nektar_uv`, seeds 0, 1, 2, 10 test draws per window:
the 8 `dropout_baseline` runs, plus `shred`, `robust_shred_v1` and `robust_shred_v2`
trained on `garbage` (p = 0.3, N(0.5, 0.5) in scaled space, silent: masked models get an all-zero mask).
Conditions: clean; `missing`/`oracle` at p = 0, 0.3; `garbage` with `oracle` and with `silent` at p = 0.3;
and all three at exactly k = 0..7. Missing and garbage use the same faulty-probe sets per severity.
Models without a mask input (SHRED, SDN) behave the same under oracle and silent.

```
uv run python scripts/run.py +experiment=random_readings
```

Figures: `notebooks/random_readings.ipynb` (saved to `outputs/random_readings/nektar_uv/figures/`).

## Results

Rerun 2026-11-07, `outputs/random_readings/nektar_uv/summary.json`. Relative L2 error,
mean ± std over 3 seeds, p = 0.3. Meeting columns from
`outputs/_legacy/meeting_2026-10-14/random_readings/summary.json` (local only).

| Run | Clean | Missing (oracle) | Garbage (oracle) | Garbage (silent) | Meeting oracle | Meeting silent |
|---|---:|---:|---:|---:|---:|---:|
| `shred__noise` | 0.0040 ± 0.0006 | 0.1589 ± 0.0062 | 0.1188 ± 0.0075 | 0.1188 ± 0.0075 | – | 0.1188 ± 0.0075 |
| `shred__missing` | 0.0102 ± 0.0027 | 0.0145 ± 0.0035 | 0.1505 ± 0.0115 | 0.1505 ± 0.0115 | – | 0.1505 ± 0.0115 |
| `shred_mask__missing` | 0.0099 ± 0.0013 | 0.0113 ± 0.0018 | 0.1281 ± 0.0095 | 0.1084 ± 0.0059 | 0.1281 ± 0.0095 | 0.1084 ± 0.0059 |
| `sdn__noise` | 0.0040 ± 0.0004 | 0.1803 ± 0.0061 | 0.1679 ± 0.0102 | 0.1679 ± 0.0102 | – | 0.1679 ± 0.0102 |
| `sdn__missing` | 0.0101 ± 0.0012 | 0.0243 ± 0.0017 | 0.1900 ± 0.0206 | 0.1900 ± 0.0206 | – | 0.1900 ± 0.0206 |
| `robust_shred_v1__missing` | 0.0101 ± 0.0008 | 0.0146 ± 0.0012 | 0.0146 ± 0.0012 | 0.1489 ± 0.0131 | 0.0150 ± 0.0011 | 0.1488 ± 0.0122 |
| `robust_shred_v2__missing` | 0.0098 ± 0.0007 | 0.0143 ± 0.0016 | 0.0143 ± 0.0016 | 0.1381 ± 0.0180 | 0.0151 ± 0.0019 | 0.1411 ± 0.0186 |
| `gappy_pod__clean` | 0.0007 ± 0.0000 | 0.0916 ± 0.0089 | 0.0916 ± 0.0089 | 0.2349 ± 0.0085 | 0.0916 ± 0.0089 | 0.2349 ± 0.0085 |
| `shred__garbage` | 0.0197 ± 0.0025 | 0.0881 ± 0.0157 | 0.0490 ± 0.0072 | 0.0490 ± 0.0072 | – | – |
| `robust_shred_v1__garbage` | 0.0318 ± 0.0026 | 0.0878 ± 0.0101 | 0.0878 ± 0.0101 | 0.0736 ± 0.0055 | – | – |
| `robust_shred_v2__garbage` | 0.0162 ± 0.0001 | 0.0423 ± 0.0022 | 0.0423 ± 0.0022 | 0.0329 ± 0.0031 | – | – |

Verification: every meeting number reproduces bit for bit except RobustSHREDv1/v2,
which differ within the seed spread (see `experiments/dropout_baseline/README.md`).
The meeting's seed-0 garbage-trained numbers (SHRED 0.045, v1 0.058, v2 0.031 silent)
are now 0.049 ± 0.007, 0.074 ± 0.006 and 0.033 ± 0.003 over 3 seeds.

Exactly k of 8 probes emit garbage, silent, mean over 3 seeds:

| Run | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shred__noise` | 0.0047 | 0.0561 | 0.0901 | 0.1275 | 0.1707 | 0.2179 | 0.2620 | 0.3132 |
| `shred__missing` | 0.0108 | 0.0497 | 0.0969 | 0.1579 | 0.2415 | 0.3085 | 0.3736 | 0.4203 |
| `shred_mask__missing` | 0.0102 | 0.0435 | 0.0763 | 0.1173 | 0.1644 | 0.2185 | 0.2826 | 0.3425 |
| `sdn__noise` | 0.0071 | 0.0893 | 0.1405 | 0.1877 | 0.2295 | 0.2720 | 0.3061 | 0.3442 |
| `sdn__missing` | 0.0127 | 0.1083 | 0.1661 | 0.2163 | 0.2617 | 0.2996 | 0.3412 | 0.3755 |
| `robust_shred_v1__missing` | 0.0110 | 0.0715 | 0.1231 | 0.1687 | 0.2153 | 0.2513 | 0.3003 | 0.3415 |
| `robust_shred_v2__missing` | 0.0107 | 0.0648 | 0.1069 | 0.1514 | 0.1963 | 0.2340 | 0.2801 | 0.3277 |
| `gappy_pod__clean` | 0.0139 | 0.1548 | 0.2157 | 0.2633 | 0.3037 | 0.3421 | 0.3732 | 0.4045 |
| `shred__garbage` | 0.0198 | 0.0238 | 0.0304 | 0.0393 | 0.0562 | 0.0976 | 0.1859 | 0.2981 |
| `robust_shred_v1__garbage` | 0.0319 | 0.0405 | 0.0551 | 0.0747 | 0.1110 | 0.1472 | 0.2136 | 0.3044 |
| `robust_shred_v2__garbage` | 0.0163 | 0.0192 | 0.0225 | 0.0302 | 0.0390 | 0.0633 | 0.1200 | 0.2407 |

Under garbage with oracle flags, the missing-trained RobustSHREDs equal their
missing numbers; the garbage-trained ones (trained without flags) are worse with
flags than without (v1 0.088 vs 0.074, v2 0.042 vs 0.033 at p = 0.3).

## Conclusion

Tentative (3 seeds, one flow).

1. **Silent garbage is ≈ 10× worse than dropout.** Every dropout-trained model lands at
   0.11–0.15 under silent garbage at p = 0.3 (vs 0.011–0.024 for missing). One silent
   garbage probe (k = 1) costs 0.04–0.11, more than five missing probes.
2. **Learned robustness is fault-specific.** `shred__missing` (0.151) is worse than
   `shred__noise` (0.119) under silent garbage, and worse at every k ≥ 2: dropout training
   seems to teach "0 means ignore", which garbage never triggers. The reverse holds too:
   garbage-trained models are worse on missing (v2 0.042) than on garbage (0.033), and far
   worse than dropout-trained models (0.014).
3. **Hard masking is fault-agnostic; flags as input are not.** With oracle flags,
   RobustSHREDv1/v2 give exactly their missing numbers under garbage (0.0146, 0.0143), as
   does gappy POD (0.092): excluded probes have no influence. SHRED+mask is worse *with*
   true flags (0.128) than without (0.108): it learned "flag + value 0" jointly, and a flag
   next to a nonzero value is out of distribution.
4. **Detection is the bottleneck.** For RobustSHRED, oracle 0.015 vs silent 0.14: a perfect
   detector is worth ≈ 10×.
5. **Training on silent garbage helps a lot; the architecture ordering is suggestive only.**

   | Garbage-trained | Silent p = 0.3 | Clean | Silent k = 4 |
   |---|---:|---:|---:|
   | RobustSHREDv2 | 0.033 ± 0.003 | 0.016 | 0.039 |
   | SHRED | 0.049 ± 0.007 | 0.020 | 0.056 |
   | RobustSHREDv1 | 0.074 ± 0.006 | 0.032 | 0.111 |
   | best dropout-trained (SHRED+mask) | 0.108 | 0.010 | 0.164 |

   3–4× better than dropout-trained models in every seed, still ≈ 2× worse than oracle
   (0.015). The order v2 < SHRED < v1 holds in each seed (silent, seed 0/1/2: v2
   0.032/0.029/0.037, SHRED 0.056/0.039/0.051, v1 0.076/0.079/0.066), but 3/3 is only
   p = 1/8 on a sign test, and the encoders are untuned. v1's clean error (0.032, 2× SHRED)
   points to an optimisation or underfitting problem rather than an inability to detect
   faults. Invariance is in line with the ordering: under silent garbage v2's
   reconstruction moves 2 % from clean (`d_recon` 0.020), against ≈ 10 % for
   dropout-trained models.
6. **Oracle flags hurt garbage-trained models** (v2 0.042 vs 0.033, v1 0.088 vs 0.074):
   they never saw a flag in training.
7. **The meeting overstated v1.** Seed-0 numbers 0.045 / 0.058 / 0.031 (SHRED / v1 / v2)
   become 0.049 / 0.074 / 0.033 over 3 seeds.
8. **What the model knows about the faults is a main factor, next to the training
   corruption.** Three levels, at p = 0.3:

   | Information | Example | Missing | Garbage |
   |---|---|---:|---:|
   | none (silent), trained on dropout | RobustSHREDv1 / v2 | 0.015 | 0.149 / 0.138 |
   | none (silent), trained on garbage | RobustSHREDv2 | 0.042 | 0.033 |
   | flags as an input | SHRED+mask | 0.011 | 0.128 (0.108 without flags) |
   | flags exclude probes | RobustSHREDv1 / v2 | 0.0146 / 0.0143 | 0.0146 / 0.0143 |
   | flags exclude probes | gappy POD | 0.092 | 0.092 |

   - Information is worth ≈ 10× (the same dropout-trained v1/v2: 0.015 with flags vs
     0.14 without under garbage).
   - How it enters matters: exclusion makes faulty probes irrelevant by construction,
     whatever they emit; flags as an input are learned, and only help for the fault seen
     with them in training.
   - The information regime must match training, even with exclusion: models trained
     with flags collapse without them (0.14), models trained without flags get worse
     with them (v2 0.033 → 0.042). Exclusion protects against a change of fault type,
     not a change of flag regime.
   - Under dropout flags barely matter (SHRED 0.0145 vs SHRED+mask 0.0113), because a
     constant-zero probe is self-flagging; the axis only shows under garbage.
   - Caveat: information route and architecture are confounded (only v1/v2 and gappy POD
     exclude, only SHRED+mask takes flags as input). The exclusion invariance is
     structural; "flags as input generalise badly" rests on one model.

**Validity.** The limits in `experiments/dropout_baseline/README.md` apply here too
(≈ 153 training windows for 4.2 M parameters, 99 % in a decoder shared by every method;
periodic wake with an interleaved split, so the task is close to recovering the shedding
phase; 19-window test and validation sets; hyperparameters untuned per method; a seed
varies everything at once). What stands: the large or structural effects (silent vs
oracle, dropout- vs garbage-trained under silent garbage, no transfer between fault
types, exact invariance under hard masking). What does not: the ranking of v1, v2 and SHRED.

Other caveats: both faults are easy to detect (garbage is i.i.d. per lag, so temporally
incoherent, and outside [0, 1] ≈ 32 % of the time); `stuck`/`drift` would be harder.
Garbage-trained models are tested on their own training distribution and p. 3 seeds, one
flow, interleaved split. The v1/v2 divergence from the meeting run is unexplained.
