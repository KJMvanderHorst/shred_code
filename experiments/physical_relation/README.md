# physical_relation

**Status:** tentative
**Code:** branch `exp/physical-relation`

## Question

Do the models need a physical relation between the probe readings and the output field?
Hypothesis: no. With ≈ 153 training windows and a 4.2 M-parameter decoder, the decoder can
memorise the training snapshots, and the input only has to tell windows apart (a clock).
If so, this would also explain why implicitly trained RobustSHRED (silent garbage) loses to
flags: implicit fault detection needs physically consistent probes, which an
overparametrised decoder never has to exploit.

## Protocol deviations

- `runs[].input` other than `probes` (see `docs/protocol.md`, Input row):
  - `wave_input`: the probe series replaced by 16 evenly spaced interior columns of wave trajectory 1,
    row i aligned with Nektar snapshot i. Deterministic and smooth, but physically unrelated.
  - `random_input`: the probe series replaced by one fixed seeded N(0, 1) series (T, 16),
    windowed like the probes (neighbouring windows share 9 of 10 lags; SDN sees one random key).
  - `shuffled_targets`: real probes, but window i's target is the snapshot at a fixed seeded
    permutation of time (same pairing in train, validation and test). The target scaler is
    fit on the unpermuted training rows (same distribution, negligible).
- `early_stopping: false` for the decoupled runs (`__nostop`): all 1000 epochs, final weights
  kept. Under early stopping, validation never improves for random pairings, so training would
  stop after ≈ 100 epochs and cut memorisation short.
- New metric `train_err`: relative L2 on the clean training windows.

## Run set

`configs/experiment/physical_relation.yaml` on `nektar_uv`, seeds 0, 1, 2, 10 test draws per window:
`shred`, `sdn`, `robust_shred_v2`, all trained on `noise`, each with `input` ∈ {probes, wave_input,
random_input, shuffled_targets} (12 runs), plus the 3 decoupled inputs with `early_stopping: false`
(9 runs). Conditions: clean, and noise only (`noise`/`oracle`, p = 0).

```
uv run python scripts/run.py +experiment=physical_relation
```

How to read it: low `train_err` with high test error = memorisation (capacity); `wave_input`
test error ≈ `probes` = a non-physical clock suffices; `random_input`/`shuffled_targets` near the
mean-field floor (error of predicting the training mean) = generalisation needs a smooth,
consistent input, not physics per se.

## Results

Run 2026-11-08, `outputs/physical_relation/nektar_uv/summary.json`. Relative L2 error,
mean ± std over 3 seeds. Train = clean training windows; test = clean test windows (the
noise-only condition is within 0.001–0.007 of clean everywhere). **Mean-field floor**
(predict the training mean): 0.328 ± 0.004.

Protocol training (early stopping):

| Input | SHRED train / test | SDN train / test | RobustSHREDv2 train / test |
|---|---|---|---|
| `probes` | 0.0035 / 0.0040 | 0.0038 / 0.0040 | 0.0071 / 0.0069 |
| `wave_input` | 0.0091 / 0.0105 | 0.0125 / 0.0156 | 0.0120 / 0.0134 |
| `random_input` | 0.0144 / 0.078 ± 0.019 | 0.316 / 0.340 | 0.324 / 0.329 |
| `shuffled_targets` | 0.320 / 0.330 | 0.319 / 0.333 | 0.323 / 0.329 |

No early stopping (1000 epochs, final weights):

| Input | SHRED train / test | SDN train / test | RobustSHREDv2 train / test |
|---|---|---|---|
| `wave_input` | 0.0135 / 0.0137 | 0.0104 / 0.0112 | 0.0245 / 0.028 |
| `random_input` | 0.0139 / 0.070 | 0.018 / 0.397 | 0.219 ± 0.149 / 0.243 ± 0.118 |
| `shuffled_targets` | 0.254 / 0.406 | 0.263 / 0.398 | 0.274 / 0.393 |

Per seed, notable: `robust_shred_v2__random_input__nostop` fits on seed 1 only (train 0.009,
test 0.076; seeds 0, 2 stay at the floor). `wave_input` is consistent across seeds
(SHRED test 0.009–0.011, v2 0.013–0.014). With early stopping, every run at the floor stopped
after 7–15 validation checks.

## Conclusion

Tentative (3 seeds, one flow, noise training only). Revised 2026-11-10 after measuring how
unique the input windows are (seed 0):

| Input | Neighbouring windows differ by | Nearest window ≥ 15 steps away | Noise per window |
|---|---:|---:|---:|
| probes | 0.74 | 0.012 (one cycle = 59 steps later) | 0.23 |
| wave | 0.18 | 2.4 (never repeats) | 0.18 |
| random | 3.4 | 3.0 | 0.12 |

The wake repeats every 59 snapshots (≈ 3.4 cycles in 200): probe windows one cycle apart are
20× closer than the noise. Probes encode the **phase**; the wave input encodes **time**.

1. **Under the interleaved split, an input that only encodes time works nearly as well as the
   probes.** `wave_input` gives 0.010–0.016 against 0.004–0.007, for every method and seed.
   The interleaved split is the intended SHRED setting (reconstruction within the observed
   dynamics), but it puts every test window between training windows one step away, so it
   cannot tell an input that tracks the flow state from one that tracks time. Whether a
   physical relation is needed is therefore **not yet tested**.
2. **The probes' advantage is exact phase matches.** Every probe test window has nearly
   identical training windows one cycle away; the wave input only has neighbours one step
   away (as far apart as the noise) and must interpolate. That gap, not "physics", is the
   likely source of the 2–4× difference.
3. **The models can memorise arbitrary pairs.** SDN on random keys, without early stopping,
   fits the training set (train 0.018) and fails on test (0.40, ≈ a random training
   snapshot), confirming the overparametrisation hypothesis.
4. **Overlapping windows carry information between training and test.** SHRED on random input
   reaches 0.07 (SDN: 0.40) because a test window shares 9 of 10 lags with neighbouring
   training windows and the GRU recognises the shared values. This applies to every
   experiment with the interleaved split, by design.
5. **Shuffled targets is not a capacity test.** Windows one cycle apart have practically
   identical probe inputs but, after shuffling, unrelated targets, so the mapping is not a
   function. The best fit averages the ≈ 3 targets per phase, ≈ 0.33 × √(2/3) ≈ 0.27, which
   matches the observed train error 0.25–0.27 (back-of-envelope). Capacity is shown by point 3.
6. **Link to implicit fault detection (interpretation, weaker than first written).** If a
   time-like signal suffices for reconstruction, training does not force the encoder to
   learn cross-probe physical consistency. Not tested; depends on point 1.
7. **Side observations.** v2 is less stable without early stopping (`wave_input` 0.028 vs
   0.013; random input fits on one seed of three), an optimisation issue.

Next test (not run): a block split, testing on a later cycle, with the same inputs plus a
periodic phase clock (sin/cos with the 59-step period). Expected: probes and the phase clock
keep working, wave and random inputs fail; that would show the input has to encode where in
the cycle the flow is, not what time it is. Needs enough cycles to give a gap of ≥ `lags`
between blocks and a test block that covers the whole cycle (see `experiments/LOG.md`).

Caveats: 3 seeds, one periodic flow, noise training only, uniqueness measured on seed 0.
