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

Tentative (3 seeds, one flow, noise training only).

1. **A physical relation is not needed for low error.** A physically unrelated but smooth
   signal (`wave_input`) gives 0.010–0.016, 20–30× below the floor and only 2–4× above the
   real probes (0.004–0.007), for every method, consistently across seeds. Physics buys a
   factor 2–4; most of the performance comes from having *some* smooth, time-unique input.
2. **What the models need is a smooth, consistent input-to-time map.** Random but temporally
   overlapping input still lets SHRED generalise partly (test 0.07–0.08, train 0.014): its
   GRU exploits that neighbouring windows share 9 of 10 lags. SDN, which sees one random
   key per window, memorises it without stopping (train 0.018) but does not generalise at all
   (test 0.40, above the floor): pure memorisation.
3. **The capacity to memorise is real, but not arbitrary.** Shuffled targets are not fitted
   even in 1000 epochs (train 0.25–0.27, test 0.39–0.41): with real probes, nearby inputs
   (plus 5 % training noise) must map to unrelated fields, which the models cannot do.
   The models interpolate a smooth input-to-field map; they do not memorise arbitrary pairs.
4. **Consequence for the fault experiments (interpretation).** Since reconstruction only
   needs a smooth clock, nothing in training forces the encoder to learn the cross-probe
   physical consistency that implicit fault detection would rely on. This fits the gap
   between implicit (silent, garbage-trained) and oracle RobustSHRED, but these runs do
   not test it directly.
5. **Side observations.** v2 is less stable without early stopping (`wave_input` 0.028 vs
   0.013; random input fits on one seed of three). Early stopping on decoupled inputs ends
   at the floor after 7–15 checks, as expected.

Caveats: the interleaved split puts test windows one step from training windows, which is
what lets a smooth clock work; a block split or a different flow might need physics more.
`random_input` windows overlap in time by construction. Noise training only; no faults.
