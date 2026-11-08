# physical_relation

**Status:** planned
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

Not run yet.

## Conclusion

Not written yet.
