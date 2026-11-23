# residual_v1

**Status:** tentative
**Code:** branch `exp/residual-v1`, PR #<n>

## Question

Can v1 learn to use its own residual as a fault signal? `residual_flags` showed that the residual score ranks
faulty probes well but that a fixed test-time threshold turns it into mostly wrong flags. Here the model is trained
with the residual as an input, so it can learn how much to trust it.

## Protocol deviations

None for existing runs. New row in `docs/protocol.md`: *Residual v1*. `fit` gains a `target_aux_loss` hook that no
existing model defines; all cached runs of random_readings, dropout_baseline, residual_flags_train,
pod_shred_nektar and physical_relation still match their fingerprints.

## Run set

`configs/experiment/residual_v1.yaml`: `robust_shred_v1_residual` trained on `garbage` (p = 0.3, no flags), and a
check added after the first results: `robust_shred_v1` on `garbage` with `early_stopping: false` (all 1000 epochs,
final weights), × seeds 0, 1, 2; eval conditions copied from `random_readings` (identical masks, noise and garbage values).

```
uv run python scripts/run.py +experiment=residual_v1
```

Device: `mps`, 3 workers (the device of every run it is compared with).

Model (`ResidualSHREDv1`): one RobustSHRED v1 network with v1's hyperparameters and 4 input channels per probe
(u, v, r_u, r_v), no mask input. Pass 1 with the residual channels at 0; r = last-lag reading − pass-1 prediction at
the probe (MinMax-scaled, gradient stopped); pass 2 with r at every lag is the output. Loss MSE(pass 2) + 1.0 ×
MSE(pass 1). Ablation `pass1__*`: the same trained network with r = 0.

Compared with (cached, same draws, MPS): v1-garbage (random_readings; same training data, no residual), v1-mixed
(residual_flags_train; half its training faults flagged, evaluated without flags), v1-missing with and without
oracle flags (random_readings).

## Results

From `outputs/residual_v1/nektar_uv/` and the cached runs, mean ± std over 3 seeds; `notebooks/residual_v1.ipynb`.
Garbage, exactly k of 8 probes faulty.

| method | clean | k = 1 | k = 2 | k = 4 | k = 7 |
|---|---|---|---|---|---|
| residual v1 (pass 2) | 0.0246 ± 0.0045 | 0.0325 ± 0.0032 | 0.0413 ± 0.0029 | 0.0810 ± 0.0118 | 0.2840 ± 0.0110 |
| residual v1, r = 0 (pass 1) | 0.0259 ± 0.0046 | 0.0336 ± 0.0031 | 0.0424 ± 0.0033 | 0.0793 ± 0.0103 | 0.2847 ± 0.0086 |
| v1-garbage, no flags | 0.0318 ± 0.0026 | 0.0405 ± 0.0038 | 0.0551 ± 0.0072 | 0.1110 ± 0.0131 | 0.3044 ± 0.0079 |
| v1-garbage, no flags, 1000 epochs | 0.0269 ± 0.0045 | 0.0293 ± 0.0046 | 0.0341 ± 0.0048 | 0.0667 ± 0.0127 | 0.3082 ± 0.0031 |
| v1-mixed, no flags | 0.0169 ± 0.0049 | 0.0220 ± 0.0025 | 0.0301 ± 0.0023 | 0.0678 ± 0.0078 | 0.2925 ± 0.0053 |
| v1-missing, no flags | 0.0101 ± 0.0008 | 0.0715 ± 0.0129 | 0.1231 ± 0.0268 | 0.2153 ± 0.0259 | 0.3415 ± 0.0306 |
| v1-missing, oracle flags | 0.0101 ± 0.0008 | 0.0115 ± 0.0009 | 0.0125 ± 0.0011 | 0.0187 ± 0.0025 | 0.1258 ± 0.0145 |

- Residual v1 is below v1-garbage on every seed at k = 1, 2, 4 (k = 2: 0.038 / 0.045 / 0.041 vs 0.045 / 0.057 / 0.063),
  and above v1-mixed on every seed (k = 2: 0.028 / 0.033 / 0.029).
- Pass 2 vs pass 1 of the same network: 2–5 % lower at k ≤ 2 (lower on every seed), 2 % higher at k = 4.
- Training: 513 ± 47 epochs, 157 ± 18 s per run (v1-garbage 353 ± 47 epochs, 50 ± 15 s); 64 more parameters than
  v1 (the wider probe embedding).
- Why pass 1 already beat v1-garbage: v1-garbage stopped early. Its best validation check came at ≈ 200 epochs on
  seeds 1 and 2 (≈ 300 on seed 0) and it stopped 100 epochs later; its training error (0.028–0.036, recomputed on CPU
  from the checkpoints) equals its clean test error, so it underfits. Residual v1 learns more slowly but kept
  improving to ≈ 360–460 epochs. The gain also appears on clean data (0.032 → 0.026), not only under faults.
- v1-garbage trained all 1000 epochs: 0.029 / 0.034 / 0.067 at k = 1 / 2 / 4, better than residual v1 at all three
  (on every seed at k = 2 and 4), close to v1-mixed (0.022 / 0.030 / 0.068; v1-mixed itself early-stopped at
  700 ± 220 epochs) and better than it on seed 2 at k = 2. Training error 0.023 ± 0.004.

## Conclusion

Tentative: one flow, 3 seeds, 19 test windows per seed, untuned pass-1 loss weight.

1. **Training v1 with its own residual gives no measurable gain.** The residual changes the same network's output by
   2–5 % at most (pass 2 vs pass 1), and plain v1 on the same data, trained 1000 epochs, is better than residual v1
   at k = 1, 2 and 4. The apparent gain over v1-garbage (0.041 vs 0.055 at k = 2) came from training longer, not
   from the residual. Pass 1 is already trained on the same garbage, so it seems to spot bad readings internally and
   the residual adds little.
2. **Early stopping held back the garbage-trained v1.** With garbage in the validation set (19 windows, one
   realisation) the validation curve is noisy, and patience 5 stopped v1-garbage at ≈ 300–400 epochs while it
   underfitted. Without early stopping its error at k = 2 drops from 0.055 to 0.034.
3. **This weakens earlier comparisons.** In `residual_flags` (conclusion point 5), v1-mixed's lead over v1-garbage
   (0.030 vs 0.055 at k = 2) is mostly this effect: against v1-garbage trained 1000 epochs it is 0.030 vs 0.034
   (k = 4: equal), and v1-mixed itself was early-stopped. The `random_readings` ranking under silent garbage was
   already marked suggestive; it rests on the same early stopping.

Not done: v1-mixed and residual v1 without early stopping (a fair comparison of training regimes), other pass-1
loss weights, other ways of feeding the residual.
