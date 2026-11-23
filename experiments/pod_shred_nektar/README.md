# pod_shred_nektar

**Status:** tentative
**Code:** branch `exp/pod-shred-nektar`, PR #<n>

## Question

Does SHRED do better, or train faster, when it predicts POD coefficients instead of the full field (as in
masked-shred), and does the loss space matter: the decoded field (as SHRED) or the MinMax-scaled coefficients (as the
paper)?

## Protocol deviations

New row in `docs/protocol.md`: POD SHRED. Training and evaluation are clean (no noise, no faults).

Design choices (user, 2026-11-23): the basis follows the paper end to end for both POD variants (per field
StandardScaler on the training snapshots, SVD, r modes per field, MinMax per coefficient); ranks r = 16, 32, 64 per
field; both loss spaces as separate methods; the coefficient-loss variant early-stops in coefficient space. Not asked,
stated in the plan: exact SVD instead of the paper's randomized SVD; the SHRED encoder and 350/400 decoder are
unchanged, only the output width changes (10 000 → 2r).

## Run set

`configs/experiment/pod_shred_nektar.yaml`: train clean × eval clean × seeds 0, 1, 2, 1000 epochs, default early
stopping: shred, pod_shred (r = 16), pod_shred_r32, pod_shred_r64 (field loss), pod_shred_coef_r16 / _r32 / _r64
(coefficient loss). 21 runs.

```
uv run python scripts/run.py +experiment=pod_shred_nektar
```

Device: `cpu`, 6 workers. Smoke runs (20 epochs, seed 0, wave and Nektar): `outputs/pod_shred_nektar_smoke/`.
Notebook: `notebooks/pod_shred_nektar.ipynb`.

## Results

From `outputs/pod_shred_nektar/nektar_uv/summary.json` (mean ± std over seeds 0, 1, 2); epochs ≈ 20 × (validation
checks − 1); wall time is per run with 6 runs sharing the CPU.

| run | output | r / field | loss | test err | train err | ≈ epochs | wall s | params |
|---|---|---:|---|---:|---:|---:|---:|---:|
| shred | full field | – | field | 0.0021 ± 0.0003 | 0.0017 ± 0.0002 | 713 | 43 | 4,213,854 |
| pod_shred | POD | 16 | field | 0.0025 ± 0.0003 | 0.0022 ± 0.0001 | 487 | 12 | 216,686 |
| pod_shred_r32 | POD | 32 | field | 0.0027 ± 0.0003 | 0.0025 ± 0.0002 | 513 | 14 | 229,518 |
| pod_shred_r64 | POD | 64 | field | 0.0028 ± 0.0008 | 0.0026 ± 0.0007 | 453 | 13 | 255,182 |
| pod_shred_coef_r16 | POD | 16 | coef | 0.0028 ± 0.0002 | 0.0022 ± 0.0004 | 413 | 8 | 216,686 |
| pod_shred_coef_r32 | POD | 32 | coef | 0.0106 ± 0.0021 | 0.0087 ± 0.0014 | 467 | 8 | 229,518 |
| pod_shred_coef_r64 | POD | 64 | coef | 0.0197 ± 0.0036 | 0.0182 ± 0.0042 | 407 | 7 | 255,182 |

Projection floor (test snapshots projected on the training basis): 2.9e-4 at r = 16, 2.4e-6 at r = 32, 1.6e-6 at
r = 64, so the basis limits no run. v error ≈ 6× u error in every run.

## Conclusion

Tentative. With the loss on the decoded field, POD coefficients cost ≈ 3× less per run (≈ 2.4× per epoch, and the runs
stop earlier) and are ≈ 20 % less accurate than the full-field output (0.0025 vs 0.0021, about one seed std apart).
Rank barely matters, since the wake is nearly rank ≤ 32 per field. With the paper's coefficient loss, accuracy depends
strongly on the rank: equal at r = 16, but 5× and 9× worse at r = 32 and 64. Modes beyond the flow's real rank carry
almost no energy but are still scaled to [0, 1], so they weigh as much in the loss as the dominant modes, and the
energetic modes are underfit (train error ≈ test error). On this setup, the output layer, not the encoder, sets
SHRED's cost per epoch. Clean data only; robustness to faults is untested.
