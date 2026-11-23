# pod_shred_nektar

**Status:** tentative
**Code:** branch `exp/pod-shred-nektar`, PR #<n>

## Question

Does SHRED do better, or train faster, when it predicts POD coefficients instead of the full field (as in
masked-shred), and does the loss space matter: the decoded field (as SHRED) or the MinMax-scaled coefficients (as the
paper)?

Prototype extension (user's proposal, 2026-11-23): put the r = 16 POD basis into a continuous trunk T(x, y), so the
field is differentiable and a continuity loss can be added at POD SHRED's speed (`pod_trunk_shred*`).

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
(coefficient loss). 21 runs. Prototype POD trunk runs (r = 16, Nektar only): pod_trunk_shred (frozen trunk, field
loss), pod_trunk_shred_coef (frozen, coefficient loss), pod_trunk_shred_div (frozen, field loss + continuity λ = 1e-3),
pod_trunk_shred_train / pod_trunk_shred_train_div (trunk trained with SHRED, same lr 1e-3, λ = 0 / 1e-3). 15 runs.

```
uv run python scripts/run.py +experiment=pod_shred_nektar
```

Device: `cpu`, 6 workers. Smoke runs (20 epochs, seed 0, wave and Nektar): `outputs/pod_shred_nektar_smoke/`; trained-trunk runs (Nektar):
`outputs/pod_trunk_train_smoke/`.
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

### POD trunk (prototype)

Notebook section 5. Trunk: tanh 2 × 256 on 16 Fourier frequencies per axis, fitted to the grid basis (values only,
5000 Adam steps, ≈ 85 s). div_rel = rms (u_x + v_y) / rms |∇u| of the clean test predictions, by autograd through the
trunk; the data's own finite-difference value is 0.003.

| run | trunk | loss | λ | test err | err u | err v | val checks | wall s | div_rel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pod_trunk_shred | frozen | field | 0 | 0.0077 ± 0.0003 | 0.0052 | 0.0202 | 21 | 141 | 0.126 |
| pod_trunk_shred_coef | frozen | coef | 0 | 0.0098 ± 0.0012 | 0.0062 | 0.0267 | 13 | 136 | 0.126 |
| pod_trunk_shred_div | frozen | field | 1e-3 | 0.0074 ± 0.0000 | 0.0051 | 0.0191 | 33 | 210 | 0.126 |
| pod_trunk_shred_train | trained | field | 0 | 0.0067 ± 0.0008 | 0.0038 | 0.0193 | 20 | 182 | 0.102 |
| pod_trunk_shred_train_div | trained | field | 1e-3 | 0.0068 ± 0.0013 | 0.0038 | 0.0197 | 21 | 231 | 0.095 ± 0.012 |

Diagnostics from the saved models (mean over seeds): the test truth's exact coefficients decoded through the frozen
trunk give 0.0071 (exact basis: 2.9e-4), of which the mean field alone is ≈ 0.004. The frozen trunk's values fit to
≈ 0.5 % but its gradients are ≈ 10 % off: its divergence is 0.13 (autograd) / 0.085 (FD on the grid), 0.12 from the
mean field alone, and the best possible coefficients cancel only 0.7 % of the mean field's divergence. A trained
trunk drifts far from the POD modes (mode error 0.019 → 0.27), so the exact coefficients no longer apply to it.

Reading (prototype, not a result): the frozen trunk's error is ≈ 90 % trunk misfit, not training. The coefficient loss
is worse because the exact coefficients are decoded through the misfit basis, while the field loss can compensate.
With the trunk frozen, the continuity loss can only move the coefficients, which cannot cancel the basis's own
divergence: div_rel is identical in all frozen runs. Training the trunk lowers error (u only) and divergence a little;
the continuity term adds ≈ 7 % lower divergence, within the seed spread. Early stopping on the field error ends every
run after ≈ 400 epochs, and at λ = 1e-3 the term is ≈ 7 % of the loss. Untested: larger λ, no early stopping, a
derivative-aware trunk fit, a stream-function trunk on a joint (u, v) POD (divergence free by construction; departs
from the paper's basis).

## Conclusion

Tentative. With the loss on the decoded field, POD coefficients cost ≈ 3× less per run (≈ 2.4× per epoch, and the runs
stop earlier) and are ≈ 20 % less accurate than the full-field output (0.0025 vs 0.0021, about one seed std apart).
Rank barely matters, since the wake is nearly rank ≤ 32 per field. With the paper's coefficient loss, accuracy depends
strongly on the rank: equal at r = 16, but 5× and 9× worse at r = 32 and 64. Modes beyond the flow's real rank carry
almost no energy but are still scaled to [0, 1], so they weigh as much in the loss as the dominant modes, and the
energetic modes are underfit (train error ≈ test error). On this setup, the output layer, not the encoder, sets
SHRED's cost per epoch. Clean data only; robustness to faults is untested. The POD trunk prototype does not yet give
working physics: the fitted trunk's derivatives, not SHRED, set its divergence (0.10–0.13 vs the data's 0.003), and
its error stays ≈ 3× POD SHRED's.
