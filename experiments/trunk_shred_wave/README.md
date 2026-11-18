# trunk_shred_wave

**Status:** tentative
**Code:** branch `feat/trunk-shred`, PR #13 (part 1); branch `exp/trunk-probes`, PR #14 (part 2, probe supervision)

## Question

Does a DeepONet-style SHRED (GRU branch, (x, τ) trunk) train on the 1D wave, and can a
wave-equation physics loss be satisfied without hurting reconstruction?

## Protocol deviations

None to existing rows. New method rows: `docs/protocol.md` (Trunk SHRED). Training is clean (no noise).

## Run set

`configs/experiment/trunk_shred_wave.yaml`: {shred, trunk_shred, trunk_shred_pinn, and the search variants
trunk_shred_noff, trunk_shred_ff8, trunk_shred_pinn_strong, trunk_shred_pinn_weak} × train clean × eval clean ×
seeds 0, 1, 2, wave trajectory 0 (21 runs).

```
uv run python scripts/run.py +experiment=trunk_shred_wave data=wave
```

## Results

From `outputs/trunk_shred_wave/wave/summary.json` (mean ± std over 3 seeds); residual and dispersion from
`notebooks/trunk_shred_wave.ipynb`. Test relative L2; residual = rms(u_ττ − c²u_xx)/rms(c²u_xx) over the window.

| run | Fourier freq. | λ | test err | train err | residual | params |
|---|---:|---:|---:|---:|---:|---:|
| shred | – | – | 0.0285 ± 0.0065 | 0.0252 | – | 305k |
| trunk_shred | 16 | 0 | 0.0285 ± 0.0019 | 0.0268 | 0.96 | 54k |
| trunk_shred_pinn | 16 | 1e-3 | 0.0255 ± 0.0040 | 0.0254 | 0.024 | 54k |
| trunk_shred_noff | 0 | 0 | 0.4438 ± 0.0225 | 0.4449 | 8.7 | 52k |
| trunk_shred_ff8 | 8 | 0 | 0.0352 ± 0.0055 | 0.0332 | 0.92 | 53k |
| trunk_shred_pinn_strong | 16 | 1e-2 | 0.1291 ± 0.0046 | 0.1298 | 0.022 | 54k |
| trunk_shred_pinn_weak | 16 | 1e-4 | 0.0354 ± 0.0083 | 0.0343 | 0.089 | 54k |

Per seed (test): shred 0.030 / 0.020 / 0.036; trunk_shred 0.026 / 0.030 / 0.030; trunk_shred_pinn 0.022 / 0.031 / 0.023.
Dispersion: median |ω̂ₖ/ωₖ − 1| over k = 1..10 is 0.010 with physics vs 0.79 without.
Over the window (seed 0, τ = −9 dt): 0.385 (no physics) and 0.391 (physics) vs 0.381 for copying the true τ = 0
field back; velocity û_t at τ = 0: relative error 1.02 ± 0.02 (no physics) and 1.08 ± 0.02 (physics), 3 seeds.
Selection: 16 frequencies and λ = 1e-3 were picked on seed-0 test error; mean best validation over 3 seeds
(scaled space) picks the same (0.0118 for λ = 1e-3, 0.0135 for λ = 0, 0.0176 for 1e-4, 0.0573 for 1e-2).
Wall time: trunk_shred_pinn ≈ 5× shred (3 seeds in parallel, rough).

## Conclusion

The trunk can be made to satisfy the PDE at no accuracy cost, but the physics adds nothing the model uses; it
only makes the unsupervised τ-direction self-consistent.

- Accuracy at τ = 0 matches SHRED (difference within seed spread) with 6× fewer parameters.
- The PDE is enforced (residual 0.024, correct dispersion for every mode with energy).
- Neither model learned time: over the window both equal copying the τ = 0 field back, and û_t is uncorrelated
  with the truth. Only τ = 0 is supervised, so u_t is free; physics then only makes u_ττ consistent with it.
- The design cannot show a physics benefit: interleaved split (interpolation), clean data. "No gain" means
  "not testable here".
- Fourier features are necessary (0.44 without). λ = 1e-3, confirmed on validation.

Caveats: the Fourier basis contains the wave's own modes (easy case); thin slab (9 dt). Details and figures:
`notebooks/trunk_shred_wave.ipynb`. Leads (not tried): supervise the trunk at the probe locations for all 10 lags
(gives u_t, no new targets), then test a block split, noise and faults.

## Part 2: probe supervision at every lag (PR #14)

**Question.** Part 1 left the velocity u_t unsupervised. Does fitting the trunk to the probe readings at every
lag (the model's own clean input; μ = 1 per probe-lag point) make the field over the window correct, and does
physics then add something?

**Run set.** Four runs added to the same experiment (the 21 part-1 runs are cached, not retrained):
`trunk_shred_probes` (λ = 0), `trunk_shred_pinn_weak_probes` (1e-4), `trunk_shred_pinn_probes` (1e-3),
`trunk_shred_pinn_strong_probes` (1e-2), train clean, seeds 0, 1, 2. Clean training only (the runner refuses
noise or faults for `probe_weight > 0`).

**Results** (mean ± std over 3 seeds; window = mean test error over frames τ < 0; copy-back baseline ≈ 0.21):

| λ | probes | test err (τ = 0) | window err | û_t err at probes | û_t err between | best val (scaled) |
|---|---|---:|---:|---:|---:|---:|
| 0 | no | 0.0285 ± 0.0019 | 0.220 | 1.01 | 1.02 | 0.0135 |
| 0 | yes | 0.0432 ± 0.0103 | 0.284 | 0.38 | 1.44 | 0.0191 |
| 1e-3 | no | 0.0255 ± 0.0040 | 0.227 | 1.08 | 1.08 | 0.0118 |
| 1e-3 | yes | 0.0354 ± 0.0083 | 0.263 | 0.26 | 1.27 | 0.0177 |
| 1e-4 | yes | 0.0435 ± 0.0192 | 0.264 | | | 0.0189 |
| 1e-2 | yes | 0.1131 ± 0.0253 | 0.256 | | | 0.0505 |

Probe-reading fit over all lags: 0.016 with probe supervision vs 0.084 without. Dispersion with probes:
36 % (no physics) vs 1 % (physics). Seed-0 probes: x ≈ 0.46–0.55 and 1.40–1.58 (gaps up to 0.85); c · 9 dt ≈ 0.07.

**Conclusion.** Probe supervision pins the velocity at the probes but not between them, so the field over the
window gets worse (above the copy-back baseline) and τ = 0 accuracy drops (0.043 / 0.035 vs 0.029 / 0.026).
Within one window the PDE cannot carry velocity across the probe gaps (domain of dependence ≈ 0.07 vs gaps up
to 0.85), and 8 clustered velocities cannot determine 10 modal velocities. Physics helps relative to probes
alone (τ = 0 better or equal on 3/3 seeds; better u_t and window error), the first measurable physics effect,
but does not recover the loss. Leads (not tried): probes placed for dynamics, a longer window, or full-field
supervision beyond τ = 0. Details: `notebooks/trunk_shred_wave.ipynb`, part 2.
