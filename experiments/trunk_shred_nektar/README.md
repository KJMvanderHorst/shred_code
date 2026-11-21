# trunk_shred_nektar

**Status:** running
**Code:** branch `exp/trunk-nektar`, PR #<n>

## Question

Does the DeepONet-style trunk SHRED from `trunk_shred_wave` train on the Nektar cylinder wake (u, v), and what does
a continuity loss ∇·u = 0 change?

## Protocol deviations

New rows in `docs/protocol.md`: Flow trunk SHRED (Nektar u, v only), Per-method lr and schedule. Training is clean
(no noise). Regime B (`trunk_shred_nektar_5k`) raises `train.epochs` to 5000 for every run, SHRED included.

Design choices (user, 2026-11-19): the trunk takes (x, y) only, without τ, since only the last frame is supervised.
Physics is continuity only: NS momentum needs u_t, and the time-free pressure Poisson equation is nearly vacuous
with an unsupervised pressure. No probe-supervision variants.

## Run set

Both regimes: train clean × eval clean × seeds 0, 1, 2.

- **A, standard protocol** (`configs/experiment/trunk_shred_nektar.yaml`, 1000 epochs, early stopping): shred,
  flow_trunk_shred_w256_cos_lr3 (trunk), flow_trunk_shred_div_weak / _div / _div_strong (trunk + λ = 1e-4 / 1e-3 /
  1e-2). 15 runs.
- **B, long training** (`configs/experiment/trunk_shred_nektar_5k.yaml`, 5000 epochs, early stopping): the runs of A
  plus flow_trunk_shred_w256 (same trunk, fixed lr 1e-3). 18 runs.

```
uv run python scripts/run.py +experiment=trunk_shred_nektar
uv run python scripts/run.py +experiment=trunk_shred_nektar_5k
```

The trunk (width 256, p = 128, 16 frequencies, cosine lr from 3e-3) was picked by a seed-0 search: width 64 / 128 /
256, p 64 / 128, 16 / 32 frequencies, fixed vs cosine lr, lr 1e-3 / 3e-3, and a 5000-epoch diagnostic without early
stopping. Search results: `notebooks/trunk_shred_nektar.ipynb`, sections 1–5.

## Results

From `outputs/trunk_shred_nektar/nektar_uv/summary.json` and `outputs/trunk_shred_nektar_5k/nektar_uv/summary.json`
(mean ± std over 3 seeds). Not run yet.

## Conclusion

Not written.
