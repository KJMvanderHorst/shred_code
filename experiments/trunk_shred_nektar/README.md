# trunk_shred_nektar

**Status:** tentative
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
- **B, long training** (`configs/experiment/trunk_shred_nektar_5k.yaml`, 5000 epochs, early stopping off, last-epoch weights): the runs of A
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
(mean ± std over 3 seeds; per seed for runs with a diverged seed). Test relative L2, clean.

| run | λ | A: 1000 epochs, early stopping | B: 5000 epochs, no early stopping |
|---|---:|---:|---:|
| shred | – | 0.0026 ± 0.0002 | 0.0013 ± 0.0005 |
| trunk, fixed lr 1e-3 | 0 | (seed 0: 0.0436) | 0.0108 ± 0.0006 |
| trunk, cosine from 3e-3 | 0 | 0.0400 ± 0.0021 | 0.0550 / 0.0066 / 0.0065 |
| + continuity | 1e-4 | 0.0401 ± 0.0022 | 0.0544 / 0.0066 / 0.0064 |
| + continuity | 1e-3 | 0.0398 ± 0.0021 | 0.0755 / 0.0358 / 0.0064 |
| + continuity | 1e-2 | 0.0392 ± 0.0018 | 0.0526 / 0.0062 / 0.0062 |

Train error equals test error for every trunk run (underfitting). In A every trunk run uses all 1000 epochs. In B
seed 0 diverges between epoch ≈ 1200 and 2000 in every cosine run (validation 0.03 → 0.30–0.47, lr still ≈ 2.7e-3)
and does not recover; so does seed 1 at λ = 1e-3. B's fixed-lr seed 0 reproduces the seed-0 diagnostic exactly.
A first attempt at B with early stopping on stopped every run after 440–880 epochs (validation spikes break
patience 5) and is uninformative; it is kept in the notebook, section 6.

## Conclusion

Tentative. The trunk SHRED trains on Nektar but far slower than SHRED; continuity has no clear effect.

- Under the standard protocol the trunk is ≈ 15× worse than SHRED (0.040 vs 0.0026), and it underfits: it is still
  improving at 1000 epochs. Continuity changes nothing (≤ 0.001 per seed).
- With 5000 epochs the gap shrinks to ≈ 8× (fixed lr, stable) or ≈ 5× (cosine from 3e-3, 2 of 3 seeds). Slow
  training, not capacity, is the main limit (seed-0 search: width and frequencies barely matter).
- The 3e-3 start is unstable: seed 0 diverges in all four cosine runs at the same point, so this is a seed effect at a
  high lr, not physics. The cosine means in B are not meaningful.
- Where training is stable, λ = 1e-2 gives 0.0062 / 0.0062 vs 0.0066 / 0.0065 without (≈ 5 %, 2 seeds): consistent,
  but too small and too few to establish.

Caveats: interleaved split and clean data, as in every experiment so far; the trunk and its lr were chosen on
seed-0 test error; B keeps last-epoch weights. Leads (not tried): a lower cosine start or warm-up, or gradient
clipping; best-checkpoint restore without early stopping. Details: `notebooks/trunk_shred_nektar.ipynb`.
