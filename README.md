# get-shredded

Reimplementation of **SHRED (SHallow REcurrent Decoder)** from Williams, Zahn, Kutz (2024), *"Sensing with shallow recurrent decoder networks"*, applied to the cylinder-vortex dataset.

The repo trains SHRED end-to-end to reconstruct the full vorticity field from a handful of point sensor measurements over a `lags`-long time window, and compares it against two baselines used in the paper:

- **SDN** — static shallow decoder (same MLP, no recurrence; takes only the most recent sensor snapshot)
- **Senseiver** — coordinate-aware latent encoder and coordinate-conditioned decoder
- **SenseiverSDN** — project-specific ablation using the Senseiver encoder and an SDN-style MLP decoder
- **QR/POD** — linear gappy-POD reconstruction with QR-pivoted sensor placement: `x̂ = U_r (C U_r)^{-1} y`

SHRED now uses a GRU internally rather than an LSTM. This is a breaking change:
previous SHRED checkpoints and committed robustness results were produced by the
LSTM implementation and must be regenerated.

## Project layout

```
get-shredded/
  data/                                # place CYLINDER_ALL.mat here
  configs/
    cylinder_baseline.yaml             # single-run config
    sweep_num_sensors.yaml             # error-vs-num_sensors sweep config
    default.yaml                       # Hydra plumbing
  scripts/
    run_cylinder_baseline.py           # train SHRED + SDN + QR/POD, save plots & GIF
    sweep_num_sensors.py               # sweep num_sensors, save aggregate plot
  src/get_shredded/
    model.py                           # SHRED, SDN, fit(), forecast()
    robust_shred.py                    # RobustSHREDv1 / RobustSHREDv2
    data.py                            # cylinder loader, qr_place, sensor windowing
    baseline.py                        # QR/POD baseline wrapper
    experiment.py                      # reusable single-run pipeline → RunResult
    plotting.py                        # panel/animation/curve/sweep plot helpers
  outputs/                             # all results land here (created on first run)
```

## Senseiver baseline

A separate `Senseiver` baseline is provided in `src/get_shredded/senseiver.py` and exported through the package (`get_shredded.Senseiver`). It follows the paper's single-frame sparse reconstruction pipeline:

- sine/cosine spatial positional encoding on continuous coordinates,
- sensor encoder with a latent bottleneck and shared recurrent attention blocks,
- decoder cross-attention from query coordinates to the latent representation,
- a forward interface of `model(sensor_values, sensor_coordinates, query_coordinates)`.

This baseline is kept separate from the SHRED/SDN codepaths and does not alter the current experimental behavior.

`SenseiverSDN` is a project-specific comparison architecture, not a paper-defined
model. It reuses the same coordinate-aware `SenseiverEncoder` and maps its fixed
latent tensor `(batch, num_latents, latent_dim)` through the existing three-layer
SDN-style MLP to `(batch, full_state_size)`. It does not use query coordinates or
the standard Senseiver decoder. Both models receive instantaneous sensor values
and sensor coordinates; SHRED additionally receives a temporal sensor history,
while SDN receives only the latest sensor snapshot.

The single-run result records trainable parameter counts for SHRED, SDN,
Senseiver, and SenseiverSDN (including encoder/decoder counts for the two
Senseiver variants), and the baseline script prints them alongside the errors.

## Robust SHRED variants

`RobustSHREDv1` and `RobustSHREDv2` combine temporal recurrence with
coordinate-aware sensor attention. Both currently use the fixed `num_sensors`
tensor convention; dropout is represented by corrupted or zero-filled readings,
not by removing sensors from the attention set.

- **RobustSHREDv1** — at each timestep, the attention query comes from the
  previous recurrent hidden state, so memory actively selects which sensors to
  trust. The recurrent cell is a GRU.
- **RobustSHREDv2** — fixed learned Senseiver-style latent queries encode each
  timestep independently of the past; only the resulting latent sequence is
  passed to a GRU. The attention step has no previous-timestep memory.

Both variants are trained and evaluated by `run_robustness_comparison.py` under
clean, gaussian, dropout, hybrid, and burst conditions using the same protocol
as SHRED and SDN. Disable either variant with
`robust_shred.v2_enabled=false`.

## Setup

Requires Python ≥ 3.10. Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

Place `CYLINDER_ALL.mat` (the standard Brunton/Kutz cylinder-vortex dataset, contains a `VORTALL` array of shape `(m, T)`) at `data/CYLINDER_ALL.mat`.

## Single run: train + evaluate + plot

```bash
uv run python scripts/run_cylinder_baseline.py
```

This trains SHRED, SDN, Senseiver, and SenseiverSDN with early stopping, computes
the QR/POD baseline, prints relative L2 test errors, and writes plots under
`outputs/cylinder/`:

- **`reconstructions/panel.png`** — 3 test snapshots × 4 columns (truth+sensor positions, SHRED, SDN, QR/POD). Sensor locations overlaid as lime dots on the truth column.
- **`reconstructions/comparison.gif`** — animated side-by-side reconstruction across the entire test set.
- **`curves/per_snapshot_error.png`** — relative L2 error per test snapshot, one line per method.
- **`curves/training_curves.png`** — log-scale validation error vs epoch for SHRED and SDN.

The previously committed checkpoints and files under
`outputs/cylinder/robustness/` predate the GRU switch and the robust variants;
regenerate them with `run_cylinder_baseline.py` and
`run_robustness_comparison.py` before using their numbers.

Hydra overrides work as usual:

```bash
# more sensors, random placement
uv run python scripts/run_cylinder_baseline.py model.num_sensors=10 model.placement=random

# longer history window, smaller recurrent cell
uv run python scripts/run_cylinder_baseline.py model.lags=20 model.hidden_size=32

# compare GRU vs LSTM for the same SHRED setup
uv run python scripts/run_cylinder_baseline.py model.recurrent_cell=lstm
uv run python scripts/run_recurrent_cell_benchmark.py

# different output directory
uv run python scripts/run_cylinder_baseline.py outputs.root=outputs/experiment_2
```

Key knobs in [configs/cylinder_baseline.yaml](configs/cylinder_baseline.yaml):

| Setting | Default | Meaning |
|---|---|---|
| `model.num_sensors` | 3 | Number of point sensors |
| `model.lags` | 10 | Length of sensor history window fed to the recurrent cell |
| `model.placement` | `QR` | `QR` (greedy QR-pivot) or `random` |
| `model.recurrent_cell` | `gru` | `gru` or `lstm` for the SHRED recurrent core |
| `model.hidden_size` / `hidden_layers` | 64 / 2 | Recurrent hidden size and stack depth |
| `model.l1` / `l2` | 350 / 400 | Decoder MLP widths |
| `data.test_size` | 10 | Last N windows held out for testing |
| `data.val_size` | 20 | Validation windows preceding the test set |
| `train.epochs` / `patience` | 1000 / 5 | Max epochs + patience (× 20 epochs of no improvement) |

## Sensor noise model (per sensor)

You can inject sensor corruption with a dedicated Hydra config group under `configs/noise/`.

- `noise=disabled` (default): all sensors return true measurements.
- `noise=per_sensor`: per-sensor modes are applied from `noise.modes`.

Supported per-sensor modes:

- `"true"` — return true sensor value.
- `"white"` — add Gaussian white noise (`noise.white_std`).
- `"none"` — sensor still returns a noisy reading; currently this uses the same Gaussian noise path as `"white"`.

Example (three sensors: true / white / dead):

```bash
uv run python scripts/run_cylinder_baseline.py noise=per_sensor
```

Or override inline:

```bash
uv run python scripts/run_cylinder_baseline.py \
  noise.enabled=true \
  noise.modes='["true","white","none"]' \
  noise.white_std=0.02 \
  noise.none_fill_value=0.0
```

Notes:

- Noise is applied to SHRED/SDN sensor inputs and to QR/POD sensor measurements.
- `noise.auto_extend=true` lets short mode lists be padded with `noise.default_mode`.
- `noise.seed` controls noise reproducibility (falls back to main `seed` when null).

## Sweep: error vs number of sensors × placement

Reproduces the paper's Fig 2B / 3B / 4B style plot — relative L2 error vs sensor count, with **both QR-pivoted and random placement**, for SHRED, SDN, and the QR/POD baseline.

```bash
uv run python scripts/sweep_num_sensors.py
```

Writes to `outputs/cylinder/sweep/`:

- **`error_vs_num_sensors.png`** — median test error vs `num_sensors`. Up to 5 lines: SHRED (QR), SHRED (random), SDN (QR), SDN (random), and QR/POD (linear baseline; only QR placement is plotted since the linear inverse becomes ill-conditioned with random sensors).
- **`sweep_results.npz`** — raw error arrays of shape `(sensor_count, seed)` for every `(method, placement)` combination, keyed as e.g. `shred_QR`, `sdn_random`, etc.

Defaults sweep `num_sensors ∈ {1, 2, 3, 5, 8, 12, 20}` × placements `{QR, random}` × seeds `{0, 1, 2}`. Tune in [configs/sweep_num_sensors.yaml](configs/sweep_num_sensors.yaml):

```bash
# faster (single seed, fewer sensor counts)
uv run python scripts/sweep_num_sensors.py sweep.seeds=[0] sweep.sensor_counts=[1,3,10,20]

# QR placement only (skip the random comparison)
uv run python scripts/sweep_num_sensors.py sweep.placements=[QR]
```

The sweep is the long-running job — runtime scales as `len(sensor_counts) × len(placements) × len(seeds) × 2 networks × min(epochs, patience-stopped)`. Three nested tqdm bars show progress: `sensor sweep` → `n_sensors=N <placement>` → `train SHRED/SDN`. The training bar shows running `loss`, `val` (relative L2 every 20 epochs), `best`, and `patience` counter.

## Implementation notes

- The split is **sequential** over sliding windows (last `test_size` for test, preceding `val_size` for val, rest for train) — chosen for the small cylinder dataset (~150 snapshots). Paper experiments on SST/turbulence use random interleaved splits since they have many more frames.
- `MinMaxScaler` is fit on training rows only and applied globally (matches paper).
- `fit()` validates every 20 epochs, restores best parameters on early stopping (matches paper's `models.fit`).
- QR/POD uses the *unscaled* training POD basis and reconstructs from unscaled sensor measurements at the test timestamps.
- SHRED architecture defaults (`hidden_size=64, hidden_layers=2, l1=350, l2=400`) match paper's `models.SHRED`.

## Reference

Williams, J. P., Zahn, O., & Kutz, J. N. (2024). *Sensing with shallow recurrent decoder networks.* arXiv:2301.12011. Original code: [github.com/JanWilliams/pyshred](https://github.com/JanWilliams/pyshred).
