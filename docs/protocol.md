# Protocol

Every scientific assumption of the pipeline, in one place. An experiment that
deviates from a row says so in its `experiments/<name>/README.md`. Change a
default only on purpose, and update this file in the same PR.

All results so far are **tentative**: the protocol is still being settled.

## Data and splits

| Choice | Config key | Default | Rationale | Caveat |
|---|---|---|---|---|
| Dataset | `data` | `nektar_uv`: Nektar cylinder wake, state `[u; v]` | Research dataset; each probe reads both velocity channels | One flow, one Reynolds number |
| Wave dataset | `data=wave` | 1D wave, trajectory 0, one channel | Quick check that a new method or fault runs at all | Never compared with Nektar results. With `sensors.rank=16` > its 10 modes, gappy POD amplifies noise |
| Split | `data.split` | 80 / 10 / 10 % of lag windows, interleaved, seeded by the run seed | Every split covers the whole time range | Neighbouring windows overlap in time, so test windows are close to training windows; optimistic compared with a block split |
| Lags | `data.lags` | 10 | SHRED paper setting | |
| Scaler and POD fit | — | MinMax scaler and POD basis fit on `load_X[train_idx]` | No test data in the fit | Rows are snapshots at training-window start indices, so snapshots inside test windows can still be near-duplicates of fitted rows. Kept for comparability |
| Seeds | `eval.seeds` | `[0, 1, 2]` | The seed drives split, placement, initialisation, noise and faults | Three seeds give rough error bars only |

## Sensors

| Choice | Config key | Default | Rationale | Caveat |
|---|---|---|---|---|
| Number of probes | `sensors.num` | 8 | | |
| Placement | `sensors.rank` | Block QR pivoting over grid points on the rank-16 (= 2N) POD basis of the training data | A probe measures every channel at its point, so pivoting picks whole points; naive stacked QR gave cond ~1e3 | Placement depends on the seed through the training split |
| Sensor coordinates | — | Physical coordinates rescaled jointly to [0, 1] by the larger span | Positional encodings for the attention models | |

## Corruption

| Choice | Config key | Default | Rationale | Caveat |
|---|---|---|---|---|
| Noise | `noise_frac`, `corruptions.*.noise` | σ = 0.05 × per-channel training std, unscaled | Realistic sensor noise; on in every condition except `clean` | Fresh per batch in training; one fixed realisation per test draw, shared by all methods |
| Test draws | `eval.draws` | 10 noise and fault realisations per test window | Averages out the luck of one fault pattern | |
| Fault types | `corruptions.*.fault` | `none`, `missing`, `garbage` | | `stuck` and `drift` are planned, not built (see below) |
| Fault scope | — | A faulty probe is faulty in every channel, for the whole lag window | One physical probe fails, not one reading | |
| Training severity | `corruptions.*.p` | Each probe faulty with probability p = 0.3; draws where every probe fails are resampled | A different fault pattern every batch | |
| Evaluation severity | `eval.conditions[].ks` | Exactly k of N probes faulty, k = 0..N-1 (`ks: all`); a fresh uniformly random subset per test window and draw, shared by all methods | Error as a function of k, without mixing severities in one number | Never all N faulty. `ps` (per-probe p, as in training) is still available; `dropout_baseline` uses it |
| Missing fill | `corruptions.missing.fill` | 0.0 in scaled input space | | 0 is the per-sensor training minimum, not "no signal" |
| Garbage | `corruptions.garbage.garbage_*` | N(0.5, 0.5) per channel per lag, scaled input space | A broken probe emits plausible-range nonsense | |
| Flags | `corruptions.*.flags`, `eval.conditions[].flags` | `oracle`: models with a mask input get the true fault mask. `silent`: they get an all-zero mask | Separates "fault known" from "fault undetected" | Models without a mask input (SHRED, SDN) behave the same under both. Gappy POD drops faulty rows under `oracle` and uses them under `silent` |
| Validation | — | Corrupted with the run's own training corruption, one fixed seeded realisation | Early stopping selects for robustness, not clean accuracy | |

## Training and metrics

| Choice | Config key | Default | Rationale | Caveat |
|---|---|---|---|---|
| Training | `train.*` | Adam, lr 1e-3, batch 64, MSE, up to 1000 epochs | SHRED paper setting | |
| Early stopping | `train.patience` | Validation relative L2 every 20 epochs, patience 5, best weights restored | | |
| Gappy POD | `methods.gappy_pod.clip` | Least squares on the leading `sensors.rank` modes, clipped to the per-pixel training range | Linear baseline | Not trained; the run's train corruption is ignored |
| Error metric | — | Relative L2 over all test snapshots in unscaled units (`err`), also per channel (`err_u`, `err_v`) and per snapshot | | |
| Invariance | — | `d_latent`, `d_recon`: relative change of latent and reconstruction from clean to corrupted input | Tests whether models ignore faulty probes | |

## Planned, not built

Proposed defaults, to be confirmed before the first experiment that needs them.

| Fault | Proposed behaviour |
|---|---|
| `stuck` | The probe holds its reading from the first lag of the window |
| `drift` | Offset growing linearly over the lags at a configurable rate; rate 0 with an offset gives a constant bias |
