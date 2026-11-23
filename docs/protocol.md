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
| Placement | `sensors.rank`, `sensors.placement`, `sensors.locations` | `qr`: block QR pivoting over grid points on the rank-16 (= 2N) POD basis of the training data | A probe measures every channel at its point, so pivoting picks whole points; naive stacked QR gave cond ~1e3 | Placement depends on the seed through the training split. Options (unset = `qr`, not in config.yaml because `sensors` is fingerprinted): `random` = N distinct grid points from stream `[seed, 7]`; `explicit` = the grid points in `sensors.locations`. Gappy POD uses the same rank-`sensors.rank` basis for every placement |
| Sensor coordinates | — | Physical coordinates rescaled jointly to [0, 1] by the larger span | Positional encodings for the attention models | |

## Corruption

| Choice | Config key | Default | Rationale | Caveat |
|---|---|---|---|---|
| Noise | `noise_frac`, `corruptions.*.noise` | σ = 0.05 × per-channel training std, unscaled | Realistic sensor noise; on in every condition except `clean` | Fresh per batch in training; one fixed realisation per test draw, shared by all methods |
| Test draws | `eval.draws` | 10 noise and fault realisations per test window | Averages out the luck of one fault pattern | |
| Fault types | `corruptions.*.fault` | `none`, `missing`, `garbage` | | `stuck` and `drift` are planned, not built (see below) |
| Fault scope | — | A faulty probe is faulty in every channel, for the whole lag window | One physical probe fails, not one reading | |
| Training severity | `corruptions.*.p` | Each probe faulty with probability p = 0.3; draws where every probe fails are resampled | A different fault pattern every batch | |
| Evaluation severity | `eval.conditions[].ks` | Exactly k of N probes faulty, k = 0..N-1 (`ks: all`); a fresh uniformly random subset per test window and draw, shared by all methods | Error as a function of k, without mixing severities in one number | Never all N faulty. `ps` (per-probe p, as in training) is still available; `dropout_baseline` and `random_readings` report `ps: [0, 0.3]` next to `ks: all`, for comparison with the meeting |
| Input | `runs[].input` | `probes`: the real probe readings | | Decoupling tests only: `wave_input` (16 columns of wave trajectory `wave_input.trajectory_index`), `random_input` (fixed seeded N(0,1) series), `shuffled_targets` (real probes, targets at a fixed seeded time permutation). Placement, target scaler and sensor coordinates stay those of the real probes |
| Missing fill | `corruptions.missing.fill` | 0.0 in scaled input space | | 0 is the per-sensor training minimum, not "no signal" |
| Garbage | `corruptions.garbage.garbage_*` | N(0.5, 0.5) per channel per lag, scaled input space | A broken probe emits plausible-range nonsense | |
| Flags | `corruptions.*.flags`, `eval.conditions[].flags` | `oracle`: models with a mask input get the true fault mask. `silent`: they get an all-zero mask | Separates "fault known" from "fault undetected" | Models without a mask input (SHRED, SDN) behave the same under both. Gappy POD drops faulty rows under `oracle` and uses them under `silent` |
| Partial flags | `corruptions.*.flag_frac` | 1.0: under `oracle` every faulty probe is flagged | | `< 1` (`garbage_mixed`: 0.5, residual_flags only): each faulty probe is flagged with that probability, the rest go in unflagged, so a masked model learns to use flags and to cope with silent garbage. Drawn per probe per batch from the global torch RNG in training; in validation and test after the garbage values from the same rng (streams [seed, 1] / [seed, 3]). Not in the fingerprint at 1.0, so earlier runs keep their config ids |
| Residual flags | `residual_flags.*` | Off (test time only, `residual_flags` experiment) | A bad reading should disagree with the field the model predicts at its own probe, so flags can come from the model instead of an oracle | Pass 1 runs the trained model with no flags; r = last-lag reading − prediction at the probe, physical units; score d_j = sqrt(Σ_c (r_cj / s_cj)²), s = residual std on the clean validation windows with the fixed validation noise; flag the probe for the whole window if d_j > τ_j = quantile q of d_j on the same windows (q = 0.95 / 0.99 / 0.999). Soft (v1 only): log(max(exp(−0.5 d²), 1e-6)) added to the probe attention logits at every lag, null token unbiased. Pass 2 reruns with the flags. Faults stay whole-window, so the last lag is the only residual a window has (no per-lag flags). Thresholds come from ≈ 19 validation windows and one noise draw |
| Residual v1 | `methods.robust_shred_v1_residual` | Off (only `residual_v1`) | A model that learns to use its own residual, instead of a fixed test-time rule (`residual_flags`) | One RobustSHRED v1 with 2C input channels per probe (readings and residuals), no mask input. Pass 1: residual channels 0. r = last-lag reading − pass-1 prediction at the probe, MinMax-scaled units, gradient stopped. Pass 2: the same network with r at every lag (faults are whole-window); its output is the prediction. Training loss MSE(pass 2) + `pass1_weight` (1.0, untuned) × MSE(pass 1) via `fit`'s `target_aux_loss` hook (absent on every other model). `pass1__<condition>` in result.json: the same network evaluated with r = 0 (ablation). ≈ 2× v1 per step |
| Validation | — | Corrupted with the run's own training corruption, one fixed seeded realisation | Early stopping selects for robustness, not clean accuracy | |
| Evaluation split | `eval.split` | `test`: every metric, per-snapshot error, panel and invariance check uses the test windows | Test results are what we report | `valid` (overnight tuning only) evaluates on the validation windows instead, with its own draws from stream `[seed, 6]`; test windows never pass through a model. The validation set also drives early stopping, so valid-split scores are optimistically biased: fine for ranking trials, never for reporting. Recorded as `eval_split` in result.json / summary.json and in the fingerprint (valid only) |
| RNG streams | — | `default_rng([seed, i])`: 1 validation noise + training-corruption validation masks, 2 test draws, 3 garbage values, 4 shuffled targets, 5 random input, 6 valid-split eval draws, 7 random placement, 8 held-out divergence points (POD trunk SHRED) | Each use has its own stream, so adding one never shifts another | |

## Training and metrics

| Choice | Config key | Default | Rationale | Caveat |
|---|---|---|---|---|
| Training | `train.*` | Adam, lr 1e-3, batch 64, MSE, up to 1000 epochs | SHRED paper setting | |
| Device | `device`, `jobs`, `threads` | `cpu`, 6 worker processes × 1 torch thread, one (seed, run) per worker | speed_benchmark: 4.5× the runs/min of one MPS job on the M1 Pro | Not in the run fingerprint; CPU and MPS results differ (shred 0.0104 vs 0.0086 at 100 epochs), so never mix devices within an experiment. Experiments before speed_benchmark are pinned to `device: mps`, `jobs: 3` |
| Per-method lr and schedule | `methods.*.lr`, `methods.*.lr_schedule` | Off: every method uses `train.lr`, fixed (`lr_schedule: none`) | Lets a slow-training method (Nektar trunk) use a higher start with decay without changing other methods | `cosine` decays lr to 0 over `train.epochs`, stepped per epoch, so early stopping can end a run before lr is small. Only the `flow_trunk_shred*cos*` variants set them |
| Extra fit options | `methods.*.train_options` | Absent: `fit` gets only the runner's arguments | Lets new off-by-default training options in `model.py` (`fit` keyword arguments) be set per method without editing the runner | Keys must be `fit` parameters the runner does not already set (lr, schedule, epochs, batch, patience, early stopping have their own keys); unknown keys raise. Part of the method dict, so fingerprinted |
| Early stopping | `train.patience`, `runs[].early_stopping` | Validation relative L2 every 20 epochs, patience 5, best weights restored | | `early_stopping: false` (decoupling tests only) trains all epochs and keeps the final weights |
| Gappy POD | `methods.gappy_pod.clip` | Least squares on the leading `sensors.rank` modes, clipped to the per-pixel training range | Linear baseline | Not trained; the run's train corruption is ignored |
| Trunk SHRED (wave only) | `methods.trunk_shred*` | SHRED GRU branch → p = 64 coefficients; tanh trunk (2 × 64) over (x, τ), τ = time relative to the window's last frame, x through 16 Fourier frequencies. Output at the grid x and τ = 0, MinMax-scaled inside the model, so the target, loss and metrics equal SHRED's. `trunk_shred_pinn` adds λ = 1e-3 × (mean ((u_ττ − c²u_xx)/ω₁²)² + mean u(0 or L, τ)²) on 64 random (x, τ) points per sample, in physical units | DeepONet-style decoder whose derivatives give a PDE residual; the wave solution is exactly a 20-function trunk | λ and the frequency count were chosen on the wave by seed-0 test error; validation over 3 seeds picks the same (`experiments/trunk_shred_wave`). The Fourier features sin(kπx/L) include the wave's own spatial modes. Only τ = 0 is supervised, so u_t is unconstrained: the field at earlier τ is no better than copying τ = 0 back, with or without physics. `*_probes` variants add μ = 1 × mean over probes and lags of (scaled û(x_probe, τ_j) − reading)², the model's own input, so clean training only (the runner refuses noise or faults). Raises on data other than the wave |
| Flow trunk SHRED (Nektar u, v only) | `methods.flow_trunk_shred*` | SHRED GRU branch → 2 × p coefficients (u, v); tanh trunk over (x, y) only (no τ: only the last frame is supervised), each axis scaled to [0, 1] through 16 Fourier frequencies, one basis per channel. Main trunk: width 256, p = 128, cosine lr from 3e-3. Output at the grid, MinMax-scaled inside the model, so target, loss and metrics equal SHRED's. `*_div*` add λ × mean (u_x + v_y)² on 64 random (x, y) points per sample in the grid's box (physical units), λ = 1e-4 / 1e-3 / 1e-2 | DeepONet-style decoder whose derivatives give the continuity residual; NS momentum needs u_t, which a trunk without τ cannot give | Size, frequencies and lr were picked on seed-0 test error (`experiments/trunk_shred_nektar`). The trunk trains far slower than SHRED (seed 0: 0.044 vs 0.0027 at 1000 epochs, 0.011 vs 0.0019 at 5000), so `trunk_shred_nektar_5k` runs every method, SHRED included, with `train.epochs: 5000`. The data's own finite-difference divergence is 0.3 % of rms ∇u, the floor for the residual. Raises on data other than Nektar u, v |
| POD SHRED | `methods.pod_shred*` | SHRED (same GRU encoder and 350/400 decoder) predicting 2r POD coefficients, the masked-shred pipeline: per field (u, v separately) StandardScaler on `load_X[train_idx]`, exact SVD, r modes per field, MinMax per coefficient on the same rows. `forward` decodes to the field, MinMax-scaled like SHRED's targets, so evaluation and metrics equal SHRED's. `loss_space: field` trains on the decoded field (loss and early stopping as SHRED); `loss_space: coef` trains the network on the scaled coefficients of the clean targets, with loss and early stopping in coefficient space | Tests whether the 10 000-wide output layer, or the loss weighting, drives accuracy and speed (`experiments/pod_shred_nektar`) | The paper uses randomized SVD; the exact SVD gives the same modes up to that approximation. `coef` weighs every mode equally, whatever its energy. Error is floored by the rank-r projection of the test snapshots. `wall_time_s` is measured with 6 parallel workers, so it is noisy |
| POD trunk SHRED (Nektar u, v only) | `methods.pod_trunk_shred*` | POD SHRED at r = 16 whose basis (per-field mean, std and modes) is a frozen continuous trunk T(x, y): tanh 2 × 256 on 16 Fourier frequencies per axis (the flow trunk's architecture), fitted once per seed to the grid basis (Adam lr 1e-3, full batch, 5000 steps, each output column standardised). Reconstruction uses the trunk at the grid; `loss_space: coef` trains on the true POD coefficients. `pod_trunk_shred_div` adds λ = 1e-3 × mean (u_x + v_y)² through the trunk on 64 random points per sample (physical units). `pod_trunk_shred_train*` (`train_trunk: true`) keep training the trunk with SHRED, same optimiser and lr (the run's lr, 1e-3), so the field and continuity losses can reshape the basis; the grid basis is then recomputed from the trunk at every decode. Records `trunk_fit` (error per mode, mean/std error, max \|Gram − I\|; of the initial fit), `trunk_parameters` (frozen: not in `parameters`; trained: included) and `div_rel` (rms divergence / rms \|∇u\| of the clean test predictions, on 1024 held-out points and on the grid) | Continuous, differentiable POD modes at POD training speed (user's proposal, `experiments/pod_shred_nektar`) | The trunk misfit (seed 0: modes 1.6–2.3 %, mean field 0.4 %) adds to the error, so it can floor the accuracy above SHRED's. The trunk fit takes ≈ 85 s per run on one thread |
| Error metric | — | Relative L2 over all test snapshots in unscaled units (`err`), also per channel (`err_u`, `err_v`) and per snapshot | | |
| Training error | — | Relative L2 on the clean (noise-free) training windows, unscaled (`train_err`); trained models only | Separates memorisation from generalisation | Recorded from `physical_relation` on; earlier runs lack it |
| Invariance | — | `d_latent`, `d_recon`: relative change of latent and reconstruction from clean to corrupted input | Tests whether models ignore faulty probes | |

## Overnight mode

`scripts/autoresearch.py` tunes on the validation split in a separate worktree (branch `auto/<date>`, outputs in
`outputs_auto/`); its rules are in `AUTORESEARCH.md`. It always forces `eval.split=valid` and `device=cpu`, so
its numbers are for ranking trials only. A promoted candidate is re-run on the test split in daytime mode next to
its incumbent, both on CPU, before any result is reported.

Within a night a candidate counts as better than its incumbent only on three seeds, with the same problem
(data, lags, sensors, placement, noise, conditions, draws), when its mean is lower by more than the larger seed
std and it is lower on every seed; the driver enforces this. Trials may change method hyperparameters, training
settings (epochs, batch, lr, patience, early stopping, training corruption), `eval.conditions`, `sensors.*`,
`noise_frac` and `data.lags`; never `data.split`, seeds, `eval.draws`, the evaluation split or the device.

## Planned, not built

Proposed defaults, to be confirmed before the first experiment that needs them.

| Fault | Proposed behaviour |
|---|---|
| `stuck` | The probe holds its reading from the first lag of the window |
| `drift` | Offset growing linearly over the lags at a configurable rate; rate 0 with an offset gives a constant bias |
