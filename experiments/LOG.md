# Experiment log

Lab notebook of everything tried: experiments (`exp`), features (`feat`) and
fixes (`fix`). Newest first. Every PR adds or updates its entry here before
merging. Details live in the linked files; the PR (or commit) is the exact code
that produced a result.

Entry format:

```
## YYYY-MM-DD · <name> · exp|feat|fix · PR #<n> · <status: planned|tentative|settled>
<one line: what was tried, or the question>
Result: <facts and numbers>
Conclusion: <what we take from it, or "not written">
→ <links: experiments/<name>/README.md, docs/protocol.md rows, outputs path>
```

---

## 2026-11-23 · pod_trunk_shred (prototype) · exp · PR #20 · tentative
Prototype, user's proposal: POD SHRED at r = 16 whose basis (per-field mean, std, modes) is a continuous trunk T(x, y) (tanh 2 × 256, 16 Fourier frequencies per axis) fitted to the grid basis, so a continuity loss λ × mean (u_x + v_y)² can be added. Adds `PODTrunkSHRED` (kind `pod_trunk_shred`, `train_trunk`, RNG stream 8 for held-out divergence points). Frozen trunk (field loss, coefficient loss, field + λ = 1e-3) and trained trunk (same lr as SHRED, λ = 0 / 1e-3). Nektar (u, v), clean, standard protocol, 3 seeds, CPU.
Result: test error frozen 0.0077 ± 0.0003 / coef 0.0098 ± 0.0012 / + continuity 0.0074 ± 0.0000; trained 0.0067 ± 0.0008 / + continuity 0.0068 ± 0.0013 (POD SHRED 0.0025). div_rel (autograd) 0.126 in every frozen run, 0.102 trained, 0.095 ± 0.012 trained + continuity; data's own FD divergence 0.003. Exact coefficients through the frozen trunk: 0.0071 (exact basis 2.9e-4); trunk values ≈ 0.5 % off, gradients ≈ 10 %; the best coefficients cancel 0.7 % of the mean field's divergence.
Conclusion: The trunk fit, not SHRED, sets the error and the divergence. Frozen, the continuity loss cannot act (it only moves coefficients); trained, it adds ≈ 7 % lower divergence, within the seed spread. Prototype only; leads: larger λ, no early stopping, derivative-aware trunk fit, stream-function trunk on a joint (u, v) POD.
→ experiments/pod_shred_nektar/README.md (POD trunk section), notebooks/pod_shred_nektar.ipynb (section 5), docs/protocol.md (POD trunk SHRED row, RNG streams), outputs/pod_shred_nektar/nektar_uv/ (local only)

## 2026-11-23 · pod_shred_nektar · exp · PR #20 · tentative
SHRED predicting POD coefficients instead of the full field (masked-shred pipeline: per field StandardScaler + exact SVD, r = 16 / 32 / 64 modes per field, MinMax per coefficient), with the loss on the decoded field or on the scaled coefficients (as the paper). Adds `PODSHRED` (kind `pod_shred`, `loss_space: field | coef`). Nektar (u, v), clean, 1000 epochs, early stopping, 3 seeds, CPU.
Result: test error SHRED 0.0021 ± 0.0003 (4.2M params, 43 s/run); POD field loss 0.0025 ± 0.0003 / 0.0027 ± 0.0003 / 0.0028 ± 0.0008 at r = 16 / 32 / 64 (217k–255k params, 12–14 s/run); POD coefficient loss 0.0028 ± 0.0002 / 0.0106 ± 0.0021 / 0.0197 ± 0.0036 (7–8 s/run). Projection floor 2.9e-4 / 2.4e-6 / 1.6e-6. Train error ≈ test error everywhere; POD runs stop at ≈ 410–510 epochs vs ≈ 710.
Conclusion: Field-loss POD is ≈ 3× cheaper and ≈ 20 % less accurate than the full-field output (≈ one seed std). The paper's coefficient loss degrades with rank on this low-rank wake: near-empty modes are scaled to [0, 1] and weigh as much as the energetic ones. On this setup the output layer, not the encoder, sets SHRED's cost per epoch.
→ experiments/pod_shred_nektar/README.md, notebooks/pod_shred_nektar.ipynb, docs/protocol.md (POD SHRED row), outputs/pod_shred_nektar/nektar_uv/ (local only)

## 2026-11-22 · trunk_shred_nektar · exp · PR #19 · tentative
DeepONet-style trunk SHRED on Nektar (u, v): SHRED GRU branch, tanh trunk over (x, y) only (no τ: only the last frame is supervised), continuity loss λ × mean (u_x + v_y)² (NS momentum needs u_t). Adds `FlowTrunkSHRED`, `divergence`, per-method `lr` and `lr_schedule: cosine` (code merged with PR #15). Two regimes, 3 seeds, clean: A = standard protocol (1000 epochs, early stopping), B = `trunk_shred_nektar_5k` (5000 epochs, early stopping off).
Result: A: SHRED 0.0026 ± 0.0002, trunk (width 256, p 128, cosine from 3e-3) 0.0400 ± 0.0021, continuity λ = 1e-4 / 1e-3 / 1e-2 0.0401 / 0.0398 / 0.0392 (≤ 0.001 per seed); train = test error, still improving at 1000 epochs. B: SHRED 0.0013 ± 0.0005, trunk fixed lr 1e-3 0.0108 ± 0.0006; cosine trunk 0.0066 / 0.0065 on seeds 1, 2 but seed 0 diverges at epoch ≈ 1200–2000 in all four cosine runs (0.053–0.076), and seed 1 at λ = 1e-3 (0.036); λ = 1e-2 0.0062 / 0.0062 on the stable seeds. Seed-0 search: width / p / frequencies barely matter (0.053 → 0.044), cosine from 1e-3 hurts, from 3e-3 helps (0.037). A first B attempt with early stopping stopped every run after 440–880 epochs (validation spikes vs patience 5). The data's own divergence is 0.3 % of rms ∇u.
Conclusion: The trunk underfits because it trains slowly (the user's hypothesis), not for lack of capacity: ≈ 15× worse than SHRED at 1000 epochs, ≈ 5–8× at 5000, still improving. A 3e-3 start is faster but unstable (one seed diverges in every cosine run, so a seed × lr effect, not physics). Continuity has no effect on the undertrained trunk; on the stable long runs λ = 1e-2 is ≈ 5 % better on 2 seeds, not established. Early stopping with patience 5 does not suit this method's noisy validation curve. Leads: lower cosine start or warm-up, gradient clipping, best-checkpoint restore without early stopping.
→ experiments/trunk_shred_nektar/README.md, notebooks/trunk_shred_nektar.ipynb, docs/protocol.md (Flow trunk SHRED, Per-method lr and schedule rows), outputs/trunk_shred_nektar/nektar_uv/, outputs/trunk_shred_nektar_5k/nektar_uv/ (local only)

## 2026-11-22 · autoresearch driver · feat · PR #17 · tentative
Overnight mode: `scripts/autoresearch.py` (init / run / report / status) alternates stateless `claude -p` planning steps (AUTORESEARCH.md rules, `--restricted`, no shell, edit allowlist) with plain run.py batches (one run.py per trial, ≤ 6 CPU jobs, `eval.split=valid`), in a worktree on `auto/<date>`. Guard on changed paths, experiment-file allowlist, pytest + exact canary on code changes, driver-enforced 3-seed gate, time budget from speed_benchmark priors.
Result: dry run (wave, 20 epochs, 3 steps, $1.06): intake → one full batch of 6 jobs (2 ideas, incumbent re-measured on 3 seeds) recorded with `device=cpu`, `eval_split=valid` and per-problem ids; an injected edit to configs/config.yaml was caught and reverted; an injected failing test was caught by pytest and reverted; report written. Canary repeats exactly across runs. A direct permission probe: edit of configs/config.yaml and a read of the main checkout's outputs/ denied, auto_*.yaml allowed, no shell.
Conclusion: Ready for a first real night; validation numbers rank trials only, promotion is a daytime test-split run.
→ AUTORESEARCH.md, scripts/autoresearch.py, docs/protocol.md (Overnight mode), docs/ideas_template.md, CLAUDE.md (Overnight mode), tests/test_autoresearch.py

## 2026-11-22 · autoresearch core options · feat · PR #16 · settled
Runner options for overnight tuning, all off by default: `eval.split: valid` (evaluate on the validation windows, own RNG stream `[seed, 6]`, test windows never through a model), `methods.*.train_options` (extra `fit` kwargs, validated against `fit`'s signature), `sensors.placement: random | explicit` (stream `[seed, 7]` / `sensors.locations`). Methods can be defined inside an experiment file.
Result: wave + Nektar 20-epoch smoke runs (7 runs) give identical metrics and config_ids before and after; all 168 cached runs of dropout_baseline, random_readings, physical_relation, trunk_shred_nektar and trunk_shred_wave still match their fingerprints. A poisoning test (test windows set to NaN) shows `eval.split=valid` never touches them.
Conclusion: Daytime mode is unchanged; valid-split scores are optimistically biased (validation also drives early stopping) and are for ranking only.
→ docs/protocol.md (Evaluation split, RNG streams, Extra fit options, Placement rows, Overnight mode), src/get_shredded/runner.py, tests/test_core_options.py

## 2026-11-21 · parallel CPU default · feat · PR #15 · settled
`run.py` now trains on CPU in 6 worker processes × 1 torch thread by default (`device: cpu`, `jobs: 6`, `threads: 1`); each worker trains one (seed, run). Experiments that already have MPS results are pinned to `device: mps`, `jobs: 3`.
Result: parallel and serial give the same test errors (test + Nektar smoke, difference 0.0). Nektar, 6 runs × 100 epochs: 29.6 s with 6 workers vs 47.1 s serial on CPU; the longest run sets the total when there are few runs. Pinned experiments still skip every cached run.
Conclusion: Use the default for new experiments; never mix devices within one experiment.
→ docs/protocol.md (Device row), configs/config.yaml, src/get_shredded/runner.py (run_all), experiments/speed_benchmark/README.md

## 2026-11-21 · speed_benchmark · exp · PR #15 · tentative
Which device / parallelism trains fastest on the M1 Pro? Adds a `device` key (default `auto` = MPS, unchanged) and times 18 runs (shred, flow_trunk_shred_w256, flow_trunk_shred_div; 100 fixed epochs) under 9 setups, 2 repeats.
Result: one MPS job (what we used so far) 4.6 runs/min; 6 CPU jobs × 1 thread 20.4 (4.5×); hybrid 2 MPS + 4 CPU jobs 21.5 (4.7×). A single CPU job (6 threads) also beats MPS (shred 4.1 vs 8.0 s). Test error is the same across repeats per device but differs between devices (shred 0.0104 CPU vs 0.0086 MPS at 100 epochs).
Conclusion: The models are too small for the GPU; run sweeps as ~6 single-thread CPU jobs (split by seed). Do not mix devices within one comparison.
→ experiments/speed_benchmark/README.md, notebooks/speed_benchmark.ipynb, docs/protocol.md (Device row), outputs/speed_benchmark/ (local only)

## 2026-11-18 · trunk_shred_wave (probe supervision) · exp · PR #14 · tentative
Part 1 left u_t unsupervised. Fit the trunk also to the probe readings at every lag (the model's own clean input, μ = 1), for λ = 0 / 1e-4 / 1e-3 / 1e-2; adds `probe_weight` and `aux_loss` (renamed from `physics_loss`) to `TrunkSHRED`.
Result (wave, clean, 3 seeds): τ = 0 test 0.0432 ± 0.0103 (no physics) and 0.0354 ± 0.0083 (λ = 1e-3) vs 0.0285 / 0.0255 without probe supervision. û_t at the probes improves (0.38 / 0.26 vs ≈ 1.0) but between them gets worse than predicting no motion (1.44 / 1.27); window error 0.284 / 0.263 vs 0.220, above the copy-back baseline (≈ 0.21). QR probes sit in two clusters (gaps up to 0.85) while a wave travels ≈ 0.07 per window.
Conclusion: Probe supervision pins velocity only at the probes; neither the data nor the PDE can carry it across the gaps within one window, so it hurts both the window and τ = 0. Physics helps relative to probes alone (first measurable physics effect) but does not recover the loss. Leads: probes placed for dynamics, a longer window, full-field supervision beyond τ = 0.
→ experiments/trunk_shred_wave/README.md (part 2), notebooks/trunk_shred_wave.ipynb, docs/protocol.md (Trunk SHRED row), outputs/trunk_shred_wave/wave/ (local only)

## 2026-11-10 · trunk_shred_wave · feat · PR #13 · tentative
DeepONet-style SHRED with a physics loss, verified on the 1D wave: SHRED GRU branch, tanh trunk over (x, τ) with 16 Fourier frequencies on x, wave residual + Dirichlet BCs on random collocation points (λ = 1e-3). Adds `TrunkSHRED`, `wave_residual`, a `physics_loss` hook in `fit`, methods `trunk_shred` / `trunk_shred_pinn` and four search variants.
Result (wave, clean, 3 seeds): test 0.0255 ± 0.0040 (physics) vs 0.0285 ± 0.0019 (trunk, λ = 0) vs 0.0285 ± 0.0065 (SHRED); wave residual 0.024 vs 0.96; dispersion within 1 % vs 79 %. Over the window both trunk models equal copying the τ = 0 field back (τ = −9 dt: 0.39 vs 0.38) and their û_t is uncorrelated with the truth (rel. error 1.02 / 1.08). No Fourier features: 0.44; λ = 1e-2: 0.129; λ = 1e-4: 0.035. Settings picked on seed-0 test error; validation picks the same.
Conclusion: The trunk can be made to satisfy the PDE at no accuracy cost, but the physics adds nothing the model uses: only τ = 0 is supervised, so u_t is free and physics only makes the τ-direction self-consistent. The design (interleaved split, clean data) cannot show a physics benefit. The Fourier basis contains the wave's own modes, so this is the easy case. Lead: supervise the trunk at the probes for all lags, then test block split, noise, faults.
→ experiments/trunk_shred_wave/README.md, notebooks/trunk_shred_wave.ipynb, docs/protocol.md (Trunk SHRED row), outputs/trunk_shred_wave/wave/ (local only)

## 2026-11-08 · physical_relation · exp · PR #12 · tentative
Does reconstruction need a physical relation between probe readings and the field? SHRED, SDN, RobustSHREDv2 (noise training) with real probes, a physically unrelated wave signal, a fixed random series, or real probes with shuffled targets; with and without early stopping. Adds per-run `input` and `early_stopping`, and `train_err` for every trained run.
Result (test, mean over 3 seeds; mean-field floor 0.328): probes 0.004–0.007; wave input 0.010–0.016 for every method; random input: SHRED 0.07–0.08, SDN memorises without early stopping (train 0.018) but tests at 0.40, v2 at the floor (fits on 1 of 3 seeds without early stopping); shuffled targets at the floor, and not memorised in 1000 epochs either (train 0.25–0.27).
Conclusion (revised 2026-11-10): Under the interleaved split (the intended SHRED setting), an input that only encodes time (wave) works nearly as well as the probes, so whether a physical relation is needed is not yet tested; the probes' 2–4× advantage is likely exact phase matches across the ≈ 3.4 cycles (period 59 snapshots). The models can memorise arbitrary pairs (SDN on random keys: train 0.018, test 0.40). SHRED's 0.07 on random input comes from overlapping windows. Shuffled targets is not a capacity test: windows one cycle apart share near-identical inputs but get unrelated targets, and the observed train error (0.25–0.27) matches averaging ≈ 3 targets per phase. Next: block split with a periodic phase clock, which needs more cycles.
→ experiments/physical_relation/README.md, docs/protocol.md (Input, Early stopping, Training error rows), outputs/physical_relation/nektar_uv/ (local only)

## 2026-11-07 · random_readings (rerun) · exp · PR #11 · tentative
Rerun of the 2026-10-15 garbage experiment on `runner.py`, 3 seeds throughout (the garbage-trained models were seed 0 only), plus exactly-k evaluation. Faulty probes emit N(0.5, 0.5), flagged (oracle) or undetected (silent).
Result (p = 0.3, mean over 3 seeds): every meeting number reproduces bit for bit except RobustSHREDv1/v2 (within seed std). Silent garbage: 0.11–0.23 for every dropout-trained method (SHRED+mask 0.108, SHRED-noise 0.119, RobustSHREDv2 0.138, v1 0.149, SHRED-dropout 0.151, gappy POD 0.235); with oracle flags the masked RobustSHREDs equal dropout (0.015). Trained on silent garbage: RobustSHREDv2 0.033 ± 0.003, SHRED 0.049 ± 0.007, RobustSHREDv1 0.074 ± 0.006 under silent garbage, at a clean cost of 0.016–0.032. Exact k, silent: dropout-trained models reach ≈ 0.04–0.11 already at k = 1; garbage-trained RobustSHREDv2 stays ≤ 0.04 up to k = 4.
Conclusion: Undetected garbage is the hard case: dropout-trained models degrade ~10× (0.11–0.15), and dropout training even hurts compared with noise-only SHRED (0.151 vs 0.119), so learned robustness is fault-specific. Given flags, hard masking (attention exclusion, gappy POD) is fully fault-agnostic (garbage oracle = missing oracle), while flags as an input (SHRED+mask) are not (0.128 with flags vs 0.108 without). Detection is the bottleneck (oracle 0.015 vs silent 0.14): what the model knows about the faults is a main factor next to the training corruption, and the flag regime at test must match training even with exclusion (trained with flags, tested without: 0.14; trained without, tested with: v2 0.033 → 0.042). Training on silent garbage closes most of the gap (3–4×, every seed). Ordering v2 (0.033) < SHRED (0.049) < v1 (0.074) holds in 3/3 seeds, but that is not established (sign test p = 1/8, encoders untuned; v1's clean 0.032 suggests an optimisation problem, not an inability to detect). Robustness does not transfer between fault types in either direction (garbage-trained v2 on missing: 0.042). Validity: only the large or structural effects stand; architecture differences do not (see dropout_baseline entry). Both faults are also easy to detect (constant 0; i.i.d. per lag, often outside [0, 1]).
→ experiments/random_readings/README.md, notebooks/random_readings.ipynb, outputs/random_readings/nektar_uv/ (local only)

## 2026-11-07 · dropout_baseline (rerun) · exp · PR #11 · tentative
Rerun of the meeting experiment on `runner.py`: `senseiver_sdn` dropped, `shred_mask__missing` added, exactly-k evaluation added. Adds `plots.plot_exact_k`.
Result (p = 0.3, mean over 3 seeds): SHRED, SHRED+mask, SDN and gappy POD reproduce the meeting bit for bit; RobustSHREDv1/v2 move 0.0150 → 0.0146 and 0.0151 → 0.0143 (per-seed differences, within seed std; cause not identified, the restructure check covered 20 epochs). First exact-k run on Nektar: dropout-trained models stay ≈ 0.01–0.05 up to k = 5 of 8, then rise to 0.13–0.21 at k = 7; noise-trained SHRED/SDN reach ≈ 0.08–0.09 at k = 1; gappy POD ≈ 0.07 at k = 1.
Conclusion: Fault-agnostic training is not viable: one missing probe raises error 20× (0.005 → 0.08). Dropout training makes every model robust (≤ 0.03 up to k = 5 of 8) at a ~2.5× clean cost (0.004 → 0.010). No architecture ranking under dropout is supported: SHRED, RobustSHREDv1/v2 swap order across seeds (SHRED+mask marginally best), possibly because a constant-zero probe is self-flagging. Time history matters (SDN 2× worse). Gappy POD with 8 probes × 2 channels = 16 measurements for 16 modes has no redundancy and fails at k = 1. Validity: ~153 training windows (200 snapshots) for 4.2 M parameters, 99 % in a decoder shared by all methods (encoders 41–100 k); periodic wake plus interleaved split, so the task is close to recovering the shedding phase and the absolute errors are interpolation errors. Test and validation sets are ~19 windows each, the hyperparameters are untuned per method, and a seed varies split, placement, init and faults at once. Only large (≥ 3×) or structural effects stand.
→ experiments/dropout_baseline/README.md, notebooks/dropout_baseline.ipynb, outputs/dropout_baseline/nektar_uv/ (local only)

## 2026-10-28 · exact-k evaluation · feat · PR #9
Evaluate at exactly k of N faulty probes (`ks`, default `ks: all` = 0..N-1) instead of per-probe p; `ps` stays available.
Result: `dropout_baseline` (still on `ps`) reproduces the meeting run bit for bit. No exact-k run on Nektar yet.
Conclusion: not written.
→ docs/protocol.md (evaluation severity)

## 2026-10-28 · restructure · feat · PR #8
One runner (`scripts/run.py`), one config tree (data / method / corruption / experiment), `missing` and `garbage` faults with oracle/silent flags, wave dataset for quick checks. Old pipeline archived in tag `pre-restructure`.
Result: the meeting run reproduces bit for bit (20 epochs, seed 0: all metrics, per-sample errors, split, probe locations, test noise and masks).
Conclusion: results from `meeting_runner` stand for the new pipeline.
→ README.md, docs/protocol.md

## 2026-10-15 · random_readings · exp · PR #7 · tentative
Faulty probes emit garbage N(0.5, 0.5) at p = 0.3 instead of dropping out. How do the meeting models cope, flagged (oracle) and undetected (silent)? And does training on silent garbage help?
Result, meeting models (3 seeds, relative L2, mean): silent 0.11–0.24 for every method (SHRED-dropout+mask 0.108, SHRED-noise 0.119, RobustSHREDv2 0.141, RobustSHREDv1 0.149, SHRED-dropout 0.151, SenseiverSDN 0.170, SDN-noise 0.168, SDN-dropout 0.190, GappyPOD 0.235); oracle equals dropout for masked models (RobustSHREDv1/v2 0.015, SenseiverSDN 0.028, GappyPOD 0.092), but 0.128 for SHRED-dropout+mask.
Result, trained on silent garbage (seed 0 only): SHRED 0.045, RobustSHREDv1 0.058, RobustSHREDv2 0.031 under silent garbage; clean 0.019–0.022.
Conclusion: not written. Not yet re-expressed as a `configs/experiment/` file.
→ outputs/_legacy/meeting_2026-10-14/random_readings/, outputs/_legacy/meeting_random_readings/ (local only); code in tag `pre-restructure` (`random_readings.py`, notebook 05)

## 2026-10-14 · dropout_baseline · exp · PR #7 · tentative
How much does reconstruction degrade when probes drop out (p = 0.3), and which methods stay robust when trained with dropout?
Result: trained with noise only, SHRED and SDN go from ≈0.005 to ≈0.16–0.18 at p = 0.3; every dropout-trained or masked method stays at 0.011–0.028; gappy POD 0.092.
Conclusion: not written.
→ experiments/dropout_baseline/README.md
