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

## 2026-11-07 · random_readings (rerun) · exp · PR #11 · tentative
Rerun of the 2026-10-15 garbage experiment on `runner.py`, 3 seeds throughout (the garbage-trained models were seed 0 only), plus exactly-k evaluation. Faulty probes emit N(0.5, 0.5), flagged (oracle) or undetected (silent).
Result (p = 0.3, mean over 3 seeds): every meeting number reproduces bit for bit except RobustSHREDv1/v2 (within seed std). Silent garbage: 0.11–0.23 for every dropout-trained method (SHRED+mask 0.108, SHRED-noise 0.119, RobustSHREDv2 0.138, v1 0.149, SHRED-dropout 0.151, gappy POD 0.235); with oracle flags the masked RobustSHREDs equal dropout (0.015). Trained on silent garbage: RobustSHREDv2 0.033 ± 0.003, SHRED 0.049 ± 0.007, RobustSHREDv1 0.074 ± 0.006 under silent garbage, at a clean cost of 0.016–0.032. Exact k, silent: dropout-trained models reach ≈ 0.04–0.11 already at k = 1; garbage-trained RobustSHREDv2 stays ≤ 0.04 up to k = 4.
Conclusion: Undetected garbage is the hard case: dropout-trained models degrade ~10× (0.11–0.15), and dropout training even hurts compared with noise-only SHRED (0.151 vs 0.119), so learned robustness is fault-specific. Given flags, hard masking (attention exclusion, gappy POD) is fully fault-agnostic (garbage oracle = missing oracle), while flags as an input (SHRED+mask) are not (0.128 with flags vs 0.108 without). Detection is the bottleneck (oracle 0.015 vs silent 0.14). Training on silent garbage closes most of the gap (3–4×, every seed). Ordering v2 (0.033) < SHRED (0.049) < v1 (0.074) holds in 3/3 seeds, but that is not established (sign test p = 1/8, encoders untuned; v1's clean 0.032 suggests an optimisation problem, not an inability to detect). Robustness does not transfer between fault types in either direction (garbage-trained v2 on missing: 0.042). Validity: only the large or structural effects stand; architecture differences do not (see dropout_baseline entry). Both faults are also easy to detect (constant 0; i.i.d. per lag, often outside [0, 1]).
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
