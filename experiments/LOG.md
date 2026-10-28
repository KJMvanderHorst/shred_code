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
