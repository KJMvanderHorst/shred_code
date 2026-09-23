# Agent Instructions — get-shredded

## Project

This repository reimplements SHRED and compares it with SDN,
Senseiver/SenseiverSDN, RobustSHRED variants, and QR/POD on the
cylinder-vortex dataset.

The project is a scientific research codebase. Preserve experimental
semantics and reproducibility unless the task explicitly requests a
scientific/protocol change.

## Before changing code

- Read the relevant README and configuration before modifying experiment code.
- Identify the smallest set of files required for the requested change.
- Do not modify experimental defaults merely to make an experiment faster.
- Do not change seeds, train/validation/test splits, model architecture,
  evaluation metrics, or experiment protocol unless explicitly requested.

## Experiment execution

Treat agent context/output as scarce.

- Do not stream epoch-by-epoch training output during automated experiments.
- Use the repository's quiet/summary execution mode for sweeps.
- Do not use verbose/tqdm training output unless debugging is explicitly required.
- Prefer one batched experiment process over many separately supervised commands.
- Do not repeatedly poll a running experiment.
- Let long-running processes complete and inspect their compact result artifact.
- Do not print large arrays, NPZ contents, tensors, tables, logs, or datasets.
- Write detailed information to files instead of stdout.

## Experiment scope

- Execute exactly the requested experiment.
- Do not expand a sweep because a result appears interesting.
- Do not add extra seeds, sensor counts, placements, architectures, or
  robustness conditions unless explicitly requested.
- Do not rerun a completed configuration unless the configuration changed
  or the user explicitly requests a rerun.
- Prefer resumable experiment execution.

## Results

Every completed experiment should have a compact machine-readable result
artifact.

When reviewing experiment results:

1. Read the compact summary first.
2. Inspect detailed files only when necessary.
3. Do not load raw result arrays into the conversation unless requested.
4. Report metrics and output paths concisely.

## Visualizations

- Do not generate per-run plots or GIFs during sweeps unless requested.
- Prefer aggregate plots and tables.
- Generate detailed visualizations only for selected runs.

## Agent workflow

For expensive experiments:

1. Determine the exact requested run set.
2. Verify the configuration.
3. Execute the run set.
4. Read the compact result summary.
5. Report the results.
6. Stop.

Do not autonomously design follow-up experiments.

## Code changes

- Prefer small, localized changes.
- Preserve existing Hydra configuration patterns.
- Preserve existing output formats unless the task requires a change.
- Add tests for behavioral changes where practical.
- Do not rewrite working scientific code merely for stylistic reasons.

## Verification

After code changes:

- Run the smallest relevant test or smoke test first.
- For experiment runners, verify the command/configuration before launching
  expensive computation.
- Do not run a full sweep merely to verify a small code change unless required.