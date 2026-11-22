# Ideas for tonight

<!-- Copied to auto/ideas.md by `scripts/autoresearch.py init`. Highest priority first; placeholders below. -->

## 1. Learning rate for METHOD_A
What to try: train METHOD_A with a few learning rates around the current one (e.g. LR_LOW .. LR_HIGH).
Why: METHOD_A seems to train slowly; a better lr could close part of the gap to METHOD_B.
Objective: clean_err (mean).
Priority: high

## 2. Regularisation option for METHOD_B
What to try: add an off-by-default OPTION_X to fit (via train_options) and try two strengths.
Why: train_err is much lower than the validation error, which suggests overfitting.
Objective: missing_oracle_k2_err and missing_oracle_k4_err (mean).
Priority: medium
Notes: budget 3 screens + 1 confirmation.

## 3. Training corruption rate for METHOD_C
What to try: train METHOD_C on missing faults with p = P_LOW and P_HIGH instead of 0.3.
Why: robustness at high k may depend on how often training sees faults.
Objective: garbage_silent_k5_err (mean), conditions garbage/silent with ks: all.
Priority: low
