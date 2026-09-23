# get-shredded Copilot Instructions

- This is a scientific Python repository using Python >= 3.10, uv, Hydra, NumPy, SciPy, PyTorch, scikit-learn, matplotlib, and tqdm.
- Preserve scientific reproducibility: do not change seeds, dataset splits, model architecture, evaluation metrics, or experiment defaults unless the task explicitly requests it.
- Use the existing Hydra configuration system for experiment parameters rather than introducing parallel configuration mechanisms.
- Prefer small, localized changes over broad refactors.
- Run the smallest relevant validation command before expensive experiments.
- Do not expose large arrays, tensors, datasets, logs, or generated artifacts in chat output; summarize them or write them to files.
- For long-running experiments, prefer compact summary output and machine-readable result artifacts over verbose terminal output.
- Do not automatically expand an experiment beyond the configuration explicitly requested by the user.
