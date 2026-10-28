"""Run an experiment's run set (cfg.runs x cfg.eval.seeds); see configs/config.yaml.

Quiet and resumable: one JSON line per run; details go to outputs/<name>/<data>/.
  uv run python scripts/run.py +experiment=dropout_baseline
  uv run python scripts/run.py +experiment=dropout_baseline data=wave train.epochs=20 eval.seeds=[0]
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from get_shredded.runner import run_all


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_all(cfg)


if __name__ == "__main__":
    main()
