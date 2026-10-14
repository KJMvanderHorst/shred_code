"""Run the meeting experiment set (methods x seeds in configs/meeting.yaml).

Quiet by default: one JSON line per run; details go to outputs.root.
  uv run python scripts/run_meeting.py
  uv run python scripts/run_meeting.py meeting.seeds=[0] train.epochs=20 outputs.root=outputs/meeting_2026-10-14_quick
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.meeting_runner import run_all


@hydra.main(version_base=None, config_path="../configs", config_name="meeting")
def main(cfg: DictConfig) -> None:
    run_all(cfg, verbose=str(cfg.execution.mode).lower() == "verbose")


if __name__ == "__main__":
    main()
