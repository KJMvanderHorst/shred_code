from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.experiment import (
    _result_summary_payload,
    _stable_config_fingerprint,
    run_experiment,
    write_result_summary,
)


@hydra.main(version_base=None, config_path="../configs", config_name="recurrent_cell_benchmark")
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(cfg.comparison.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = [str(cell).lower() for cell in cfg.comparison.cells]
    verbose = str(cfg.execution.mode).lower() == "verbose"

    records: list[dict[str, object]] = []
    for cell in cells:
        cell_dir = output_dir / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "experiment": "recurrent_cell_benchmark",
            "recurrent_cell": cell,
            "num_sensors": int(cfg.model.num_sensors),
            "placement": str(cfg.model.placement),
            "seed": int(cfg.seed),
            "lags": int(cfg.model.lags),
            "hidden_size": int(cfg.model.hidden_size),
            "hidden_layers": int(cfg.model.hidden_layers),
            "train_epochs": int(cfg.train.epochs),
            "batch_size": int(cfg.train.batch_size),
            "lr": float(cfg.train.lr),
            "patience": int(cfg.train.patience),
            "shred_only": bool(cfg.comparison.shred_only),
        }
        config_id = _stable_config_fingerprint(config)
        summary_path = cell_dir / "result.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
            if payload.get("status") == "completed" and payload.get("config_id") == config_id:
                records.append({
                    "cell": cell,
                    "status": "skipped",
                    "config_id": config_id,
                    "final_metrics": payload.get("final_metrics", {}),
                    "output_dir": str(cell_dir),
                })
                continue
        result = run_experiment(
            mat_path=Path(to_absolute_path(cfg.data.mat)),
            num_sensors=int(cfg.model.num_sensors),
            lags=int(cfg.model.lags),
            placement=str(cfg.model.placement),
            test_size=int(cfg.data.test_size),
            val_size=int(cfg.data.val_size),
            hidden_size=int(cfg.model.hidden_size),
            hidden_layers=int(cfg.model.hidden_layers),
            l1=int(cfg.model.l1),
            l2=int(cfg.model.l2),
            dropout=float(cfg.model.dropout),
            epochs=int(cfg.train.epochs),
            batch_size=int(cfg.train.batch_size),
            lr=float(cfg.train.lr),
            patience=int(cfg.train.patience),
            seed=int(cfg.seed),
            recurrent_cell=cell,
            shred_only=bool(cfg.comparison.shred_only),
            verbose=verbose,
        )
        summary = _result_summary_payload(
            config=config,
            result=result,
            output_dir=cell_dir,
            status="completed",
        )
        write_result_summary(summary_path, summary)
        records.append({
            "cell": cell,
            "status": "completed",
            "config_id": config_id,
            "final_metrics": summary.get("final_metrics", {}),
            "output_dir": str(cell_dir),
        })

    benchmark_summary = {
        "experiment": "recurrent_cell_benchmark",
        "requested_cells": cells,
        "records": records,
        "output_dir": str(output_dir),
    }
    write_result_summary(output_dir / "benchmark_summary.json", benchmark_summary)
    print(json.dumps({
        "status": "completed",
        "experiment": "recurrent_cell_benchmark",
        "requested_cells": cells,
        "records": records,
        "output_dir": str(output_dir),
    }, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
