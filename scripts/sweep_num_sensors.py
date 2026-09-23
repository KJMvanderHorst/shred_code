"""Sweep `num_sensors` × {QR, random} placement and plot test relative L2 error.

Mirrors Fig 2B / 3B / 4B of Williams, Zahn, Kutz (2024):
- SHRED, SHRED-QR, SDN, SDN-QR, QR/POD (linear; only QR placement is well-conditioned).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.experiment import (
    _result_summary_payload,
    _stable_config_fingerprint,
    require_matching_result_summary,
    run_experiment,
    write_result_summary,
)
from get_shredded.plotting import plot_sweep_error_vs_sensors


@hydra.main(version_base=None, config_path="../configs", config_name="sweep_num_sensors")
def main(cfg: DictConfig) -> None:
    execution_mode = str(cfg.execution.mode).lower()
    verbose = execution_mode == "verbose"
    sensor_counts = list(cfg.sweep.sensor_counts)
    seeds = list(cfg.sweep.seeds)
    placements = list(cfg.sweep.placements)
    out_dir = Path(to_absolute_path(cfg.outputs.root)) / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw[(method, placement)] -> list (over sensor_counts) of lists (over seeds)
    raw: dict[tuple[str, str], list[list[float]]] = {
        (method, p): [] for method in ("shred", "sdn", "qrpod") for p in placements
    }
    completed_runs = 0
    skipped_runs = 0
    failed_runs = 0
    expected_runs = len(sensor_counts) * len(placements) * len(seeds)
    run_records: list[dict[str, object]] = []

    outer = tqdm(sensor_counts, desc="sensor sweep", disable=not verbose)
    for n_s in outer:
        per_count: dict[tuple[str, str], list[float]] = {k: [] for k in raw}
        for placement in placements:
            inner = tqdm(seeds, desc=f"  n_sensors={n_s} {placement}", leave=False, disable=not verbose)
            for seed in inner:
                config = {
                    "experiment": "sensor_sweep",
                    "num_sensors": int(n_s),
                    "placement": str(placement),
                    "seed": int(seed),
                    "lags": int(cfg.model.lags),
                    "hidden_size": int(cfg.model.hidden_size),
                    "hidden_layers": int(cfg.model.hidden_layers),
                    "train_epochs": int(cfg.train.epochs),
                    "batch_size": int(cfg.train.batch_size),
                    "lr": float(cfg.train.lr),
                    "patience": int(cfg.train.patience),
                    "noise_enabled": bool(cfg.noise.enabled),
                    "noise_modes": [str(mode) for mode in cfg.noise.modes],
                }
                config_id = _stable_config_fingerprint(config)
                run_dir = out_dir / f"n_{n_s}_{placement}_seed_{seed}"
                result_path = run_dir / "result.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                existing = require_matching_result_summary(result_path, config_id)
                if existing is not None:
                    skipped_runs += 1
                    run_records.append({"status": "skipped", "config_id": config_id, "num_sensors": int(n_s), "placement": placement, "seed": int(seed), "final_metrics": existing.get("final_metrics", {})})
                    if not verbose:
                        print(json.dumps({"status": "skipped", "config_id": config_id, "num_sensors": int(n_s), "placement": placement, "seed": int(seed)}, separators=(",", ":"), sort_keys=True))
                    continue

                try:
                    r = run_experiment(
                        mat_path=Path(to_absolute_path(cfg.data.mat)),
                        num_sensors=int(n_s),
                        lags=int(cfg.model.lags),
                        placement=str(placement),
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
                        seed=int(seed),
                        noise_enabled=bool(cfg.noise.enabled),
                        noise_modes=[str(mode) for mode in cfg.noise.modes],
                        noise_white_std=float(cfg.noise.white_std),
                        noise_none_fill_value=float(cfg.noise.none_fill_value),
                        noise_auto_extend=bool(cfg.noise.auto_extend),
                        noise_default_mode=str(cfg.noise.default_mode),
                        noise_seed=(int(cfg.noise.seed) + int(seed)) if cfg.noise.seed is not None else None,
                        recurrent_cell=str(cfg.model.recurrent_cell).lower(),
                        verbose=verbose,
                    )
                    summary = _result_summary_payload(
                        config=config,
                        result=r,
                        output_dir=run_dir,
                        status="completed",
                    )
                    write_result_summary(result_path, summary)
                    completed_runs += 1
                    per_count[("shred", placement)].append(r.shred_err)
                    per_count[("sdn", placement)].append(r.sdn_err)
                    per_count[("qrpod", placement)].append(r.qrpod_err)
                    run_records.append({"status": "completed", "config_id": config_id, "num_sensors": int(n_s), "placement": placement, "seed": int(seed), "final_metrics": summary.get("final_metrics", {})})
                    if not verbose:
                        print(json.dumps({"status": "completed", "config_id": config_id, "num_sensors": int(n_s), "placement": placement, "seed": int(seed), "final_metrics": summary.get("final_metrics", {})}, separators=(",", ":"), sort_keys=True))
                    if verbose:
                        inner.set_postfix(shred=f"{r.shred_err:.3f}", sdn=f"{r.sdn_err:.3f}", qrpod=f"{r.qrpod_err:.3f}")
                except Exception as exc:  # pragma: no cover - large scientific run failure path
                    failed_runs += 1
                    error_summary = str(exc)
                    failed_summary = _result_summary_payload(
                        config={**config, "config_id": config_id},
                        result=None,
                        output_dir=run_dir,
                        status="failed",
                        error_summary=error_summary,
                    )
                    write_result_summary(result_path, failed_summary)
                    run_records.append({"status": "failed", "config_id": config_id, "num_sensors": int(n_s), "placement": placement, "seed": int(seed), "error_summary": error_summary})
                    if not verbose:
                        print(json.dumps({"status": "failed", "config_id": config_id, "num_sensors": int(n_s), "placement": placement, "seed": int(seed), "error_summary": error_summary}, separators=(",", ":"), sort_keys=True))
                    raise

        for key, vals in per_count.items():
            raw[key].append(vals)

        if verbose:
            outer.set_postfix(
                shred_QR=f"{float(np.median(per_count[('shred', 'QR')])):.3f}"
                    if ('shred', 'QR') in per_count and per_count[('shred', 'QR')] else "—",
                shred_R=f"{float(np.median(per_count[('shred', 'random')])):.3f}"
                    if ('shred', 'random') in per_count and per_count[('shred', 'random')] else "—",
            )

    # Build plot series. QR/POD only shown for QR placement (random is ill-conditioned).
    series: dict[str, list[float]] = {}
    for placement in placements:
        suffix = f" ({placement})"
        if ("shred", placement) in raw:
            series[f"SHRED{suffix}"] = [float(np.median(v)) for v in raw[("shred", placement)]]
        if ("sdn", placement) in raw:
            series[f"SDN{suffix}"] = [float(np.median(v)) for v in raw[("sdn", placement)]]
    if "QR" in placements:
        series["QR/POD"] = [float(np.median(v)) for v in raw[("qrpod", "QR")]]

    plot_sweep_error_vs_sensors(
        sensor_counts=sensor_counts,
        series=series,
        save_path=out_dir / "error_vs_num_sensors.png",
    )

    np.savez(
        out_dir / "sweep_results.npz",
        sensor_counts=np.asarray(sensor_counts),
        seeds=np.asarray(seeds),
        placements=np.asarray(placements),
        **{f"{m}_{p}": np.asarray(raw[(m, p)]) for (m, p) in raw},
    )

    sweep_summary = {
        "experiment": "sensor_sweep",
        "requested": {"sensor_counts": sensor_counts, "placements": placements, "seeds": seeds},
        "expected_runs": expected_runs,
        "completed": completed_runs,
        "skipped": skipped_runs,
        "failed": failed_runs,
        "output_dir": str(out_dir),
        "final_metrics": {
            "shred_QR": float(np.median(raw[("shred", "QR")])) if ("shred", "QR") in raw and raw[("shred", "QR")] else None,
            "sdn_QR": float(np.median(raw[("sdn", "QR")])) if ("sdn", "QR") in raw and raw[("sdn", "QR")] else None,
            "qrpod_QR": float(np.median(raw[("qrpod", "QR")])) if ("qrpod", "QR") in raw and raw[("qrpod", "QR")] else None,
        },
        "runs": run_records,
    }
    write_result_summary(out_dir / "sweep_summary.json", sweep_summary)
    if not verbose:
        print(json.dumps({
            "status": "completed",
            "experiment": "sensor_sweep",
            "expected_runs": expected_runs,
            "completed": completed_runs,
            "skipped": skipped_runs,
            "failed": failed_runs,
            "output_dir": str(out_dir),
            "final_metrics": sweep_summary["final_metrics"],
        }, separators=(",", ":"), sort_keys=True))
    else:
        print(f"\nSaved sweep results to {out_dir}")


if __name__ == "__main__":
    main()
