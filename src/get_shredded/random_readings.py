"""Evaluation-only robustness to faulty probes that emit random readings.

Reuses the trained models cached by :mod:`meeting_runner` (no retraining). The
faulty-probe sets are the cached ``test_mask[p]`` draws, so results line up with
the dropout ``p{p}`` condition. Measurement noise is added as in the main run;
every channel of a faulty probe is then replaced, at every lag, by fixed seeded
garbage ~ N(mean, std) in the scaled [0, 1] input space.

Conditions:
- ``garbage_silent``: the fault is undetected (mask flags all zero; plain models
  simply read the garbage; gappy POD uses the garbage rows).
- ``garbage_oracle``: the fault is flagged (masked/flag models and gappy POD
  only); masked models then ignore the values, so this equals dropout up to
  float rounding.

Explicit run set: ``cfg.meeting.methods`` x ``cfg.meeting.seeds``; results are
cached in ``<outputs.root>/random_readings/seed{s}/{method}.json`` (+ ``.npz``).

:func:`train_random_readings` additionally trains ``cfg.random_readings.train_methods``
on the same silent fault model (fault rate ``p``, fresh garbage per batch, no
flags; validation corrupted with a fixed realisation at the same rate) and
evaluates them on clean inputs and the same garbage test set. Cached in
``<outputs.root>/seed{s}/{method}/`` (result.json, arrays.npz, model.pt).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .corruption import expand_mask, make_garbage_augmenter, sample_dropout_mask
from .data import gappy_pod_reconstruct
from .experiment import (
    _aggregate_rel_error,
    _git_commit_short,
    _parameter_count,
    _per_snapshot_rel_error,
    _stable_config_fingerprint,
    require_matching_result_summary,
    write_result_summary,
)
from .model import TimeSeriesDataset, fit
from .meeting_runner import (
    METHODS,
    SeedData,
    _abs,
    _build_model,
    _device,
    _predict,
    prepare_seed,
    run_config,
)

CONDITIONS = ["garbage_silent", "garbage_oracle"]

# Trained on silent random-reading faults; masked models always receive all-zero flags.
TRAINED_METHODS: dict[str, dict[str, Any]] = {
    "SHRED-garbage": {"kind": "shred", "input": "zero_fill"},
    "RobustSHREDv1-garbage": {"kind": "robust_v1", "input": "masked"},
    "RobustSHREDv2-garbage": {"kind": "robust_v2", "input": "masked"},
}


def garbage_windows(windows: np.ndarray, mask: np.ndarray, channels: int, mean: float, std: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Replace every channel of the failed probes (mask True) at every lag by N(mean, std) values.

    windows: (B, lags, C*N) scaled; mask: (B, N) bool. Returns a new array.
    """
    garbage = (mean + std * rng.standard_normal(windows.shape)).astype(windows.dtype)
    full = expand_mask(mask, channels)[:, None, :]
    return np.where(full, garbage, windows)


def rr_config(cfg, method: str, seed: int) -> dict[str, Any]:
    rr = cfg.random_readings
    return {
        "experiment": "meeting_random_readings",
        "model_run": run_config(cfg, method, seed),
        "p": float(rr.p),
        "mean": float(rr.mean),
        "std": float(rr.std),
    }


def _inputs(sd: SeedData, noise: np.ndarray, mask: np.ndarray, rng_seed: list[int], rr) -> np.ndarray:
    """Noisy test windows (repeated over draws) with faulty probes replaced by garbage."""
    draws = sd.test_noise.shape[0]
    base = np.tile(sd.windows[sd.test_idx], (draws, 1, 1)) + noise / sd.sensor_range
    return garbage_windows(base, mask, sd.channels, float(rr.mean), float(rr.std), np.random.default_rng(rng_seed))


def evaluate_random_readings(cfg, sd: SeedData, method: str,
                             model: torch.nn.Module | None = None) -> tuple[dict, dict]:
    """Garbage-fault metrics; loads the cached meeting model unless ``model`` is given."""
    spec = METHODS[method] if method in METHODS else TRAINED_METHODS[method]
    kind, input_kind = spec["kind"], spec["input"]
    rr = cfg.random_readings
    p = float(rr.p)
    draws, n_test = sd.test_noise.shape[:2]
    noise = sd.test_noise.reshape(draws * n_test, *sd.test_noise.shape[2:])
    mask = sd.test_mask[p].reshape(draws * n_test, -1)
    x = _inputs(sd, noise, mask, [sd.seed, 3], rr)  # same garbage for every method
    truth = np.tile(sd.truth, (draws, 1))
    root = _abs(cfg.outputs.root)

    recons: dict[str, np.ndarray] = {}
    if kind == "pod":
        lo, hi = sd.scaler.data_min_, sd.scaler.data_max_
        y = x[:, -1, :] * sd.sensor_range + lo[sd.state_indices]  # back to unscaled units
        for cond, missing in (("garbage_silent", None), ("garbage_oracle", expand_mask(mask, sd.channels))):
            recon = gappy_pod_reconstruct(y, missing, sd.state_indices, sd.U_r)
            recons[cond] = np.clip(recon, lo, hi) if bool(cfg.meeting.clip_pod) else recon
    else:
        if model is None:
            model = _build_model(cfg, sd, kind, "cpu")
            state = torch.load(root / f"seed{sd.seed}" / method / "model.pt", map_location="cpu")
            model.load_state_dict(state)
        model.eval()
        if input_kind in ("masked", "flags"):
            for cond, flags in (("garbage_silent", np.zeros_like(mask)), ("garbage_oracle", mask)):
                f = np.repeat(flags.astype(np.float32)[:, None, :], x.shape[1], axis=1)
                pred, _ = _predict(model, np.concatenate([x, f], axis=-1).astype(np.float32), "cpu", False)
                recons[cond] = sd.scaler.inverse_transform(pred)
        else:
            xin = x[:, -1, :] if kind == "sdn" else x
            pred, _ = _predict(model, xin.astype(np.float32), "cpu", False)
            recons["garbage_silent"] = sd.scaler.inverse_transform(pred)

    m_block = sd.n_points
    k = mask.sum(axis=1)
    metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"k": k}
    panel = np.arange(min(int(cfg.meeting.panel_windows), n_test))
    for cond, recon in recons.items():
        err = _per_snapshot_rel_error(recon, truth)
        metrics[cond] = {
            "err": _aggregate_rel_error(recon, truth),
            "err_u": _aggregate_rel_error(recon[:, :m_block], truth[:, :m_block]),
            "err_v": _aggregate_rel_error(recon[:, m_block:], truth[:, m_block:]) if sd.channels > 1 else None,
            "err_by_k": {int(kk): [float(err[k == kk].mean()), int((k == kk).sum())] for kk in np.unique(k)},
        }
        arrays[f"{cond}_err"] = err
        arrays[f"{cond}_panel_recon"] = recon[panel].astype(np.float32)
    return metrics, arrays


def run_random_readings(cfg) -> list[dict[str, Any]]:
    """Evaluate (or skip if cached) every method x seed; one compact JSON line per run."""
    root = _abs(cfg.outputs.root)
    out_root = root / "random_readings"
    methods = [str(m) for m in cfg.meeting.methods]
    seeds = [int(s) for s in cfg.meeting.seeds]
    records = []
    for seed in seeds:
        sd = None
        for method in methods:
            config = rr_config(cfg, method, seed)
            config_id = _stable_config_fingerprint(config)
            path = out_root / f"seed{seed}" / f"{method}.json"
            if require_matching_result_summary(path, config_id) is not None:
                records.append({"status": "skipped", "method": method, "seed": seed})
                print(json.dumps(records[-1]))
                continue
            model_result = root / f"seed{seed}" / method / "result.json"
            if require_matching_result_summary(model_result, _stable_config_fingerprint(config["model_run"])) is None:
                raise FileNotFoundError(f"no completed meeting run matching the config at {model_result}")
            if sd is None:
                sd = prepare_seed(cfg, seed)
            started = time.time()
            metrics, arrays = evaluate_random_readings(cfg, sd, method)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            write_result_summary(path, {
                "experiment": config["experiment"], "method": method, "seed": seed, "config": config,
                "config_id": config_id, "status": "completed", "metrics": metrics,
                "wall_time_s": round(time.time() - started, 2),
            })
            records.append({"status": "completed", "method": method, "seed": seed,
                            **{c: round(v["err"], 5) for c, v in metrics.items()}})
            print(json.dumps(records[-1]))
    write_result_summary(out_root / "summary.json", build_rr_summary(root, methods, seeds))
    return records


def load_rr_results(root: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for path in sorted((_abs(root) / "random_readings").glob("seed*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            out[(payload["method"], int(payload["seed"]))] = payload
    return out


def load_rr_arrays(root: str | Path, method: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(_abs(root) / "random_readings" / f"seed{seed}" / f"{method}.npz") as data:
        return {k: data[k] for k in data.files}


def build_rr_summary(root: str | Path, methods: list[str], seeds: list[int]) -> dict[str, Any]:
    results = load_rr_results(root)
    summary: dict[str, Any] = {"methods": {}, "missing": [], "seeds": seeds}
    for method in methods:
        runs = [results[(method, s)] for s in seeds if (method, s) in results]
        summary["missing"] += [f"{method}/seed{s}" for s in seeds if (method, s) not in results]
        entry: dict[str, Any] = {"n_seeds": len(runs)}
        for cond in CONDITIONS:
            for key in ("err", "err_u", "err_v"):
                vals = [r["metrics"][cond][key] for r in runs if cond in r["metrics"]]
                if vals and all(v is not None for v in vals):
                    entry[f"{cond}_{key}"] = [float(np.mean(vals)), float(np.std(vals))]
        summary["methods"][method] = entry
    return summary


def train_config(cfg, method: str, seed: int) -> dict[str, Any]:
    rr = cfg.random_readings
    spec = TRAINED_METHODS[method]
    config = run_config(cfg, "RobustSHREDv1", seed)  # shared data/model/train fields
    config.update({
        "experiment": "meeting_random_readings_trained", "method": method, "kind": spec["kind"],
        "input": spec["input"], "train_p": float(rr.p), "fault": "garbage_silent",
        "garbage_mean": float(rr.mean), "garbage_std": float(rr.std),
    })
    config.pop("clip_pod", None)
    return config


def _train_one(cfg, sd: SeedData, method: str, device: str) -> tuple[dict, dict, torch.nn.Module]:
    rr = cfg.random_readings
    spec = TRAINED_METHODS[method]
    masked = spec["input"] == "masked"
    p = float(rr.p)
    lags = sd.lags

    def with_zero_flags(x: np.ndarray) -> np.ndarray:
        if not masked:
            return x.astype(np.float32)
        return np.concatenate([x, np.zeros((*x.shape[:-1], sd.num_sensors), dtype=x.dtype)], axis=-1).astype(np.float32)

    def targets(idx: np.ndarray) -> torch.Tensor:
        return torch.tensor(sd.transformed_X[idx + lags - 1], device=device)

    rng_val = np.random.default_rng([sd.seed, 4])
    valid_mask = sample_dropout_mask(len(sd.valid_idx), sd.num_sensors, p, rng_val)
    valid_x = garbage_windows(sd.windows[sd.valid_idx] + sd.valid_noise / sd.sensor_range, valid_mask,
                              sd.channels, float(rr.mean), float(rr.std), rng_val)

    torch.manual_seed(sd.seed)
    np.random.seed(sd.seed)
    model = _build_model(cfg, sd, spec["kind"], device)
    augment = make_garbage_augmenter(
        sd.sigma / sd.sensor_range, p, sd.num_sensors, channels=sd.channels,
        mean=float(rr.mean), std=float(rr.std), append_zero_flags=masked,
    )
    history = fit(
        model,
        TimeSeriesDataset(torch.tensor(sd.windows[sd.train_idx], device=device), targets(sd.train_idx)),
        TimeSeriesDataset(torch.tensor(with_zero_flags(valid_x), device=device), targets(sd.valid_idx)),
        batch_size=int(cfg.train.batch_size), num_epochs=int(cfg.train.epochs), lr=float(cfg.train.lr),
        verbose=False, patience=int(cfg.train.patience), augment_fn=augment,
    )
    model = model.cpu().eval()
    clean_pred, _ = _predict(model, with_zero_flags(sd.windows[sd.test_idx]), "cpu", False)
    clean = sd.scaler.inverse_transform(clean_pred)
    metrics, arrays = evaluate_random_readings(cfg, sd, method, model)
    metrics["clean"] = {"err": _aggregate_rel_error(clean, sd.truth)}
    arrays["clean_err"] = _per_snapshot_rel_error(clean, sd.truth)
    arrays["val_history"] = np.asarray(history)
    metrics["epochs_used"] = int(len(history))
    metrics["parameters"] = _parameter_count(model)
    return metrics, arrays, model


def train_random_readings(cfg) -> list[dict[str, Any]]:
    """Train + evaluate (or skip if cached) every train method x seed; one JSON line per run."""
    root = _abs(cfg.outputs.root)
    device = _device()
    methods = [str(m) for m in cfg.random_readings.train_methods]
    records = []
    for seed in [int(s) for s in cfg.meeting.seeds]:
        sd = None
        for method in methods:
            config = train_config(cfg, method, seed)
            config_id = _stable_config_fingerprint(config)
            out_dir = root / f"seed{seed}" / method
            if require_matching_result_summary(out_dir / "result.json", config_id) is not None:
                records.append({"status": "skipped", "method": method, "seed": seed})
                print(json.dumps(records[-1]))
                continue
            if sd is None:
                sd = prepare_seed(cfg, seed)
            started = time.time()
            metrics, arrays, model = _train_one(cfg, sd, method, device)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / "arrays.npz", **arrays)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
            payload = {
                "experiment": config["experiment"], "method": method, "seed": seed, "config": config,
                "config_id": config_id, "device": device, "git_commit": _git_commit_short(),
                "status": "completed", "metrics": metrics, "wall_time_s": round(time.time() - started, 2),
            }
            write_result_summary(out_dir / "result.json", payload)
            records.append({"status": "completed", "method": method, "seed": seed,
                            "wall_time_s": payload["wall_time_s"],
                            **{c: round(v["err"], 5) for c, v in metrics.items() if isinstance(v, dict)}})
            print(json.dumps(records[-1]))
    return records
