"""Resumable experiment runner: train each (method, train corruption) run per seed and
evaluate it on every evaluation condition.

Explicit run set: every entry of ``cfg.runs`` x every seed in ``cfg.eval.seeds``.
Each run is cached under ``<outputs_root>/<name>/<data.name>/seed{s}/{method}__{train}/``
as ``result.json`` (compact summary) + ``arrays.npz`` (per-sample details) +
``model.pt``; completed runs with a matching config are skipped.

Protocol (identical for every method within a seed; see docs/protocol.md):
- Sensors are probes at block-QR-selected grid points; inputs use the layout
  [ch0_1..ch0_N, ch1_1..ch1_N]. A faulty probe is faulty in all channels for
  the whole lag window.
- Noise: sigma = noise_frac x per-channel training std, drawn in unscaled units
  and divided by the MinMax range before entering the networks (equivalent to
  adding it before the scaler). Fresh per batch in training; a fixed seeded
  realisation per test draw, shared by all methods and by gappy POD.
- Faults are applied in the networks' (scaled) input space, where 0 = per-sensor
  training minimum, not "no signal".
- Models with a mask input receive the fault mask (oracle) or an all-zero mask
  (silent); gappy POD solves with the non-faulty rows (oracle) or all rows (silent).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

from .corruption import (
    CorruptionSpec,
    apply_fault,
    expand_mask,
    make_augmenter,
    noise_std_from_training,
    sample_dropout_mask,
    sample_exact_k_mask,
    split_masked_input,
)
from .data import (
    build_sensor_windows,
    gappy_pod_reconstruct,
    load_dataset,
    nektar_point_coordinates,
    qr_place_probes,
    split_window_indices,
)
from .model import SDN, SHRED, TimeSeriesDataset, fit
from .robust_shred import RobustSHREDv1, RobustSHREDv2
from .senseiver import SenseiverSDN
from .wave_data import wave_point_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LATENT_KINDS = ("shred", "senseiver_sdn", "robust_v1", "robust_v2")


def load_config(overrides: list[str] | None = None, config_name: str = "config"):
    """Compose the Hydra config for notebook use, e.g. ``load_config(["+experiment=dropout_baseline"])``."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def output_root(cfg) -> Path:
    return _abs(cfg.outputs_root) / str(cfg.name) / str(cfg.data.name)


def run_name(run) -> str:
    return f"{run.method}__{run.train}"


def corruption_spec(cfg, name: str, **overrides) -> CorruptionSpec:
    return CorruptionSpec(**{**OmegaConf.to_container(cfg.corruptions[name]), **overrides})


def eval_conditions(cfg) -> dict[str, tuple[CorruptionSpec, str]]:
    """{condition key: (spec, severity)} for every corruption x flags x severity in ``cfg.eval.conditions``.

    A condition lists either ``ps`` (each probe faulty with probability p; severity ``p<p>``) or
    ``ks`` (exactly k probes faulty; severity ``k<k>``; ``ks: all`` means 0..N-1).
    """
    out = {}
    for cond in cfg.eval.conditions:
        if "ks" in cond:
            ks = range(int(cfg.sensors.num)) if cond.ks == "all" else cond.ks
            severities = [(f"k{int(k)}", 0.0) for k in ks]
        else:
            severities = [(f"p{float(p):g}", float(p)) for p in cond.ps]
        for severity, p in severities:
            spec = corruption_spec(cfg, cond.corruption, flags=str(cond.flags), p=p)
            out[f"{spec.fault}_{spec.flags}_{severity}"] = (spec, severity)
    return out


def _sample_test_mask(severity: str, num_samples: int, num_sensors: int, rng: np.random.Generator) -> np.ndarray:
    if severity.startswith("k"):
        return sample_exact_k_mask(num_samples, num_sensors, int(severity[1:]), rng)
    return sample_dropout_mask(num_samples, num_sensors, float(severity[1:]), rng)


# ---------------------------------------------------------------- result files

def _git_commit_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip() or None
    except Exception:
        return None


def _stable_config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def write_result_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def load_result_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def require_matching_result_summary(summary_path: Path, expected_config_id: str) -> dict[str, Any] | None:
    summary = load_result_summary(summary_path)
    if summary is None or summary.get("status") != "completed":
        return None
    actual_id = summary.get("config_id")
    if actual_id is not None and actual_id != expected_config_id:
        raise ValueError(
            f"Found completed result at {summary_path} with config_id={actual_id}, "
            f"but requested config_id={expected_config_id}. Refusing to reuse mismatched output."
        )
    return summary


def _per_snapshot_rel_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - truth, axis=1)
    den = np.linalg.norm(truth, axis=1)
    return num / np.where(den == 0, 1.0, den)


def _aggregate_rel_error(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ---------------------------------------------------------------- data + models

class MaskedInputAdapter(torch.nn.Module):
    """Adapts ``[values, mask flags]`` inputs to a masked model's (values, mask) call."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_sensors: int,
        channels: int,
        sensor_coordinates: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_sensors = num_sensors
        self.channels = channels
        self.snapshot = sensor_coordinates is not None
        if self.snapshot:
            self.register_buffer("sensor_coordinates", sensor_coordinates)

    def _call(self, fn, x: torch.Tensor) -> torch.Tensor:
        values, mask = split_masked_input(x, self.channels * self.num_sensors)
        if self.snapshot:  # SenseiverSDN: last snapshot, one token per probe with C channels
            values = values[:, -1, :].reshape(x.shape[0], self.channels, self.num_sensors).transpose(1, 2)
            return fn(values, self.sensor_coordinates, mask)
        return fn(values, mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(self.model, x)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(self.model.latent, x)


@dataclass
class SeedData:
    seed: int
    load_X: np.ndarray            # (n, m) unscaled
    transformed_X: np.ndarray     # (n, m) MinMax scaled
    scaler: MinMaxScaler
    nx: int
    ny: int
    n_points: int
    channels: int
    locations: np.ndarray         # (N,) grid points of the probes
    state_indices: np.ndarray     # (C*N,) state index of each input column
    point_coordinates: np.ndarray # (n_points, d) physical coordinates
    sensor_coordinates: np.ndarray  # (N, d) jointly rescaled physical coords
    U_r: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    windows: np.ndarray           # (n - lags, lags, C*N) scaled, clean
    sigma: np.ndarray             # (C*N,) unscaled noise std
    sensor_range: np.ndarray      # (C*N,) MinMax range per input column
    valid_noise: np.ndarray       # (V, lags, C*N) unscaled
    valid_rng_state: dict         # validation RNG state after valid_noise; fault masks continue from it
    test_noise: np.ndarray        # (D, T, lags, C*N) unscaled
    test_mask: dict[str, np.ndarray]  # severity ('p0.3', 'k2') -> (D, T, N)
    truth: np.ndarray             # (T, m) unscaled test targets

    @property
    def lags(self) -> int:
        return self.windows.shape[1]

    @property
    def num_sensors(self) -> int:
        return len(self.locations)


def prepare_seed(cfg, seed: int) -> SeedData:
    """Load data, split, place probes, fit scaler, and draw all fixed corruptions."""
    data_path = _abs(cfg.data.file)
    source = str(cfg.data.source)
    load_X, nx, ny = load_dataset(source, data_path, trajectory_index=int(cfg.data.get("trajectory_index", 0)),
                                  nektar_field=str(cfg.data.get("field", "uv")))
    point_coordinates = nektar_point_coordinates(data_path) if source == "nektar" else wave_point_coordinates(data_path)
    n_points = point_coordinates.shape[0]
    channels = load_X.shape[1] // n_points
    lags = int(cfg.data.lags)
    num_sensors = int(cfg.sensors.num)

    train_idx, valid_idx, test_idx = split_window_indices(
        load_X.shape[0] - lags,
        train_percentage=float(cfg.data.split[0]),
        val_percentage=float(cfg.data.split[1]),
        test_percentage=float(cfg.data.split[2]),
        seed=seed,
    )
    locations, U_r = qr_place_probes(load_X[train_idx].T, num_sensors, int(cfg.sensors.rank), n_points)
    state_indices = np.concatenate([locations + c * n_points for c in range(channels)])
    scaler = MinMaxScaler().fit(load_X[train_idx])
    transformed_X = scaler.transform(load_X).astype(np.float32)
    windows = build_sensor_windows(transformed_X, state_indices, lags)

    lo = point_coordinates.min(axis=0)
    span = float((point_coordinates.max(axis=0) - lo).max())
    sensor_coordinates = ((point_coordinates[locations] - lo) / span).astype(np.float32)

    sigma = noise_std_from_training(load_X[train_idx][:, state_indices], float(cfg.noise_frac))
    sensor_range = (scaler.data_max_ - scaler.data_min_)[state_indices].astype(np.float32)

    rng_val = np.random.default_rng([seed, 1])
    valid_noise = (rng_val.standard_normal((len(valid_idx), lags, len(state_indices))) * sigma).astype(np.float32)

    draws = int(cfg.eval.draws)
    rng_test = np.random.default_rng([seed, 2])
    test_noise = (rng_test.standard_normal((draws, len(test_idx), lags, len(state_indices))) * sigma).astype(np.float32)
    test_mask = {}
    for _, severity in eval_conditions(cfg).values():
        if severity not in test_mask:  # one mask set per severity, shared by every fault and flags setting
            test_mask[severity] = _sample_test_mask(severity, draws * len(test_idx), num_sensors, rng_test).reshape(
                draws, len(test_idx), num_sensors
            )
    truth = load_X[test_idx + lags - 1]
    return SeedData(
        seed=seed, load_X=load_X, transformed_X=transformed_X, scaler=scaler, nx=nx, ny=ny,
        n_points=n_points, channels=channels, locations=locations, state_indices=state_indices,
        point_coordinates=point_coordinates, sensor_coordinates=sensor_coordinates, U_r=U_r,
        train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, windows=windows, sigma=sigma,
        sensor_range=sensor_range, valid_noise=valid_noise, valid_rng_state=rng_val.bit_generator.state,
        test_noise=test_noise, test_mask=test_mask, truth=truth,
    )


def _format_inputs(sd: SeedData, windows: np.ndarray, noise: np.ndarray | None, mask: np.ndarray | None,
                   spec: CorruptionSpec, method, rng: np.random.Generator | None = None) -> np.ndarray:
    """Corrupt scaled windows (B, lags, C*N) and present them in the method's input format."""
    x = windows
    if noise is not None and spec.noise:
        x = x + (noise / sd.sensor_range).astype(x.dtype, copy=False)
    x = apply_fault(x, mask, sd.channels, spec, rng)
    if method.mask_input:
        flags = np.zeros((x.shape[0], sd.num_sensors), dtype=np.float32)
        if mask is not None and spec.flags == "oracle":
            flags = mask.astype(np.float32)
        x = np.concatenate([x, np.repeat(flags[:, None, :], x.shape[1], axis=1)], axis=-1)
    elif method.kind == "sdn":
        x = x[:, -1, :]
    return x.astype(np.float32)


def _build_model(sd: SeedData, method, device: str) -> torch.nn.Module:
    n, c, m = sd.num_sensors, sd.channels, sd.load_X.shape[1]
    kind = method.kind
    coords = torch.tensor(sd.sensor_coordinates)
    if kind == "shred":
        inputs = c * n + (n if method.mask_input else 0)
        model = SHRED(inputs, m, hidden_size=int(method.hidden_size), hidden_layers=int(method.hidden_layers),
                      l1=int(method.l1), l2=int(method.l2), dropout=float(method.dropout),
                      recurrent_cell=str(method.recurrent_cell))
    elif kind == "sdn":
        model = SDN(c * n, m, l1=int(method.l1), l2=int(method.l2), dropout=float(method.dropout))
    elif kind == "senseiver_sdn":
        inner = SenseiverSDN(
            output_size=m, input_channels=c, spatial_dim=coords.shape[1], num_latents=int(method.num_latents),
            latent_dim=int(method.latent_dim), num_frequencies=int(method.num_frequencies),
            num_heads=int(method.num_heads), l1=int(method.l1), l2=int(method.l2), dropout=float(method.dropout),
            null_token=bool(method.null_token),
        )
        model = MaskedInputAdapter(inner, n, c, coords)
    elif kind == "robust_v1":
        inner = RobustSHREDv1(
            n, m, coords, hidden_size=int(method.hidden_size), num_heads=int(method.num_heads),
            embed_dim=int(method.embed_dim), num_frequencies=int(method.num_frequencies), l1=int(method.l1),
            l2=int(method.l2), dropout=float(method.dropout), input_channels=c, null_token=bool(method.null_token),
        )
        model = MaskedInputAdapter(inner, n, c)
    elif kind == "robust_v2":
        inner = RobustSHREDv2(
            n, m, coords, num_latents=int(method.num_latents), latent_dim=int(method.latent_dim),
            num_frequencies=int(method.num_frequencies), num_heads=int(method.num_heads),
            hidden_size=int(method.hidden_size), hidden_layers=int(method.hidden_layers), l1=int(method.l1),
            l2=int(method.l2), dropout=float(method.dropout), input_channels=c, null_token=bool(method.null_token),
        )
        model = MaskedInputAdapter(inner, n, c)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return model.to(device)


def run_config(cfg, run, seed: int) -> dict[str, Any]:
    """The configuration that identifies one cached run."""
    method = cfg.methods[run.method]
    config = {
        "experiment": str(cfg.name),
        "run": run_name(run),
        "method": OmegaConf.to_container(method),
        "seed": int(seed),
        "data": OmegaConf.to_container(cfg.data),
        "sensors": OmegaConf.to_container(cfg.sensors),
        "placement": "QR-probe-block",
        "noise_frac": float(cfg.noise_frac),
        "eval": {"draws": int(cfg.eval.draws),
                 "conditions": {k: asdict(spec) for k, (spec, _) in eval_conditions(cfg).items()}},
    }
    if method.kind != "pod":
        config["train"] = OmegaConf.to_container(cfg.train)
        config["train_corruption"] = asdict(corruption_spec(cfg, run.train))
    return json.loads(json.dumps(config, default=str))


# ---------------------------------------------------------------- train + evaluate

def _rel_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(b, axis=1)
    return np.linalg.norm(a - b, axis=1) / np.where(den == 0, 1.0, den)


def _predict(model: torch.nn.Module, x: np.ndarray, device: str, latent: bool, batch: int = 512):
    recon, lat = [], []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch):
            xb = torch.tensor(x[start:start + batch], device=device)
            recon.append(model(xb).cpu().numpy())
            if latent:
                lat.append(model.latent(xb).cpu().numpy())
    return np.concatenate(recon), (np.concatenate(lat) if latent else None)


def _evaluate(cfg, sd: SeedData, recon_fn, panel_windows: np.ndarray) -> tuple[dict, dict]:
    """recon_fn(noise, mask, spec, rng) -> (recon unscaled (B, m), latent or None) for the test windows."""
    draws, n_test = sd.test_noise.shape[:2]
    truth_rep = np.tile(sd.truth, (draws, 1))
    m_block = sd.n_points
    metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    recon_clean, latent_clean = recon_fn(None, None, CorruptionSpec(noise=False), None)
    metrics["clean"] = {"err": _aggregate_rel_error(recon_clean, sd.truth)}
    arrays["clean_err"] = _per_snapshot_rel_error(recon_clean, sd.truth)
    noise = sd.test_noise.reshape(draws * n_test, *sd.test_noise.shape[2:])
    for tag, (spec, severity) in eval_conditions(cfg).items():
        flat_mask = sd.test_mask[severity].reshape(draws * n_test, -1)
        # Same seeded garbage for every method within a condition.
        recon, latent = recon_fn(noise, flat_mask, spec, np.random.default_rng([sd.seed, 3]))
        k = flat_mask.sum(axis=1)
        err = _per_snapshot_rel_error(recon, truth_rep)
        entry = {
            "err": _aggregate_rel_error(recon, truth_rep),
            "err_u": _aggregate_rel_error(recon[:, :m_block], truth_rep[:, :m_block]),
            "err_v": _aggregate_rel_error(recon[:, m_block:], truth_rep[:, m_block:]) if sd.channels > 1 else None,
            "err_by_k": {int(kk): [float(err[k == kk].mean()), int((k == kk).sum())] for kk in np.unique(k)},
        }
        arrays[f"{tag}_err"] = err
        arrays[f"{tag}_err_u"] = _per_snapshot_rel_error(recon[:, :m_block], truth_rep[:, :m_block])
        if sd.channels > 1:
            arrays[f"{tag}_err_v"] = _per_snapshot_rel_error(recon[:, m_block:], truth_rep[:, m_block:])
        arrays[f"{tag}_k"] = k
        clean_rep = np.tile(recon_clean, (draws, 1))
        arrays[f"{tag}_d_recon"] = _rel_rows(recon, clean_rep)
        if latent is not None:
            arrays[f"{tag}_d_latent"] = _rel_rows(latent, np.tile(latent_clean, (draws, 1)))
            entry["d_latent_mean"] = float(arrays[f"{tag}_d_latent"].mean())
        entry["d_recon_mean"] = float(arrays[f"{tag}_d_recon"].mean())
        metrics[tag] = entry
        # Reconstructions of draw 0 for the panel windows.
        arrays[f"{tag}_panel_recon"] = recon[panel_windows].astype(np.float32)
    arrays["clean_panel_recon"] = recon_clean[panel_windows].astype(np.float32)
    if latent_clean is not None:
        arrays["latent_clean"] = latent_clean.astype(np.float32)
    return metrics, arrays


def _train_and_evaluate(cfg, sd: SeedData, run, device: str) -> tuple[dict, dict, Any]:
    method = cfg.methods[run.method]
    kind = method.kind
    lags = sd.lags
    panel_windows = np.arange(min(int(cfg.eval.panel_windows), len(sd.test_idx)))

    if kind == "pod":
        lo, hi = sd.scaler.data_min_, sd.scaler.data_max_
        y_clean = sd.load_X[sd.test_idx + lags - 1][:, sd.state_indices]
        draws = sd.test_noise.shape[0]
        test_windows = np.tile(sd.windows[sd.test_idx], (draws, 1, 1))

        def recon_fn(noise, mask, spec, rng):
            if noise is None:
                y, missing = y_clean, None
            else:
                y = np.tile(y_clean, (draws, 1)) + noise[:, -1, :]
                full = expand_mask(mask, sd.channels)
                if spec.fault != "none":  # faulty rows read the scaled fault values, mapped back to unscaled units
                    x = _format_inputs(sd, test_windows, noise, mask, spec, method, rng)[:, -1, :]
                    y = np.where(full, x * sd.sensor_range + lo[sd.state_indices], y)
                missing = full if spec.flags == "oracle" else None
            recon = gappy_pod_reconstruct(y, missing, sd.state_indices, sd.U_r)
            return (np.clip(recon, lo, hi) if bool(method.clip) else recon), None

        metrics, arrays = _evaluate(cfg, sd, recon_fn, panel_windows)
        return metrics, arrays, None

    spec = corruption_spec(cfg, run.train)
    torch.manual_seed(sd.seed)
    np.random.seed(sd.seed)
    model = _build_model(sd, method, device)

    def targets(idx: np.ndarray) -> torch.Tensor:
        return torch.tensor(sd.transformed_X[idx + lags - 1], device=device)

    train_x = sd.windows[sd.train_idx][:, -1, :] if kind == "sdn" else sd.windows[sd.train_idx]
    rng_val = np.random.default_rng()
    rng_val.bit_generator.state = sd.valid_rng_state
    valid_mask = sample_dropout_mask(len(sd.valid_idx), sd.num_sensors, spec.p, rng_val) if spec.p > 0 else None
    valid_x = _format_inputs(sd, sd.windows[sd.valid_idx], sd.valid_noise, valid_mask, spec, method, rng_val)
    augment = make_augmenter(spec, sd.sigma / sd.sensor_range, sd.num_sensors, channels=sd.channels,
                             mask_input=bool(method.mask_input))
    history = fit(
        model,
        TimeSeriesDataset(torch.tensor(train_x, device=device), targets(sd.train_idx)),
        TimeSeriesDataset(torch.tensor(valid_x, device=device), targets(sd.valid_idx)),
        batch_size=int(cfg.train.batch_size), num_epochs=int(cfg.train.epochs), lr=float(cfg.train.lr),
        verbose=False, patience=int(cfg.train.patience), augment_fn=augment,
    )
    model.eval()
    test_windows = sd.windows[sd.test_idx]
    draws = sd.test_noise.shape[0]
    with_latent = kind in LATENT_KINDS

    def recon_fn(noise, mask, spec, rng):
        base = test_windows if noise is None else np.tile(test_windows, (draws, 1, 1))
        x = _format_inputs(sd, base, noise, mask, spec, method, rng)
        pred, latent = _predict(model, x, device, with_latent)
        return sd.scaler.inverse_transform(pred), latent

    metrics, arrays = _evaluate(cfg, sd, recon_fn, panel_windows)
    arrays["val_history"] = np.asarray(history)
    metrics["epochs_used"] = int(len(history))  # validation checks (every 20 epochs)
    metrics["parameters"] = _parameter_count(model)
    if with_latent:
        metrics["latent_dim"] = int(arrays["latent_clean"].shape[1])
    return metrics, arrays, model


def save_seed_info(sd: SeedData, out_dir: Path, panel_windows: int) -> None:
    path = out_dir / "seed_info.npz"
    if path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = np.arange(min(panel_windows, len(sd.test_idx)))
    np.savez_compressed(
        path,
        locations=sd.locations, state_indices=sd.state_indices, sensor_coordinates=sd.sensor_coordinates,
        point_coordinates=sd.point_coordinates, nx=sd.nx, ny=sd.ny, n_points=sd.n_points, channels=sd.channels,
        sigma=sd.sigma, train_idx=sd.train_idx, valid_idx=sd.valid_idx, test_idx=sd.test_idx,
        test_noise=sd.test_noise, **{f"test_mask_{severity}": mask for severity, mask in sd.test_mask.items()},
        panel_truth=sd.truth[panel].astype(np.float32), panel_windows=panel,
    )


def run_all(cfg) -> list[dict[str, Any]]:
    """Run (or skip if cached) every run x seed; one compact JSON line per run."""
    root = output_root(cfg)
    device = _device()
    runs = list(cfg.runs)
    if not runs:
        raise ValueError("cfg.runs is empty; select an experiment with +experiment=<name>")
    for run in runs:
        if run.method not in cfg.methods or run.train not in cfg.corruptions:
            raise ValueError(f"unknown method or train corruption in run {dict(run)}")
    seeds = [int(s) for s in cfg.eval.seeds]
    records = []
    for seed in seeds:
        sd = None
        for run in runs:
            name = run_name(run)
            config = run_config(cfg, run, seed)
            config_id = _stable_config_fingerprint(config)
            out_dir = root / f"seed{seed}" / name
            if require_matching_result_summary(out_dir / "result.json", config_id) is not None:
                record = {"status": "skipped", "run": name, "seed": seed}
                records.append(record)
                print(json.dumps(record))
                continue
            if sd is None:
                sd = prepare_seed(cfg, seed)
                save_seed_info(sd, root / f"seed{seed}", int(cfg.eval.panel_windows))
            started = time.time()
            payload: dict[str, Any] = {
                "experiment": str(cfg.name), "run": name, "seed": seed, "config": config,
                "config_id": config_id, "device": device, "git_commit": _git_commit_short(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                metrics, arrays, model = _train_and_evaluate(cfg, sd, run, device)
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_dir / "arrays.npz", **arrays)
                if model is not None:
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
                payload.update(status="completed", metrics=metrics, wall_time_s=round(time.time() - started, 2))
            except Exception as exc:  # recorded so the sweep continues; failed runs are retried next time
                payload.update(status="failed", error_summary=f"{type(exc).__name__}: {exc}",
                               wall_time_s=round(time.time() - started, 2))
                if bool(cfg.verbose):
                    raise
            write_result_summary(out_dir / "result.json", payload)
            record = {"status": payload["status"], "run": name, "seed": seed, "wall_time_s": payload["wall_time_s"]}
            if payload["status"] == "completed":
                record.update({k: round(v["err"], 5) for k, v in payload["metrics"].items()
                               if isinstance(v, dict) and "err" in v})
            else:
                record["error"] = payload["error_summary"]
            records.append(record)
            print(json.dumps(record))
    write_result_summary(root / "summary.json", build_summary(root, [run_name(r) for r in runs], seeds))
    return records


# ---------------------------------------------------------------- loading results

def load_results(root: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """{(run, seed): result.json payload} for every completed run under root."""
    out = {}
    for path in sorted(_abs(root).glob("seed*/*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            out[(payload["run"], int(payload["seed"]))] = payload
    return out


def load_arrays(root: str | Path, run: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(_abs(root) / f"seed{seed}" / run / "arrays.npz") as data:
        return {k: data[k] for k in data.files}


def load_seed_info(root: str | Path, seed: int) -> dict[str, np.ndarray]:
    with np.load(_abs(root) / f"seed{seed}" / "seed_info.npz") as data:
        return {k: data[k] for k in data.files}


def build_summary(root: str | Path, runs: list[str], seeds: list[int]) -> dict[str, Any]:
    """Mean/std over seeds per run and condition, plus missing/failed runs."""
    results = load_results(root)
    summary: dict[str, Any] = {"runs": {}, "missing": [], "seeds": seeds}
    for name in runs:
        done = [results[(name, s)] for s in seeds if (name, s) in results]
        summary["missing"] += [f"{name}/seed{s}" for s in seeds if (name, s) not in results]
        if not done:
            continue
        entry: dict[str, Any] = {"n_seeds": len(done)}
        for cond in [k for k, v in done[0]["metrics"].items() if isinstance(v, dict)]:
            for key in ("err", "err_u", "err_v", "d_latent_mean", "d_recon_mean"):
                vals = [r["metrics"][cond].get(key) for r in done]
                if all(v is not None for v in vals):
                    entry[f"{cond}_{key}"] = [float(np.mean(vals)), float(np.std(vals))]
        if "parameters" in done[0]["metrics"]:
            entry["parameters"] = int(done[0]["metrics"]["parameters"])
        entry["wall_time_s"] = float(np.mean([r["wall_time_s"] for r in done]))
        summary["runs"][name] = entry
    return summary


def masked_invariance_check(cfg, run, seed: int) -> float:
    """Max |output change| of a trained masked model when faulty sensors' values are set to 1e3.

    Uses draw 0 of the test mask with the most faults, with oracle flags. Should be ~0 (float rounding) for masked methods.
    """
    method = cfg.methods[run.method]
    if not method.mask_input or method.kind == "shred":
        raise ValueError(f"{run_name(run)} is not a masked network")
    sd = prepare_seed(cfg, seed)
    model = _build_model(sd, method, "cpu")
    path = output_root(cfg) / f"seed{seed}" / run_name(run) / "model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    mask = max(sd.test_mask.values(), key=lambda m: m.sum())[0]
    spec = CorruptionSpec(fault="missing")
    x = _format_inputs(sd, sd.windows[sd.test_idx], sd.test_noise[0], mask, spec, method)
    perturbed = x.copy()
    failed = expand_mask(mask, sd.channels)[:, None, :].repeat(sd.lags, axis=1)
    perturbed[..., : sd.channels * sd.num_sensors][failed] = 1e3
    with torch.no_grad():
        diff = model(torch.tensor(perturbed)) - model(torch.tensor(x))
    return float(diff.abs().max())
