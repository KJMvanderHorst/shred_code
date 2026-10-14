"""Noise + dropout robustness and latent-invariance runs for the meeting notebooks.

Explicit run set: every method in ``cfg.meeting.methods`` x every seed in
``cfg.meeting.seeds``. Each run is cached under ``<outputs.root>/seed{s}/{method}/``
as ``result.json`` (compact summary) + ``arrays.npz`` (per-sample details) +
``model.pt``; completed runs with a matching config are skipped.

Protocol (identical for every method within a seed):
- Sensors are (u, v) probes at QR-selected grid points; inputs use the layout
  [ch0_1..ch0_N, ch1_1..ch1_N]. A failed probe loses all channels for the whole
  lag window.
- Noise: sigma = noise_frac x per-channel training std, drawn in unscaled units
  and divided by the MinMax range before entering the networks (equivalent to
  adding it before the scaler). Fresh per batch in training; a fixed seeded
  realisation per test draw, shared by all methods and by gappy POD.
- Zero-fill happens in the networks' (scaled) input space, where 0 = per-sensor
  training minimum, not "no signal".
- Masked methods additionally receive the mask and exclude failed sensors from
  attention; gappy POD solves with the available rows only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .corruption import (
    corrupt_numpy,
    expand_mask,
    make_corruption_augmenter,
    noise_std_from_training,
    sample_dropout_mask,
    split_masked_input,
)
from .data import (
    build_sensor_windows,
    gappy_pod_reconstruct,
    load_dataset,
    nektar_point_coordinates,
    qr_place_probes,
)
from .experiment import (
    _aggregate_rel_error,
    _git_commit_short,
    _parameter_count,
    _per_snapshot_rel_error,
    _stable_config_fingerprint,
    require_matching_result_summary,
    split_window_indices,
    write_result_summary,
)
from .model import SDN, SHRED, TimeSeriesDataset, fit
from .robust_shred import RobustSHREDv1, RobustSHREDv2
from .senseiver import SenseiverSDN

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# kind: architecture; train_p: training dropout rate; input: how dropout is presented.
METHODS: dict[str, dict[str, Any]] = {
    "SHRED-noise": {"kind": "shred", "train_p": 0.0, "input": "zero_fill"},
    "SHRED-dropout": {"kind": "shred", "train_p": "p", "input": "zero_fill"},
    "SDN-noise": {"kind": "sdn", "train_p": 0.0, "input": "zero_fill"},
    "SDN-dropout": {"kind": "sdn", "train_p": "p", "input": "zero_fill"},
    "SenseiverSDN": {"kind": "senseiver_sdn", "train_p": "p", "input": "masked"},
    "RobustSHREDv1": {"kind": "robust_v1", "train_p": "p", "input": "masked"},
    "RobustSHREDv2": {"kind": "robust_v2", "train_p": "p", "input": "masked"},
    # Tier B: SHRED fed [zero-filled readings, 0/1 missing flags].
    "SHRED-dropout+mask": {"kind": "shred_flags", "train_p": "p", "input": "flags"},
    "GappyPOD": {"kind": "pod", "train_p": None, "input": "masked"},
}
LATENT_METHODS = [name for name, spec in METHODS.items() if spec["kind"] not in ("sdn", "pod")]


def load_config(overrides: list[str] | None = None, config_name: str = "meeting"):
    """Compose the Hydra config for notebook use."""
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
    point_coordinates: np.ndarray # (n_points, 2) physical (x, y)
    sensor_coordinates: np.ndarray  # (N, 2) jointly rescaled physical coords
    U_r: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    windows: np.ndarray           # (n - lags, lags, C*N) scaled, clean
    sigma: np.ndarray             # (C*N,) unscaled noise std
    sensor_range: np.ndarray      # (C*N,) MinMax range per input column
    valid_noise: np.ndarray       # (V, lags, C*N) unscaled
    valid_mask: np.ndarray        # (V, N) at training dropout rate
    test_noise: np.ndarray        # (D, T, lags, C*N) unscaled
    test_mask: dict[float, np.ndarray]  # p -> (D, T, N)
    truth: np.ndarray             # (T, m) unscaled test targets

    @property
    def lags(self) -> int:
        return self.windows.shape[1]

    @property
    def num_sensors(self) -> int:
        return len(self.locations)


def prepare_seed(cfg, seed: int) -> SeedData:
    """Load data, split, place probes, fit scaler, and draw all fixed corruptions."""
    data_path = _abs(cfg.data.nektar_file)
    field = str(cfg.data.nektar_field).lower()
    load_X, nx, ny = load_dataset("nektar", data_path, nektar_field=field)
    point_coordinates = nektar_point_coordinates(data_path)
    n_points = point_coordinates.shape[0]
    channels = load_X.shape[1] // n_points
    lags = int(cfg.model.lags)
    num_sensors = int(cfg.meeting.num_sensors)

    train_idx, valid_idx, test_idx = split_window_indices(
        load_X.shape[0] - lags,
        train_percentage=float(cfg.data.train_percentage),
        val_percentage=float(cfg.data.val_percentage),
        test_percentage=float(cfg.data.test_percentage),
        seed=seed,
    )
    # Fitting on load_X[train_idx] mirrors run_experiment / run_robustness_comparison.
    locations, U_r = qr_place_probes(load_X[train_idx].T, num_sensors, int(cfg.meeting.qr_rank), n_points)
    state_indices = np.concatenate([locations + c * n_points for c in range(channels)])
    scaler = MinMaxScaler().fit(load_X[train_idx])
    transformed_X = scaler.transform(load_X).astype(np.float32)
    windows = build_sensor_windows(transformed_X, state_indices, lags)

    lo = point_coordinates.min(axis=0)
    span = float((point_coordinates.max(axis=0) - lo).max())
    sensor_coordinates = ((point_coordinates[locations] - lo) / span).astype(np.float32)

    sigma = noise_std_from_training(load_X[train_idx][:, state_indices], float(cfg.meeting.noise_frac))
    sensor_range = (scaler.data_max_ - scaler.data_min_)[state_indices].astype(np.float32)

    train_p = float(cfg.meeting.train_p)
    rng_val = np.random.default_rng([seed, 1])
    valid_noise = (rng_val.standard_normal((len(valid_idx), lags, len(state_indices))) * sigma).astype(np.float32)
    valid_mask = sample_dropout_mask(len(valid_idx), num_sensors, train_p, rng_val)

    draws = int(cfg.meeting.test_draws)
    rng_test = np.random.default_rng([seed, 2])
    test_noise = (rng_test.standard_normal((draws, len(test_idx), lags, len(state_indices))) * sigma).astype(np.float32)
    test_mask = {}
    for p in cfg.meeting.test_ps:
        p = float(p)
        test_mask[p] = sample_dropout_mask(draws * len(test_idx), num_sensors, p, rng_test).reshape(
            draws, len(test_idx), num_sensors
        )
    truth = load_X[test_idx + lags - 1]
    return SeedData(
        seed=seed, load_X=load_X, transformed_X=transformed_X, scaler=scaler, nx=nx, ny=ny,
        n_points=n_points, channels=channels, locations=locations, state_indices=state_indices,
        point_coordinates=point_coordinates, sensor_coordinates=sensor_coordinates, U_r=U_r,
        train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, windows=windows, sigma=sigma,
        sensor_range=sensor_range, valid_noise=valid_noise, valid_mask=valid_mask,
        test_noise=test_noise, test_mask=test_mask, truth=truth,
    )


def _format_inputs(sd: SeedData, windows: np.ndarray, noise: np.ndarray | None, mask: np.ndarray | None,
                   input_kind: str, snapshot: bool) -> np.ndarray:
    """Corrupt scaled windows (B, lags, C*N) and present them in the method's input format."""
    scaled_noise = None if noise is None else noise / sd.sensor_range
    x = corrupt_numpy(windows, scaled_noise, mask, sd.channels)
    if input_kind in ("masked", "flags"):
        flags = np.zeros((x.shape[0], sd.num_sensors), dtype=np.float32) if mask is None else mask.astype(np.float32)
        x = np.concatenate([x, np.repeat(flags[:, None, :], x.shape[1], axis=1)], axis=-1)
    elif snapshot:
        x = x[:, -1, :]
    return x.astype(np.float32)


def _build_model(cfg, sd: SeedData, kind: str, device: str) -> torch.nn.Module:
    n, c, m = sd.num_sensors, sd.channels, sd.load_X.shape[1]
    mc, rc, sc = cfg.model, cfg.robust_shred, cfg.senseiver_sdn
    null_token = bool(cfg.meeting.null_token)
    coords = torch.tensor(sd.sensor_coordinates)
    if kind in ("shred", "shred_flags"):
        inputs = c * n + (n if kind == "shred_flags" else 0)
        model = SHRED(inputs, m, hidden_size=int(mc.hidden_size), hidden_layers=int(mc.hidden_layers),
                      l1=int(mc.l1), l2=int(mc.l2), dropout=float(mc.dropout), recurrent_cell=str(mc.recurrent_cell))
    elif kind == "sdn":
        model = SDN(c * n, m, l1=int(mc.l1), l2=int(mc.l2), dropout=float(mc.dropout))
    elif kind == "senseiver_sdn":
        inner = SenseiverSDN(
            output_size=m, input_channels=c, spatial_dim=2, num_latents=int(sc.num_latents),
            latent_dim=int(sc.latent_dim), num_frequencies=int(sc.num_frequencies), num_heads=int(sc.num_heads),
            l1=int(sc.l1), l2=int(sc.l2), dropout=float(sc.dropout), null_token=null_token,
        )
        model = MaskedInputAdapter(inner, n, c, coords)
    elif kind == "robust_v1":
        inner = RobustSHREDv1(
            n, m, coords, hidden_size=int(rc.hidden_size), num_heads=int(rc.num_heads), embed_dim=int(rc.embed_dim),
            num_frequencies=int(rc.num_frequencies), l1=int(rc.l1), l2=int(rc.l2), dropout=float(rc.dropout),
            input_channels=c, null_token=null_token,
        )
        model = MaskedInputAdapter(inner, n, c)
    elif kind == "robust_v2":
        inner = RobustSHREDv2(
            n, m, coords, num_latents=int(rc.num_latents), latent_dim=int(rc.latent_dim),
            num_frequencies=int(rc.num_frequencies), num_heads=int(rc.num_heads), hidden_size=int(rc.hidden_size),
            hidden_layers=int(rc.hidden_layers), l1=int(rc.l1), l2=int(rc.l2), dropout=float(rc.dropout),
            input_channels=c, null_token=null_token,
        )
        model = MaskedInputAdapter(inner, n, c)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return model.to(device)


def run_config(cfg, method: str, seed: int) -> dict[str, Any]:
    """The configuration that identifies one cached run."""
    spec = METHODS[method]
    train_p = float(cfg.meeting.train_p) if spec["train_p"] == "p" else spec["train_p"]
    config = {
        "experiment": "meeting_dropout_robustness",
        "method": method,
        "kind": spec["kind"],
        "input": spec["input"],
        "train_p": train_p,
        "seed": int(seed),
        "data_source": "nektar",
        "data_file": str(cfg.data.nektar_file),
        "nektar_field": str(cfg.data.nektar_field),
        "split": [float(cfg.data.train_percentage), float(cfg.data.val_percentage), float(cfg.data.test_percentage)],
        "num_sensors": int(cfg.meeting.num_sensors),
        "placement": "QR-probe-block",
        "qr_rank": int(cfg.meeting.qr_rank),
        "lags": int(cfg.model.lags),
        "noise_frac": float(cfg.meeting.noise_frac),
        "test_ps": [float(p) for p in cfg.meeting.test_ps],
        "test_draws": int(cfg.meeting.test_draws),
        "clip_pod": bool(cfg.meeting.clip_pod),
    }
    if spec["kind"] != "pod":
        config.update({
            "train": {k: cfg.train[k] for k in ("epochs", "batch_size", "lr", "patience")},
            "model": dict(cfg.model),
            "robust_shred": dict(cfg.robust_shred),
            "senseiver_sdn": dict(cfg.senseiver_sdn),
            "null_token": bool(cfg.meeting.null_token),
        })
    return json.loads(json.dumps(config, default=str))


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


def _evaluate(sd: SeedData, recon_fn, panel_windows: np.ndarray) -> tuple[dict, dict]:
    """recon_fn(noise, mask) -> (recon unscaled (B, m), latent or None) for the test windows."""
    draws, n_test = sd.test_noise.shape[:2]
    truth_rep = np.tile(sd.truth, (draws, 1))
    m_block = sd.n_points
    metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    recon_clean, latent_clean = recon_fn(None, None)
    metrics["clean"] = {"err": _aggregate_rel_error(recon_clean, sd.truth)}
    arrays["clean_err"] = _per_snapshot_rel_error(recon_clean, sd.truth)
    for p, mask in sd.test_mask.items():
        tag = f"p{p:g}"
        noise = sd.test_noise.reshape(draws * n_test, *sd.test_noise.shape[2:])
        flat_mask = mask.reshape(draws * n_test, -1)
        recon, latent = recon_fn(noise, flat_mask)
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


def _train_and_evaluate(cfg, sd: SeedData, method: str, config: dict, device: str) -> tuple[dict, dict, Any]:
    spec = METHODS[method]
    kind, input_kind = spec["kind"], spec["input"]
    snapshot = kind == "sdn"
    lags = sd.lags
    panel_windows = np.arange(min(int(cfg.meeting.panel_windows), len(sd.test_idx)))

    if kind == "pod":
        lo, hi = sd.scaler.data_min_, sd.scaler.data_max_
        y_clean = sd.load_X[sd.test_idx + lags - 1][:, sd.state_indices]
        draws = sd.test_noise.shape[0]

        def recon_fn(noise, mask):
            y = y_clean if noise is None else np.tile(y_clean, (draws, 1)) + noise[:, -1, :]
            missing = None if mask is None else expand_mask(mask, sd.channels)
            recon = gappy_pod_reconstruct(y, missing, sd.state_indices, sd.U_r)
            return (np.clip(recon, lo, hi) if config["clip_pod"] else recon), None

        metrics, arrays = _evaluate(sd, recon_fn, panel_windows)
        return metrics, arrays, None

    torch.manual_seed(sd.seed)
    np.random.seed(sd.seed)
    model = _build_model(cfg, sd, kind, device)
    train_p = config["train_p"]

    def targets(idx: np.ndarray) -> torch.Tensor:
        return torch.tensor(sd.transformed_X[idx + lags - 1], device=device)

    train_x = sd.windows[sd.train_idx][:, -1, :] if snapshot else sd.windows[sd.train_idx]
    valid_mask = sd.valid_mask if train_p > 0 else None
    valid_x = _format_inputs(sd, sd.windows[sd.valid_idx], sd.valid_noise, valid_mask, input_kind, snapshot)
    augment = make_corruption_augmenter(
        sd.sigma / sd.sensor_range, train_p, sd.num_sensors, channels=sd.channels,
        append_mask=input_kind in ("masked", "flags"),
    )
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
    with_latent = method in LATENT_METHODS

    def recon_fn(noise, mask):
        base = test_windows if noise is None else np.tile(test_windows, (draws, 1, 1))
        x = _format_inputs(sd, base, noise, mask, input_kind, snapshot)
        pred, latent = _predict(model, x, device, with_latent)
        return sd.scaler.inverse_transform(pred), latent

    metrics, arrays = _evaluate(sd, recon_fn, panel_windows)
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
        test_noise=sd.test_noise, **{f"test_mask_p{p:g}": mask for p, mask in sd.test_mask.items()},
        panel_truth=sd.truth[panel].astype(np.float32), panel_windows=panel,
    )


def run_all(cfg, *, verbose: bool = False) -> list[dict[str, Any]]:
    """Run (or skip if cached) every method x seed; one compact JSON line per run."""
    root = _abs(cfg.outputs.root)
    device = _device()
    methods = [str(m) for m in cfg.meeting.methods]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    records = []
    for seed in [int(s) for s in cfg.meeting.seeds]:
        sd = None
        for method in methods:
            config = run_config(cfg, method, seed)
            config_id = _stable_config_fingerprint(config)
            out_dir = root / f"seed{seed}" / method
            if require_matching_result_summary(out_dir / "result.json", config_id) is not None:
                record = {"status": "skipped", "method": method, "seed": seed}
                records.append(record)
                print(json.dumps(record))
                continue
            if sd is None:
                sd = prepare_seed(cfg, seed)
                save_seed_info(sd, root / f"seed{seed}", int(cfg.meeting.panel_windows))
            started = time.time()
            payload: dict[str, Any] = {
                "experiment": config["experiment"], "method": method, "seed": seed, "config": config,
                "config_id": config_id, "device": device, "git_commit": _git_commit_short(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                metrics, arrays, model = _train_and_evaluate(cfg, sd, method, config, device)
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_dir / "arrays.npz", **arrays)
                if model is not None:
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
                payload.update(status="completed", metrics=metrics, wall_time_s=round(time.time() - started, 2))
            except Exception as exc:  # recorded so the sweep continues; failed runs are retried next time
                payload.update(status="failed", error_summary=f"{type(exc).__name__}: {exc}",
                               wall_time_s=round(time.time() - started, 2))
                if verbose:
                    raise
            write_result_summary(out_dir / "result.json", payload)
            record = {"status": payload["status"], "method": method, "seed": seed,
                      "wall_time_s": payload["wall_time_s"]}
            if payload["status"] == "completed":
                record.update({k: round(v["err"], 5) for k, v in payload["metrics"].items()
                               if isinstance(v, dict) and "err" in v})
            else:
                record["error"] = payload["error_summary"]
            records.append(record)
            print(json.dumps(record))
    write_result_summary(root / "summary.json", build_summary(root, methods, [int(s) for s in cfg.meeting.seeds]))
    return records


def load_results(root: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """{(method, seed): result.json payload} for every completed run under root."""
    out = {}
    for path in sorted(_abs(root).glob("seed*/*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            out[(payload["method"], int(payload["seed"]))] = payload
    return out


def load_arrays(root: str | Path, method: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(_abs(root) / f"seed{seed}" / method / "arrays.npz") as data:
        return {k: data[k] for k in data.files}


def load_seed_info(root: str | Path, seed: int) -> dict[str, np.ndarray]:
    with np.load(_abs(root) / f"seed{seed}" / "seed_info.npz") as data:
        return {k: data[k] for k in data.files}


def build_summary(root: str | Path, methods: list[str], seeds: list[int]) -> dict[str, Any]:
    """Mean/std over seeds per method and condition, plus missing/failed runs."""
    results = load_results(root)
    summary: dict[str, Any] = {"methods": {}, "missing": [], "seeds": seeds}
    for method in methods:
        runs = [results[(method, s)] for s in seeds if (method, s) in results]
        summary["missing"] += [f"{method}/seed{s}" for s in seeds if (method, s) not in results]
        if not runs:
            continue
        entry: dict[str, Any] = {"n_seeds": len(runs)}
        for cond in [k for k, v in runs[0]["metrics"].items() if isinstance(v, dict)]:
            for key in ("err", "err_u", "err_v", "d_latent_mean", "d_recon_mean"):
                vals = [r["metrics"][cond].get(key) for r in runs]
                if all(v is not None for v in vals):
                    entry[f"{cond}_{key}"] = [float(np.mean(vals)), float(np.std(vals))]
        if "parameters" in runs[0]["metrics"]:
            entry["parameters"] = int(runs[0]["metrics"]["parameters"])
        entry["wall_time_s"] = float(np.mean([r["wall_time_s"] for r in runs]))
        summary["methods"][method] = entry
    return summary
