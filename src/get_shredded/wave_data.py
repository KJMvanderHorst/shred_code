from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WaveGeneratorConfig:
    """Canonical 1D wave-equation dataset configuration."""

    L: float = 2.0
    c: float = 2.0
    num_modes: int = 10
    num_trajectories: int = 1000
    nx: int = 256
    nt: int = 500
    t_start: float = 0.0
    t_end: float = 2.0
    endpoint: bool = True
    seed: int = 42
    amplitude_scale: float = 1.0


DEFAULT_WAVE_CONFIG = WaveGeneratorConfig()


def _wave_mode_numbers(num_modes: int) -> np.ndarray:
    return np.arange(1, num_modes + 1, dtype=np.float64)


def _validate_wave_dataset(
    field: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    *,
    L: float,
    endpoint: bool,
) -> None:
    if field.ndim != 3:
        raise ValueError(f"field must have shape (N_trajectories, Nt, Nx); got {field.shape}")
    if field.shape[1] != t.size or field.shape[2] != x.size:
        raise ValueError(
            f"field shape mismatch: expected ({field.shape[0]}, {t.size}, {x.size}), got {field.shape}"
        )
    if not np.allclose(x[0], 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"x[0] should be 0, got {x[0]}")
    if not np.allclose(x[-1], L, atol=1e-12, rtol=0.0):
        raise ValueError(f"x[-1] should be L={L}, got {x[-1]}")
    if not np.isfinite(field).all():
        raise ValueError("Generated field contains NaN or Inf values.")
    if not np.allclose(field[..., 0], 0.0, atol=1e-8, rtol=0.0):
        raise ValueError("Boundary condition failed at x=0.")
    if not np.allclose(field[..., -1], 0.0, atol=1e-8, rtol=0.0):
        raise ValueError("Boundary condition failed at x=L.")
    if endpoint and not np.allclose(field[:, 0, :], field[:, -1, :], atol=1e-8, rtol=1e-8):
        raise ValueError("For endpoint=True the first and final snapshots should match up to tolerance.")


def generate_wave_dataset(
    config: WaveGeneratorConfig | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Generate a raw 1D wave-equation field dataset and optionally save it to .npz."""
    cfg = config or DEFAULT_WAVE_CONFIG
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0.0, cfg.L, cfg.nx, endpoint=True, dtype=np.float64)
    t = np.linspace(cfg.t_start, cfg.t_end, cfg.nt, endpoint=cfg.endpoint, dtype=np.float64)

    mode_numbers = _wave_mode_numbers(cfg.num_modes)
    omega = (np.pi * cfg.c / cfg.L) * mode_numbers
    amplitudes = rng.normal(0.0, 1.0, size=(cfg.num_trajectories, cfg.num_modes))
    amplitudes = amplitudes / mode_numbers[np.newaxis, :]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(cfg.num_trajectories, cfg.num_modes))
    spatial_modes = np.sin((np.pi / cfg.L) * np.outer(mode_numbers, x))

    field = np.zeros((cfg.num_trajectories, cfg.nt, cfg.nx), dtype=np.float64)
    for traj_idx in range(cfg.num_trajectories):
        for mode_idx in range(cfg.num_modes):
            amplitude = amplitudes[traj_idx, mode_idx] * cfg.amplitude_scale
            phase = phases[traj_idx, mode_idx]
            temporal_term = np.cos(omega[mode_idx] * t + phase)
            field[traj_idx] += (temporal_term[:, None] * spatial_modes[mode_idx][None, :]) * amplitude

    _validate_wave_dataset(field, x, t, L=cfg.L, endpoint=cfg.endpoint)

    payload = {
        "field": field.astype(np.float32),
        "x": x.astype(np.float64),
        "t": t.astype(np.float64),
        "L": np.array(cfg.L, dtype=np.float64),
        "c": np.array(cfg.c, dtype=np.float64),
        "num_modes": np.array(cfg.num_modes, dtype=np.int64),
        "num_trajectories": np.array(cfg.num_trajectories, dtype=np.int64),
        "Nx": np.array(cfg.nx, dtype=np.int64),
        "Nt": np.array(cfg.nt, dtype=np.int64),
        "t_start": np.array(cfg.t_start, dtype=np.float64),
        "t_end": np.array(cfg.t_end, dtype=np.float64),
        "endpoint": np.array(cfg.endpoint),
        "seed": np.array(cfg.seed, dtype=np.int64),
        "amplitudes": amplitudes.astype(np.float64),
        "phases": phases.astype(np.float64),
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **payload)

    return payload


def load_wave_data(npz_path: str | Path, *, trajectory_index: int = 0) -> tuple[np.ndarray, int, int]:
    """Load a 1D wave dataset and return (load_X, nx, ny) with load_X shape (Nt, Nx)."""
    path = Path(npz_path)
    with np.load(path) as data:
        field = np.asarray(data["field"])
        if field.ndim == 2:
            field = field[None, ...]
        if field.ndim != 3:
            raise ValueError(f"Wave field in {path} must have shape (N_trajectories, Nt, Nx); got {field.shape}")
        if not 0 <= trajectory_index < field.shape[0]:
            raise IndexError(
                f"trajectory_index={trajectory_index} is out of bounds for {field.shape[0]} trajectories"
            )
        selected = field[trajectory_index].astype(np.float32, copy=False)
    return selected, 1, selected.shape[1]
