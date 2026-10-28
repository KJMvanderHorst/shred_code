"""Per-sample sensor corruption: additive Gaussian noise plus probe faults.

Sensor arrays use the layout (batch, [lags,] channels * num_sensors) with the
channel blocks concatenated, e.g. [u_1..u_N, v_1..v_N] for (u, v) probes. A
fault mask has shape (batch, num_sensors), True = faulty; a faulty probe is
faulty in every channel for the whole lag window.

Faults (``CorruptionSpec.fault``):
- ``none``: noise only.
- ``missing``: faulty probes read ``fill`` (scaled input space).
- ``garbage``: faulty probes read fresh N(garbage_mean, garbage_std) values at
  every lag (scaled input space).

Flags (``CorruptionSpec.flags``), for models that take a mask input:
``oracle`` appends the true fault mask, ``silent`` appends an all-zero mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "CorruptionSpec",
    "sample_dropout_mask",
    "sample_exact_k_mask",
    "sample_dropout_mask_torch",
    "expand_mask",
    "noise_std_from_training",
    "apply_fault",
    "make_augmenter",
    "split_masked_input",
]

FAULTS = ("none", "missing", "garbage")
FLAGS = ("oracle", "silent")


@dataclass(frozen=True)
class CorruptionSpec:
    noise: bool = True
    fault: str = "none"
    flags: str = "oracle"
    p: float = 0.0
    fill: float = 0.0
    garbage_mean: float = 0.5
    garbage_std: float = 0.5

    def __post_init__(self) -> None:
        if self.fault not in FAULTS:
            raise ValueError(f"fault must be one of {FAULTS}, got {self.fault!r}")
        if self.flags not in FLAGS:
            raise ValueError(f"flags must be one of {FLAGS}, got {self.flags!r}")
        if self.fault == "none" and self.p > 0:
            raise ValueError("fault 'none' requires p = 0")


def sample_dropout_mask(
    num_samples: int, num_sensors: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Bernoulli(p) failures per sensor; rows where every sensor failed are resampled."""
    if p <= 0:
        return np.zeros((num_samples, num_sensors), dtype=bool)
    if p >= 1:
        raise ValueError("p must be < 1 so at least one sensor can remain")
    mask = rng.random((num_samples, num_sensors)) < p
    all_failed = mask.all(axis=1)
    while all_failed.any():
        mask[all_failed] = rng.random((int(all_failed.sum()), num_sensors)) < p
        all_failed = mask.all(axis=1)
    return mask


def sample_exact_k_mask(num_samples: int, num_sensors: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly k failed sensors per row, a uniformly random subset per row."""
    if not 0 <= k < num_sensors:
        raise ValueError(f"k must be in [0, {num_sensors - 1}] so at least one sensor remains, got {k}")
    return rng.random((num_samples, num_sensors)).argsort(axis=1).argsort(axis=1) < k


def sample_dropout_mask_torch(
    num_samples: int, num_sensors: int, p: float, device: torch.device | str = "cpu"
) -> Tensor:
    """Torch version of :func:`sample_dropout_mask` (uses the global torch RNG)."""
    if p <= 0:
        return torch.zeros(num_samples, num_sensors, dtype=torch.bool, device=device)
    if p >= 1:
        raise ValueError("p must be < 1 so at least one sensor can remain")
    mask = torch.rand(num_samples, num_sensors, device=device) < p
    all_failed = mask.all(dim=1)
    while bool(all_failed.any()):
        mask[all_failed] = torch.rand(int(all_failed.sum()), num_sensors, device=device) < p
        all_failed = mask.all(dim=1)
    return mask


def expand_mask(mask, channels: int):
    """(batch, N) sensor mask -> (batch, channels * N) mask over the channel blocks."""
    if isinstance(mask, Tensor):
        return mask.repeat(1, channels)
    return np.tile(mask, (1, channels))


def noise_std_from_training(train_values: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Per-column sigma = frac * std of the (unscaled) training sensor readings."""
    return (frac * np.asarray(train_values, dtype=np.float64).std(axis=0)).astype(np.float32)


def apply_fault(
    x: np.ndarray, mask: np.ndarray | None, channels: int, spec: CorruptionSpec, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Replace the faulty probes of x (B, [lags,] C*N) according to ``spec.fault``. Returns a new array."""
    if mask is None or spec.fault == "none":
        return np.array(x, copy=True)
    full = expand_mask(mask, channels)
    if x.ndim == 3:
        full = full[:, None, :]
    if spec.fault == "missing":
        replacement = np.asarray(spec.fill, dtype=x.dtype)
    else:
        replacement = (spec.garbage_mean + spec.garbage_std * rng.standard_normal(x.shape)).astype(x.dtype)
    return np.where(full, replacement, x)


def make_augmenter(
    spec: CorruptionSpec,
    noise_std: np.ndarray | Tensor,
    num_sensors: int,
    *,
    channels: int = 1,
    mask_input: bool = False,
) -> Callable[[Tensor], Tensor]:
    """Per-batch training augmenter: fresh noise and a fresh per-sample fault mask on
    every call (so masks are resampled every epoch). With ``mask_input`` the float
    mask (B, [lags,] N) is concatenated on the last axis (all zero if ``spec.flags``
    is ``silent``) so masked models can read it via :func:`split_masked_input`."""
    std = torch.as_tensor(np.asarray(noise_std), dtype=torch.float32)
    if std.numel() != channels * num_sensors:
        raise ValueError(f"noise_std must have {channels * num_sensors} entries, got {std.numel()}")
    if not spec.noise:
        std = torch.zeros_like(std)

    def _augment(x: Tensor) -> Tensor:
        out = x + torch.randn_like(x) * std.to(x.device)
        mask = sample_dropout_mask_torch(x.shape[0], num_sensors, spec.p, device=x.device)
        full = expand_mask(mask, channels)
        if out.ndim == 3:
            full = full.unsqueeze(1)
        if spec.fault == "missing":
            out = out.masked_fill(full, spec.fill)
        elif spec.fault == "garbage":
            out = torch.where(full, spec.garbage_mean + spec.garbage_std * torch.randn_like(out), out)
        if mask_input:
            flags = mask.to(out.dtype) if spec.flags == "oracle" else torch.zeros_like(mask, dtype=out.dtype)
            if out.ndim == 3:
                flags = flags.unsqueeze(1).expand(-1, out.shape[1], -1)
            out = torch.cat([out, flags], dim=-1)
        return out

    return _augment


def split_masked_input(x: Tensor, num_values: int) -> tuple[Tensor, Tensor]:
    """Inverse of the mask appended by :func:`make_augmenter`: returns (values, bool mask of shape (B, N))."""
    values = x[..., :num_values]
    flags = x[..., num_values:]
    if flags.ndim == 3:
        flags = flags[:, 0, :]
    return values, flags > 0.5
