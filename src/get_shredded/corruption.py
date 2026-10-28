"""Per-sample sensor corruption: additive Gaussian noise plus sensor dropout.

Sensor arrays use the layout (batch, [lags,] channels * num_sensors) with the
channel blocks concatenated, e.g. [u_1..u_N, v_1..v_N] for (u, v) probes. A
dropout mask has shape (batch, num_sensors), True = missing; a failed sensor
loses every channel and stays failed for the whole lag window.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "sample_dropout_mask",
    "sample_dropout_mask_torch",
    "expand_mask",
    "noise_std_from_training",
    "corrupt_numpy",
    "make_corruption_augmenter",
    "make_garbage_augmenter",
    "split_masked_input",
]


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


def corrupt_numpy(
    x: np.ndarray, noise: np.ndarray | None, mask: np.ndarray | None, channels: int, fill: float = 0.0
) -> np.ndarray:
    """Add a given noise realisation and fill failed sensors. x: (B, [lags,] C*N)."""
    out = np.array(x, copy=True)
    if noise is not None:
        out = out + noise.astype(out.dtype, copy=False)
    if mask is not None:
        full = expand_mask(mask, channels)
        if out.ndim == 3:
            full = full[:, None, :]
        out = np.where(full, np.asarray(fill, dtype=out.dtype), out)
    return out


def make_corruption_augmenter(
    noise_std: np.ndarray | Tensor,
    p: float,
    num_sensors: int,
    *,
    channels: int = 1,
    append_mask: bool = False,
    fill: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Per-batch training augmenter: fresh noise and a fresh per-sample mask on
    every call (so masks are resampled every epoch). Failed sensors are
    filled with ``fill`` in the model's input space. With ``append_mask`` the
    float mask (B, [lags,] N) is concatenated on the last axis so masked
    models can read it via :func:`split_masked_input`."""
    std = torch.as_tensor(np.asarray(noise_std), dtype=torch.float32)
    if std.numel() != channels * num_sensors:
        raise ValueError(f"noise_std must have {channels * num_sensors} entries, got {std.numel()}")

    def _augment(x: Tensor) -> Tensor:
        out = x + torch.randn_like(x) * std.to(x.device)
        mask = sample_dropout_mask_torch(x.shape[0], num_sensors, p, device=x.device)
        full = expand_mask(mask, channels)
        if out.ndim == 3:
            full = full.unsqueeze(1)
        out = out.masked_fill(full, fill)
        if append_mask:
            flags = mask.to(out.dtype)
            if out.ndim == 3:
                flags = flags.unsqueeze(1).expand(-1, out.shape[1], -1)
            out = torch.cat([out, flags], dim=-1)
        return out

    return _augment


def make_garbage_augmenter(
    noise_std: np.ndarray | Tensor,
    p: float,
    num_sensors: int,
    *,
    channels: int = 1,
    mean: float = 0.5,
    std: float = 0.5,
    append_zero_flags: bool = False,
) -> Callable[[Tensor], Tensor]:
    """Like :func:`make_corruption_augmenter`, but failed sensors emit fresh
    N(mean, std) garbage at every lag instead of a fill value, and the failure
    is silent: with ``append_zero_flags`` an all-zero mask is appended."""
    std_noise = torch.as_tensor(np.asarray(noise_std), dtype=torch.float32)
    if std_noise.numel() != channels * num_sensors:
        raise ValueError(f"noise_std must have {channels * num_sensors} entries, got {std_noise.numel()}")

    def _augment(x: Tensor) -> Tensor:
        out = x + torch.randn_like(x) * std_noise.to(x.device)
        mask = sample_dropout_mask_torch(x.shape[0], num_sensors, p, device=x.device)
        full = expand_mask(mask, channels)
        if out.ndim == 3:
            full = full.unsqueeze(1)
        out = torch.where(full, mean + std * torch.randn_like(out), out)
        if append_zero_flags:
            shape = (*out.shape[:-1], num_sensors)
            out = torch.cat([out, torch.zeros(shape, dtype=out.dtype, device=out.device)], dim=-1)
        return out

    return _augment


def split_masked_input(x: Tensor, num_values: int) -> tuple[Tensor, Tensor]:
    """Inverse of ``append_mask``: returns (values, bool mask of shape (B, N))."""
    values = x[..., :num_values]
    flags = x[..., num_values:]
    if flags.ndim == 3:
        flags = flags[:, 0, :]
    return values, flags > 0.5
