from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .senseiver import Senseiver, SenseiverSDN, spatial_positional_encoding

__all__ = [
    "SHRED",
    "SDN",
    "TrunkSHRED",
    "FlowTrunkSHRED",
    "PODSHRED",
    "PODTrunkSHRED",
    "divergence",
    "wave_residual",
    "TimeSeriesDataset",
    "fit",
    "forecast",
    "Senseiver",
    "SenseiverSDN",
    "spatial_positional_encoding",
]


class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return self.len


class SHRED(nn.Module):
    """SHallow REcurrent Decoder: stacked recurrent cell over a trajectory of
    sensor measurements followed by a 3-layer fully-connected decoder that maps
    the final hidden state to the full high-dimensional state.

    Mirrors the architecture from Williams, Zahn, Kutz (2024), with a configurable
    recurrent cell type used by the experiment runner.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
        recurrent_cell: str = "gru",
    ) -> None:
        super().__init__()
        cell_type = str(recurrent_cell).lower()
        if cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=hidden_layers,
                batch_first=True,
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=hidden_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent_cell={recurrent_cell!r}; expected 'gru' or 'lstm'.")
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.recurrent_cell = cell_type

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """Final hidden state of the last recurrent layer, (batch, hidden_size)."""
        device = next(self.parameters()).device
        h_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        if self.recurrent_cell == "gru":
            _, h_out = self.rnn(x, h_0)
        else:
            c_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
            _, (h_out, _) = self.rnn(x, (h_0, c_0))
        return h_out[-1].view(-1, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_out = self.latent(x)
        out = torch.relu(self.dropout(self.linear1(h_out)))
        out = torch.relu(self.dropout(self.linear2(out)))
        return self.linear3(out)


def wave_residual(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, c: float) -> torch.Tensor:
    """u_tt - c^2 u_xx by autograd; u[j] must depend on x[j], t[j] only (pointwise queries)."""
    u_x, u_t = torch.autograd.grad(u, (x, t), torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    return u_tt - c**2 * u_xx


class TrunkSHRED(SHRED):
    """DeepONet-style SHRED for the 1D wave: the SHRED recurrent encoder is the branch
    (window -> b in R^p), a tanh MLP over (x, tau) is the trunk, u(x, tau) = b . T(x, tau) + bias.
    tau is time relative to the window's last frame (tau in [-(lags-1) dt, 0]).

    ``forward`` returns the field at the grid x and tau = 0, MinMax-scaled like SHRED's targets
    (u * out_scale + out_offset), so training, validation and metrics are unchanged. The model
    itself works in physical units, so ``aux_loss`` (wave residual + Dirichlet BCs on random
    collocation points, weighted by ``physics_weight``) uses true derivatives. With ``probe_weight``,
    ``aux_loss`` also fits the trunk to the probe readings at every lag (the model's own clean input).
    """

    def __init__(
        self,
        input_size: int,
        x_grid,
        out_scale,
        out_offset,
        c: float,
        L: float,
        dt: float,
        lags: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        recurrent_cell: str = "gru",
        p: int = 64,
        trunk_width: int = 64,
        trunk_layers: int = 2,
        physics_weight: float = 0.0,
        n_colloc: int = 64,
        num_frequencies: int = 0,
        probe_idx=None,
        probe_weight: float = 0.0,
    ) -> None:
        super().__init__(input_size, 1, hidden_size=hidden_size, hidden_layers=hidden_layers, l1=1, l2=1,
                         recurrent_cell=recurrent_cell)
        del self.linear1, self.linear2, self.linear3, self.dropout  # replaced by branch + trunk
        self.branch = nn.Linear(hidden_size, p)
        self.num_frequencies = int(num_frequencies)  # Fourier features of x (0: raw x) against spectral bias
        layers, width = [], 1 + (2 * self.num_frequencies if self.num_frequencies > 0 else 1)
        for _ in range(trunk_layers):
            layers += [nn.Linear(width, trunk_width), nn.Tanh()]  # smooth: ReLU has zero 2nd derivative
            width = trunk_width
        self.trunk = nn.Sequential(*layers, nn.Linear(width, p))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("x_grid", torch.as_tensor(x_grid, dtype=torch.float32))
        self.register_buffer("out_scale", torch.as_tensor(out_scale, dtype=torch.float32))
        self.register_buffer("out_offset", torch.as_tensor(out_offset, dtype=torch.float32))
        self.c, self.L = float(c), float(L)
        self.tau_span = max(int(lags) - 1, 1) * float(dt)
        self.physics_weight = float(physics_weight)
        self.n_colloc = int(n_colloc)
        self.lags = int(lags)
        self.probe_weight = float(probe_weight)
        if self.probe_weight > 0 and probe_idx is None:
            raise ValueError("probe_weight > 0 needs probe_idx")
        self.register_buffer("probe_idx", torch.as_tensor([] if probe_idx is None else probe_idx, dtype=torch.long),
                             persistent=False)  # from the data (placement), not a trained parameter

    def field(self, b: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Physical u at (x, tau): b (B, p); x, tau (Q,) or (B, Q). Returns (B, Q)."""
        x = (x / self.L).unsqueeze(-1)
        if self.num_frequencies > 0:
            x = spatial_positional_encoding(x, self.num_frequencies)
        T = self.trunk(torch.cat([x, (tau / self.tau_span).unsqueeze(-1)], dim=-1))
        return (b.unsqueeze(1) * T).sum(-1) + self.bias

    def coefficients(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(self.latent(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.field(self.coefficients(x), self.x_grid, torch.zeros_like(self.x_grid))
        return u * self.out_scale + self.out_offset

    def probe_field(self, b: torch.Tensor) -> torch.Tensor:
        """Scaled u at the probes for every lag: (B, lags, N); lag j is tau_j = (j - lags + 1) dt."""
        n = len(self.probe_idx)
        tau = torch.arange(1 - self.lags, 1, device=b.device) * (self.tau_span / max(self.lags - 1, 1))
        x = self.x_grid[self.probe_idx].repeat(self.lags)  # lag-major: [all probes at lag 0, ...]
        u = self.field(b, x, tau.repeat_interleave(n)).view(-1, self.lags, n)
        return u * self.out_scale[self.probe_idx] + self.out_offset[self.probe_idx]

    def aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Training-only terms next to the data loss: physics (wave residual + BCs) and the probe readings at
        every lag (x itself, so only valid for clean inputs; the runner enforces that)."""
        if self.physics_weight == 0.0 and self.probe_weight == 0.0:
            return x.new_zeros(())
        b = self.coefficients(x)
        loss = x.new_zeros(())
        if self.probe_weight > 0.0:
            n = len(self.probe_idx)
            loss = loss + self.probe_weight * (self.probe_field(b) - x[..., :n]).square().mean()
        if self.physics_weight == 0.0:
            return loss
        shape = (x.shape[0], self.n_colloc)
        xs = (torch.rand(shape, device=x.device) * self.L).requires_grad_()
        ts = (-torch.rand(shape, device=x.device) * self.tau_span).requires_grad_()
        omega1 = torch.pi * self.c / self.L  # residual / omega1^2 is O(1) per unit amplitude
        residual = wave_residual(self.field(b, xs, ts), xs, ts, self.c) / omega1**2
        xb = torch.randint(0, 2, shape, device=x.device).float() * self.L
        boundary = self.field(b, xb, -torch.rand(shape, device=x.device) * self.tau_span)
        return loss + self.physics_weight * (residual.square().mean() + boundary.square().mean())


def divergence(u: torch.Tensor, v: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """u_x + v_y by autograd; xy (..., 2) with u[j], v[j] depending on xy[j] only (pointwise queries)."""
    u_x = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0][..., 0]
    v_y = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0][..., 1]
    return u_x + v_y


class FlowTrunkSHRED(SHRED):
    """DeepONet-style SHRED for the Nektar (u, v) wake: SHRED recurrent branch (window -> b in R^{2 x p}),
    tanh trunk over Fourier features of (x, y) -> one basis per channel; u_c(x, y) = b_c . T_c(x, y) + bias_c.
    No time input: only the last frame is supervised. ``forward`` returns [u block; v block] at the grid,
    MinMax-scaled like SHRED's targets. ``aux_loss`` = ``physics_weight`` x mean (u_x + v_y)^2 on random points
    in the grid's bounding box, in physical units.
    """

    def __init__(
        self,
        input_size: int,
        coords,
        out_scale,
        out_offset,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        recurrent_cell: str = "gru",
        p: int = 64,
        trunk_width: int = 64,
        trunk_layers: int = 2,
        num_frequencies: int = 16,
        physics_weight: float = 0.0,
        n_colloc: int = 64,
    ) -> None:
        super().__init__(input_size, 1, hidden_size=hidden_size, hidden_layers=hidden_layers, l1=1, l2=1,
                         recurrent_cell=recurrent_cell)
        del self.linear1, self.linear2, self.linear3, self.dropout  # replaced by branch + trunk
        self.p, self.num_frequencies = int(p), int(num_frequencies)
        self.branch = nn.Linear(hidden_size, 2 * self.p)
        layers, width = [], 2 * (2 * self.num_frequencies if self.num_frequencies > 0 else 1)
        for _ in range(trunk_layers):
            layers += [nn.Linear(width, trunk_width), nn.Tanh()]
            width = trunk_width
        self.trunk = nn.Sequential(*layers, nn.Linear(width, 2 * self.p))
        self.bias = nn.Parameter(torch.zeros(2))
        coords = torch.as_tensor(coords, dtype=torch.float32)
        self.register_buffer("coords", coords)
        self.register_buffer("lo", coords.min(0).values)
        self.register_buffer("span", coords.max(0).values - coords.min(0).values)
        self.register_buffer("out_scale", torch.as_tensor(out_scale, dtype=torch.float32))
        self.register_buffer("out_offset", torch.as_tensor(out_offset, dtype=torch.float32))
        self.physics_weight = float(physics_weight)
        self.n_colloc = int(n_colloc)

    def basis(self, xy: torch.Tensor) -> torch.Tensor:
        """Trunk at physical (..., 2) points -> (..., 2, p)."""
        z = (xy - self.lo) / self.span
        if self.num_frequencies > 0:
            z = spatial_positional_encoding(z, self.num_frequencies)
        return self.trunk(z).unflatten(-1, (2, self.p))

    def coefficients(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(self.latent(x)).unflatten(-1, (2, self.p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uv = torch.einsum("bcp,qcp->bcq", self.coefficients(x), self.basis(self.coords)) + self.bias[:, None]
        return uv.flatten(1) * self.out_scale + self.out_offset  # state order [u block; v block]

    def aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.physics_weight == 0.0:
            return x.new_zeros(())
        b = self.coefficients(x)
        xy = (self.lo + torch.rand(x.shape[0], self.n_colloc, 2, device=x.device) * self.span).requires_grad_()
        uv = torch.einsum("bcp,bqcp->bqc", b, self.basis(xy)) + self.bias
        return self.physics_weight * divergence(uv[..., 0], uv[..., 1], xy).square().mean()


class PODSHRED(nn.Module):
    """SHRED predicting POD coefficients, with the basis built the masked-shred way: per field (channel block of
    the state) StandardScaler on the training snapshots, SVD, ``rank`` modes per field, MinMax per coefficient.

    ``self.shred`` maps a window to the 2r (C r) MinMax-scaled coefficients; ``forward`` decodes them to the field,
    MinMax-scaled like SHRED's targets (u * out_scale + out_offset), so it can be trained on the field (loss in field
    space) or ``self.shred`` on ``coefficient_targets`` (loss in coefficient space), with the same evaluation.
    """

    def __init__(self, input_size: int, train_X, channels: int, rank: int, out_scale, out_offset,
                 **shred_kwargs) -> None:
        super().__init__()
        X = torch.as_tensor(train_X, dtype=torch.float64)
        n, m = X.shape
        self.channels, self.rank, self.points = int(channels), int(rank), m // int(channels)
        if self.rank > min(n, self.points):
            raise ValueError(f"rank {rank} exceeds the {n} training snapshots or {self.points} points per field")
        mean, std = X.mean(0), X.std(0, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)  # as StandardScaler: constant columns unscaled
        z = ((X - mean) / std).unflatten(-1, (self.channels, self.points))
        Vt = torch.stack([torch.linalg.svd(z[:, c], full_matrices=False)[2][: self.rank]
                          for c in range(self.channels)])  # (C, r, points), exact SVD (the paper: randomized)
        a = torch.einsum("ncp,crp->ncr", z, Vt).flatten(1)
        lo, span = a.min(0).values, a.max(0).values - a.min(0).values
        span = torch.where(span == 0, torch.ones_like(span), span)
        for name, value in (("mean", mean), ("std", std), ("Vt", Vt), ("coef_min", lo), ("coef_range", span),
                            ("out_scale", torch.as_tensor(out_scale)), ("out_offset", torch.as_tensor(out_offset))):
            self.register_buffer(name, value.float())
        self.shred = SHRED(input_size, self.channels * self.rank, **shred_kwargs)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.shred.latent(x)

    def decode(self, scaled_coefficients: torch.Tensor) -> torch.Tensor:
        a = (scaled_coefficients * self.coef_range + self.coef_min).unflatten(-1, (self.channels, self.rank))
        u = torch.einsum("bcr,crp->bcp", a, self.Vt).flatten(1) * self.std + self.mean
        return u * self.out_scale + self.out_offset

    def coefficient_targets(self, X: torch.Tensor) -> torch.Tensor:
        """Unscaled snapshots (B, m) -> MinMax-scaled coefficients (B, C r)."""
        z = ((X - self.mean) / self.std).unflatten(-1, (self.channels, self.points))
        return (torch.einsum("bcp,crp->bcr", z, self.Vt).flatten(1) - self.coef_min) / self.coef_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.shred(x))


class PODTrunkSHRED(PODSHRED):
    """PODSHRED whose basis (per-field mean, std and modes) is a frozen continuous trunk T(x, y): the flow trunk's
    tanh MLP on Fourier features of (x, y), fitted once in ``__init__`` to the grid POD basis (full batch, Adam).
    ``decode`` uses the trunk at the grid (cached), ``coefficient_targets`` the true POD coefficients, and
    ``field_at`` evaluates u, v anywhere, so ``aux_loss`` (``physics_weight`` x mean (u_x + v_y)^2 on random points
    in the grid's box, physical units) differentiates through the trunk."""

    def __init__(self, input_size: int, train_X, channels: int, rank: int, out_scale, out_offset, coords,
                 trunk_width: int = 256, trunk_layers: int = 2, num_frequencies: int = 16, fit_steps: int = 5000,
                 fit_lr: float = 1e-3, physics_weight: float = 0.0, n_colloc: int = 64, train_trunk: bool = False,
                 **shred_kwargs) -> None:
        super().__init__(input_size, train_X, channels, rank, out_scale, out_offset, **shred_kwargs)
        C, r, P = self.channels, self.rank, self.points
        coords = torch.as_tensor(coords, dtype=torch.float32)
        self.register_buffer("coords", coords)
        self.register_buffer("lo", coords.min(0).values)
        self.register_buffer("span", coords.max(0).values - coords.min(0).values)
        self.num_frequencies = int(num_frequencies)
        self.physics_weight, self.n_colloc = float(physics_weight), int(n_colloc)
        self.train_trunk = bool(train_trunk)
        # targets per point: [mean, std, sqrt(P) * modes] per field, so every column is O(1)
        target = torch.cat([self.mean.view(C, P, 1), self.std.view(C, P, 1),
                            self.Vt.permute(0, 2, 1) * P**0.5], dim=-1).permute(1, 0, 2).flatten(1)
        t_mu, t_sd = target.mean(0), target.std(0)
        t_sd = torch.where(t_sd == 0, torch.ones_like(t_sd), t_sd)
        self.register_buffer("t_mu", t_mu)
        self.register_buffer("t_sd", t_sd)
        layers, width = [], 2 * 2 * self.num_frequencies
        for _ in range(trunk_layers):
            layers += [nn.Linear(width, trunk_width), nn.Tanh()]
            width = trunk_width
        self.trunk = nn.Sequential(*layers, nn.Linear(width, C * (r + 2)))
        z, goal = self._encode(coords), (target - t_mu) / t_sd
        optimizer = torch.optim.Adam(self.trunk.parameters(), lr=fit_lr)
        for _ in range(int(fit_steps)):
            optimizer.zero_grad()
            loss = (self.trunk(z) - goal).square().mean()
            loss.backward()
            optimizer.step()
        self.trunk.requires_grad_(self.train_trunk)  # train_trunk: SHRED's field and physics losses keep training it
        with torch.no_grad():
            out = self.basis_at(coords)  # (P, C, r + 2)
        self.register_buffer("mean_fit", out[..., 0].T.reshape(-1))
        self.register_buffer("std_fit", out[..., 1].T.reshape(-1))
        self.register_buffer("Vt_fit", out[..., 2:].permute(1, 2, 0) / P**0.5)
        gram = torch.einsum("crp,csp->crs", self.Vt_fit, self.Vt_fit) - torch.eye(r)
        rel = lambda a, b: float(torch.linalg.norm(a - b) / torch.linalg.norm(b))  # noqa: E731
        self.fit_report = {
            "mode_error": torch.linalg.norm(self.Vt_fit - self.Vt, dim=-1).tolist(),  # (C, r); true modes unit norm
            "mean_error": rel(self.mean_fit, self.mean), "std_error": rel(self.std_fit, self.std),
            "gram_max_dev": float(gram.abs().max()), "final_fit_loss": float(loss.detach()),
        }

    def _encode(self, xy: torch.Tensor) -> torch.Tensor:
        return spatial_positional_encoding((xy - self.lo) / self.span, self.num_frequencies)

    def basis_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Trunk at physical (..., 2) points -> (..., C, r + 2): [mean, std, sqrt(P) * modes] per field."""
        return (self.trunk(self._encode(xy)) * self.t_sd + self.t_mu).unflatten(-1, (self.channels, self.rank + 2))

    def grid_basis(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(modes (C, r, P), std (C*P,), mean (C*P,)) at the grid: cached when frozen, from the trunk when trained."""
        if not self.train_trunk:
            return self.Vt_fit, self.std_fit, self.mean_fit
        out = self.basis_at(self.coords)
        return out[..., 2:].permute(1, 2, 0) / self.points**0.5, out[..., 1].T.reshape(-1), out[..., 0].T.reshape(-1)

    def decode(self, scaled_coefficients: torch.Tensor) -> torch.Tensor:
        a = (scaled_coefficients * self.coef_range + self.coef_min).unflatten(-1, (self.channels, self.rank))
        modes, std, mean = self.grid_basis()
        u = torch.einsum("bcr,crp->bcp", a, modes).flatten(1) * std + mean
        return u * self.out_scale + self.out_offset

    def field_at(self, scaled_coefficients: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Physical fields at points xy (B, Q, 2) -> (B, Q, C)."""
        a = (scaled_coefficients * self.coef_range + self.coef_min).unflatten(-1, (self.channels, self.rank))
        T = self.basis_at(xy)
        return T[..., 0] + T[..., 1] * torch.einsum("bcr,bqcr->bqc", a, T[..., 2:]) / self.points**0.5

    def aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.physics_weight == 0.0:
            return x.new_zeros(())
        xy = (self.lo + torch.rand(x.shape[0], self.n_colloc, 2, device=x.device) * self.span).requires_grad_()
        uv = self.field_at(self.shred(x), xy)
        return self.physics_weight * divergence(uv[..., 0], uv[..., 1], xy).square().mean()

    def divergence_rel(self, x: torch.Tensor, frac: torch.Tensor) -> dict[str, float]:
        """rms(u_x + v_y) / rms|grad u| of the predicted fields at points lo + frac * span (held out) and at the grid."""
        with torch.no_grad():
            s = self.shred(x)
        out = {}
        for name, pts in (("held_out", self.lo + frac * self.span), ("grid", self.coords)):
            xy = pts.expand(x.shape[0], -1, -1).clone().requires_grad_()
            uv = self.field_at(s, xy)
            gu = torch.autograd.grad(uv[..., 0].sum(), xy, retain_graph=True)[0]
            gv = torch.autograd.grad(uv[..., 1].sum(), xy)[0]
            div = gu[..., 0] + gv[..., 1]
            out[name] = float(div.square().mean().sqrt() / (gu.square().sum(-1) + gv.square().sum(-1)).mean().sqrt())
        return out


class SDN(nn.Module):
    """Static Shallow Decoder Network: 3-layer FC mapping a single snapshot of
    sensor measurements to the full state. Used as a non-temporal ablation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.dropout(self.linear1(x)))
        out = torch.relu(self.dropout(self.linear2(out)))
        return self.linear3(out)


def fit(
    model: nn.Module,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    batch_size: int = 64,
    num_epochs: int = 4000,
    lr: float = 1e-3,
    verbose: bool = False,
    patience: int = 5,
    augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    early_stopping: bool = True,
    lr_schedule: str = "none",
) -> torch.Tensor:
    """Training loop matching the paper: Adam + MSE, validation on relative
    L2 error every 20 epochs, early stopping on patience, restores best params.
    early_stopping=False trains all epochs and keeps the final params.

    augment_fn is applied to each training batch only (not validation/test).
    lr_schedule "cosine" decays lr to 0 over num_epochs (stepped per epoch); "none" keeps lr fixed.
    """
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_schedule not in ("none", "cosine"):
        raise ValueError(f"lr_schedule must be 'none' or 'cosine', got {lr_schedule!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs) if lr_schedule == "cosine" else None

    val_error_list: list[torch.Tensor] = []
    patience_counter = 0
    best_params = deepcopy(model.state_dict())

    pbar = tqdm(
        range(1, num_epochs + 1),
        desc=f"train {model.__class__.__name__}",
        disable=not verbose,
        leave=False,
    )
    last_val = float("nan")
    best_so_far = float("nan")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            if augment_fn is not None:
                batch_x = augment_fn(batch_x)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            if hasattr(model, "aux_loss"):
                loss = loss + model.aux_loss(batch_x)
            if hasattr(model, "target_aux_loss"):  # a training-only term that needs the targets (ResidualSHREDv1)
                loss = loss + model.target_aux_loss(batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix(loss=f"{train_loss:.4e}", val=f"{last_val:.4f}",
                         best=f"{best_so_far:.4f}", patience=f"{patience_counter}/{patience}")

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y) / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error.detach().cpu())

            last_val = val_error_list[-1].item()
            best_so_far = float(torch.min(torch.tensor(val_error_list)))
            pbar.set_postfix(loss=f"{train_loss:.4e}", val=f"{last_val:.4f}",
                             best=f"{best_so_far:.4f}", patience=f"{patience_counter}/{patience}")

            if val_error_list[-1] == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if early_stopping and patience_counter == patience:
                pbar.close()
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list)

    pbar.close()
    if early_stopping:
        model.load_state_dict(best_params)
    return torch.tensor(val_error_list)


def forecast(forecaster: SHRED, reconstructor: SHRED, test_dataset: TimeSeriesDataset):
    """Two-step forecasting (paper §IV.B): an LSTM forecasts sensor
    trajectories autoregressively, then SHRED reconstructs the high-dimensional
    state from the forecasted sensor windows."""
    initial_in = test_dataset.X[0:1].clone()
    vals = [initial_in[0, i, :].detach().cpu().clone().numpy() for i in range(test_dataset.X.shape[1])]

    for _ in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()
        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(scaled_output, device=initial_in.device)

    import numpy as np
    device = next(reconstructor.parameters()).device
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    seq_len = test_dataset.X.shape[1]
    n_sensors = test_dataset.X.shape[2]
    for i in range(len(forecasted_vals) - seq_len):
        recon = reconstructor(
            forecasted_vals[i : i + seq_len].reshape(1, seq_len, n_sensors)
        ).detach().cpu().numpy()
        reconstructions.append(recon)
    return forecasted_vals, np.array(reconstructions)
