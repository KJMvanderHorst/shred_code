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
    itself works in physical units, so ``physics_loss`` (wave residual + Dirichlet BCs on random
    collocation points, weighted by ``physics_weight``) uses true derivatives.
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

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.physics_weight == 0.0:
            return x.new_zeros(())
        b = self.coefficients(x)
        shape = (x.shape[0], self.n_colloc)
        xs = (torch.rand(shape, device=x.device) * self.L).requires_grad_()
        ts = (-torch.rand(shape, device=x.device) * self.tau_span).requires_grad_()
        omega1 = torch.pi * self.c / self.L  # residual / omega1^2 is O(1) per unit amplitude
        residual = wave_residual(self.field(b, xs, ts), xs, ts, self.c) / omega1**2
        xb = torch.randint(0, 2, shape, device=x.device).float() * self.L
        boundary = self.field(b, xb, -torch.rand(shape, device=x.device) * self.tau_span)
        return self.physics_weight * (residual.square().mean() + boundary.square().mean())


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
) -> torch.Tensor:
    """Training loop matching the paper: Adam + MSE, validation on relative
    L2 error every 20 epochs, early stopping on patience, restores best params.
    early_stopping=False trains all epochs and keeps the final params.

    augment_fn is applied to each training batch only (not validation/test).
    """
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            if hasattr(model, "physics_loss"):
                loss = loss + model.physics_loss(batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
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
