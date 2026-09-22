from __future__ import annotations

import torch
from torch import Tensor, nn

from .senseiver import SenseiverEncoder, spatial_positional_encoding

__all__ = ["RobustSHREDv1", "RobustSHREDv2"]


class _SHREDDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, l1: int, l2: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        out = torch.relu(self.dropout(self.linear1(hidden)))
        out = torch.relu(self.dropout(self.linear2(out)))
        return self.linear3(out)


class RobustSHREDv1(nn.Module):
    """SHRED decoder with hidden-state-conditioned sensor cross-attention."""

    def __init__(
        self,
        num_sensors: int,
        output_size: int,
        sensor_coordinates: Tensor,
        hidden_size: int = 64,
        num_heads: int = 4,
        embed_dim: int = 32,
        num_frequencies: int = 8,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        coordinates = torch.as_tensor(sensor_coordinates, dtype=torch.float32)
        if coordinates.ndim != 2 or coordinates.shape[0] != num_sensors:
            raise ValueError("sensor_coordinates must have shape (num_sensors, spatial_dim).")
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        self.num_frequencies = num_frequencies
        self.spatial_dim = coordinates.shape[-1]
        positional_dim = 2 * self.spatial_dim * num_frequencies
        self.register_buffer("sensor_coordinates", coordinates)
        self.sensor_embed = nn.Linear(1 + positional_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, hidden_size)
        self.value_projection = nn.Linear(embed_dim, hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.decoder = _SHREDDecoder(hidden_size, output_size, l1, l2, dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_sensors:
            raise ValueError("x must have shape (batch, lags, num_sensors).")
        batch_size = x.shape[0]
        device = x.device
        coords = self.sensor_coordinates.to(device)
        positions = spatial_positional_encoding(coords, self.num_frequencies)
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        for timestep in range(x.shape[1]):
            values = x[:, timestep, :].unsqueeze(-1)
            embedded = self.sensor_embed(torch.cat([values, positions], dim=-1))
            keys = self.key_projection(embedded)
            sensor_values = self.value_projection(embedded)
            query = self.query_projection(hidden).unsqueeze(1)
            context, _ = self.attention(query, keys, sensor_values, need_weights=False)
            hidden = self.gru_cell(context.squeeze(1), hidden)
        return self.decoder(hidden)


class RobustSHREDv2(nn.Module):
    """SHRED decoder consuming a recurrent sequence of Senseiver latents."""

    def __init__(
        self,
        num_sensors: int,
        output_size: int,
        sensor_coordinates: Tensor,
        num_latents: int = 8,
        latent_dim: int = 32,
        num_frequencies: int = 8,
        num_heads: int = 4,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        coordinates = torch.as_tensor(sensor_coordinates, dtype=torch.float32)
        if coordinates.ndim != 2 or coordinates.shape[0] != num_sensors:
            raise ValueError("sensor_coordinates must have shape (num_sensors, spatial_dim).")
        self.num_sensors = num_sensors
        self.register_buffer("sensor_coordinates", coordinates)
        self.encoder = SenseiverEncoder(
            input_channels=1,
            spatial_dim=coordinates.shape[-1],
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_frequencies=num_frequencies,
            num_heads=num_heads,
            dropout=dropout,
        )
        latent_size = num_latents * latent_dim
        self.rnn = nn.GRU(latent_size, hidden_size, num_layers=hidden_layers, batch_first=True)
        self.decoder = _SHREDDecoder(hidden_size, output_size, l1, l2, dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_sensors:
            raise ValueError("x must have shape (batch, lags, num_sensors).")
        batch_size, lags, _ = x.shape
        values = x.reshape(batch_size * lags, self.num_sensors, 1)
        coords = self.sensor_coordinates.to(x.device)
        coords = coords.unsqueeze(0).expand(batch_size * lags, -1, -1)
        latents = self.encoder(values, coords).flatten(start_dim=1)
        latent_sequence = latents.reshape(batch_size, lags, -1)
        _, hidden = self.rnn(latent_sequence)
        return self.decoder(hidden[-1])
