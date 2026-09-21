from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


__all__ = ["Senseiver", "spatial_positional_encoding", "sine_cosine_spatial_encoding"]


def spatial_positional_encoding(coords: Tensor, num_frequencies: int) -> Tensor:
    """Compute the sine/cosine spatial encoding used by Senseiver.

    The input coordinates have shape (..., spatial_dim). The output has shape
    (..., 2 * spatial_dim * num_frequencies) and remains differentiable with
    respect to the coordinates.
    """
    if num_frequencies <= 0:
        raise ValueError(f"num_frequencies must be positive, got {num_frequencies}.")
    if coords.dim() == 0:
        raise ValueError("coords must have at least one spatial dimension.")

    coords = torch.as_tensor(coords)
    freqs = torch.arange(1, num_frequencies + 1, device=coords.device, dtype=coords.dtype)
    scale_shape = [1] * (coords.dim() - 1) + [-1]
    scaled = coords.unsqueeze(-1) * freqs.reshape(*scale_shape)
    sin_part = torch.sin(math.pi * scaled)
    cos_part = torch.cos(math.pi * scaled)
    encoded = torch.cat([sin_part, cos_part], dim=-1)
    return encoded.reshape(*coords.shape[:-1], coords.shape[-1] * 2 * num_frequencies)


sine_cosine_spatial_encoding = spatial_positional_encoding


class _LatentAttentionBlock(nn.Module):
    """Multi-head attention block with a residual connection."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: Tensor, key_value: Tensor | None = None) -> Tensor:
        if key_value is None:
            key_value = query
        q = self.norm(query)
        k = self.norm(key_value)
        v = self.norm(key_value)
        out, _ = self.mha(q, k, v, need_weights=False)
        return self.out_proj(out) + query


class Senseiver(nn.Module):
    """Senseiver-style sparse reconstruction model for a single frame."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        spatial_dim: int = 2,
        num_latents: int = 8,
        latent_dim: int = 32,
        num_frequencies: int = 8,
        num_heads: int = 4,
        encoder_hidden_dim: int | None = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if "n_latents" in kwargs:
            num_latents = int(kwargs["n_latents"])
        if "n_heads" in kwargs:
            num_heads = int(kwargs["n_heads"])
        if "hidden_dim" in kwargs:
            latent_dim = int(kwargs["hidden_dim"])

        if encoder_hidden_dim is None:
            encoder_hidden_dim = latent_dim

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.spatial_dim = int(spatial_dim)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.num_frequencies = int(num_frequencies)
        self.num_heads = int(num_heads)
        self.positional_dim = 2 * self.spatial_dim * self.num_frequencies

        self.latent_query = nn.Parameter(torch.empty(self.num_latents, self.latent_dim))
        self.decoder_query = nn.Parameter(torch.empty(1, 1, self.latent_dim))
        nn.init.xavier_uniform_(self.latent_query)
        nn.init.xavier_uniform_(self.decoder_query)

        self.sensor_projection = nn.Linear(
            self.input_channels + self.positional_dim,
            encoder_hidden_dim,
        )
        self.sensor_to_latent = nn.Linear(encoder_hidden_dim, self.latent_dim)

        self.encoder_cross_attention = _LatentAttentionBlock(self.latent_dim, num_heads, dropout)
        self.encoder_recurrent_block = _LatentAttentionBlock(self.latent_dim, num_heads, dropout)

        self.decoder_input_projection = nn.Linear(
            self.positional_dim + self.latent_dim,
            self.latent_dim,
        )
        self.decoder_cross_attention = _LatentAttentionBlock(self.latent_dim, num_heads, dropout)
        self.output_projection = nn.Linear(self.latent_dim, self.output_channels)

    def _prepare_sensor_values(self, sensor_values: Tensor) -> Tensor:
        if sensor_values.dim() == 2:
            return sensor_values.unsqueeze(0)
        if sensor_values.dim() != 3:
            raise ValueError(
                "sensor_values must have shape (batch, Ns, input_channels) or (Ns, input_channels)."
            )
        return sensor_values

    def _prepare_coordinates(self, coordinates: Tensor, expected_dim: int, name: str) -> Tensor:
        if coordinates.dim() == 1:
            coordinates = coordinates.reshape(1, 1, expected_dim)
        elif coordinates.dim() == 2:
            coordinates = coordinates.unsqueeze(0)
        if coordinates.dim() != 3:
            raise ValueError(f"{name} must have shape (..., spatial_dim).")
        if coordinates.shape[-1] != self.spatial_dim:
            raise ValueError(f"{name} last dimension must be {self.spatial_dim}, got {coordinates.shape[-1]}.")
        return coordinates

    def encode(self, sensor_values: Tensor, sensor_coordinates: Tensor) -> Tensor:
        sensor_values = self._prepare_sensor_values(sensor_values)
        sensor_coordinates = self._prepare_coordinates(sensor_coordinates, self.spatial_dim, "sensor_coordinates")

        batch_size = sensor_values.shape[0]
        if sensor_values.shape[-1] != self.input_channels:
            raise ValueError(
                f"sensor_values last dimension must be {self.input_channels}, got {sensor_values.shape[-1]}."
            )
        if sensor_coordinates.shape[0] == 1 and batch_size > 1:
            sensor_coordinates = sensor_coordinates.expand(batch_size, -1, -1)
        if sensor_coordinates.shape[0] != batch_size:
            raise ValueError(
                f"sensor_coordinates batch mismatch: expected {batch_size}, got {sensor_coordinates.shape[0]}."
            )
        if sensor_coordinates.shape[1] not in (1, sensor_values.shape[1]):
            raise ValueError(
                "sensor_coordinates must match the number of sensors or be shared across the batch."
            )
        if sensor_coordinates.shape[1] == 1 and sensor_values.shape[1] > 1:
            sensor_coordinates = sensor_coordinates.expand(batch_size, sensor_values.shape[1], self.spatial_dim)

        sensor_pos = spatial_positional_encoding(sensor_coordinates, self.num_frequencies)
        sensor_input = torch.cat([sensor_values, sensor_pos], dim=-1)
        sensor_features = self.sensor_to_latent(self.sensor_projection(sensor_input))
        latent = self.latent_query.unsqueeze(0).expand(batch_size, -1, -1)
        latent = self.encoder_cross_attention(latent, sensor_features)
        latent = self.encoder_recurrent_block(latent)
        latent = self.encoder_recurrent_block(latent)
        return latent

    def decode(self, latent: Tensor, query_coordinates: Tensor) -> Tensor:
        query_coordinates = self._prepare_coordinates(query_coordinates, self.spatial_dim, "query_coordinates")
        batch_size = latent.shape[0]
        if query_coordinates.shape[0] == 1 and batch_size > 1:
            query_coordinates = query_coordinates.expand(batch_size, -1, -1)
        if query_coordinates.shape[0] != batch_size:
            raise ValueError(
                f"query_coordinates batch mismatch: expected {batch_size}, got {query_coordinates.shape[0]}."
            )

        query_pos = spatial_positional_encoding(query_coordinates, self.num_frequencies)
        decoder_query = self.decoder_query.expand(batch_size, query_coordinates.shape[1], -1)
        decoder_input = torch.cat([query_pos, decoder_query], dim=-1)
        decoder_tokens = self.decoder_input_projection(decoder_input)
        decoded = self.decoder_cross_attention(decoder_tokens, latent)
        return self.output_projection(decoded)

    def forward(self, sensor_values: Tensor, sensor_coordinates: Tensor, query_coordinates: Tensor) -> Tensor:
        single_example = (
            sensor_values.dim() == 2 and sensor_coordinates.dim() == 2 and query_coordinates.dim() == 2
        )

        sensor_values = self._prepare_sensor_values(sensor_values)
        sensor_coordinates = self._prepare_coordinates(sensor_coordinates, self.spatial_dim, "sensor_coordinates")
        query_coordinates = self._prepare_coordinates(query_coordinates, self.spatial_dim, "query_coordinates")

        batch_size = sensor_values.shape[0]
        if sensor_values.shape[-1] != self.input_channels:
            raise ValueError(
                f"sensor_values last dimension must be {self.input_channels}, got {sensor_values.shape[-1]}."
            )
        if sensor_coordinates.shape[0] == 1 and batch_size > 1:
            sensor_coordinates = sensor_coordinates.expand(batch_size, -1, -1)
        if sensor_coordinates.shape[0] != batch_size:
            raise ValueError(
                f"sensor_coordinates batch mismatch: expected {batch_size}, got {sensor_coordinates.shape[0]}."
            )
        if sensor_coordinates.shape[1] not in (1, sensor_values.shape[1]):
            raise ValueError(
                "sensor_coordinates must match the number of sensors or be shared across the batch."
            )
        if sensor_coordinates.shape[1] == 1 and sensor_values.shape[1] > 1:
            sensor_coordinates = sensor_coordinates.expand(batch_size, sensor_values.shape[1], self.spatial_dim)

        if query_coordinates.shape[0] == 1 and batch_size > 1:
            query_coordinates = query_coordinates.expand(batch_size, -1, -1)
        if query_coordinates.shape[0] != batch_size:
            raise ValueError(
                f"query_coordinates batch mismatch: expected {batch_size}, got {query_coordinates.shape[0]}."
            )

        latent = self.encode(sensor_values, sensor_coordinates)
        output = self.decode(latent, query_coordinates)
        if single_example:
            return output[0]
        return output
