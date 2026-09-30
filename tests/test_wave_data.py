from __future__ import annotations

import numpy as np

from get_shredded.wave_data import WaveGeneratorConfig, generate_wave_dataset


def test_generate_wave_dataset_shapes_and_boundaries() -> None:
    cfg = WaveGeneratorConfig(
        L=2.0,
        c=2.0,
        num_modes=6,
        num_trajectories=7,
        nx=32,
        nt=48,
        t_start=0.0,
        t_end=2.0,
        endpoint=True,
        seed=7,
    )
    payload = generate_wave_dataset(cfg)
    field = payload["field"]
    assert field.shape == (7, 48, 32)
    assert np.allclose(field[..., 0], 0.0, atol=1e-8)
    assert np.allclose(field[..., -1], 0.0, atol=1e-8)
    assert np.allclose(field[:, 0, :], field[:, -1, :], atol=1e-8, rtol=1e-8)
    assert np.isfinite(field).all()
    assert payload["x"][0] == 0.0
    assert payload["x"][-1] == cfg.L
