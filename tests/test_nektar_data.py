from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from get_shredded.data import load_dataset, load_nektar_data

NX, NY, NT = 6, 4, 5


def _write_nektar(path: Path, *, shuffle: bool = False) -> None:
    xs = np.linspace(1.0, 2.0, NX)
    ys = np.linspace(-1.0, 1.0, NY)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([gx.ravel(order="F"), gy.ravel(order="F")], axis=1)
    if shuffle:
        coords = coords[np.random.default_rng(0).permutation(coords.shape[0])]
    scale = np.arange(1, NT + 1, dtype=np.float64)
    # Solid-body rotation u = -y, v = x has vorticity 2 everywhere; scaled per time step.
    u = -coords[:, 1:2] * scale
    v = coords[:, 0:1] * scale
    savemat(
        str(path),
        {
            "X_star": coords,
            "t": scale[:, None],
            "U_star": np.stack([u, v], axis=1),
            "p_star": coords[:, 0:1] + coords[:, 1:2] * scale,
        },
    )


def test_nektar_vorticity_from_rotation(tmp_path: Path) -> None:
    path = tmp_path / "nektar.mat"
    _write_nektar(path)
    load_X, nx, ny = load_nektar_data(path, field="vorticity")
    assert (nx, ny) == (NY, NX)
    assert load_X.shape == (NT, NX * NY)
    assert load_X.dtype == np.float32
    expected = 2.0 * np.arange(1, NT + 1)[:, None] * np.ones((1, NX * NY))
    np.testing.assert_allclose(load_X, expected, rtol=1e-5)


@pytest.mark.parametrize(("field", "m", "ny"), [("u", NX * NY, NX), ("v", NX * NY, NX), ("p", NX * NY, NX), ("uv", 2 * NX * NY, 2 * NX)])
def test_nektar_field_shapes(tmp_path: Path, field: str, m: int, ny: int) -> None:
    path = tmp_path / "nektar.mat"
    _write_nektar(path)
    load_X, nx_out, ny_out = load_dataset("nektar", path, nektar_field=field)
    assert load_X.shape == (NT, m)
    assert (nx_out, ny_out) == (NY, ny)
    assert nx_out * ny_out == m


def test_nektar_uv_stacks_u_then_v(tmp_path: Path) -> None:
    path = tmp_path / "nektar.mat"
    _write_nektar(path)
    uv, _, _ = load_nektar_data(path, field="uv")
    u, _, _ = load_nektar_data(path, field="u")
    v, _, _ = load_nektar_data(path, field="v")
    np.testing.assert_array_equal(uv, np.concatenate([u, v], axis=1))


def test_nektar_rejects_unordered_grid(tmp_path: Path) -> None:
    path = tmp_path / "nektar.mat"
    _write_nektar(path, shuffle=True)
    with pytest.raises(ValueError, match="regular grid"):
        load_nektar_data(path)


def test_nektar_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "nektar.mat"
    _write_nektar(path)
    with pytest.raises(ValueError, match="Unsupported nektar field"):
        load_nektar_data(path, field="omega")


def test_nektar_state_follows_vortall_layout(tmp_path: Path) -> None:
    """Cross-stream y varies fastest, so reshape(nx, ny, order="F") gives rows=y, cols=x."""
    path = tmp_path / "nektar.mat"
    _write_nektar(path)
    load_X, nx, ny = load_nektar_data(path, field="v")
    field = load_X[0].reshape(nx, ny, order="F")
    xs = np.linspace(1.0, 2.0, NX)
    # v = x at t=1: constant down each column, increasing left to right.
    np.testing.assert_allclose(field, np.broadcast_to(xs, (NY, NX)), rtol=1e-6)


def test_point_coordinates_match_state_layout(tmp_path: Path) -> None:
    from get_shredded.data import nektar_point_coordinates

    path = tmp_path / "nektar.mat"
    _write_nektar(path)
    coords = nektar_point_coordinates(path)
    load_u, _, _ = load_nektar_data(path, field="u")
    load_v, _, _ = load_nektar_data(path, field="v")
    # At t=1: u = -y and v = x at every state index.
    np.testing.assert_allclose(load_u[0], -coords[:, 1], rtol=1e-6)
    np.testing.assert_allclose(load_v[0], coords[:, 0], rtol=1e-6)
