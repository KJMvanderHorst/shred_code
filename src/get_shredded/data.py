from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.io import loadmat

from .wave_data import load_wave_data


def load_cylinder_data(mat_path: str | Path) -> tuple[np.ndarray, int, int]:
    """Returns (load_X, nx, ny) where load_X has shape (N, m): N temporal
    snapshots of an m-dimensional state — the convention used in the paper."""
    data = loadmat(str(mat_path))
    if "VORTALL" not in data:
        raise KeyError("Expected key 'VORTALL' in .mat file")
    vortall = np.asarray(data["VORTALL"], dtype=np.float32)
    load_X = vortall.T
    nx = int(data.get("nx", [[0]])[0][0])
    ny = int(data.get("ny", [[0]])[0][0])
    return load_X, nx, ny


NEKTAR_FIELDS = ("vorticity", "u", "v", "uv", "p")


def load_nektar_data(mat_path: str | Path, field: str = "vorticity") -> tuple[np.ndarray, int, int]:
    """Load the Nektar cylinder-wake `.mat` (X_star, U_star, p_star, t).

    The file stores points on a regular grid with streamwise x varying fastest.
    The state is re-ordered to the VORTALL convention: cross-stream y varies
    fastest and the returned (nx, ny) are (cross-stream, streamwise) counts, so
    plots and grid coordinates match the CYLINDER_ALL data (flow left to right).
    Vorticity is derived from u, v with second-order finite differences. `uv`
    stacks [u; v] and reports ny doubled so plots show u and v side by side.
    Returns (load_X, nx, ny) with load_X of shape (N, m).
    """
    field = str(field).lower()
    if field not in NEKTAR_FIELDS:
        raise ValueError(f"Unsupported nektar field '{field}'. Expected one of: {', '.join(NEKTAR_FIELDS)}.")
    data = loadmat(str(mat_path))
    for key in ("X_star", "U_star", "p_star"):
        if key not in data:
            raise KeyError(f"Expected key '{key}' in nektar .mat file")
    coords = np.asarray(data["X_star"], dtype=np.float64)
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    n_stream, n_cross = xs.size, ys.size
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    if (
        n_stream * n_cross != coords.shape[0]
        or not np.allclose(coords[:, 0], grid_x.ravel(order="F"))
        or not np.allclose(coords[:, 1], grid_y.ravel(order="F"))
    ):
        raise ValueError("X_star is not a regular grid stored x-fastest; cannot map points to the grid.")

    U = np.asarray(data["U_star"], dtype=np.float64)
    n_t = U.shape[2]
    # Grids indexed (x, y, t) in physical coordinates.
    u = U[:, 0, :].reshape(n_stream, n_cross, n_t, order="F")
    v = U[:, 1, :].reshape(n_stream, n_cross, n_t, order="F")

    def to_state(grid: np.ndarray) -> np.ndarray:
        # (x, y, t) -> (m, t) with y fastest, matching reshape(nx, ny, order="F").
        return grid.transpose(1, 0, 2).reshape(n_cross * n_stream, n_t, order="F")

    nx, ny = n_cross, n_stream
    if field == "vorticity":
        dv_dx = np.gradient(v, xs, axis=0, edge_order=2)
        du_dy = np.gradient(u, ys, axis=1, edge_order=2)
        state = to_state(dv_dx - du_dy)
    elif field == "u":
        state = to_state(u)
    elif field == "v":
        state = to_state(v)
    elif field == "uv":
        state = np.concatenate([to_state(u), to_state(v)], axis=0)
        ny = 2 * ny
    else:
        p = np.asarray(data["p_star"], dtype=np.float64).reshape(n_stream, n_cross, n_t, order="F")
        state = to_state(p)
    return state.T.astype(np.float32), int(nx), int(ny)


def nektar_point_coordinates(mat_path: str | Path) -> np.ndarray:
    """Physical (x, y) of each grid point in the state order of :func:`load_nektar_data`.

    Returns (n_points, 2). For field ``uv`` the state holds two blocks of
    n_points (u then v) that share these coordinates.
    """
    coords = np.asarray(loadmat(str(mat_path))["X_star"], dtype=np.float64)
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    # (x, y) grid -> y fastest, matching to_state() in load_nektar_data.
    return np.stack([grid_x.T.reshape(-1, order="F"), grid_y.T.reshape(-1, order="F")], axis=1)


def load_dataset(
    data_source: str,
    data_path: str | Path,
    *,
    trajectory_index: int = 0,
    nektar_field: str = "vorticity",
) -> tuple[np.ndarray, int, int]:
    """Load a raw dataset and return a training-ready (N_time, N_state) matrix.

    Supported sources are the cylinder-vortex `.mat` dataset, the Nektar
    cylinder-wake `.mat` dataset and the synthetic 1D wave `.npz` prototype. The downstream model code stays neutral to the
    dataset origin as long as the raw state matrix follows the repo contract.
    """
    source = str(data_source).lower()
    if source in {"cylinder", "cylinder_data", "mat", "cylinder_mat"}:
        return load_cylinder_data(data_path)
    if source == "nektar":
        return load_nektar_data(data_path, field=nektar_field)
    if source in {"wave", "wave_1d", "synthetic_wave", "synthetic"}:
        return load_wave_data(data_path, trajectory_index=trajectory_index)
    raise ValueError(
        f"Unsupported data_source='{data_source}'. Expected one of: "
        "cylinder, nektar, wave."
    )


def qr_place(data_matrix: np.ndarray, num_sensors: int) -> tuple[np.ndarray, np.ndarray]:
    """QR-pivoting sensor selection on the rank-`num_sensors` POD basis.

    data_matrix: (m, N) — m-dimensional state, N snapshots.
    Returns (sensor_locs, U_r) with U_r the leading `num_sensors` POD modes.
    """
    u, _, _ = np.linalg.svd(data_matrix, full_matrices=False)
    U_r = u[:, :num_sensors]
    _, _, pivot = scipy.linalg.qr(U_r.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, U_r


def build_sensor_windows(
    transformed_X: np.ndarray, sensor_locations: np.ndarray, lags: int
) -> np.ndarray:
    """transformed_X: (N, m). Returns (N - lags, lags, num_sensors)."""
    n = transformed_X.shape[0]
    num_sensors = len(sensor_locations)
    out = np.zeros((n - lags, lags, num_sensors), dtype=np.float32)
    for i in range(n - lags):
        out[i] = transformed_X[i : i + lags, sensor_locations]
    return out


def qrpod_reconstruct(
    sensor_measurements: np.ndarray, sensor_locations: np.ndarray, U_r: np.ndarray, m: int
) -> np.ndarray:
    """Linear QR/POD reconstruction: x_hat = U_r (C U_r)^{-1} y.

    sensor_measurements: (T, num_sensors) in the original (unscaled) units.
    Returns (T, m).
    """
    num_sensors = len(sensor_locations)
    C = np.zeros((num_sensors, m), dtype=U_r.dtype)
    for i in range(num_sensors):
        C[i, sensor_locations[i]] = 1.0
    return (U_r @ np.linalg.inv(C @ U_r) @ sensor_measurements.T).T


def qr_place_probes(
    data_matrix: np.ndarray, num_locations: int, rank: int, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Block QR pivoting: choose grid points for multi-channel probes.

    data_matrix: (C * n_points, N) — channel blocks stacked, e.g. [u; v].
    A probe at a grid point measures every channel there, so pivoting runs over
    points: each point is the block of its C columns of U_r^T, and at every
    step the point whose residual block spans the largest volume
    (sqrt det of its Gram matrix) is chosen, then its block is projected out.
    This is the greedy volume criterion of QR column pivoting applied to whole
    probes; with C == 1 it reduces to standard pivoted QR (see :func:`qr_place`).
    Returns (locations in [0, n_points), U_r).
    """
    u, _, _ = np.linalg.svd(data_matrix, full_matrices=False)
    U_r = u[:, :rank]
    channels = U_r.shape[0] // n_points
    # residual[:, c, p] = column of U_r^T for channel c at point p.
    residual = U_r.T.reshape(rank, channels, n_points).astype(np.float64).copy()
    available = np.ones(n_points, dtype=bool)
    locations: list[int] = []
    for _ in range(num_locations):
        gram = np.einsum("icp,idp->pcd", residual, residual)
        # det for C <= rank remaining; fall back to the trace once the residual
        # space is too small to hold a full block.
        score = np.linalg.det(gram) if channels * (len(locations) + 1) <= rank else np.trace(gram, axis1=1, axis2=2)
        score[~available] = -np.inf
        point = int(np.argmax(score))
        locations.append(point)
        available[point] = False
        q, _ = np.linalg.qr(residual[:, :, point])
        residual -= np.einsum("ic,jc,jdp->idp", q, q, residual)
    return np.asarray(locations), U_r


def gappy_pod_reconstruct(
    sensor_measurements: np.ndarray,
    missing: np.ndarray | None,
    sensor_state_indices: np.ndarray,
    U_r: np.ndarray,
) -> np.ndarray:
    """Gappy POD using only the available measurements of each sample.

    Per sample: x_hat = U_r' (C_m U_r')^+ y_m (least squares), where C_m keeps
    the available rows and r' = min(r, number of available measurements).
    sensor_measurements: (T, M) in unscaled units; missing: bool (T, M) or None
    (True = measurement unavailable); sensor_state_indices: (M,) state index of
    each measurement row. Returns (T, m). With nothing missing and M == r this
    equals :func:`qrpod_reconstruct`.
    """
    y = np.asarray(sensor_measurements, dtype=np.float64)
    basis = np.asarray(U_r, dtype=np.float64)
    CU = basis[np.asarray(sensor_state_indices)]
    if missing is None:
        missing = np.zeros(y.shape, dtype=bool)
    rank = basis.shape[1]
    out = np.empty((y.shape[0], basis.shape[0]), dtype=np.float64)
    patterns, inverse = np.unique(missing, axis=0, return_inverse=True)
    for pattern_id, pattern in enumerate(patterns):
        rows = np.flatnonzero(inverse.ravel() == pattern_id)
        available = ~pattern
        r_eff = min(rank, int(available.sum()))
        coef, *_ = np.linalg.lstsq(CU[available][:, :r_eff], y[rows][:, available].T, rcond=None)
        out[rows] = (basis[:, :r_eff] @ coef).T
    return out.astype(np.asarray(sensor_measurements).dtype, copy=False)
