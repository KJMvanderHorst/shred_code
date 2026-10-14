import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from get_shredded.meeting_runner import METHODS, load_arrays, load_config, load_results, run_all


def _write_random_nektar(path: Path, nx: int = 6, ny: int = 4, nt: int = 40) -> None:
    xs = np.linspace(1.0, 2.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([gx.ravel(order="F"), gy.ravel(order="F")], axis=1)
    rng = np.random.default_rng(0)
    savemat(str(path), {
        "X_star": coords,
        "t": np.arange(nt, dtype=float)[:, None],
        "U_star": rng.normal(size=(coords.shape[0], 2, nt)),
        "p_star": rng.normal(size=(coords.shape[0], nt)),
    })


def test_run_all_writes_artifacts_and_skips_completed(tmp_path: Path, capsys) -> None:
    data = tmp_path / "nektar.mat"
    _write_random_nektar(data)
    root = tmp_path / "out"
    methods = list(METHODS)
    cfg = load_config([
        f"data.nektar_file={data}", f"outputs.root={root}", "model.lags=3", "train.epochs=2",
        "meeting.num_sensors=3", "meeting.qr_rank=6", "meeting.seeds=[0]", "meeting.test_draws=2",
        "meeting.panel_windows=2", f"meeting.methods=[{','.join(methods)}]",
    ])
    records = run_all(cfg, verbose=True)
    assert [r["status"] for r in records] == ["completed"] * len(methods)
    results = load_results(root)
    assert set(results) == {(m, 0) for m in methods}
    for method in methods:
        metrics = results[(method, 0)]["metrics"]
        assert np.isfinite(metrics["p0"]["err"]) and np.isfinite(metrics["p0.3"]["err"])
        arrays = load_arrays(root, method, 0)
        assert arrays["p0.3_err"].shape == (2 * 4,)  # draws x test windows
        assert ("p0.3_d_latent" in arrays) == (METHODS[method]["kind"] not in ("sdn", "pod"))
    summary = json.loads((root / "summary.json").read_text())
    assert summary["missing"] == []
    capsys.readouterr()
    again = run_all(cfg)
    assert [r["status"] for r in again] == ["skipped"] * len(methods)
